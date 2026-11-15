"""Isolated SQLite and fake-runner checks for scheduler safety and shared memory."""
import concurrent.futures
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ironasu.orchestrator import LimitedBackend, Orchestrator, WorkerResult


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **kw):
        return Orchestrator(self.root, max_agents=4, max_active_inference=2,
                            max_parallel_tools=2, lease_seconds=4, **kw)

    def test_dependency_order_and_parallel_execution(self):
        observed = []
        lock = threading.Lock()

        def runner(agent, tid, prompt):
            with lock:
                observed.append((tid, 'start'))
            time.sleep(0.025)
            with lock:
                observed.append((tid, 'end'))
            return f'completed {tid}'

        o = self.build(runner=runner)
        o.submit([{'id': 'inspect', 'prompt': 'inspect'},
                  {'id': 'design', 'prompt': 'design'},
                  {'id': 'build', 'prompt': 'build', 'depends_on': ['inspect', 'design']},
                  {'id': 'verify', 'prompt': 'verify', 'depends_on': ['build']}])
        result = o.run()
        self.assertEqual({x['state'] for x in result}, {'complete'})
        self.assertLess(observed.index(('inspect', 'end')), observed.index(('build', 'start')))
        self.assertLess(observed.index(('design', 'end')), observed.index(('build', 'start')))
        self.assertLess(observed.index(('build', 'end')), observed.index(('verify', 'start')))
        self.assertEqual(o.status()['counts']['complete'], 4)

    def test_atomic_submission_unknown_dependency_and_cycle(self):
        o = self.build(runner=lambda a, b, c: 'ok')
        with self.assertRaisesRegex(ValueError, 'cycle'):
            o.submit([{'id': 'a', 'prompt': 'a', 'depends_on': ['b']},
                      {'id': 'b', 'prompt': 'b', 'depends_on': ['a']}])
        self.assertEqual(o.status()['tasks'], [])
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            o.submit([{'id': 'a', 'prompt': 'a', 'depends_on': ['missing']}])
        self.assertEqual(o.status()['tasks'], [])

    def test_task_idempotency_and_conflict(self):
        o = self.build(runner=lambda a, b, c: 'ok')
        task = {'id': 'a', 'prompt': 'a', 'idempotency_key': 'request-a'}
        self.assertEqual(o.submit([task]), ['a'])
        self.assertEqual(o.submit([task]), ['a'])
        self.assertEqual(len(o.status()['tasks']), 1)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            o.submit([{**task, 'prompt': 'different'}])
        with self.assertRaisesRegex(ValueError, 'Idempotency'):
            o.submit([{'id': 'b', 'prompt': 'b', 'idempotency_key': 'request-a'}])

    def test_work_limit_and_multiple_iterations(self):
        o = self.build(runner=lambda a, tid, prompt: 'done')
        o.submit([{'id': f't{i}', 'prompt': 'p'} for i in range(8)])
        self.assertEqual(len(o.run(max_tasks=3)), 3)
        self.assertEqual(o.status()['counts']['pending'], 5)
        self.assertEqual(len(o.run()), 5)
        self.assertEqual(o.status()['counts']['complete'], 8)

    def test_failed_dependency_blocks_downstream(self):
        o = self.build(runner=lambda a, tid, p: WorkerResult('failed', 'failure') if tid == 'a' else 'ok')
        o.submit([{'id': 'a', 'prompt': 'a'},
                  {'id': 'b', 'prompt': 'b', 'depends_on': ['a']},
                  {'id': 'c', 'prompt': 'c', 'depends_on': ['b']}])
        o.run()
        self.assertEqual(o.status()['counts'].get('failed'), 1)
        self.assertEqual(o.status()['counts'].get('blocked'), 2)

    def test_repair_unblocks_transitive_dependents(self):
        o = self.build(runner=lambda a, tid, p: WorkerResult('failed', 'bad') if tid == 'a' else 'ok')
        o.submit([{'id': 'a', 'prompt': 'a'},
                  {'id': 'b', 'prompt': 'b', 'depends_on': ['a']},
                  {'id': 'c', 'prompt': 'c', 'depends_on': ['b']}])
        o.run()
        self.assertEqual(o.status()['counts']['blocked'], 2)
        o.resolve('a', action='complete')
        self.assertEqual(o.status()['counts'].get('pending'), 2)
        self.assertEqual(len(o.run()), 2)

    def test_shared_event_diff_and_ack(self):
        o = self.build(runner=lambda a, b, c: 'ok')
        ev1 = o.publish('code.changed', 'changed scheduler', 'full diff', critical=True)
        ev2 = o.publish('code.changed', 'changed cache', 'full diff2')
        page = o.updates('agent-01')
        self.assertEqual([x['id'] for x in page['events']], [ev1, ev2])
        self.assertEqual(o.critical_status()[0]['unacked'], 4)
        self.assertEqual(o.detail(ev1)['body'], 'full diff')
        o.ack('agent-01', ev1)
        self.assertEqual(o.updates('agent-01')['events'][0]['id'], ev2)
        self.assertEqual(o.critical_status()[0]['unacked'], 3)
        with self.assertRaises(ValueError):
            o.ack('agent-01', 0)
        with self.assertRaises(ValueError):
            o.ack('agent-01', 500)
        # A different worker cannot acknowledge an event it has never fetched.
        with self.assertRaises(ValueError):
            o.ack('agent-02', ev2)

    def test_event_idempotency(self):
        o = self.build()
        first = o.publish('state.change', 'new state', 'body', idempotency_key='once')
        self.assertEqual(first, o.publish('state.change', 'new state', 'body', idempotency_key='once'))
        with self.assertRaises(ValueError):
            o.publish('state.change', 'new state2', 'body', idempotency_key='once')

    def test_backlog_pauses_without_false_ack(self):
        calls = []
        o = self.build(context_chars=1024, runner=lambda a, b, c: calls.append(b) or 'ok')
        for i in range(7):
            o.publish('event', 'x' * 390 + str(i), 'payload')
        o.submit([{'id': 'job', 'prompt': 'do job'}])
        results = o.run()
        self.assertEqual(results[0]['state'], 'paused')
        self.assertEqual(calls, [])
        self.assertEqual(o.updates('agent-01')['from'], 0)

    def test_crash_lease_needs_operator_resolution(self):
        o = self.build(runner=lambda a, b, c: 'ok')
        o.submit([{'id': 'job', 'prompt': 'do job'}])
        claim = o._claim()
        self.assertIsNotNone(claim)
        with o._connect() as con:
            con.execute('UPDATE tasks SET lease_until=? WHERE id=?', (time.time() - 20, 'job'))
        self.assertEqual(o.recover_expired(), 1)
        self.assertFalse(o._complete(claim, WorkerResult('complete', 'zombie success')))
        self.assertEqual(o.status()['counts'].get('uncertain'), 1)
        self.assertEqual(o.run(), [])
        o.resolve('job', action='retry')
        self.assertEqual(o.run()[0]['state'], 'complete')

    def test_uncertain_failure_no_blind_retry(self):
        def runner(a, b, c):
            raise RuntimeError('unknown whether write happened')
        o = self.build(runner=runner)
        o.submit([{'id': 'job', 'prompt': 'work'}])
        self.assertEqual(o.run()[0]['state'], 'uncertain')
        self.assertEqual(o.run(), [])
        o.resolve('job', action='complete')
        self.assertEqual(o.status()['counts']['complete'], 1)

    def test_bounded_model_concurrency(self):
        gate = threading.BoundedSemaphore(2)
        lock = threading.Lock()
        active, maximum = [0], [0]

        class Backend:
            def complete(self, messages, tools):
                with lock:
                    active[0] += 1
                    maximum[0] = max(maximum[0], active[0])
                time.sleep(0.015)
                with lock:
                    active[0] -= 1
                return {'role': 'assistant', 'content': 'ok'}

        model = LimitedBackend(Backend(), gate)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: model.complete([], []), range(12)))
        self.assertEqual(len(results), 12)
        self.assertEqual(maximum[0], 2)

    def test_runner_may_not_claim_consumed_updates_on_resume(self):
        o = self.build(runner=lambda a, b, c: WorkerResult('complete', 'done', ack_events=False))
        o.publish('critical', 'must be seen', critical=True)
        o.submit([{'id': 'job', 'prompt': 'work'}])
        result = o.run()
        self.assertEqual(o.updates(result[0]['agent'])['from'], 0)

    def test_cursor_moves_only_on_completed_task(self):
        o = self.build(runner=lambda a, b, c: WorkerResult('uncertain', 'not committed'))
        ev = o.publish('important', 'must review', critical=True)
        o.submit([{'id': 'a', 'prompt': 'a'}])
        result = o.run()
        self.assertEqual(result[0]['state'], 'uncertain')
        self.assertEqual(o.updates(result[0]['agent'])['from'], 0)
        o2 = self.build(runner=lambda a, b, c: 'pass')
        o2.resolve('a', action='retry')
        result2 = o2.run()
        self.assertEqual(result2[0]['state'], 'complete')
        self.assertGreaterEqual(o2.updates(result2[0]['agent'])['from'], ev)

    def test_real_harness_end_to_end_with_both_gates(self):
        counters = {'model': 0, 'tool': 0, 'peak_model': 0, 'peak_tool': 0}
        lock = threading.Lock()

        class FakeBackend:
            def __init__(self, settings):
                pass

            def complete(self, messages, specs):
                with lock:
                    counters['model'] += 1
                    counters['peak_model'] = max(counters['peak_model'], counters['model'])
                time.sleep(0.018)
                with lock:
                    counters['model'] -= 1
                if any(m.get('role') == 'tool' for m in messages):
                    return {'role': 'assistant', 'content': 'done'}
                return {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'call-1', 'function': {'name': 'list_files',
                                               'arguments': '{"path":"."}'}}]}

        def fake_tool(tools, name, args):
            with lock:
                counters['tool'] += 1
                counters['peak_tool'] = max(counters['peak_tool'], counters['tool'])
            time.sleep(0.04)
            with lock:
                counters['tool'] -= 1
            return 'ok'

        o = Orchestrator(self.root, max_agents=16, max_active_inference=2,
                         max_parallel_tools=4, lease_seconds=4)
        o.submit([{'id': f'agent-task-{i:02}', 'prompt': f'Work {i}'} for i in range(16)])
        with patch('ironasu.orchestrator.OpenAIBackend', FakeBackend), \
             patch('ironasu.harness.Tools.execute', fake_tool):
            output = o.run()
        self.assertEqual(len(output), 16)
        self.assertEqual({r['state'] for r in output}, {'complete'})
        self.assertLessEqual(counters['peak_model'], 2)
        self.assertLessEqual(counters['peak_tool'], 4)
        self.assertEqual(o.status()['counts']['complete'], 16)

    def test_persistent_across_instances(self):
        o1 = self.build(runner=lambda a, b, c: 'complete')
        o1.submit([{'id': 'a', 'prompt': 'durable'}])
        o2 = self.build(runner=lambda a, b, c: 'complete')
        self.assertEqual(len(o2.run()), 1)
        self.assertEqual(o1.status()['counts']['complete'], 1)


if __name__ == '__main__':
    unittest.main()
