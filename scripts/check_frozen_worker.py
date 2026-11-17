"""Exercise redirected std handles and cancellation in the built windowless EXE."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from threading import Event, Thread
import time

from ironasu.execution import run_command

seen = Event()
release = Event()
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        if self.server.stall:
            self.send_header("Content-Length", "500000")
            self.end_headers()
            self.wfile.write(b"{")
            self.wfile.flush()
            seen.set()
            release.wait(5)
        else:
            body = json.dumps({"choices": [{"message": {"content": "frozen worker"}}]}).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
server.daemon_threads = True
server.stall = False
thread = Thread(target=server.serve_forever, daemon=True)
thread.start()
exe = Path(sys.argv[1]).resolve(strict=True)
request = json.dumps({"settings": {
    "endpoint": f"http://127.0.0.1:{server.server_port}/v1",
    "model": "test", "api_key": "", "timeout_s": 10},
    "messages": [], "tools": []}).encode()
try:
    argv = [str(exe), "--ironasu-model-worker"]
    result = run_command(argv, cwd=exe.parent, timeout=10, input_bytes=request)
    assert result.passed, result
    assert json.loads(result.output)["message"]["content"] == "frozen worker", result
    server.stall = True
    stop = Event()
    cancel = Thread(target=lambda: (seen.wait(5), stop.set()))
    cancel.start()
    before = time.monotonic()
    result = run_command(argv, cwd=exe.parent, timeout=10, input_bytes=request, stop=stop)
    cancel.join(timeout=6)
    assert seen.is_set() and result.reason == "cancelled", result
    assert time.monotonic() - before < 3
    print("Frozen HTTP worker completion and in-flight cancellation passed")
finally:
    release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)
