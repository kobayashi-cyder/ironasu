"""Correct-by-construction *greedy* speculative decoding for paged models.

Only target-model argmax tokens are emitted. Draft tokens are untrusted until
compared with target logits. No sampled decoding or cross-request KV sharing.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, Protocol, Sequence

from .scheduling import AdaptiveSpeculationScheduler, NgramDraftIndex


@dataclass(frozen=True)
class Verification:
    accepted: int
    emitted: tuple[int, ...]


def verify_greedy(draft: Sequence[int], target: Sequence[int]) -> Verification:
    """target[i] predicts the token at draft[i], target[-1] is the bonus.

    For an early mismatch, emit matching draft prefix and the TARGET's next
    token. The unchecked remainder is never emitted or reused.
    """
    if len(target) != len(draft) + 1:
        raise ValueError("target must contain one next-token prediction per draft token plus bonus")
    if any(type(x) is not int or x < 0 for x in (*draft, *target)):
        raise ValueError("token ids must be nonnegative integers")
    for i, candidate in enumerate(draft):
        if candidate != target[i]:
            return Verification(i, tuple(draft[:i]) + (target[i],))
    return Verification(len(draft), tuple(draft) + (target[-1],))


class GreedyTarget(Protocol):
    def __call__(self, context: tuple[int, ...], draft: tuple[int, ...]) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class DecodeReport:
    new_tokens: tuple[int, ...]
    target_calls: int
    draft_calls: int
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    speculative_policy: dict


class GreedySpeculativeDecoder:
    """Per-request speculative target verifier and bounded N-gram proposer.

    The target must return EXACT greedy predictions, not probabilities or
    assistant predictions. No autoregressive temperature/sampling support.
    """

    def __init__(
        self, target: GreedyTarget,
        scheduler: AdaptiveSpeculationScheduler | None = None,
        index: NgramDraftIndex | None = None,
        synchronize: Callable[[], None] | None = None,
    ):
        self.target = target
        self.scheduler = scheduler or AdaptiveSpeculationScheduler(
            draft_lengths=(2, 4, 6, 8), min_samples=3)
        self.index = index or NgramDraftIndex(order=4, max_draft=8)
        self.synchronize = synchronize or (lambda: None)

    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: Iterable[int] = (),
        memory_headroom: bool = True,
        queue_busy: bool = False,
    ) -> DecodeReport:
        prompt = tuple(prompt_ids)
        if not prompt or any(type(x) is not int or x < 0 for x in prompt):
            raise ValueError("prompt_ids must be a nonempty sequence of nonnegative integers")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be between 1 and 1024")
        eos = frozenset(eos_token_ids)
        if any(type(x) is not int or x < 0 for x in eos):
            raise ValueError("eos_token_ids must contain nonnegative integers")
        # The draft index is session-local. No candidate tokens are added here.
        history = list(prompt)
        self.index.remember(history)
        emitted: list[int] = []
        draft_calls = accepted = proposed = target_calls = 0
        while len(emitted) < max_new_tokens:
            choice = self.scheduler.choose(
                backend_supports_speculation=True, queue_busy=queue_busy,
                memory_headroom=memory_headroom)
            remaining = max_new_tokens - len(emitted)
            # A target pass can emit k+1 tokens. Restrict draft to remaining-1.
            limit = min(choice.draft_tokens, max(0, remaining - 1))
            proposal = self.index.propose(history[-self.index.order:], limit) if limit else ()
            self.synchronize()
            started = perf_counter()
            target_predictions = self.target(tuple(history), proposal)
            self.synchronize()
            elapsed = perf_counter() - started
            target_calls += 1
            result = verify_greedy(proposal, target_predictions)
            # Proposals have already been verified here. Never count speculative
            # tokens as accepted before this check.
            if proposal:
                draft_calls += 1
                proposed += len(proposal)
                accepted += result.accepted
                if len(proposal) in self.scheduler.draft_lengths:
                    self.scheduler.record_speculative(
                        len(proposal), result.accepted, elapsed, len(result.emitted))
                else:
                    # An uncalibrated partial draft is neither a target-only
                    # baseline nor an observation for a configured draft arm.
                    pass
            else:
                self.scheduler.record_baseline(elapsed, 1)
            # Stop at EOS; never append tokens beyond EOS even if the verifier
            # returned a longer accepted draft prefix.
            finished = False
            for token in result.emitted[:remaining]:
                history.append(token)
                emitted.append(token)
                if token in eos:
                    finished = True
                    break
            # Update recent index after VERIFIED tokens only, bounded workload.
            span = self.index.order + self.index.max_draft + 1
            self.index.remember(history[-span:])
            if finished:
                break
        return DecodeReport(tuple(emitted), target_calls, draft_calls,
                            accepted, proposed, self.scheduler.snapshot())


class PagedTorchGreedyTarget:
    """Hugging Face forward verifier; KV reuse when safe, full replay otherwise.

    The pager.session() context must remain active while this object is used.
    This class never persists KV across requests or uses cache from other models.
    Unsupported cache structures automatically trigger a full-prefix replay.
    """

    def __init__(self, model: object, torch: object, device: str):
        self.model = model
        self.torch = torch
        self.device = device
        self.cache = None
        self.processed_prefix: tuple[int, ...] = ()
        self.cache_reuse_count = 0
        self.cache_fallback_count = 0

    def _invalidate(self) -> None:
        self.cache = None
        self.processed_prefix = ()

    def synchronize(self) -> None:
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def __call__(self, context: tuple[int, ...], draft: tuple[int, ...]) -> tuple[int, ...]:
        if not context:
            raise ValueError("empty context")
        # KV contains processed tokens up to but excluding the last emitted one.
        reuse = (self.cache is not None and self.processed_prefix == context[:-1])
        if reuse:
            suffix = context[-1:] + draft
            try:
                seq_length = int(self.cache.get_seq_length())
                if seq_length != len(self.processed_prefix):
                    reuse = False
            except (AttributeError, TypeError, ValueError, RuntimeError):
                reuse = False
        if not reuse:
            self._invalidate()
            suffix = context + draft
            self.cache_fallback_count += 1
        else:
            self.cache_reuse_count += 1
            # Sliding window / linear attention must record history for crop.
            if hasattr(self.cache, "activate_past_recording"):
                try:
                    self.cache.activate_past_recording()
                except (RuntimeError, ValueError):
                    self._invalidate()
                    suffix = context + draft
                    reuse = False
                    self.cache_fallback_count += 1

        ids = self.torch.tensor([suffix], dtype=self.torch.long, device=self.device)
        total_length = len(context) + len(draft)
        mask = self.torch.ones((1, total_length), dtype=self.torch.long, device=self.device)
        kwargs = {"input_ids": ids, "attention_mask": mask, "use_cache": True}
        if reuse:
            kwargs["past_key_values"] = self.cache
        with self.torch.inference_mode():
            output = self.model(**kwargs)
        logits = output.logits
        wanted = len(draft) + 1
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < wanted:
            self._invalidate()
            raise RuntimeError("target did not provide all causal verification logits")
        predictions = tuple(int(x) for x in logits[0, -wanted:].argmax(dim=-1).tolist())
        verified = verify_greedy(draft, predictions)
        # Retain only the processed prefix. The final emitted token is NOT
        # processed yet and must enter the next forward invocation.
        keep_length = len(context) + verified.accepted
        new_cache = getattr(output, "past_key_values", None)
        try:
            if new_cache is None or not hasattr(new_cache, "crop"):
                raise RuntimeError("cache does not support safe rollback")
            length = int(new_cache.get_seq_length())
            if length != total_length or keep_length > length:
                raise RuntimeError("inconsistent KV cache length")
            discard = length - keep_length
            if discard:
                new_cache.crop(-discard)
                if int(new_cache.get_seq_length()) != keep_length:
                    raise RuntimeError("KV cache did not crop to accepted prefix")
            self.cache = new_cache
            self.processed_prefix = context + draft[:verified.accepted]
        except (AttributeError, TypeError, ValueError, RuntimeError):
            self._invalidate()
        return predictions
