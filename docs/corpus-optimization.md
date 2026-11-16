# Parallel corpus optimisation — ironasu Studio 1.6.1

This release optimises the native **エージェント ×16** corpus analysis tab. Existing 4/16 partitions, 1–16 lightweight workers and model-call admission limits are retained.

## Changes

1. **Adaptive lossless subchunks:** ordered 4/16 partitions are subdivided into at most 8,000 characters. Regions containing multiple headings, issue or warning markers are broken into approximately 4,096-character segments. All characters are retained, in original order. Importance signals are heuristics, not factual judgements.
2. **In-run exact single-flight reuse:** identical segment text in the same run uses a shared result, including concurrent in-flight work. Different text is never skipped through approximate matching. Failed shared requests remain failed. The cache is in-memory and discarded after each run; objectives/models cannot leak across runs.
3. **Bounded hierarchical map/reduce:** each segment is mapped independently, then up to 4 findings are reduced per request with a maximum 9,000-character input budget. Partition summaries are similarly merged in a hierarchy instead of an unbounded 16-way prompt. Oversize outputs cause explicit failures, not silent truncation.
4. **Traceable shared evidence:** each original segment occurrence gets a frozen reference containing partition, segment, original character start/end offsets, SHA-256 and model-produced finding. Model summaries are unverified, even when accompanied by hashes. Cross-agent sharing occurs through the final reviewer, not through unrestricted cross-worker prompts.
5. **Measured call accounting:** exported reports and the desktop UI record total_segments, unique_segments, cache_hits, model_calls and evidence. Cache hits count avoided duplicate **map requests**, not guaranteed GPU speedup. Model calls include maps, reductions and synthesis.

## Runtime and safety

Sixteen agents are **not** sixteen GPU model instances. Model call slots default to 2 and are capped in the UI by IRONASU_MAX_CONCURRENCY. Benchmark one, two and four model slots before raising the cap on a single Radeon GPU.

This pipeline is read-only. Corpus text and summaries are untrusted data, not instructions; no proposed tools or shell commands are executed. Cancel stops new stages, but an existing inference request can remain in flight until the backend returns. Failed partitions never receive a misleading successful final report.

**Tradeoff:** Smaller high-density chunks can mean *more* calls, while the exact-match single-flight and hierarchical reduction can reduce repeated work and prevent overlong synthesis prompts. Semantic compression remains lossy; check original source spans for high-stakes conclusions.

## Test

Run: python -m unittest tests.test_corpus_agents -v

Tests cover lossless 4/16 splits, dense section segmentation, sixteen worker / two model slot limits, deduplication, evidence spans and SHA, bounded synthesis, run isolation, cancellation and failure.
