# Adaptive, bounded stage prefetch (experimental)

Default mode is unchanged: prefetch=True performs at most one queued CPU stage read-ahead; prefetch=False disables it.

Opt in to adaptive gating:
    ironasu-stream LOCAL_MODEL_DIR --prompt "..." --prefetch --prefetch-mode adaptive

The gate records used versus wasted reads, wall-time spent reading and waiting, and a heuristic estimate of overlap. It samples 16 prefetch outcomes. If more than 25% were wasted or less than 10% of reading appears overlapped, it suspends the next 8 speculative submissions, then probes again. No model weights are approximated or skipped, and the actual forward is executed normally. At most one read-ahead task can be in flight.

The prefetched tensor data is subject to the existing LayerPager cache and residency rules. Policy state is bounded (16 observations). This feature does not predict sparse expert routing; MoE expert paging is unchanged.

Metrics: pager.snapshot()["prefetch_scheduler"] includes considered, submitted, skipped, used, wasted, pauses, read_seconds, wait_seconds, overlap_seconds. Beware: reading from OS page cache is not physical SSD IO. Overlap is only a heuristic, not measured end-to-end performance improvement.

The existing tuner should compare fixed/adaptive/no prefetch with the same prompt, target device and output. Since runtime adaptive gating modifies a timing tradeoff, it is strictly opt-in. The inference pathway and correctness remain identical even if the gate suspends speculative I/O.
