# Independent scheduler and simulation

`model.py` supplies the compute/memory crossing and graph catalogue; `scheduler.py`
handles admission/completion; `controller.py` manages pool allocation;
`simulator.py` runs a deterministic event clock.

The CPU implementation recreates separate queues, graph buckets, adaptive wait/depth,
deadline dispatch, long-prefill chunks, and pool control from the
[original paper](https://arxiv.org/html/2601.11589v1). The released
[SGLang scheduler](https://github.com/Jianshu-She/LAPS/blob/main/docs/laps_scheduler.md)
differs in some policies; cluster runs use that runtime rather than injecting
the independent simulator code.

## Assumptions

Times are seconds. Compute and memory estimates are summed. Batch scaling uses
exposed hypothetical exponents, without an extra mixed-batch penalty. Supply
profiled coefficients via `--latency-model file.json` before drawing hardware
conclusions. Automatic runtime fitting is not implemented.

```json
{
  "alpha": 2e-9,
  "beta": 1e-6,
  "gamma_write": 1.5e-6,
  "gamma_read": 4e-7,
  "eager_launch": 0.00015,
  "graph_launch": 0.00002,
  "compute_batch_exponent": 0.45,
  "memory_batch_exponent": 0.65
}
```

The default cutoff is 256 new tokens. `classify_by_model: true` enables a
history-dependent crossing. Long requests retain their original class through
their final chunk and accumulate cached history as chunks complete.

`kv_token_budget` counts history and new tokens within a worker's admitted batch,
including graph padding/dummy positions. It is a token proxy, not a byte-accurate
global cache allocator. Queued contexts are assumed externally available.
Oversized individual contexts are rejected. Eager simulation is ragged by
default; `eager_padding` accounts for rectangular input padding.

AWD state is per worker and bucket. Underfilled dispatches reduce depth; the
literal algorithm supplies no recovery rule, so sparse workloads may settle at
depth one. Deadline/age checks override the minimum wait. Urgency and aging
supplement short-first single-worker arbitration, an explicit choice where the
paper is underspecified.

Pool pressure uses P90 signals, cooldown, minimum allocations, and one migration
per decision. Global backlog is apportioned per pool; utilization is cumulative.
A difference/absolute-baseline comparison prevents false switches when both
pressures are negative. Busy workers drain before reassignment; at most one
move is pending.

## Traces and metrics

```json
{"id":"r1","arrival":0.0,"tokens":32,"history":512,"deadline":0.4}
```

Times are relative to the run origin. `tokens` counts new input; `history` counts
cached input. Omit deadlines or use `null` for offline work. IDs must be unique.
Requests are independent, with no inferred prior-turn dependencies.

Latency is completion minus arrival, including queue/chunk waits; CPU latency
is a TTFT proxy without decode or transport. Percentiles use nearest rank.
SLO rates count only deadline-bearing requests and are `null` otherwise.
Throughput includes arrival span plus final drain; low-rate runs can be
arrival-limited. Queue time subtracts service across all chunks. Padding metrics
count new-token positions only.

## Reference GPU experiment

The optional Transformers backend runs small pretrained Qwen first-turn prefill
through PyTorch SDPA and predicts one token. It rejects cached input history.
Persistent KV caching, chunking, HTTP serving, PD transfer, and spatial allocation
belong to the simulator or official runtime.

CUDA Graph inputs use fixed addresses, explicit masks, and left-padding. Capture
checks eager logits, changed-input replay, and unpadded predictions. Only accepted
shapes remain eligible; capture limits include a soft memory threshold.

Model download, graph initialization, and warmup precede timing. Measured service
includes packing, copies, GPU execution, synchronization, and token selection.
Policies share seeded token sequences/arrivals. This is a mechanism smoke test,
not an SGLang performance baseline.
