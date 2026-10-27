# LAPS prefill scheduling

`--scheduler laps` selects length-aware prefill scheduling from
[LAPS](https://arxiv.org/abs/2601.11589), Sections 2.1 and 3.1-3.2.
Short and long requests use separate queues. Short requests form padded,
length-bucketed batches with CUDA Graph execution. Long requests execute one
request per instance in fixed-size chunks. The implementation uses native
SGLang weights and KV slots through the shared asynchronous prefill backend.

## Scheduling

The default short-prefill boundary is 256 new tokens, as in the
[released implementation](https://github.com/Jianshu-She/LAPS/tree/aa29962d501236ecd2a0c7a8470c5977df3fe057).
`boundary_coefficients` optionally supplies the independently fitted
`[alpha, beta, gamma_w, gamma_r]` costs from Section 2.1. The default threshold
is an explicit configuration value, not a measured hardware boundary.

Adaptive Wait-Depth batching maintains a waiting window and target batch depth
for each instance. Requests are selected bucket-first under the KV and short
batch token budgets. A batch dispatches when it reaches its target depth or
token budget, its waiting window expires, its oldest request reaches the maximum
queue wait, or its deadline slack reaches the safety threshold. After dispatch,
a full batch updates the waiting window using its fill time; an underfilled
batch reduces the target depth. Deadline-aware waiting uses the tighter of the
remaining deadline budget and estimated time to fill the graph shape.

Single-instance execution alternates short batches and long chunks. Long
chunks may execute while short requests accumulate. Urgent requests take
priority when deadlines are supplied. Multiple instances are divided into
short and long pools. The controller compares nearest-rank P90 pressures,
using queue depth, deadline deviation and GPU utilization. Virtual
round-robin routing converts each global queue into per-instance backlog
observations without allocating KV to queued requests. At most one idle
instance changes pools per control interval, subject to hysteresis, cooldown
and minimum pool sizes. Instances with resident KV drain before reassignment.
GPU utilization is read through NVML at control intervals.
`utilization_signal: "occupancy"` instead uses committed operation intervals and
active leases. This alternative avoids reliance on hardware utilization counters
but is a different pressure signal.

## Execution

The default graph grid is the Cartesian product of prompt lengths
`[8, 16, 32, 64, 128, 256]` and batch sizes `[1, 2, 4, 8, 16, 32, 64]`.
Memory-infeasible shapes are excluded. Capture precedes workload arrivals.
Replay updates token IDs, positions, sequence lengths and KV slot indices in
stable device buffers. Padding is excluded from KV writes and causal attention.
Attention reads historical KV directly from the native slot pool and uses FP32
softmax accumulation. Graph execution uses BF16 or FP16 weights. Shapes outside the captured grid use standard execution.
Graphs and scratch storage persist across benchmark phases; prompt KV does not.

An instance contains one GPU by default. `instance_degree` configures a fixed
SP group for long requests. Short batches execute on that group's first GPU;
other group members remain reserved for the instance. This SP mapping is a
backend extension, not a placement policy proposed by LAPS. Instance degree is
not selected dynamically per request. A full prompt must fit the instance's
aggregate KV capacity even when it is chunked. Unsupported inputs fail without
truncation.

Workload records are independent by default. A record with `retain_kv: true`
retains its prefix for a later record naming it through `prefix_request_id`.
The continuation's `token_ids` must contain the complete parent prefix followed
by new tokens. Its class is determined by new-token length and historical
context. Prefix ownership transfers atomically, including between short and
long pools; historical tokens are not recomputed. Each prefix has one successor.
Retained KV is released after its final use or when the arrival stream is exhausted.
Graphs read retained histories directly from paged storage. A history that does
not fit on one GPU uses the instance's SP execution path. Branching conversations,
decoding, HTTP routing and PD KV transfer are outside this prefill baseline.

## Configuration

```bash
./launch_dynamic_sp_8gpu.sh --scheduler laps \
  --eval-dataset emozilla/pg19 --eval-split test \
  --laps-config python/sglang/srt/dynamic_sp/laps_config.json
```

`--laps-config PATH` supplies a JSON object; omitted fields use the defaults
below. No GPU calibration or dataset fitting runs automatically.

| Field | Default |
|---|---:|
| `instance_degree` | 1 |
| `length_threshold` | 256 |
| `long_chunk_tokens` | 8192 |
| `short_token_budget` | 16384 |
| `wait_min_seconds` | 0 |
| `wait_max_seconds` | 0.005 |
| `max_queue_wait_seconds` | 0.05 |
| `deadline_seconds` | null |
| `slack_seconds` | 0.001 |
| `safety_seconds` | 0.001 |
| `initial_service_seconds` | 0.005 |
| `control_period_seconds` | 1 |
| `cooldown_seconds` | 5 |
| `hysteresis` | 0.2 |
| `minimum_instances` | 1 |
| `queue_weight`, `deadline_weight`, `utilization_weight` | 1 |
| `utilization_signal` | `gpu` |

`graph_lengths` and `graph_batches` override the captured shapes. Both must
contain ascending distinct powers of two. Controller timing, weights and safety
values are configurable choices because the paper does not specify a complete
parameter set. Service estimates use an exponential moving average of completed
operations. Arrival-rate estimates use the most recent 256 short arrivals.

A workload record may set `ttft_slo_seconds`; otherwise `deadline_seconds`
applies. A null deadline selects deadline-free batching. All deadlines are
relative to scheduled arrival, preserving producer and queueing delays.
TTFT ends at committed first-token sampling, as for the other SP policies.
Throughput uses unpadded new input tokens and the common first-submission-to-last-completion
measurement window. Results retain the LAPS configuration and controller state.

The launcher excludes whole dataset records exceeding the fixed instance KV
capacity. When LAPS is included in a benchmark sweep, the same capacity filter
is applied to the shared workload before any scheduler runs. Exclusions are
recorded in the manifest. Reusing a prepared workload does not silently modify
it; unsupported lengths are rejected before execution.

For independent boundary measurements, `laps_profile` fits the two Section 2.1
cost equations using nonnegative least squares. Input JSON contains `samples`
with `new_tokens`, `history_tokens`, `compute_seconds` and `memory_seconds`, plus
optional measurement `metadata`. Durations must be separately measured phases;
a total-latency sample cannot identify compute and memory costs. The command
writes a configuration containing `boundary_coefficients` and a companion fit
report. Fitting is explicit and does not inspect evaluation requests.

```bash
python -m sglang.srt.dynamic_sp.laps_profile \
  --samples boundary-samples.json --output laps-config.json
```
