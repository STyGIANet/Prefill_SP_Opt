# LAPS native baseline

`--scheduler laps` runs the released LAPS scheduling policy on this checkout's
native SGLang Engine. It uses the existing Python environment and model
installation. No separate environment or vendor installation is required. If
`CUDA_HOME` is unset, the runner selects an existing `/usr/local/cuda-VERSION`
toolkit matching PyTorch when available. An explicit `CUDA_HOME` is preserved.

The scheduling source is [Jianshu-She/LAPS at aa29962d](https://github.com/Jianshu-She/LAPS/tree/aa29962d501236ecd2a0c7a8470c5977df3fe057).
The server policy is adapted from `_get_next_laps_batch_to_run` and
`_build_batch_from_queue` in `disaggregation/prefill.py`. The instance policy is
adapted from `_laps_maybe_rebalance`, `select_pair_laps` and `laps_request_done`
in the authors' `mini_lb.py`.

## Scheduling

Each serving replica separates requests at 256 input tokens. It attempts the
short queue first and then the long queue when no short batch can be admitted.
Native SGLang performs KV admission, chunk continuation, batching and execution.
A resident chunk remains under the native chunk scheduler, as in the released
implementation.

The coordinator divides replicas into short and long pools. It routes each
request randomly within its class pool and tracks outstanding requests by
class. Every five seconds, evaluated on an arrival, it moves one instance when
one class has more than twice the other's outstanding requests. Both counts
are floored at one for this comparison, and each pool retains at least one
instance. Reassignment changes routing for subsequent requests; it does not
migrate requests already assigned. The routing seed is fixed by default.

These are the released implementation's policies. They do not implement the
paper's runtime boundary fitting, adaptive wait/depth algorithm or P90 pressure
controller. The default released benchmark also does not enable the separate
waiting-window branch. See the [paper](https://arxiv.org/abs/2601.11589) for those
designs.

## Native execution

The default deployment has eight replicas, one per GPU. `tp_size` can be 1, 2,
4 or 8; the number of replicas is `8 / tp_size`. With one replica, only the
server's temporal queue policy applies. LAPS does not partition a sequence or
change its SP degree.

Prefill uses the released `PiecewiseCudaGraphRunner`, including rectangular
batch graphs with batch sizes `[1, 2, 4, 8]` and query lengths
`[16, 32, 64, 128, 256]`. Real queries are right-aligned in each padded row.
Batch graphs take priority when eligible; otherwise execution uses the released
piecewise token buckets, followed by eager execution outside those buckets.
The piecewise grid is generated from the chunk size using the released rule.
The graph runner and queue and routing methods are vendored under
`laps_released/`, with the source revision recorded in `UPSTREAM.json`.

The compatibility layer maps the released runner's batch, attention metadata,
compilation context and residual bookkeeping APIs to this checkout. The
piecewise attention boundary passes only real query rows to FlashInfer, whose
current API requires an exact match with its planned row count. Padding remains
in the tokenwise graph operations and is excluded from returned logits. FlashInfer
uses an isolated header and compilation cache under `work/laps-flashinfer/`.
The header adaptation saturates unsigned causal tile bounds when a padded query
row is longer than its real KV sequence. It preserves right-aligned causal
visibility and leaves installed packages unchanged. The installed header hash
must match the reviewed version; a different version requires review.

The supported path is dense, full-attention text inference with FlashInfer,
without LoRA or speculative decoding. The custom SP LAPS scheduler and private
attention kernel are removed.

Each request generates one token on its assigned replica. There is no separate
decode deployment, HTTP router or prefill-to-decode transfer in this prefill
baseline. Native DP routing carries the authors' instance selections. This
preserves the existing experiment's first-token completion boundary without
introducing a separate serving environment. Pending counters are released when
the one-token response completes, rather than after a remote decode response.
These deployment and completion changes are specific to the prefill-only
experiment; end-to-end serving latency is not equivalent to the released
prefill/decode deployment.

## Invocation

```bash
./launch_dynamic_sp_8gpu.sh \
  --scheduler laps \
  --eval-dataset emozilla/pg19 --eval-split test \
  --offered-load 25000 --no-verify
```

`--offered-load` is expected input tokens per second. The prepared Poisson
arrival offsets are preserved. Model, dataset, split, prompt count and warmup
options use the common launcher. SP attention and KV transport options do not
apply. Native settings are supplied through
`--laps-config python/sglang/srt/dynamic_sp/laps_native.json`.

The requested KV capacity defaults to 600000 tokens per replica. The actual
native allocation can be smaller. Before arrivals begin, the runner filters
whole records against the smallest runtime input limit across replicas and
writes `results.capacity.json` and `results.workload.json`. Inputs are never
silently truncated. Filtered records retain their original arrival offsets;
filtering therefore reduces the effective offered load.

Warmup covers every replica unless disabled with `--warmup-requests 0`. Warmup
inputs may be shortened to the warmup budget; measured inputs are unchanged.
KV state is cleared before measured arrivals, and prefix reuse is disabled.

For a dataset sweep, use `--scheduler laps --datasets "pg19 arxiv govreport"`.
Each dataset is evaluated once per repeat with the native configuration.
`--scheduler all` remains the SP-only sweep.

## Measurements

TTFT is measured from scheduled arrival to the native scheduler's first sampled
token becoming available on the CPU. It excludes response IPC and HTTP, matching
the native DP baseline. Response latency is also retained separately. Median
and nearest-rank p95 are computed over individual prompt TTFT values.

Throughput is completed input tokens divided by the interval from first actual
submission to last first-token completion. It includes queueing and arrival
gaps. Progress reports show completed prompts, input tokens per second and
TTFT percentiles.

Each run writes `results.json`, `laps-settings.json` and `metrics.csv`. Results
record the upstream revision, effective native settings, routing assignments,
pool changes and evaluated workload hash. The native benchmark does not collect
allocator retry or graph replay counters; those fields are left unset.
