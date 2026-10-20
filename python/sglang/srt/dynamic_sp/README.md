# Dynamic Sequence Parallelism for Prefill

## Scope

This module implements dynamic sequence parallelism (SP) for the prefill phase
of dense, unquantized `LlamaForCausalLM` models. Each worker holds a complete
native SGLang model replica with tensor and pipeline parallelism set to one.
Persistent key-value (KV) states are partitioned across workers by token
position. A scheduling policy may change each prompt's SP degree, worker
membership, and chunk size between execution steps.

The SP implementation is a separate execution path within SGLang. It does not
integrate with the native serving scheduler or HTTP interface and does not
perform autoregressive decoding after the first token. Supported weight types are FP32, FP16, and BF16. 
Quantization, sliding-window attention, CPU KV
offload, and changes during an in-flight kernel are unsupported.

## Execution Model

For an ordered worker set $R=(r_0,\ldots,r_{d-1})$, the SP degree is $d$ and the
owner of absolute token position $p$ is

$$
\text{owner}(p)=r_{p\bmod d}.
$$

Any nonempty ordered subset of workers is permitted. The degree need not divide
the worker count. Changing the order of $R$ also changes ownership. Different
prompts may use overlapping or disjoint worker sets.

A `PrefillPlan` specifies admissions, per-prompt steps, and releases. Each
`PromptStep(request_id, ranks, end)` selects a placement and an exclusive prefix
endpoint. Advancing from $n$ to $m$ computes positions $[n,m)$; setting $m=n$
permits migration without recomputation. Omitting a resident prompt pauses it
while retaining its KV state. In the online scheduler, releasing an unfinished
prompt evicts its KV state and requeues the original input for recomputation.
Releasing a completed prompt retires it.

The engine stages KV updates before publishing a new version. Recoverable
allocation or model errors raise `PlanRejected` collectively and preserve the
previous committed state. Transport failures, process loss, and corrupted CUDA
contexts require restarting the worker group. All workers must execute plans
in the same order.

### KV migration and attention

Placement changes pack the affected prompts into one destination-major
all-to-all transfer per plan. Migration preserves token positions and stored
values without arithmetic or prefix recomputation.

The default `gathered` backend performs one packed KV all-gather per layer for
active prompts. Each worker evaluates causal PyTorch SDPA for its local queries
using absolute positions. Attention is isolated between prompts. Persistent KV
remains sharded, but active KV for one layer is temporarily replicated.

The alternative `streaming` backend processes bounded KV blocks and accumulates
online softmax in FP32. It reduces temporary KV storage at the cost of additional
communication and error checks. Both backends use the fixed world group;
workers without local queries still participate in collectives. Communication
and computation are not overlapped, and the SP path does not use CUDA graphs.

`--max-local-tokens` bounds committed KV token slots per worker. It does not
bound total GPU memory: model weights, activations, staged KV, migration buffers,
and attention workspaces require additional capacity.

## Scheduling Policies

The root launcher selects the execution policy with `--scheduler`.

| Selection | Execution and scheduling |
|---|---|
| `greedy` | Dynamic SP with the heuristic in `GreedyScheduler.next_plan()` |
| `sglang-dp` | Native SGLang DP replicas, round-robin dispatch, and FCFS scheduling within each replica |
| `module:factory` | An importable factory that constructs a custom SP policy |

### Dynamic SP heuristic

The heuristic prioritizes resident requests, orders them by last service, and
admits pending requests in arrival order when capacity permits. For $W$ workers
and $N$ runnable requests, it initially targets $\lceil W/N\rceil$ workers per
prompt. It increases the degree when necessary to satisfy KV capacity, selects
workers using planned token load and reserved KV, and retains comparable
placements of equal degree to reduce migration.

The policy reserves full-prompt KV token capacity and enforces a global resident
request limit and a global new-token budget per step. Chunk endpoints advance
by at most the configured per-prompt chunk size. This is a capacity-aware
heuristic, not an execution-time optimization algorithm. The default policy
does not evict requests; alternative policies may use the eviction interface.

### Custom policy interface

`module:factory` denotes a Python module and callable, for example
`my_scheduler:make_scheduler`. The module must be importable through the active
Python environment or `PYTHONPATH`. The runner calls `make_scheduler(config)`
with a `SchedulerConfig`; the returned object must implement
`next_plan(state)`.

`state` is an immutable `SchedulerInput` containing the version, time, request
progress and placements, last-service versions, and previous-step measurements.
The method returns a `PrefillPlan`, or `None` after all requests have retired.
Scheduling decisions belong in this method; migration, attention, and commit
operations are implemented by the engine. Policy evaluation should not consume
speculative state because a proposed plan may be rejected.

### Native DP baseline

The baseline uses the native SGLang `Engine` with eight replicas and TP=PP=1.
Each request remains on one replica, and workers schedule independently without
SP collectives. It generates one output token to measure first-token latency;
there are no subsequent decode steps. Tokenized inputs bypass tokenizer
initialization. Prefix caching and decode CUDA graphs are disabled.

The controls have different meanings in the two execution paths:

| Control | Dynamic SP | Native DP |
|---|---|---|
| `--chunk-tokens` | Maximum new tokens per prompt per step | Native per-worker prefill batch chunk size |
| `--max-running-requests` | Global resident request limit | Limit per replica |
| `--max-batch-tokens` | Global new-token budget per step | Not applied |
| `--max-local-tokens` | Committed KV token capacity per worker | Native token-pool capacity per replica |

These differences must be reported when comparing results. In the DP baseline,
both model weights and each prompt's KV state must fit on one GPU.

## Workload and Measurement

The preparation utility tokenizes one fixed-length prefix per eligible dataset
row, including BOS when available. Rows shorter than the requested prefix are
skipped. It records token IDs, source rows, resolved model and dataset revisions,
and a workload SHA-256 digest.

For offered input load $\rho$ in tokens/s and mean prompt length $\bar L$, the
Poisson request rate is $\lambda=\rho/\bar L$. Interarrival times are sampled from
$\operatorname{Exponential}(\lambda)$ using `--arrival-seed`. Offered token load
measures input demand, not GPU utilization; attention cost also depends on
prompt length. `--arrival-pattern burst` places all arrivals at time zero.

Arrivals are submitted independently of earlier completions. The SP producer
queues requests during execution, and the scheduler observes them at execution
boundaries. The DP client submits asynchronous requests to the native Engine.
Neither path waits for one request to finish before submitting the next.

### Throughput

For completed prompts with input lengths $L_i$, actual submission times $s_i$,
and completion times $c_i$, reported wall throughput is

$$
\Theta=\frac{\sum_i L_i}{\max_i c_i-\min_i s_i}.
$$

Each completed input is counted once, including after eviction and
recomputation. The window includes queueing and idle gaps. It excludes model
loading and cleanup after the final completion and is not an estimate of
saturated device capacity. SP execution throughput, based on summed step
durations and including recomputed tokens, is a separate diagnostic.

`throughput_start_seconds` and `throughput_end_seconds` expose the window
boundaries on one coordinator monotonic clock. Their difference is
`prefill_wall_seconds`; the reported rate is `useful_tokens_per_second`.
Per-request timestamps appear in SP `request_timings` and DP `results`.

### Latency and experimental limitations

Latency uses the scheduled arrival time $a_i$, preserving producer delay:
$T_i=c_i-a_i$. In both paths, $c_i$ is the time at which the first greedily
sampled token is available on the CPU, before response IPC. SP selects the
argmax of the final prefill logits without another transformer forward pass.
DP exposes the native scheduler's prefill completion timestamp through the
existing response metadata. The median and nearest-rank p95 are computed from
individual prompt latencies. Throughput uses the same completion boundary.

DP additionally reports requester-observed latency in `response_ttft_seconds`
and response-window throughput in `response_throughput`. The interval from
scheduler completion to requester receipt is `response_delivery_seconds`; it
includes response processing and IPC, not solely transport time. DP request
IPC remains part of its arrival-to-token latency. SP retains its direct ingress
and execution paths, so aligning completion boundaries does not make the
implementations identical.

No benchmark warmup is performed. First-use kernel compilation may affect
measured latency and throughput. SP reference verification is outside step
timing but affects subsequent request latency and wall throughput; disable it
for performance experiments. Identical workloads, precision, resource limits,
and explicitly controlled warmup are necessary for meaningful comparisons.

## Reproducing Experiments

Run commands from the repository root. Install SGLang in editable mode:

```bash
./install_sglang.sh --venv .venv
source .venv/bin/activate
```

The eight-GPU launcher defaults to `NousResearch/Meta-Llama-3.1-8B-Instruct`,
`emozilla/pg19` training data, 32 prompts of 8192 tokens, BF16, 1024-token chunks,
and an offered load of 8192 tokens/s. SP reference verification is enabled by
default with absolute and relative tolerances of `1e-4`. These tolerances are
not automatically relaxed for BF16; use FP32 for the initial correctness check
and assess reduced precision separately.

Prepare one workload and reuse it for both execution paths:

```bash
work_dir=python/sglang/srt/dynamic_sp/work
./launch_dynamic_sp_8gpu.sh --prepare-only --num-prompts 1024 \
  --offered-load 12000 --arrival-seed 42 --output-dir "$work_dir/input"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./launch_dynamic_sp_8gpu.sh \
  --workload-dir "$work_dir/input" --scheduler greedy \
  --dtype bfloat16 --no-verify --output-dir "$work_dir/sp"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./launch_dynamic_sp_8gpu.sh \
  --workload-dir "$work_dir/input" --scheduler sglang-dp \
  --dtype bfloat16 --output-dir "$work_dir/dp"
```

Each run retains `manifest.json`, `workload.json`,
`run.log`, and `results.json`. SP results also contain executed plans, replayable
with `--schedule results.json` on the SP Python entry point. The console reports
metrics; `--verbose` includes diagnostic logs.

## Source Organization

| File | Responsibility |
|---|---|
| `plan.py` | Plan representation and token ownership |
| `scheduler.py` | Online request lifecycle and default SP policy |
| `engine.py` | KV migration, attention, collective execution, and commit |
| `llama.py` | Native SGLang Llama adapter |
| `arrivals.py`, `prepare_workload.py` | Arrival generation and dataset preparation |
| `__main__.py`, `dp_baseline.py` | SP runner and native DP baseline |
| `metrics.py` | Shared wall-throughput calculation |
