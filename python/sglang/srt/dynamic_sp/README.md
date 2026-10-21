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

## Tetris scheduling

Use `--scheduler tetris --fit-dataset ID --fit-split train` to prepare the prefill policy from
arXiv:2511.06247v1, or pass `--tetris-profile PATH` to reuse a frozen profile.
Choose evaluation data separately with `--eval-dataset ID --eval-split test`.
See [Tetris Prefill Scheduler](TETRIS.md) for calibration,
threshold search, launch commands, and the remaining differences from the paper.

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
contexts require restarting the worker group.

`--execution-mode async` is the default. The coordinator leases the union of
old and new KV owners for each operation. Requests connected through shared
owners form one transaction; disjoint transactions execute and commit
independently. A completed group can accept its next chunk while other groups
remain active. Migration waits only for its old and new owners to become free.
Membership-specific communicators are created on first use and cached. Their
identities do not depend on group creation by unrelated workers.

`--execution-mode sync` retains globally synchronized steps for comparison.
Both modes support arbitrary ordered placements, migration-only operations,
pause, eviction, and recomputation. The async runner currently requires one
Linux host and a shared monotonic time namespace for accurate measurements.

### KV migration and attention

Placement changes pack affected prompts into destination-major variable-split
all-to-all transfers. By default, ring and gathered attention migrate KV one
layer at a time: redistribution for the next layer overlaps the current layer's
output projection and feed-forward computation. CUDA events order consumption
and buffer reuse. `--no-migration-overlap` restores the all-layer transfer before
prefill. Migration preserves stored values without prefix recomputation.

The launcher defaults to `--attention-backend ring`. This backend circulates
cyclic KV shards among owners, uses two shard buffers, and accumulates online
softmax across hops. NCCL P2P on a separate CUDA stream overlaps current-shard
attention. Causal masking uses absolute key and query positions, including
cached prefixes, uneven shards, and grouped-query attention. The fused kernel
retains FP32 statistics between hops. See [Tetris Prefill Baseline](TETRIS.md)
for implementation differences from the paper and multi-GPU validation.

The alternative `gathered` backend reconstructs shared KV only on workers that
compute new queries for that prompt. `--kv-transport targeted` packs an
all-to-all transfer with variable per-peer sizes. A prompt's degree-one KV stays
local; shared KV is sent only to consumers of the current chunk, including when
some members have no new queries. Idle workers participate with zero transfer
sizes. Collectives use the operation's owner-union group in async mode and the
world group in sync mode. `--kv-transport all_gather` selects padded all-gather
within that execution group for controlled comparisons.
Pass runner options through the root launcher after `--`, for example
`./launch_dynamic_sp_8gpu.sh <launch options> -- --kv-transport all_gather
--shared-attention sdpa`. These switches retain the combined error checks.

For shared prompts, `--shared-attention auto` uses tiled Triton attention on
CUDA with FP16 or BF16 and head dimensions up to 256. Query positions follow the
existing cyclic ownership rule; the fused kernel applies absolute-position
causality and accumulates online softmax in FP32 without a dense mask. FP32,
FP64, CPU, and larger head dimensions use PyTorch SDPA. `--shared-attention sdpa`
selects that reference implementation explicitly. Degree-one prompts retain
SDPA with a lower-right causal bias. Triton specializes on head layout, dtype,
and SP degree, but query length, KV length, and prefix position remain runtime
arguments. First use of a new kernel specialization still incurs compilation.

Shared batches check initial projections, then combine each layer's attention
and output with the next layer's projections in one local phase. Every phase
is checked collectively before the next KV transfer. An all-private batch has
no layer transport and checks all local layers together. Recoverable errors
reject the entire staged plan; these changes do not weaken commit or rollback
semantics. For $L$ layers, layer-computation checks decrease from $2L$ to $L+1$
for shared batches, and from $L$ to one for private batches when no migration
pipeline is active. Pipelined migration separates attention and output phases
to launch the next layer's transfer between them. Setup, migration, and logit
checks are additional.

With `--no-verify`, vocabulary projection and logit broadcast occur only when a
prompt finishes prefill. Verification retains prefix logits at every advancing
step. The engine API defaults to `logits_mode="all"`; callers that need only
completed prompts can select `logits_mode="completed"`. Both modes compute the
same KV states. Per-step metrics record shared and private KV rows, workspace
bytes, per-layer send and receive bytes, fused attention calls, and collective
counts. Transport byte counters include self transfers; they are not network
traffic measurements.

The alternative `streaming` backend processes bounded KV blocks and accumulates
online softmax in FP32. It reduces temporary KV storage at the cost of additional
communication and error checks. Workers in the execution group participate
even without local queries. This reference backend does not overlap communication and computation.
The SP path does not use CUDA graphs.

`--max-local-tokens` bounds committed KV token slots per worker. It does not
bound total GPU memory: model weights, activations, staged KV, migration buffers,
and attention workspaces require additional capacity.

## Scheduling Policies

The root launcher selects the execution policy with `--scheduler`.

| Selection | Execution and scheduling |
|---|---|
| `greedy` | Dynamic SP with the heuristic in `GreedyScheduler.next_plan()` |
| `tetris` | Latency-model-based chunking and progressive SP expansion |
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
In async mode it also includes `busy_ranks`, `worker_available_at`, and a
per-request `in_flight` flag. Availability predictions guide planning only;
actual completion events release worker leases. A policy must not dispatch an
in-flight request or use a busy worker, including an old migration owner.
The method returns a `PrefillPlan`, or `None` when waiting or finished.
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

These controls describe the greedy SP policy. Tetris retains its reserved chunk
plans and does not apply greedy admission or chunk-budget limits; see its
[baseline description](TETRIS.md). These differences must be reported when comparing results. In the DP baseline,
both model weights and each prompt's KV state must fit on one GPU.

## Workload and Measurement

The preparation utility tokenizes one fixed-length prefix per eligible dataset
row, including BOS when available. Rows shorter than the requested prefix are
skipped. It records token IDs, source rows, resolved model and dataset revisions,
and a workload SHA-256 digest.

For offered input load $\rho$ in tokens/s and mean prompt length $\bar L$, the
Poisson request rate is $\lambda=\rho/\bar L$. Interarrival times are sampled from
$\text{Exponential}(\lambda)$ using `--arrival-seed`. Offered token load
measures input demand, not GPU utilization; attention cost also depends on
prompt length. `--arrival-pattern burst` places all arrivals at time zero.

Arrivals are submitted independently of earlier completions. The SP producer
queues requests during execution. The async coordinator drains arrivals while
other groups execute; sync mode observes them at global execution boundaries. The DP client submits asynchronous requests to the native Engine.
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
saturated device capacity. Sync execution throughput, based on summed step
durations and including recomputed tokens, is a separate diagnostic. Async
operation durations overlap and are never summed to form the denominator.

`throughput_start_seconds` and `throughput_end_seconds` expose the window
boundaries on the shared host monotonic clock. Their difference is
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
Async SP records completion after sampling and successful group commit. The
coordinator retains that worker timestamp unchanged, even if response delivery
or polling is delayed. Startup verifies a common Linux boot and time namespace;
no round-trip clock offset or wall-clock conversion is used. DP preserves the
original monotonic first-token timestamp through its response IPC hops.
Invalid timestamp ordering causes an error rather than a corrected measurement.
Clock resolution is recorded separately from latency; it is not an accuracy bound.

DP additionally reports requester-observed latency in `response_ttft_seconds`
and response-window throughput in `response_throughput`. The interval from
scheduler completion to requester receipt is `response_delivery_seconds`; it
includes response processing and IPC, not solely transport time. DP request
IPC remains part of its arrival-to-token latency. SP retains its direct ingress
and execution paths, so aligning completion boundaries does not make the
implementations identical.

No benchmark warmup is performed. First-use kernel compilation may affect
measured latency and throughput. SP reference verification affects latency and wall throughput. Async verification
runs before publication and is included in completion time; sync verification
is outside step timing but delays subsequent work. Use `--no-verify` for
performance experiments and run correctness validation separately. Identical workloads, precision, resource limits,
and explicitly controlled warmup are necessary for meaningful comparisons.

## Reproducing Experiments

Run commands from the repository root. Install SGLang in editable mode:

```bash
./install_sglang.sh --venv .venv
source .venv/bin/activate
```

The eight-GPU launcher defaults to `NousResearch/Meta-Llama-3.1-8B-Instruct`,
`emozilla/pg19` training data, all eligible records capped at 8192 tokens, BF16, 1024-token chunks,
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
| `async_executor.py`, `communication.py` | Worker leases, scoped groups, and acknowledgements |
| `async_scheduler.py`, `async_runner.py` | Independent completion handling and online dispatch |
| `ring.py`, `fused_attention.py` | Reusable ring attention and layer migration pipeline |
| `reservations.py` | Persistent worker/task dependency calendar used online and in simulation |

Evaluation uses the entire selected split by default (`--num-prompts all`).
Use `--num-prompts N` for a shorter run. Each eligible record becomes one prompt;
the input-token cap still applies, and fixed length mode skips shorter records.
Use `--length-mode natural` to include nonempty short records as well.
Threshold fitting retains its separate `--fit-num-prompts` limit.
