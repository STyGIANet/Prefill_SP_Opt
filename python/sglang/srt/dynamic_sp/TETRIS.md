# Tetris Prefill Baseline

This implementation is done back-and-forth using gpt-6 with 
explicit instructions and algorithm inputs. 
All code and scripts have been validated manually. 
The implementation reproduces the prefill scheduling procedure in
[Li et al., arXiv:2511.06247v1, Section 5.1](https://arxiv.org/html/2511.06247v1)
on the dynamic SP backend. It implements Algorithms 1-3, persistent instance
reservations, and offline selection of load-dependent expansion thresholds.
Decoding and prefill-to-decode KV transfer are excluded. The completion boundary
is a sampled first token after committed prefill, before response IPC or HTTP.
This is a prefill baseline, not a reproduction of the paper's end-to-end service.

## Planning and reservations

The fitted latency model is

$$T_s(C,L)=a_s+b_s L+c_s CL+d_s L^2.$$

The planner first considers one chunk, with an SP expansion accepted only when
its predicted TTFT improves sufficiently. It then enumerates smaller initial
groups, solves chunk lengths from differences in instance availability, and
recursively evaluates the remainder. Each subsequent group includes its
predecessor. Placement retains existing workers and prefers node locality.
The integer inverse is solved by bisection. This is the paper's restricted
recursive search, not an exhaustive optimum over arbitrary schedules.

Each arriving request receives its complete chunk plan in FIFO order. The
backend reserves that plan as a dependency graph: each chunk depends on the
previous chunk of its request and on previous reservations on its workers.
Only successful execution satisfies a dependency. Predictions are updated as
execution progresses, but a new arrival does not replace earlier chunk plans.
Ready chunks execute independently on disjoint groups. Cleanup of completed
requests is separate from new computation so it cannot combine independent
requests into one batch.

The online scheduler and threshold simulator use the same `ReservationCalendar`.
The simulator advances its completion events using fitted execution durations;
the online scheduler advances them using actual commits. Their equivalence is
tested under identical durations, including simultaneous arrivals and queue gaps.
Host dispatch overhead and prediction errors can still change observed times.

Tetris does not truncate chunks to `--chunk-tokens` or `--max-batch-tokens`, and
it does not apply the greedy scheduler's `--max-running-requests` admission cap.
It does not add resident priority or reserve the full input's KV capacity at
admission. Actual committed/staged token capacity is checked before dispatch.
If a reserved allocation cannot fit, the experiment fails explicitly rather
than silently changing Tetris's scheduling decisions. Configure sufficient KV
capacity for the intended workload. Other policies retain their own limits.

## Execution backend

The default is `--execution-mode async --attention-backend ring`.
The ring circulates each worker's cyclic KV shard through the request's owner
set. A tiled kernel accumulates online softmax in FP32 with absolute-position
causal masking and grouped-query attention. Two shard buffers replace full-KV
assembly. P2P exchange on a communication stream overlaps attention on the
current shard. Uneven and empty shards are supported. CPU and unsupported
CUDA dtypes use the reference online-softmax implementation.

Layer-wise migration is enabled by default. After layer l's attention, the
backend launches redistribution of layer l+1's historical KV on a separate
CUDA stream while layer l's output projection and feed-forward computation run.
The next layer waits on the migration event before consuming historical KV.
Committed KV remains unchanged until the complete operation succeeds.
Recoverable attention errors drain the prescribed P2P sequence before rejection;
pending migration is drained before staging buffers are released. CUDA context
or transport failures remain fail-stop.

The same migration facility supports gathered attention:

```bash
--attention-backend gathered --kv-transport targeted
```

This uses variable-split all-to-all exchange, effectively all-to-all-v, followed
by local attention over assembled KV. `--kv-transport all_gather` selects padded
all-gather. `--no-migration-overlap` disables the migration pipeline for an
ablation. The streaming backend remains a bounded-workspace reference path.
No transport is assumed universally fastest. Fit separate latency models for
each selected backend, precision, hardware configuration, and overlap setting.

These are reusable backend facilities. Tetris supplies plans and thresholds;
other policies can use ring attention, gathered attention, migration overlap,
and the reservation calendar without depending on the Tetris planner.

The paper uses NVSHMEM-based ring communication and a C++ planner. This backend
uses NCCL P2P and a Python planner, with explicit transactional error checks.
Async prefill phases check host errors locally without per-layer status votes.
A failure during these phases invalidates the run and requires worker restart;
it does not attempt collective recovery after peers may have advanced.
Validation, allocation, and publication retain operation-boundary consensus.
The synchronous reference path retains intermediate status votes.
Migration explicitly orders packing before transfer with a stream dependency.
A compute-stream completion wait before commit detects asynchronous execution
errors and preserves completion timestamps. 
These differences can affect implementation-specific overhead. 

## Load regulation and fitting

Calibration defaults to power-of-two SP degrees, 1, 2, 4, and 8 on eight GPUs.
Other candidate sets can be supplied to the calibration utility. The latency
coefficients default to nonnegative least squares with column scaling. This
refits all coefficients jointly subject to nonnegativity, preserving monotone
latency and chunk-budget inversion when noisy measurements yield a negative
ordinary least-squares coefficient. It does not clip an unconstrained fit.
This constraint is an implementation choice beyond the paper's stated
least-squares fitting. `--tetris-fit-method least-squares` selects strict ordinary
least squares and rejects negative coefficients. Rank-deficient data is rejected
for either method. Profiles record the method, unconstrained coefficients and
residual, constrained residual, and zero terms. Residual diagnostics are training
errors, not held-out prediction accuracy. Existing calibration samples can be
reused through `--tetris-samples PATH` without another GPU calibration.

Threshold search samples the fitting length distribution and Poisson arrivals.
Each candidate improvement rate sees identical arrivals. The default rates are
0.05 to 0.75 in increments of 0.05. The objective defaults to mean per-prompt
TTFT; `--tetris-objective mean|p50|p95` selects another objective.

User-facing offered load remains input tokens/s. The simulator converts it to
requests/s using the fitting distribution's mean length. Online regulation uses
observed requests/s, as in the paper, and selects the nearest profiled rate.
The threshold is updated every 30 seconds. The observation window defaults to
30 seconds; the paper specifies the update period but does not fix this window
in its algorithm. Existing reservations retain the threshold used at admission.
Short runs may end before the first periodic update. Use a sufficiently long
trace when evaluating load adaptation. Evaluation defaults to all eligible
records in the selected split; `--num-prompts N` explicitly limits a run.
Full records are retained by default; records exceeding the model context are
excluded with explicit provenance. Threshold fitting has its own independent
record limit. The benchmark matrix freezes one profile across all datasets and
execution backends; calibration backend differences are recorded in its CSV.

## Independent fitting and evaluation

Use the single root launcher:

```bash
./launch_dynamic_sp_8gpu.sh --scheduler tetris \
  --fit-dataset emozilla/pg19 --fit-split train \
  --eval-dataset emozilla/pg19 --eval-split test \
  --offered-load 25000 --no-verify
```

Fitting and evaluation datasets, splits, length modes, and token caps are
independently configurable. `--fit-num-prompts`, `--fit-input-tokens`, and
`--fit-length-mode` control the fitting sample. Evaluation accepts `--dataset`,
`--split`, `--num-prompts`, `--input-tokens`, and `--length-mode`, or their
`--eval-` aliases. Dataset config, revision, and text column also have separate
fitting and evaluation options. `--workload-dir` reuses evaluation token IDs and
arrival times only; it never becomes the fitting workload.

The launcher prepares the two datasets, checks provenance, calibrates the
backend, fits latency models, searches thresholds, and starts online evaluation.
It never tunes thresholds on evaluation lengths. Provenance checks reject
identical workloads and overlapping recorded rows of the same partition;
they do not establish semantic deduplication between differently named sources.

`--tetris-samples PATH` reuses hardware measurements.
`--tetris-profile PATH` reuses the entire frozen profile and skips fitting.
Profiles use schema version 2 for persistent reservations and request-rate
regulation. Older profiles must be rebuilt. Backend, precision, topology,
execution mode, and migration-overlap settings must match calibration.
First-use communicator construction remains inside online timing. Calibration
uses warmed isolated chunks and does not separately model every migration,
contention, or host scheduling cost.

Fitting defaults to 256 records with natural lengths capped at 8192 tokens.
Evaluation defaults to 32 fixed-length 8192-token prefixes. Natural mode retains
short records; fixed mode skips them. Select a held-out evaluation split
explicitly. The default threshold load grid is 4096, 8192, 16384, 25000, and
32768 input tokens/s, with 1000 simulated requests per load. Configure it with
`--tetris-loads` and `--tetris-simulation-requests`.

Artifacts are saved under the run's `tetris/` directory, including the fitting
workload, measurement samples, fit diagnostics, frozen profile, and stage logs.
Reference verification changes execution time. Use `--no-verify` for performance
experiments and validate correctness separately.

## Frozen profile reuse

The launcher reuses a compatible saved Tetris profile by default, including its
fitted thresholds. If none is available, it stops instead of starting calibration.
Use `--recalibrate-tetris` to explicitly repeat GPU calibration and threshold
selection, or `--tetris-samples PATH` to fit saved measurements and select
thresholds without GPU profiling. `--tetris-profile PATH` selects a particular
frozen profile. Training/evaluation overlap checks remain enabled when reusing
profiles. Changing fitting options alone does not invalidate a frozen profile.
The benchmark uses one frozen Tetris profile across the entire sweep, including
backend combinations. Explicit recalibration occurs only once per sweep.
