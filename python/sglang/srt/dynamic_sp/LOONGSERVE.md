# LoongServe prefill scheduler

This implementation is done back-and-forth using gpt-6 with explicit instructions and 
algorithm inputs.  All code and scripts have been validated manually. 
`--scheduler loongserve` implements the prefill-only specialization of [LoongServe, SOSP 
2024](https://arxiv.org/abs/2404.09526), Sections 5.1-5.5. We used the 
implementation for the normalized objective and admission threshold from commit 
[f6e8fc15f31ea6913b353b627545058db5eec321](https://github.com/LoongServe/LoongServe/tree/f6e8fc15f31ea6913b353b627545058db5eec321).

## Scheduling

Pending requests are considered in arrival order. Admission reserves capacity
for the complete input and stops at the first request that does not fit on the
idle workers. A batch expansion is accepted only if its predicted prefill time
is at most the configured threshold. The first request may exceed this threshold,
matching the released implementation and preventing starvation of long prompts.
The initial threshold is 0.5 seconds, corresponding to the upstream 500 ms default.
It is configurable and is not asserted to be the measured compute-bound
transition on the current hardware.

All idle workers are available to the batching optimizer. Requests are sorted
by decreasing input length, and workers by node and rank. Available workers have
empty KV pools in this prefill-only lifecycle. A compiled dynamic program
partitions contiguous request intervals and worker intervals. Its objective is

$$\sum_{B} T_{|G_B|}(B)\sum_{r\in B}\frac{1}{L_r},$$

where

$$T_s(B)=\alpha_s+\beta_s\sum_{r\in B}L_r+\gamma_s\sum_{r\in B}L_r^2.$$

The solver considers every feasible split and selects the best number of used
workers. It uses the released implementation's normalized input-latency objective,
not a makespan or mean-TTFT substitute. Exact search has time complexity
$O(n^2m^2)$ and space complexity $O(nm)$; it does not apply the paper's optional
quadrangle-inequality pruning to independently fitted coefficients.
The C++ library is compiled and cached before measured arrivals begin.

Each batch executes its full prompts in one prefill operation. Disjoint groups
complete independently. Chunk size, greedy token budgets and greedy admission
limits do not subdivide LoongServe batches. Concatenated-batch striping rotates
per-request worker order so uneven lengths distribute remainder tokens evenly.
Aggregate token capacity therefore suffices on the equally sized idle workers.

## Prefill-only reduction

There are no decoding batches. Consequently, the decoding preemption terms in
Section 5.1, reclamation of decoding workers in Section 5.2, and decoding scale-up
in Section 5.4 have no applicable state. Completed requests release their KV
instead of retaining a smaller decoding group. Proactive KV retention during the
prefill ring, whose purpose is to prepare that decoding group, has no destination
in this lifecycle. It is not replaced by a post-prefill migration.

This is the paper's prefill scheduling policy on the shared SGLang dynamic-SP
backend, not the full LoongServe serving system. The default backend uses striped
ring attention, native SGLang KV slots, and independent async worker groups.
Gathered and streaming attention are other backend options. TP remains one
per worker. 

## Calibration and launch

```bash
./launch_dynamic_sp_8gpu.sh --scheduler loongserve \
  --eval-dataset emozilla/pg19 --eval-split test \
  --offered-load 25000 --no-verify
```

Without an existing profile, the launcher measures synthetic full-prefill
batches independently of evaluation data, fits Eq. (7) by ordinary least squares,
and launches the online workload. Measurements cover every SP degree, equal and
mixed prompt lengths, and multiple batch sizes. Context- or capacity-infeasible
calibration points are excluded and recorded. Fitting requires full-rank data
for every degree. Profiles record measurement provenance and fit residuals.
Residuals are in-sample errors, not validation of extrapolation to longer prompts.

`--loongserve-profile PATH` reuses a fitted SIB. `--loongserve-samples PATH`
refits saved batch measurements without another GPU calibration.
`--loongserve-calibration-lengths "128 512 2048 8192 32768 65536"` overrides
the default SP-scaled lengths with explicit absolute lengths for every degree. `--loongserve-max-prefill-seconds S` overrides the
profile's admission threshold. Profile hardware, model, precision and backend
must match execution unless the existing explicit backend-mismatch option is used.

The benchmark's `--scheduler all` includes LoongServe. It reuses one LoongServe
profile per backend/transport pair across datasets and repetitions. 
Completion is the sampled first token after
committed prefill, excluding response IPC and HTTP, as for the other SP policies.

## Sparse calibration

The default `--loongserve-calibration-grid lean` keeps every SP degree but uses
at most six fitting shapes and two held-out validation shapes per degree.
Default candidate lengths are SP degree times
`[128, 512, 2048, 8192, 32768, 65536]`, filtered separately by model context
and aggregate KV capacity. Fitting uses four
single-request anchors plus equal and mixed two-request batches. The largest
single request and an additional mixed batch are held out of coefficient fitting.
This yields at most 64 configurations on eight workers; extending the length
range does not expand the lean grid into a Cartesian sweep.

Calibration defaults to disjoint concurrent groups: eight groups at SP1, four
at SP2, two at SP3 and SP4, and one at SP5-SP8 on eight GPUs. Shapes are assigned
by decreasing estimated work to balance group completion times. Each group
warms up and measures independently; only SP-degree boundaries synchronize all
workers. Durations are the maximum participating-rank wall time through first
token sampling, with timing aggregation and KV release outside the interval.
Completed points are checkpointed in per-group JSON files and merged into the
sample file after each degree. Worker membership and concurrency are recorded.

Concurrent groups share host and interconnect resources, so these estimates
represent concurrent execution rather than isolated-group latency. Use
`--loongserve-calibration-concurrency serial` for isolated measurements. Held-out
prediction checks remain enabled in both modes.

Each configuration receives one warmup and up to three measured executions.
If a measured execution takes at least one second, that configuration stops
after that measurement. Thus expensive points use two executions including
warmup. Small points retain repeated measurements to reduce timing noise.
Progress identifies the SP degree, fitting/validation role and current repetition.

Ordinary least squares, the normalized scheduling objective, admission logic and
candidate SP degrees are unchanged. A profile is rejected when a held-out
relative prediction error exceeds 25%. Expand the independent calibration range
or use `--loongserve-calibration-grid full` for more fitting shapes when this
check fails. The full grid retains independent validation points too. Historical
sample files without role labels remain supported as fitting data and have no
held-out accuracy claim.

With sufficient KV capacity and model context, the default largest fitting and
validation prompt lengths are:

| SP degree | Largest fitting prompt | Validation prompt |
|---|---:|---:|
| 1 | 32768 | 65536 |
| 2 | 65536 | 131072 |
| 3 | 98304 | 196608 |
| 4 | 131072 | 262144 |
| 5 | 163840 | 327680 |
| 6 | 196608 | 393216 |
| 7 | 229376 | 458752 |
| 8 | 262144 | 524288 |

