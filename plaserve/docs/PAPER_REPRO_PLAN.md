# LAPS paper reproduction plan — Phase D0

Survey date: 2026-10-17. Planning only: no GPUs queried or used, no servers launched, no datasets/models downloaded, and no packages installed. This document is a specification, not evidence that experiments passed. Earlier single-server/PD correctness is accepted; it does not validate new graph grids, model sizes, TP sizes, or router implementations.

Primary references: [paper v2, Sections 3–4](https://arxiv.org/html/2601.11589v2), [LMSys dataset and access terms](https://huggingface.co/datasets/lmsys/lmsys-chat-1m). Local code references below are relative to this document. Numeric choices marked **local choice** are not recovered paper parameters.

## Decision and priority

Start with **E3, Qwen2.5-7B, one prefill instance**, then a scaled multi-prefill layout. A faithful paper reproduction is presently blocked by missing dataset identification and incomplete artifact functionality. We can reproduce the available artifact with explicitly documented deviations; we cannot label its simple scheduler/controller as the complete paper implementation.

Priority: (1) preparation and E3 7B, (2) E3 14B and scaled allocation, (3) E4 SGLang comparisons, (4) E3 32B, (5) E5, (6) E1 after scheduler preparation, (7) E6 after a decode-cost pilot. vLLM remains blocked under the no-install rule.

Use only free GPUs 2–5 initially. GPUs 6–7 require explicit approval before any use. Never use 0–1 or stop unowned processes. GPU visibility failed late in the preceding Phase C campaign; successful visibility/occupancy checks are a prerequisite, not assumed here. Maximum approved future capacity is six 96-GB RTX PRO 6000 Blackwell GPUs, versus the paper's H200 hardware.

## Artifact provenance and inventory

Vendor HEAD: `88b986375b531b2f09261b9f44628a24a9c0d6d7`.
The tracked HEAD diff includes the FlashInfer page-size-1 batch-prefill adapter and the benchmark's FlashInfer/strict-health changes; it is intentionally uncommitted. The previously inspected binary HEAD diff SHA256 is `da2d03fadfdb9d927155798fc0b657235a0223ab9ddb78204c22c5c2baad6a35`. Recompute for every run, never copy this value as evidence.

Use [vendor_provenance.py](../scripts/vendor_provenance.py) to save `git diff --binary HEAD --` as `vendor.diff` and record HEAD, path, SHA256 in each run's `provenance.json`. Also record launcher/driver/router source hashes, exact argv/environment, package versions, model revision, dataset hash and selected conversation IDs, graph grid, GPU UUIDs, topology, kernel manifest hashes, and seeds. The outer workspace is not itself a Git repository; hash local scripts explicitly. Do not add untracked results, backups, or trial scripts to vendor Git.

| Item | What the local artifact actually supplies |
|---|---|
| Main throughput harness | [README](../vendor/LAPS/bench_laps_prefill_throughput/README.md), [bench_prefill_only.py](../vendor/LAPS/bench_laps_prefill_throughput/bench_prefill_only.py): flat LMSys text, normally 10,000 requests, one output token, cc 1/2/4/8/16/32/64; three policies, not E3's four |
| Arrival process | Main harness rate mode uses evenly spaced `i/rate` dispatch, not Poisson. Fixed concurrency eagerly queues tasks behind a semaphore |
| Timing | Main harness non-streaming response latency is called TTFT; valid as an approximation only for its one-token workload, not long-output E3/E4 |
| Filtering | First N nonempty JSONL `text` rows; no language, token-length, session, or turn filtering; shorter datasets silently reduce workload; chars/4 token estimates |
| Caching | Warmup 20 requests, concurrency 4; no cache flush between concurrency levels; identical initial dataset rows repeat |
| 7B launcher | `scripts/bench_7b_2gpu.sh`: TP1, one P plus one D; default Mooncake, IB `mlx5_0`; P/D/router ports 30300/30301/30302, bootstrap 9301/9300 |
| 14B/32B launchers | `scripts/bench_14b_8gpu.sh`, `bench_32b_8gpu.sh`: TP4 P + TP4 D = **one** prefill instance, not eight. 32B script selects `Qwen/Qwen2.5-32B-Instruct`; 7B/14B select base models |
| Current Blackwell launcher | [bench_h100_4gpu.sh](../vendor/LAPS/bench_laps_prefill_throughput/scripts/bench_h100_4gpu.sh): TP2 P=2,3 D=4,5, NIXL, corrected FlashInfer through our wrapper |
| Additional ablations | `scripts/bench_multimodel_8gpu.sh` includes piecewise and batch-prefill graph variants, but no complete paper workload/topology matrix |
| Single-server comparison | [bench_1gpu.sh](../vendor/LAPS/bench_laps_prefill_throughput/bench_1gpu.sh): flat one-token workload; not a scripted E5 mixed-decode experiment |
| vLLM/E6 | Generic vLLM benchmark references exist; no ready LAPS E4 vLLM-PD orchestration or E6 distillation pipeline found |

Do not invoke upstream shell cleanup directly: it uses broad `pkill` and tolerates some errors. Extend our owned-process wrapper, preserving original vendor files.

### Multi-turn loaders worth reusing

[hicache/data_processing.py](../vendor/LAPS/benchmark/hicache/data_processing.py) has a real ShareGPT conversation loader: at least two messages, even message count, first role human, adjacent user/assistant pairs, minimum four input and four output tokens per retained turn; random conversation shuffle unless disabled. It can repeat conversations to reach the requested count. It does not prove strict role alternation or identify ShareGPT-4. Its fallback dataset is `anon8231489123/ShareGPT_Vicuna_unfiltered`, `ShareGPT_V3_unfiltered_cleaned_split.json`, **not an established match to ShareGPT-4**.

[hicache/bench_serving.py](../vendor/LAPS/benchmark/hicache/bench_serving.py) supports `--enable-multiturn`, carries **generated** assistant text into subsequent user turns, streams chat completions, and supports exponential inter-arrivals. `--num-prompts` selects conversations in this loader; total measured requests are the retained turns. Its cache flush targets the base URL without proving all PD workers flushed. TTFT starts after client semaphore admission, so client queueing is not included. Its summaries are insufficient for our strict per-request validator.

The generic `python/sglang/bench_serving.py` ShareGPT sampler primarily samples prompt/completion pairs; the hicache synthetic multi-turn benchmark is not interchangeable with real conversations. Reuse the real-conversation driver via a local adapter, not synthetic shared-prefix prompts. Persist actual input/output token IDs and all turn records.

## Missing paper details and implementation differences

The paper does not specify enough to recover exact dataset revision/subset, all request counts, chat serialization, model base/instruct revisions, per-model TP, 8-prefill decode placement, arrival rates, or E6 subset/model from its Section 4 prose. These remain unknown, not inferred from throughput claims.

[server_args.py](../vendor/LAPS/python/sglang/srt/server_args.py) and [prefill.py](../vendor/LAPS/python/sglang/srt/disaggregation/prefill.py) show:

- `--enable-laps` expands to piecewise graphs, batch-prefill graphs, and the LAPS scheduler. Use explicit flags to separate the ablations.
- Default graph batches are 1,2,4,8; lengths 16,32,64,128,256. Paper grid adds B=16,32,64 and L=8. Set it explicitly and validate before timing; do not silently trim after OOM.
- Scheduler default threshold is 256, with **inclusive** `<=` comparison on `len(req.origin_input_ids)`, i.e. total context, not the incremental uncached suffix. This can classify a multi-turn cache-hit request as long even when little prefill work remains.
- Current scheduler is a priority short/long queue. There are no current waiting-window CLI fields and no paper AWD/SLO-aware dispatch implementation.
- Local remote ref `origin/laps-waiting-window` at `7f944099bcfb787b476af3083c0cea7ba02ac4c8` supplies static `--laps-wait-window-ms` (default 5) and `--laps-wait-max-batch` (default 8). Static waiting is not the paper's adaptive waiting. This ref was inspected without checkout.
- `origin/laps-dynamic-alloc` is `68bd2e1b5a8bb0f0d7a58c99b341c50294b74666`. Neither remote ref is an authorization to replace the validated working tree.

Vendored [router_args.py](../vendor/LAPS/sgl-model-gateway/bindings/python/src/sglang_router/router_args.py) and [mini_lb.py](../vendor/LAPS/sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py) implement `--enable-laps-alloc`, threshold 256, rebalance interval 5 seconds, pending-request ratio 2.0. They split P workers into short/long groups, preserve at least one per group, and move at most one per rebalance. **Three P instances are needed to exercise migration**; two only provide a static split. Routing is not a demonstrated session-affinity/cache-aware implementation. Text length estimates also do not measure uncached work. The controller lacks the paper's full SLO/service-time pressure model and unspecified hysteresis parameters.

Installed `sglang-router 0.3.2` lacks these allocation flags in its Python router files. Future allocation experiments must select the vendored Python router explicitly and pass a CPU import/help check before launching. Do not claim that prior installed-router PD runs exercised this controller.

Local choices: primary scheduler/router threshold **255** to implement paper short `<256`; graph max length remains 256. Record the total-context classification deviation. An artifact-default threshold-256 sensitivity run is separate. A cache-aware classifier or AWD implementation would be a new implementation requiring review and correctness checks, not a flag change.

## Data and software already available

Inspected default HF cache `/home/apal/.cache/huggingface/hub` and local vendor data. No HF cache-path override was present. This is an inventory of inspected locations, not a claim about every filesystem/mount.

| Asset | Availability |
|---|---|
| Qwen/Qwen2.5-7B | Complete 4-shard cache, revision `d149729398750b98c0af14eb82c78cfe92750796`, approximately 15.23 GB weights |
| Qwen/Qwen2.5-0.5B-Instruct | Complete single shard, revision `7ae557604adf67be50417f59c2c2f167def9a775`; not a paper target |
| Qwen2.5-14B / 32B | No matching cached model directories found; obtain/pin weights before scheduling these experiments |
| ShareGPT-4 | Not found; exact paper dataset ID/revision unresolved. Do not silently substitute ShareGPT V3 |
| LMSys full conversations | No matching HF dataset cache found. Official dataset is gated by license/contact-sharing acceptance. Account entitlement was not tested; no credentials inspected or terms accepted |
| `vendor/LAPS/data/lmsys_chat_10k.jsonl` | 10,000 valid rows, 9,998 unique texts, only `text` field; no roles/conversation IDs or reconstruction metadata. SHA256 `e57461138dc226e399a2b79c3f8e869686fc67d475004274550ef2ee701336ba`. Suitable only for explicitly labeled flat-prompt controls |
| Environment | torch 2.9.1, transformers 4.57.1, datasets 5.0.1, flashinfer-python 0.6.3, nixl 1.5.0, sglang-router 0.3.2; vLLM not installed |

Dataset preparation must retain IDs, role order, original turns, tokenizer-derived lengths, exclusions and dataset revision/hash. Reject malformed role chains. Never silently loop a short dataset. Primary local sampling: deterministic complete conversations, without replacement; 100-conversation pilot then 1,000 conversations per E3 condition, three repeats with seeds 1701/1702/1703. Record total turns as the exact request count. Do not call 1,000 conversations 1,000 requests. Increase counts only after pilot precision/runtime assessment, with a frozen manifest for all policies.

Use real user turns and generated assistant history (matching the reusable driver). Cap each response at its recorded assistant's tokenizer length and ignore EOS to equalize intended output work; use temperature 0. This output-length/serialization choice is **local**, not specified by the paper. Pin one explicit chat template for all policies; base Qwen models must not rely on an assumed default chat template. Save its file and SHA256. Filter complete conversations to a 32,768-token planned context budget, log exclusions, and fail rather than silently truncate if generated history exceeds the limit. Actual generated histories can differ across policies: save and report this, plus cached/uncached token counts. A teacher-forced fixed-history control is useful but distinct from generated-history cache reuse.

## Exact server configuration and hardware layouts

All commands below are future command templates, **not executed during D0**. Readiness depends on the preparation gates below. They use existing server flags; the strict experiment orchestration still needs a local adapter. Never treat these fragments alone as a validated benchmark launcher.

Activate `plaserve`. For **each server**, set unique event directory and retain these values in provenance:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plaserve
export LAPS_FLASHINFER_KERNEL_ROOT="$(realpath results/flashinfer-mask-experiment/saturating-4fbcdb804781)"
export LAPS_KERNEL_EVENT_DIR="$(realpath -m "$RUN/kernel-events/$ROLE")"
export PYTHONPATH="$(realpath scripts/laps_kernel_hook):${PYTHONPATH:-}"
export MAX_JOBS=4
unset LAPS_REPLAY_CONTROL_DIR LAPS_LOGITS_DIR LAPS_DIAGNOSTIC_CASE_FILE LAPS_PD_COVERAGE_DIR
```

Common server argv:

```text
python -m sglang.launch_server --model-path MODEL --tp-size TP
  --dtype bfloat16 --attention-backend flashinfer --page-size 1
  --host 127.0.0.1 --port PORT --mem-fraction-static 0.75
  --context-length 32768 --random-seed 1701
  --disaggregation-mode ROLE --disaggregation-transfer-backend nixl
  --disaggregation-bootstrap-port BOOTSTRAP
```

BF16 LM head is the primary performance setting (paper does not specify an FP32 head); retain the accepted FP32-head diagnostic as a separate correctness control, not a performance baseline. Decode CUDA graphs remain enabled identically in every policy. Do not use `--disable-cuda-graph` to construct vanilla.

| E3 configuration | Additional prefill server flags | Router allocation |
|---|---|---|
| vanilla SGLang PD | none | off |
| graphs-only | `--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph` plus grid below | off |
| disaggregation-only | `--enable-laps-scheduler --laps-length-threshold 255` | on for multi-P |
| full LAPS artifact | union of graphs-only and disaggregation-only | on for multi-P |

Graph flags: `--batch-prefill-max-seq-len 256 --batch-prefill-batch-sizes 1 2 4 8 16 32 64 --batch-prefill-seq-lengths 8 16 32 64 128 256`.
“Disaggregation-only” here means the artifact's length-aware scheduling/allocation ablation; **all four configurations use PD**. With one P, allocation cannot operate. Report this distinction, and that artifact-full is not proven paper-full.

For the router, prepend `vendor/LAPS/sgl-model-gateway/bindings/python/src` after the kernel hook in PYTHONPATH, and use:

```text
python -m sglang_router.launch_router --mini-lb --pd-disaggregation
  --host 127.0.0.1 --port 30302
  --prefill http://127.0.0.1:30300 9301
  --decode http://127.0.0.1:30301
```

Repeat `--prefill URL BOOTSTRAP` / `--decode URL` for additional instances. Allocation adds `--enable-laps-alloc --laps-alloc-threshold 255 --laps-rebalance-interval-s 5 --laps-rebalance-ratio 2`. Use the same router code/version for all four E3 variants. Validate mini-router streaming/chat/flush behavior; do not assume parity with the installed router.

| Layout | GPU placement | Purpose / deviation |
|---|---|---|
| S2, known 7B bridge | P TP2=2,3; D TP2=4,5 | Connect new driver to accepted correctness setup; four GPUs, one P, not paper TP1 |
| S1 | P TP1=2; D TP1=3 | Primary 7B and 14B single-P; 14B requires download and memory gate |
| M3 | P TP1=2 / 3 / 4; D TP1=5 | Primary scaled 7B/14B multi-P, three P rather than eight; exercises migration, may be decode limited |
| S32 | P TP2=2,3; D TP2=4,5 | 32B primary; TP2 avoids assuming 64-GB weights plus full graph/KV budget fit TP1 |
| M32 | P TP2=2,3 / 4,5; D TP2=6,7 | Requires approval; only two P, cannot exercise allocation migration; label static split |
| M5, optional | P TP1=2 / 3 / 4 / 5 / 6; D TP1=7 | Requires approval; closest instance-count scaling (5P), decode bottleneck risk |
| D22 | P TP1=2 / 3; D TP1=4 / 5 | E6 fallback without additional GPUs; 2P+2D versus paper 4P+4D |
| D33 | P TP1=2 / 3 / 4; D TP1=5 / 6 / 7 | E6 preferred scaled 3P+3D, requires approval |

Assign P instance j port `30300+10*j`, bootstrap `9301+10*j`; D j port `30301+10*j`, bootstrap `9300+10*j`; router 30302. Mixed instances use 30400/30410. Confirm all ports are free first. Same resources per policy within each comparison. No co-location of multiple model instances on one GPU to manufacture eight instances. Eight P plus decode is not faithfully available on six GPUs, and the paper does not resolve its placement sufficiently to emulate it by TP arithmetic.

## Experiment specifications

### E3: throughput, TTFT and latency

Primary grid: cc **1,2,4,8,16,32,64**, four policies, S1 then M3 for 7B; S1/M3 for 14B; S32 and approved M32 for 32B. Three repeats, frozen conversation selection per repeat and identical ordering across policies. Rotate policy order across repeats to reduce drift. Closed-loop concurrency limits active turns; a conversation's next turn waits for its prior reply. No synthetic think time (local choice). Report request/turn RPS and conversation completion rate separately, mean/P90 TTFT and mean/P90 full request latency. Report repeat means and sample standard deviations, and paired LAPS/vanilla ratios with spread.

Existing driver command for a **development smoke run**, after a dataset/template exists (not the final validated experiment):

```text
python vendor/LAPS/benchmark/hicache/bench_serving.py
  --backend sglang --base-url http://127.0.0.1:30302
  --model Qwen/Qwen2.5-7B --dataset-name sharegpt
  --dataset-path DATASET.json --enable-multiturn
  --num-prompts 100 --max-concurrency 1 --request-rate inf
  --seed 1701 --output-file RUN/driver.jsonl --chat-template TEMPLATE
```

Preparation must check the template argument's actual handling and serialize consistently in the local adapter; the presence of a CLI option alone is not proof it affects this chat path. The raw driver must not be used to claim a pass. Extend it locally for token records, strict accounting, all-worker flush, dataset no-repeat enforcement and validated SSE parsing.

Compare percentage improvements against paper +20% single / +33% eight-instance and ~20% average-latency reduction only as context. Hardware, scaled topology, scheduler/controller and dataset differences preclude an exact claim replication. Oct 16's 118.4 versus 133.6 RPS was a buggy-kernel flat-prompt result, not an E3 baseline.

### E4: TTFT SLO under arrivals

Use real LMSys conversations once access and a pinned export are available. Primary local model 7B, S1/M3 (paper model choice/rate details insufficiently specified). Compare vanilla PD and full artifact with the same router. A distinct “SGLang+router” baseline must declare routing policy and topology; do not duplicate vanilla under a new label. Evaluate installed/native router versus vendored mini-router as a separately labeled router control only after protocol and kernel checks. vLLM PD is **blocked**: not installed, no artifact launch recipe, and the SGLang sitecustomize adapter is not proof of corrected vLLM attention. No installation under present instructions.

Use frozen exponential inter-arrival traces, seeds 1701–1703. Local offered-rate grid: 0.25,0.5,0.75,1.0,1.25 times vanilla sustainable request rate measured in a separate calibration; freeze the resulting numeric rates for all policies. Cross with cc 1/2/4/8/16/32/64. Use 10,000 scheduled turns per condition with turn dependencies enforced; report realized arrival rates because dependencies can delay turns. Also run an independent first-turn Poisson control so turn dependencies cannot be mistaken for a pure Poisson process.

Primary SLO: first token minus **scheduled arrival** >0.4s, including client waiting. Also report dispatch-to-first-token to compare server-only definitions. Keep client queue timestamps, no dropped requests; errors/timeouts count as violations and invalidate the strict throughput run. Report counts and denominators, confidence intervals, and percentage-point versus relative changes separately. Zero observed violations is not a guaranteed zero rate. Paper numeric arrival rates and exact SLO origin remain unresolved deviations.

### E6: offline distillation

Use D33 if approved, otherwise D22; fixed Qwen2.5-7B (local choice because paper table does not identify the model sufficiently). Compare vanilla versus full artifact on both datasets, greedy, `ignore_eos=true`, **1,024 output tokens for every request**. Pilot 100 input prompts/dataset; main local count 1,000/dataset, three repeats; optionally expand to 10,000 only after cost calibration. Preserve sampled dataset prompt-length distributions and publish IDs; this is not an assertion about the paper's request count. Offline queue always has work; choose a common concurrency via a 16/32/64 pilot, then freeze it for both policies. Measure first dispatch to final completion, separately report startup/capture time, all tokens and all completion counts. Do not convert the table's 18,963/17,572 or 25,132/23,041 seconds into a workload-size assumption.

D33 has three P workers and can exercise migration; D22 cannot dynamically migrate while retaining one worker per group. Report that limitation. Decode length alone does not define a distillation workload: prompt selection and total jobs are mandatory provenance.

### E1: waiting-window sensitivity

Current HEAD cannot run this experiment: waiting-window flags are absent. Before implementation, extract a reviewed static-window patch from the local branch into an isolated future worktree, preserve the corrected Blackwell adapter and kernel injection, and revalidate. Do not change installed packages. Exact AWD still requires author parameters/implementation; a static sweep is an approximation.

Proposed local static sweep: 14B TP1 P=2, D=3, cc64, only tokenizer-verified new prompts of length <256, one output token to isolate prefill. W=0,0.1,0.25,0.5,1,2,5,10 ms; fixed `--laps-wait-max-batch 8` (branch default), then separate target-64 sensitivity. Use full graph grid, 10,000 requests and three repeats. These window values, batch targets and output length are local choices, not recovered paper figure-axis values. Report TTFT/RPS, realized batch sizes, waiting time and utilization. The paper's one-H200 configuration versus our extra decode worker is explicit.

### E5: PD versus mixed decode

7B TP1: one mixed worker GPU2 versus P2+D3; two mixed workers GPUs2,3 versus P2/P3+D4/D5. Compare equal **prefill instance count**, and separately state the unequal total GPU cost. Use identical tokenized prompt trace and 1,024-token generations (local workload choice), concurrency 1/2/4/8/16/32/64, three repeats, 1,000 requests. Mixed server uses common flags with PD role/bootstrap/transfer options omitted; expose graphs-only versus full-PD as named controls because the LAPS scheduler is disabled outside prefill mode. Measure actual prefill tokens/s from server counters alongside TTFT/RPS and background decode throughput. Without those counters, do not relabel response RPS as prefill throughput. Existing one-token `bench_1gpu.sh` does not establish decode interference.

## Mandatory preparation and per-run acceptance

1. Resolve ShareGPT-4 identity/revision with the artifact authors or user-supplied manifest. If unavailable, run a clearly named ShareGPT-V3 proxy, not a paper replication. Obtain LMSys access without silently accepting terms; obtain missing model weights/revisions. No downloads in D0.
2. Build local wrapper/driver adapters and CPU tests; preserve vendor and installed packages. Reuse owned-process handling and provenance from [run_bench_fixed_kernels.sh](../scripts/run_bench_fixed_kernels.sh). Existing [validate_bench_results.py](../scripts/validate_bench_results.py) expects three flat-prompt policies and TP2 evidence; generalize configuration-aware checks rather than weakening them. It cannot certify these new experiments unchanged.
3. Before each repeat check GPU visibility, UUIDs, utilization/memory, compute PIDs and owners, plus all listening ports. GPUs must be free. If another user's process appears, stop after the current repeat and report; never kill it. Monitor GPU0–1 activity read-only to annotate possible shared-host interference, never allocate there. Run approved experiments in tmux/nohup with durable logs.
4. First test TP1 and the full 49-shape paper graph grid for memory, mask correctness, graph replay and router correctness. Run full-model/FP32 near-tie checks for any new divergence. Log graph capture count, peak memory and fallback frequency. Reject OOM/grid fallback as a faithful-grid pass; a reduced-grid experiment must be a separately labeled deviation. Apply corresponding checks to 14B/32B and the vendored router before timing.
5. Warm up with disjoint inputs, drain outstanding requests, then explicitly flush radix caches on **every P and D worker** before each policy, concurrency/rate level, and repeat. Save successful flush replies and verify cache reset with available counters. Router-only flush is insufficient. Keep cache enabled throughout the measured run; never flush between turns. Measure prefix-cache hits, router placement and migrations. Source router affinity is not assumed: document cache loss from random placement. Adding affinity is a separate controlled change shared across applicable policies.
6. For every role/worker, require kernel configured and loaded events with matching corrected-kernel hashes, correct package version, expected PID/TP ranks and device mapping. Header presence or environment variables alone do not prove the fix was used. Missing events fail even if request metrics look good. Disable diagnostics during timing.
7. Validator must require every planned policy/level/repeat, exact planned conversations/turns, unique request IDs, zero failed/missing/duplicate requests, completed=requested, correct output lengths, valid finish reasons, finite metrics, valid streaming timestamps, complete provenance and kernel evidence. For E4 include every scheduled arrival in the SLO denominator. Validate per-request data against summaries; process exit code alone never passes. Fail on unexplained fallback, skipped samples or truncated turns. Archive failed runs too.
8. Save fresh output directories per repeat/configuration, with child paths for each condition and no overwritten events. Stop/report after each approved phase. Only terminate registered process groups started by that launcher.

## Budget estimates and stopping gates

These are **planning envelopes**, not measured Blackwell multi-turn performance. Actual lengths, decode throughput, graph capture and data access dominate. GPU-hours count all reserved workers during each run, including idle PD partners; CPU/data preparation is separate. No inference from the flat one-token benchmark is used to claim multi-turn speed.

Estimate each condition with `GPU-hours = allocated_GPUs * (startup_seconds + total_output_tokens / measured_aggregate_decode_tokens_per_second + nonoverlapped_prefill_seconds) / 3600`. Use actual pilot lengths, cache-hit rate and measured concurrency-specific throughput; this is a planning approximation, not a substitute for measured elapsed time. Concurrency 1 can be far more expensive than saturated batching.

| Stage | Provisional GPU-hour allowance | Basis / release gate |
|---|---:|---|
| TP1/grid/router correctness + 7B 100-conversation E3 pilot | 4–12 | First authorized GPU work; measure rather than assume full-grid fit |
| E3 7B S1+M3, four policies, seven cc, three repeats, 1,000 conversations | 100–1,000 | 168 condition-runs; final budget depends on total retained turns/output lengths, especially cc1 |
| E3 14B same matrix | 200–2,000 | Missing model; revise after 100-conversation pilot |
| E3 32B S32 plus approved M32 | 400–4,000 | More GPUs and slower decode; optional multi-P is static allocation |
| E4 two available policies, two layouts, five rates, seven cc, three repeats | 100–1,000 | 420 condition-runs at 10k scheduled turns; low arrival rates can dominate wall time; reduce only as a declared pilot |
| E5 7B | 20–200 | Long-output workload; calibrate before full grid |
| E1 static sweep | 5–30 | One-token workload; implementation/correctness prerequisite excluded |
| E6 1,000 jobs × 1,024 outputs, two policies × two datasets × three repeats, D33 | 20–205 | Assumed aggregate 100–1,000 output tokens/s: 0.28–2.84 hours/run ×6 GPUs ×12; add startup/prefill overhead |
| E6 10,000-job expansion | 205–2,050 plus overhead | Tenfold output work; not authorized by this plan |

The envelopes intentionally allow large uncertainty; they are not a commitment to spend the upper bound. After the first pilot, publish measured per-condition runtime estimates and a concrete budget before releasing the large matrices. Full-million-conversation LMSys processing is not implied by the dataset name and is outside this sampled plan. If runtime is excessive, publish a smaller declared pilot or defer the experiment; do not silently reduce counts or omit slow concurrency levels from a claimed full matrix.

## What this plan can establish

A successful first phase can establish corrected-kernel E3 behavior on 7B with real multi-turn reuse and measured cache behavior. Scaled multi-P results can test the available controller, not reproduce an eight-prefill H200 result. E1 adaptive waiting, exact ShareGPT-4 identity, a distinct paper router baseline and vLLM PD remain unresolved. The report must retain those limitations even if observed percentage gains resemble the paper.
