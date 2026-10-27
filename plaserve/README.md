# LAPS on RTX PRO 6000 Blackwell

This folder reproduces the released
[LAPS](https://github.com/Jianshu-She/LAPS) prefill-scheduling artifact
([paper](https://arxiv.org/abs/2601.11589)) on 8× NVIDIA RTX PRO 6000 Blackwell (96 GB).
It contains the benchmark harness, the fixes needed to run LAPS correctly on Blackwell,
and the measured results.

It is independent of the SGLang source in the rest of this repository. LAPS ships its own
SGLang snapshot, which is checked out into `vendor/LAPS` and patched (see [Reproduce](#reproduce)).
The separate LAPS reimplementation on the `dsp` branch (`python/sglang/srt/dynamic_sp/laps.py`)
was not used here.

## Result

LAPS raises request throughput over vanilla SGLang by **1.07× at concurrency 1, 1.15× at 4,
1.38× at 8 and 1.29× at 16**. Values are the mean of four repeats that ran with no other jobs on the GPUs.

![Throughput](results/phase-c-20261027/figures/throughput.png)

![LAPS / vanilla speedup](results/phase-c-20261027/figures/speedup.png)

![Mean latency](results/phase-c-20261027/figures/latency.png)

| Policy | Concurrency | Requests/s | Mean latency (ms) | P90 latency (ms) |
|---|---:|---:|---:|---:|
| LAPS | 1 | 46.5 ± 0.2 | 21.5 ± 0.1 | 23.3 ± 0.1 |
| Vanilla SGLang | 1 | 43.6 ± 0.5 | 22.9 ± 0.3 | 24.5 ± 0.2 |
| Disaggregation | 1 | 43.5 ± 0.4 | 22.9 ± 0.2 | 24.9 ± 0.5 |
| LAPS | 4 | 130.0 ± 1.2 | 30.6 ± 0.3 | 33.0 ± 0.3 |
| Vanilla SGLang | 4 | 113.5 ± 2.4 | 35.1 ± 0.8 | 40.2 ± 1.7 |
| Disaggregation | 4 | 92.4 ± 4.3 | 43.3 ± 1.9 | 47.5 ± 0.7 |
| LAPS | 8 | 219.0 ± 3.0 | 36.3 ± 0.5 | 43.2 ± 0.8 |
| Vanilla SGLang | 8 | 158.7 ± 6.1 | 50.2 ± 2.0 | 56.5 ± 2.4 |
| Disaggregation | 8 | 128.4 ± 10.4 | 62.2 ± 4.9 | 56.0 ± 1.3 |
| LAPS | 16 | 326.3 ± 12.3 | 48.4 ± 1.8 | 58.5 ± 1.2 |
| Vanilla SGLang | 16 | 252.7 ± 4.3 | 62.7 ± 1.1 | 69.2 ± 2.3 |
| Disaggregation | 16 | 192.5 ± 11.3 | 82.4 ± 5.1 | 72.8 ± 2.9 |

Mean ± sample standard deviation over repeats 1, 2, 5 and 6 (`results/phase-c-20261027/aggregate-clean-1-2-5-6/`).
Repeats 3 and 4 also passed every check. Another user's job shared GPUs 2–3 for part of each,
so they are plotted as hollow points but excluded from the table. An earlier two-repeat run on
2026-10-17 gave the same picture (`results/phase-c-20261017/partial-summary.md`).

### Setup

| | |
|---|---|
| Model | Qwen/Qwen2.5-7B, revision `d149729`, BF16 |
| Layout | 1 prefill instance on GPUs 2,3 (TP2) + 1 decode instance on GPUs 4,5 (TP2), NIXL KV transfer |
| Workload | 500 prompts from the artifact's `data/lmsys_chat_10k.jsonl`, 1 output token, 20 warmup requests |
| Policies | All three use the same prefill server + decode server + router. Only the prefill server's flags differ. The names are the artifact's labels. **Vanilla SGLang**: default scheduler. **Disaggregation**: short-first queue (`--enable-laps-scheduler --laps-length-threshold 256`), no prefill CUDA graphs. **LAPS**: that queue plus piecewise and batch-prefill CUDA graphs |
| Software | torch 2.9.1+cu128, flashinfer 0.6.3 (corrected kernel), nixl 1.5.0, driver 580.159.04 (`environment/`) |

"Latency" is end-to-end time for a one-token request, which the artifact reports as TTFT.

The short-first queue alone ("Disaggregation") is slower than vanilla from concurrency 4 upward.
The gain comes from adding the prefill CUDA graphs ("LAPS").

### What this does and does not show

- It reproduces the **released artifact's throughput benchmark**, on Blackwell hardware.
- It is **not** the paper's main experiment. That uses multi-turn ShareGPT-4 and LMSys
  conversations with long outputs, an H200 cluster, and larger models. The artifact's benchmark
  uses flat single-turn prompts that generate one token. The paper's dataset version is not identified.
- Only one prefill instance is used, so the router's dynamic short/long instance allocation
  (`--enable-laps-alloc`) is not exercised. The installed `sglang-router` 0.3.2 also lacks those flags.
- The plan for the remaining paper experiments is in [docs/PAPER_REPRO_PLAN.md](docs/PAPER_REPRO_PLAN.md).

## Blackwell fixes

The unmodified artifact produced wrong outputs on Blackwell. Two changes were needed:

1. **FlashInfer attention mask** ([docs/FLASHINFER_MASK_INVESTIGATION.md](docs/FLASHINFER_MASK_INVESTIGATION.md)).
   With padded queries in captured CUDA graphs, FlashInfer 0.6.3's FA2 prefill kernel computes the
   causal mask with arithmetic that wraps around. A saturating version of that arithmetic
   (`kernels/flashinfer-saturating-4fbcdb804781/kernel.patch`) brings an isolated test with
   maximum error 3.85 down to 0.009 across the full attention suite.
   The benchmark loads it through `scripts/laps_kernel_hook/`, which checks header hashes and
   logs every load. Validation fails if any worker did not load the corrected kernel.
2. **FlashInfer batch-prefill adapter** ([docs/BLACKWELL_PATCH.md](docs/BLACKWELL_PATCH.md)).
   The artifact's batch-prefill CUDA-graph path did not match the runner's capture/replay interface
   for page size 1. The fix is in `patches/laps-blackwell-vendor.diff`.

Correctness after the fixes ([docs/FULL_MODEL_VERIFICATION.md](docs/FULL_MODEL_VERIFICATION.md),
`results/correctness-20261017/`):
- Prefill/decode-disaggregated output matched single-server output for every input but one.
- That input is a near-tie: in an FP32 reference, its top two next-token logits differ by 0.105 (`fp32-near-tie.json`).

## Reproduce

Requirements: Linux, 6+ NVIDIA GPUs (the scripts use GPUs 2–5), CUDA 12.8 driver, conda.

```bash
cd plaserve

# 1. LAPS artifact at the tested commit, plus the Blackwell fixes.
git clone https://github.com/Jianshu-She/LAPS vendor/LAPS
git -C vendor/LAPS checkout 88b986375b531b2f09261b9f44628a24a9c0d6d7
git -C vendor/LAPS apply ../../patches/laps-blackwell-vendor.diff

# 2. Environment (exact versions in environment/pip-freeze.txt).
conda create -n plaserve python=3.12 -y && conda activate plaserve
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e vendor/LAPS/python flashinfer-python==0.6.3 flashinfer-cubin==0.6.3 nixl==1.5.0 sglang-router==0.3.2 transformers==4.57.1

# 3. Model weights.
huggingface-cli download Qwen/Qwen2.5-7B --revision d149729398750b98c0af14eb82c78cfe92750796

# 4. Three repeats × {vanilla, disaggregation, LAPS} × concurrency {1, 4, 8, 16}.
python -u scripts/run_phase_c.py --output results/phase-c-$(date +%Y%m%d-%H%M%S)

# 5. Plots.
python scripts/plot_phase_c.py results/phase-c-<timestamp> --clean 1 2 3
```

The corrected FlashInfer kernel compiles on first use into
`kernels/flashinfer-saturating-4fbcdb804781/cached_ops/`, which takes a few minutes.

Each repeat takes about 6 minutes. Before starting, the runner checks that GPUs 2–5 are empty.
After every repeat it validates all 12 policy/concurrency results: request counts, finite metrics,
corrected-kernel load events from every worker, and vendor diff hash. It stops if another user's
process appears on GPUs 2–5. To add repeats to an existing campaign, retrying until one runs
alone on the GPUs:

```bash
bash scripts/run_phase_c_clean_repeat.sh results/phase-c-<timestamp> 4 5
```

Notes:
- If `HF_HOME` points to a cache you cannot read, set `HF_HOME=~/.cache/huggingface`.
- To use other GPUs, edit `CUDA_VISIBLE_DEVICES` in `scripts/run_bench_fixed_kernels.sh` and
  the check in `scripts/run_phase_c.py`.

## Where the splits are configured

Paths are inside `vendor/LAPS` after step 1.

| What | Where |
|---|---|
| Prefill vs decode GPUs, TP size | `bench_laps_prefill_throughput/scripts/bench_h100_4gpu.sh`: `PREFILL_GPUS`, `DECODE_GPUS`, `TP_SIZE`. Other layouts (2-GPU 7B, 8-GPU 14B/32B/72B) in the same directory |
| Short/long queues in a prefill server | `python/sglang/srt/disaggregation/prefill.py`: `_laps_classify_requests` (short if input ≤ threshold), `_get_next_laps_batch_to_run` (short first). Flags `--enable-laps-scheduler`, `--laps-length-threshold` in `python/sglang/srt/server_args.py` |
| Short/long prefill instance groups (router) | `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py` and `router_args.py`: `--enable-laps-alloc`. Needs ≥3 prefill instances to move instances between groups. Not used in these runs |

## Contents

| Path | |
|---|---|
| `scripts/run_phase_c.py` | Three-repeat campaign runner with GPU-occupancy and interference checks |
| `scripts/run_bench_fixed_kernels.sh` | One repeat: launches the artifact benchmark with the corrected kernel |
| `scripts/validate_bench_results.py` | Per-repeat validation |
| `scripts/plot_phase_c.py` | Figures |
| `scripts/check_*.py` | Attention, full-model and disaggregation correctness checks |
| `scripts/laps_kernel_hook/` | Import hook that loads and logs the corrected FlashInfer kernel |
| `patches/` | Vendor diff against LAPS `88b9863` |
| `kernels/` | Corrected FlashInfer headers and manifest |
| `results/phase-c-20261027/` | Six repeats: validation, provenance, kernel-load events, per-policy benchmark JSON, aggregates, figures |
| `results/correctness-20261017/` | Disaggregated vs single-server output check and FP32 near-tie reference |
| `docs/` | Investigation notes and the full paper-reproduction plan |
| `plaserve/`, `tests/` | CPU scheduling simulator (illustrative timings only) and unit tests |
| `environment/` | `pip freeze` and system versions |
