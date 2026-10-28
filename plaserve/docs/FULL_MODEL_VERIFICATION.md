# Full-model validation on advent

## Remaining failures after the FA2 arithmetic correction

Corrected-kernel full-model runs reduced mismatches from14 to6. The same six
outputs differ with either eager or piecewise baseline. First-token differences
remain at cold-6 row3 (92 vs3417) and cached-0 row2 (4293 vs2549). Other cold-6
outputs first differ at generated token index3. Kernel-load evidence confirms
both TP workers in both modes used corrected attention builds. All graph/cache
coverage checks pass. Exact full-model equivalence remains unresolved.

Use `--dump-logits --cases cold-6,cached-0 --counts 1` with `--kernel-root` and
`--baseline piecewise` to capture first-prefill logits without requesting API
logprobs. The opt-in worker hook wraps ModelRunner.forward, saves input metadata
and logits after forward returns, and never records tensors inside graph capture.
No source edits or performance conclusions are involved. It stops if it cannot
identify the expected forward output. The analyzer matches rows by actual token
suffix and total sequence length, so changed batch order does not silently pair
different requests. It reports errors and top-token margins without marking the
run correct or relaxing token-equivalence requirements.

```bash
ARGS=(--kernel-root "$KR" --dump-logits)
ARGS+=(--cases cold-6,cached-0 --counts 1)
ARGS+=(--baseline piecewise)
OUT=results/model-logits-$(date +%Y%m%d-%H%M%S)
python scripts/check_blackwell_model.py "${ARGS[@]}" --output "$OUT"
python scripts/summarize_model_logits.py "$OUT"
```

Run the summarizer even when the completed comparison reports token mismatches.
It saves `logit-comparison.json`. A server startup/forward failure is different:
inspect the server log before attempting to interpret incomplete captures.

## October 16: first full-model comparison failed

Server run `results/model-check-20261016-170056` completed both modes with all
coverage checks true and 40 replay log events (including both TP ranks). It
reported 14 request-output mismatches. Four inputs differed on the first token
in both one-token and eight-token runs: cold-2 row 1, cold-3 row 1, cold-6 row 3,
and cached-0 row 2. Other differences appeared later during decoding. These
results do not establish full-model correctness; their cause is unresolved.

The original comparison changed ordinary CUDA graphs and piecewise compilation
as well as batch-prefill replay. The checker now accepts `--baseline piecewise`
to keep ordinary CUDA graphs and piecewise compilation on both sides while
adding only batch-prefill graphs to the candidate. `--baseline cuda` is also
available for further isolation. Baseline logs must contain no batch-prefill
replay instrumentation events. These controls do not replace the failing eager
comparison, which remains an outstanding correctness gate.

The attention checker now accepts `--model Qwen/Qwen2.5-7B --tp 2` to read actual
per-rank head dimensions and exercise larger shapes implicated in the failures.
The original passing attention fixture used smaller synthetic dimensions.

```bash
python scripts/check_blackwell_attention.py --model Qwen/Qwen2.5-7B
OUT=results/model-control-$(date +%Y%m%d-%H%M%S)
python scripts/check_blackwell_model.py --baseline piecewise --output "$OUT"
```

The attention-level checker passed on advent. A four-GPU FlashInfer serving
trial completed all requests, and temporary logging confirmed batch-prefill
graph replay. The 100-request comparison completed all 600 measured requests
across three policies and two concurrency levels, with zero reported failures.
At concurrency 4, vanilla/LAPS throughput was 118.4/133.6 req/s; these short
measurements are preliminary, not a complete paper reproduction.

## Full-model greedy regression

Transfer `model-check.tgz` to advent through the Mac, as with the adapter bundle.
From the PLAserve root in the existing environment:

```bash
tar -xzf ~/model-check.tgz
python scripts/instrument_laps_verification.py
OUT=results/model-check-$(date +%Y%m%d-%H%M%S)
python scripts/check_blackwell_model.py --output "$OUT"
```

Requires GPUs 2 and 3 to be available. Uses two sequential TP=2 servers on port
30400, the same Qwen/Qwen2.5-7B weights, BF16 runtime default, FlashInfer/page size
1 and LAPS scheduling. Baseline disables ordinary CUDA graphs and does not enable
piecewise/batch-prefill graphs; candidate enables piecewise/batch-prefill graphs.
The tool signals only its own spawned process group during cleanup. It requires
a fresh output directory and saves commands, inputs, raw generation responses,
server logs, token mismatches, replay coverage and a pass/fail manifest.

The workload spans graph length boundaries, batch sizes including padding,
changed inputs, longer fallback requests, and shared-prefix requests primed
before generation. Each case starts with a cache flush. Both one-token prefill
predictions and eight-token continuations use greedy sampling with EOS ignored.
No logprobs are requested: this fork excludes logprob requests from the graph
path, which would invalidate an apparent graph-versus-eager comparison.

A pass requires exact generated token IDs and evidence of multiple replay
shapes, batch padding, query padding, cached history in replay, and API-reported
cache hits. Failure due to absent evidence is not a correctness pass, even when
tokens agree. Scheduling may prevent a case from reaching an intended graph;
use the saved traces to diagnose that rather than weakening coverage checks.

After testing, remove diagnostic logging before performance measurements:

```bash
python scripts/instrument_laps_verification.py --remove
```

## Limits and remaining reproduction work

This is an end-to-end full-model greedy-token regression check on one serving
instance. It does not prove full-logit equivalence, all possible inputs, or
correctness across the prefill/decode transport. Exact tokens may also disagree
because of near-tied logits; investigate any mismatch rather than accepting it
as numerical noise. The attention check provides separate numerical evidence.

After the regression passes, validate the same workload through PD serving and
test cached/multi-turn behavior there. Preserve source revisions, configuration,
hardware metadata and dataset hashes. Repair benchmark request/result failure
propagation and process cleanup before unattended sweeps. Then run longer,
repeated, matched-policy workloads across the paper's settings; report variation,
graph coverage and any backend/hardware deviations. Blackwell with the patched
FlashInfer implementation is an adapted reproduction, not identical to the
authors' hardware and kernels. The six-GPU layout remains undesigned.
