# FA2 padded-query mask investigation

## Confirmed attention result on advent

The freshly compiled original header reproduced the isolated width64 failure
(maximum error3.84766). The saturating arithmetic correction passed that case
and the entire expanded attention suite: every compared value was within
atol0.03/rtol0.03, and direct versus captured replay was bitwise identical.
Maximum reported error across the full suite was0.00935578. Cases included
batch8/width256, dummy-sized rows, cached history, changed inputs and multiple
captured shapes.

Original header SHA256 prefix: `30375ad157ff`; corrected: `4fbcdb804781`.
The validated corrected kernel root on advent is
`results/flashinfer-mask-experiment/saturating-4fbcdb804781`.
This controlled experiment establishes the arithmetic correction fixes the
reproduced attention failure. Full-model correctness remains to be rerun.

## Full-model rerun using corrected kernels

The new `--kernel-root` option to `check_blackwell_model.py` injects an opt-in
import hook through child-process PYTHONPATH. It does not initialize CUDA at
interpreter startup. When a worker imports FlashInfer, the hook verifies installed
and corrected header hashes/version against the passed attention manifest,
redirects build/include/AOT paths, and logs successful attention kernel loads.
The harness requires at least TP distinct worker PIDs with corrected kernel load
evidence in both modes. This verifies subprocesses did not silently use the
original kernels. Hook event logs are retained alongside model results.

```bash
KR=results/flashinfer-mask-experiment
KR="$KR/saturating-4fbcdb804781"
C=scripts/check_blackwell_model.py
OUT=results/model-fixed-$(date +%Y%m%d-%H%M%S)
python "$C" --kernel-root "$KR" --output "$OUT"
```

Keep replay instrumentation installed for correctness testing. The default
baseline remains eager. Previous throughput measurements used kernels with
the now-reproduced padding failure; they cannot establish a correctness-validated
performance result and must be repeated after full-model validation passes.

On advent, Qwen per-rank attention dimensions (14 Q heads, 2 KV heads, dimension
128) fail at Q width64 with KV lengths32,33,63,64. Real queries are right-aligned.
Rows with shorter KV fail (maximum error around3); the row with KV64 passes.
Direct execution with the graph-enabled wrapper and CUDA Graph replay are
bitwise identical. This also reproduces with only that shape captured, so neither
cross-shape buffer reuse nor replay alone explains the failure.

## Source-based hypothesis, pending on-device confirmation

FlashInfer v0.6.3's FA2 prefill kernel declares Q/KV lengths unsigned. Its causal
mask iteration calculation subtracts Q length before calling a saturating
subtraction helper. A negative boundary can wrap before the clamp, causing a KV
tile to skip causal masking. Example: KV32, Q64, tile start0, tileKV32 gives
mask_iteration1 with unsigned wrap instead of0. Small KV lengths can hide this
because the subsequent tile-size division rounds down to zero.

Primary source:
https://github.com/flashinfer-ai/flashinfer/blob/v0.6.3/include/flashinfer/attention/prefill.cuh

This is a strong hypothesis, not yet proof that the installed binary has this
specific defect or that it explains all full-model mismatches. The adapter
introduced padded Q greater than real KV, so its compatibility assumption was
not sufficiently tested by the original small-shape fixture.

## Isolated experiment

`scripts/check_flashinfer_mask_fix.py` reads installed headers, checks expected
expressions, copies headers to a results subdirectory, and changes only the
boundary arithmetic in that copy. It redirects FlashInfer's include, generated
source, JIT and AOT lookup paths before importing the attention checker. The
installed package and its existing kernel caches are untouched. It verifies
that attention build commands refer to the copied include directory.

Run each in the plaserve environment, with GPU0 available:

```bash
export CUDA_VISIBLE_DEVICES=0
K=scripts/check_flashinfer_mask_fix.py
python "$K" --unpatched --case 4
python "$K" --case 4
python "$K"
```

The first command is a control: it should reproduce the numerical failure using
fresh compilation of original installed headers. The second tests the isolated
arithmetic correction; the third covers all shapes. New JIT compilation may
take time. Preserve generated manifests, source diff and compiler logs under
`results/flashinfer-mask-experiment`. If the expected source expressions are
missing, the experiment stops rather than applying a blind patch.

A pass would establish attention correctness for these cases only. Before
promoting the change, rerun full-model comparisons with the same isolated kernel
configuration, validate PD transport, and retain provenance for benchmark runs.
Do not simply rerun full-model checks using unmodified installed kernels: this
experiment's runtime overrides apply only to its current Python process.
