# Experimental FlashInfer adapter for the supplied LAPS checkout

This patch targets the source supplied from advent (reported commit 88b9863).
It does not update CUDA, PyTorch, or packages. GPU execution is not validated on
the local machine. Keep the FA4 source unchanged and select FlashInfer explicitly
when testing this adapter.

Changes in `patches/blackwell-flashinfer.patch`:

- Match the runner's batch-prefill initialization/capture/replay interface.
- Allocate independent KV indices with capacity batch size × context length.
- Retain wrappers for every captured shape and reject uncaptured replay.
- Keep padded Q lengths fixed while preserving actual KV lengths. Synthetic
  prefix lengths feed only the updater's Q-offset computation, not cache writes.
- Reject sliding-window, encoder-decoder, diffusion, and page sizes other than 1.

The patch is outside ignored `vendor/` so it can be versioned and transferred.
The extracted vendor checkout is partial; do not run setup against it locally.

## On advent

Copy `patches/blackwell-flashinfer.patch` and
`scripts/check_blackwell_attention.py` into the same paths in PLAserve.
From the PLAserve root, in the existing `plaserve` environment:

```bash
git -C vendor/LAPS apply --check ../../patches/blackwell-flashinfer.patch
git -C vendor/LAPS apply ../../patches/blackwell-flashinfer.patch
CUDA_VISIBLE_DEVICES=0 python scripts/check_blackwell_attention.py
```

The checker uses the actual upstream adapter, indices updater and FlashInfer FA2
kernels. It captures two graph shapes and alternates replays with changed Q/K/V,
request indices, lengths, padding and cached history. Real query outputs are
compared with PyTorch SDPA in FP32. A missing dependency, kernel failure, NaN or
numerical mismatch must be fixed before proceeding. It does not load a model or
validate the runner's KV-write behavior, routing, transfer or full serving.

After the checker passes, inspect the benchmark script and change BOTH prefill
and decode server commands to `--attention-backend flashinfer --page-size 1`.
Preserve physical GPU mapping 2,3 for prefill and 4,5 for decode. Commit intentional
upstream changes before using the PLAserve launcher, which requires clean tracked
files. Start with minimal model requests and verify LAPS graph capture/replay,
then compare eager and graph outputs for the full model before a sweep.

Do not interpret the benchmark's exit code or router `/health` alone as success.
The known launcher result-validation weakness is not repaired by this kernel
adapter. Require successful request counts and numeric metrics for every policy,
including LAPS. A full benchmark remains unvalidated until those checks pass.

To revert just the adapter (before later modifications):

```bash
git -C vendor/LAPS apply -R ../../patches/blackwell-flashinfer.patch
```
