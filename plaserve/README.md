# LAPS on RTX PRO 6000 Blackwell

[LAPS](https://arxiv.org/abs/2601.11589) separates short and long prefill requests
and reallocates serving instances between the two classes. The native baseline
ports the [released scheduling policies](https://github.com/Jianshu-She/LAPS/tree/aa29962d501236ecd2a0c7a8470c5977df3fe057)
to this checkout's SGLang Engine. Prefill uses native paged KV and the released
rectangular and piecewise CUDA Graph runner through a compatibility layer.

Run from the repository root:

```bash
./launch_dynamic_sp_8gpu.sh \
  --scheduler laps \
  --eval-dataset emozilla/pg19 --eval-split test \
  --offered-load 25000 --no-verify
```

The launcher uses the current Python environment. No additional installation or
cloning is required. The default deployment has eight single-GPU serving
replicas. Configure native execution and routing with
`--laps-config python/sglang/srt/dynamic_sp/laps_native.json`.

See [LAPS native baseline](../python/sglang/srt/dynamic_sp/LAPS.md) for scheduling,
configuration, workload handling and metric definitions.

## Artifact support

This directory also contains the separate upstream artifact's benchmark scripts
and Blackwell compatibility patches.

| Path | Description |
|---|---|
| `patches/` | Upstream FlashInfer batch-prefill adapter changes |
| `kernels/` | FlashInfer causal-mask correction for padded graph queries |
| `scripts/` | Upstream artifact launch and benchmark utilities |
| `environment/` | Upstream artifact dependency versions |

The root launcher uses the current SGLang runtime and does not invoke these
vendor-environment scripts.
