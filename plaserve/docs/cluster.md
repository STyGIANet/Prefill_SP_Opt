# Cluster workflow

## Setup

`scripts/setup_cluster.sh PROFILE [REVISION]` fetches the authors' source into
`vendor/LAPS`, runs its installer in the `plaserve` conda environment, and adds
nixl for the non-RDMA profiles. Conda, CUDA, and a working GPU allocation must
already be available. Set `PLASERVE_ENV` to change the environment name.

The [upstream artifact instructions](https://github.com/Jianshu-She/LAPS/tree/main/bench_laps_prefill_throughput)
define the runtime requirements. Installation and model downloads need network
access unless the checkout, environment, and model caches are staged beforehand.
PLAserve runs from its root with `python -m plaserve`; `pip install -e .` also
installs the CLI.

The launcher rejects a different SGLang installation or modified tracked upstream
files. Commit intentional runtime modifications before benchmarking. The fetch
helper leaves existing checkouts untouched and cleans up a failed initial fetch.

## Profiles

```bash
python -m plaserve cluster --profile h200 --models 14b 32b 72b --output results/h200-sweep
python -m plaserve cluster --profile h100-4 --models 7b 14b --output results/h100-sweep
python -m plaserve cluster --profile a100-2 --output results/a100-suite
```

The A100 profile runs its full upstream suite rather than assuming undocumented
model-selection arguments. For different hardware, inspect the selected upstream
scripts before extending a profile. `BACKEND`, `NUM_PROMPTS`, and `IB_DEVICE` may
be set as documented upstream. RDMA experiments need the correct IB device.

## Slurm

Set up the environment before submission. Load site-specific modules in your
submission environment or in the template. The job activates
`${PLASERVE_ENV:-plaserve}` when conda is available. Override `PLASERVE_ROOT`
if submitting outside the repository root.

```bash
PROFILE=h100-4 MODELS='7b 14b' sbatch --gpus=4 cluster/benchmark.sbatch
PROFILE=a100-2 sbatch --gpus=2 cluster/benchmark.sbatch
```

Supply your site's account, partition, and resource flags. The scripts launch
prefill, decode, and router processes on **one node**; all required GPUs must
be visible on that node. Cross-node serving/KV transport needs a separate
deployment configuration.

## Results

```text
results/<run>/
  manifest.json          # source, runtime, hardware, inputs, status
  benchmark.log          # complete launcher output
  artifacts/...          # copied upstream summaries and result directories
```

The released artifact's ablations differ from the four simulator policies.
Interpret each result by its execution label and source commit. Failures retain
partial outputs and status. Use a fresh output directory on each invocation.

Pin exact model weight revisions and repeat trials on identical allocations.
The manifest records installed package versions; upstream defaults can change.
The source is downloaded on the target cluster, not bundled in this workspace.
