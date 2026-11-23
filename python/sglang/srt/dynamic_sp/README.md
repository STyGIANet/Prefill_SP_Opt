# Dynamic sequence parallelism for prefill

This implementation studies prefill scheduling with dynamic sequence parallelism
(SP). Each GPU holds a complete model replica. A prompt's KV state is partitioned
across its assigned GPUs by token position. Placement, SP degree, and chunk size
can change after a completed execution step. Disjoint GPU groups execute
independently; paused prompts retain their KV state.

The evaluation uses dense Llama models and eight GPUs. Each request ends after
the first sampled token. Subsequent decoding and HTTP serving are excluded.

## Methods

| Method | Scheduling policy |
|---|---|
| Tetris | Latency-based chunk planning, progressive SP expansion, and persistent worker reservations |
| LoongServe | FCFS admission and dynamic programming over whole-prompt batches |
| LAPS | Released short/long queue and instance-routing policies on native SGLang replicas |
| SPSRPT | Roofline-based remaining-time estimation and joint placement selection |
| Native DP | Independent SGLang replicas with round-robin request routing |

Tetris and LoongServe use the shared SP backend. LAPS follows the released
implementation's replica-based execution and does not partition individual
sequences. Tetris additionally enforces per-worker KV capacity through feasible
chunk endpoints and conservative peak-memory reservations. This memory admission
rule extends the paper's scheduling procedure.

The SP backend uses native SGLang model kernels and token-granularity KV pools.
Ring attention reads local KV through slot mappings and exchanges remote shards
on a communication stream. Layer-wise KV migration overlaps subsequent-layer
redistribution with current-layer computation. These execution mechanisms are
shared across SP policies.

## SPSRPT

For each candidate placement, the scheduler predicts remaining service from the
processed context, remaining tokens, and SP degree. Dense layers use compute and
memory bounds. Attention includes computation, memory traffic, and ring exchange
costs, with overlap represented by the maximum of concurrent stages.

For request i and placement A, the allocation score is

    I_i(A) = 1 / [S_i (W_i(A) + T_i(A))],

where S_i is the estimated standalone SP1 service time of the original prompt,
T_i(A) includes remaining execution and migration, and W_i(A) accounts for busy
owners. At arrivals and execution-state changes, a compiled dynamic program
maximizes the sum of selected scores subject to disjoint operation masks, KV
capacity, and admission limits. Operation masks include both old and new owners.
In-flight chunks are not interrupted.

For each placement, chunking selects the largest permitted token count whose
predicted duration meets `--spsrpt-quantum` (default 0.2 seconds). The token count
is global to the SP group. Profile limits, batch limits, and alignment constrain
the candidate sizes. If no chunk meets the duration target, the scheduler selects
the chunk with the highest predicted tokens per second. Remaining-service
prediction uses an exact next chunk, adaptive numerical integration, and an
explicit final chunk.

The score is a surrogate for mean slowdown. The allocation solver is exact for
this score, not for the complete online scheduling problem. SPSRPT is distinct
from [heSRPT](https://arxiv.org/abs/2011.09676). Roofline predictions outside the
measured context range require extrapolation; degree-level transport estimates
do not capture every placement or concurrent communication pattern.

## Experiments

From the repository root:

```bash
./run_prefill_experiments.sh --suite pilot --datasets pg19 --schedulers laps tetris spsrpt --offered-load 25000 --spsrpt-quantum 1.0 --short-threshold 8192
```

The model is `gradientai/Llama-3-8B-Instruct-Gradient-1048k`. The pilot selects at
most 24 PG19 test records, targets 262,144 input tokens, and excludes prompts
above 65,536 tokens. The example classifies prompts of at most 8,192 tokens as
short; the script's default threshold is 1,024. Classification uses total prompt
length and does not affect scheduling.

`--suite full` removes prompt-count and aggregate-token limits. The available
workloads are PG19 and LongBench v2. Prompts exceeding 1,048,576 tokens are
prefix-truncated. Native LAPS and DP require each prompt to fit one replica;
the requested default KV capacity is 600,000 tokens, and actual usable capacity
may be smaller. Comparisons requiring these baselines must use a common eligible
workload that fits every method.

Arrivals are Poisson, with request rate determined from the offered input-token
load and mean prompt length. All methods receive identical token IDs and arrival
times. Each method requiring a profile uses one frozen profile across evaluation
datasets and repetitions. Tetris threshold fitting uses a separate training
split. SPSRPT fits existing roofline measurements rather than evaluation traces.

Warmup precedes measured arrivals. SP uses one full-model backend pass, two
synthetic-KV context passes, and bounded scheduler warmup. Case count is bounded
independently of dataset size, although long-context attention increases warmup
cost. Warmup KV is cleared before measurement. Native baselines use separate
warmup paths.

TTFT is measured from scheduled arrival until the first sampled token is ready
on the CPU, including queueing and producer delay but excluding response IPC and
HTTP. Throughput is completed input tokens divided by the interval from first
actual submission to last first-token completion. It includes arrival gaps and
the final drain period.

Results include overall and short/long TTFT statistics, throughput, and the full
per-prompt distribution. Estimated slowdown divides TTFT by a common modeled SP1
reference, using Tetris when available and the roofline reference otherwise. It
is not measured isolated-runtime slowdown. The experiment directory contains
`summary.csv`, `ttft.csv`, frozen profiles, workload manifests, and per-run logs.
Failed or incomplete trials are excluded from successful comparisons.

## References

- [Tetris: Optimizing Long-context LLM Serving via Fine-grained Sequence Parallelism](https://arxiv.org/abs/2511.06247v1)
- [LoongServe: Efficiently Serving Long-context Large Language Models with Elastic Sequence Parallelism](https://arxiv.org/abs/2404.09526)
- [LAPS: A Length-Aware-Prefill LLM Serving System](https://arxiv.org/abs/2601.11589)
