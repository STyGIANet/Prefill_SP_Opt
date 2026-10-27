"""Run on advent in plaserve after applying patches/blackwell-flashinfer.patch.

Exercises the actual adapter/updater/FlashInfer kernels without loading a model.
"""
import argparse
import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend, FlashInferIndicesUpdaterPrefill,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Read attention dimensions from this model's config")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--case", type=int, help="Run only this zero-based scenario in a fresh process")
    args = parser.parse_args()
    torch.manual_seed(7)
    device, dtype = "cuda", torch.bfloat16
    heads, kv_heads, dim, context, max_bs = 4, 2, 64, 32, 2
    scenarios = [
        (2, 8, [3, 12], [3, 5]), (1, 4, [7], [2]),
        (2, 8, [10, 1], [4, 1]), (1, 4, [4], [4]),
    ]
    if args.model:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model)
        if args.tp <= 0 or config.num_attention_heads % args.tp or config.num_key_value_heads % args.tp:
            raise ValueError("This checker requires Q and KV heads divisible by TP")
        heads = config.num_attention_heads // args.tp
        kv_heads = config.num_key_value_heads // args.tp
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        context, max_bs = 512, 8
        scenarios += [
            (4, 64, [32, 33, 63, 64], [32, 33, 63, 64]),
            (8, 32, [20] * 8, [20] * 8),
            (8, 256, [65, 127, 128, 129, 255, 1, 1, 1],
             [65, 127, 128, 129, 255, 1, 1, 1]),
            (4, 16, [35, 39, 41, 1], [3, 7, 9, 1]),
            (4, 64, [90, 65, 130, 64], [32, 33, 63, 64]),
        ]
    if args.case is not None:
        if not 0 <= args.case < len(scenarios):
            raise ValueError("Case index out of range")
        scenarios = [scenarios[args.case]]
    print(f"Attention dimensions: Q heads={heads}, KV heads={kv_heads}, head_dim={dim}")
    shapes = list(dict.fromkeys((bs, width) for bs, width, _, _ in scenarios))
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.skip_prefill = backend.is_dllm_model = False
    backend.dispatch_reason = None
    backend.batch_prefill_page_size = 1
    backend.max_context_len = context
    backend.prefill_backend = "fa2"
    # Match the upstream Qwen workspace allocation; larger head layouts exceed
    # the 128 MiB used by the original small synthetic fixture.
    backend.workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    backend.kv_indptr = [torch.zeros(max_bs + 1, dtype=torch.int32, device=device)]
    backend.qo_indptr = [torch.zeros(max_bs + 1, dtype=torch.int32, device=device)]
    backend.kv_last_page_len = torch.ones(max_bs, dtype=torch.int32, device=device)
    updater = FlashInferIndicesUpdaterPrefill.__new__(FlashInferIndicesUpdaterPrefill)
    updater.attn_backend = backend
    updater.num_qo_heads, updater.num_kv_heads, updater.head_dim = heads, kv_heads, dim
    updater.data_type = updater.q_data_type = dtype
    updater.kv_indptr, updater.qo_indptr = backend.kv_indptr, backend.qo_indptr
    updater.kv_last_page_len = backend.kv_last_page_len
    updater.prefill_wrapper_ragged = None
    updater.req_to_token = torch.arange(max_bs * context, dtype=torch.int32, device=device).reshape(max_bs, context)
    updater.update = updater.update_single_wrapper
    backend.indices_updater_prefill = updater
    backend.init_batch_prefill_cuda_graph_state(max_bs, max(width for _, width in shapes))
    cache = torch.randn(max_bs * context, 2, 1, kv_heads, dim, dtype=dtype, device=device)
    captured = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for bs, width in shapes:
            q = torch.randn(bs * width, heads, dim, dtype=dtype, device=device)
            req = torch.arange(bs, dtype=torch.int64, device=device)
            lens = torch.full((bs,), width, dtype=torch.int64, device=device)
            backend.init_forward_metadata_capture_batch_prefill_cuda_graph(bs, width, req, lens)
            wrapper = backend.forward_metadata.prefill_wrappers[0]
            for _ in range(3):
                wrapper.forward(q, cache, causal=True, sm_scale=dim ** -0.5)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = wrapper.forward(q, cache, causal=True, sm_scale=dim ** -0.5)
            captured[bs, width] = q, graph, output
    torch.cuda.current_stream().wait_stream(stream)
    # Return to the first graph after capturing a second shape, then alternate.
    for bs, width, lengths, real_q in scenarios:
        print(f"CHECK bs={bs} width={width} KV={lengths} real_Q={real_q}", flush=True)
        q, graph, output = captured[bs, width]
        q.normal_()
        cache.normal_()
        # Changed request mapping as well as changed Q/K/V and lengths.
        req = torch.arange(bs - 1, -1, -1, dtype=torch.int64, device=device)
        lens = torch.tensor(lengths, dtype=torch.int64, device=device)
        backend.init_forward_metadata_replay_batch_prefill_cuda_graph(
            bs, width, req, lens, lens.cpu())
        graph.replay()
        torch.cuda.synchronize()
        replay_output = output.clone()
        wrapper = backend.forward_metadata.prefill_wrappers[0]
        direct_output = wrapper.forward(q, cache, causal=True, sm_scale=dim ** -0.5)
        torch.cuda.synchronize()
        errors = []
        for row, (n, count) in enumerate(zip(lengths, real_q)):
            query = q.reshape(bs, width, heads, dim)[row, -count:].transpose(0, 1).float()
            indices = updater.req_to_token[req[row], :n].long()
            k = cache[indices, 0, 0].repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).float()
            v = cache[indices, 1, 0].repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).float()
            mask = torch.arange(n, device=device)[None, :] <= (
                torch.arange(count, device=device)[:, None] + n - count)
            reference = F.scaled_dot_product_attention(query, k, v, attn_mask=mask)
            actual = replay_output.reshape(bs, width, heads, dim)[row, -count:].transpose(0, 1).float()
            direct = direct_output.reshape(bs, width, heads, dim)[row, -count:].transpose(0, 1).float()
            for label, left, right in [("replay_vs_reference", actual, reference),
                                       ("direct_vs_reference", direct, reference),
                                       ("replay_vs_direct", actual, direct)]:
                close = torch.isclose(left, right, atol=0.03, rtol=0.03)
                print(f"  row={row} {label}: bad={(~close).sum().item()}/{close.numel()} "
                      f"max_abs={(left-right).abs().max().item():.6g}", flush=True)
                if not bool(close.all().item()):
                    errors.append(f"row={row} {label}")
        if errors:
            raise AssertionError("Attention comparison failed: " + "; ".join(errors))
        print(f"PASS bs={bs} width={width} KV={lengths} real_Q={real_q}")
    print("PASS: changed-input graph replay, multiple shapes, padding, cached history")


if __name__ == "__main__":
    main()
