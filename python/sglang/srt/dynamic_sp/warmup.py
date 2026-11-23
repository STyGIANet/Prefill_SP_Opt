# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Bounded, explicit backend warmup independent of scheduling decisions."""

from dataclasses import dataclass

from .plan import PrefillPlan, Prompt, PromptStep


def backend_warmup_plans(args, model, policy, world):
    """Cover each degree and device, empty context, cached context and final logits.

    Serial cases bound resident KV. The caller repeats this sequence in the same
    executor, releases all KV and keeps compiled kernels and workspaces alive.
    """
    profiled = getattr(policy, "max_profiled_chunk", {})
    profile = getattr(policy, "profile", {})
    calibrated = [
        int(item["length_range"][1])
        for item in profile.get("fit_diagnostics", {}).values()
        if "length_range" in item
    ]
    ceiling = min(
        args.max_batch_tokens,
        max(args.warmup_tokens, args.chunk_size, *profiled.values(), *calibrated),
    )
    plans, coverage = [], []
    for degree in range(1, world + 1):
        # Disjoint blocks plus the final overlapping block cover every device.
        groups = list(
            dict.fromkeys(
                tuple(range(start, start + degree))
                for start in [*range(0, world - degree + 1, degree), world - degree]
            )
        )
        maximum = min(
            ceiling, model.context_length // 2, args.max_local_tokens * degree // 2
        )
        if maximum < 1:
            raise ValueError("KV/context capacity cannot hold a two-chunk warmup")
        sizes = {maximum, min(args.chunk_size, maximum)}
        size = 256
        while size < maximum:
            sizes.add(size)
            size *= 2
        for chunk in sorted(sizes):
            batches, batch, occupied = [], [], set()
            for ranks in groups:
                if batch and (
                    occupied.intersection(ranks)
                    or (len(batch) + 1) * chunk > args.max_batch_tokens
                    or len(batch) >= args.max_running_requests
                ):
                    batches.append(batch)
                    batch, occupied = [], set()
                batch.append(ranks)
                occupied.update(ranks)
            if batch:
                batches.append(batch)
            for batch in batches:
                prompts, first, second = [], [], []
                for ranks in batch:
                    rid = f"backend-warmup-{len(coverage)}"
                    prompts.append(Prompt(rid, (1,) * (2 * chunk)))
                    first.append(PromptStep(rid, ranks, chunk))
                    second.append(PromptStep(rid, ranks, 2 * chunk))
                    coverage.append(
                        {
                            "ranks": ranks,
                            "chunk_tokens": chunk,
                            "processed_tokens": [0, chunk],
                        }
                    )
                plans.extend(
                    [
                        PrefillPlan(0, tuple(first), tuple(prompts)),
                        PrefillPlan(0, tuple(second)),
                        PrefillPlan(0, release=tuple(p.request_id for p in prompts)),
                    ]
                )
    return plans, coverage


@dataclass(frozen=True)
class ContextWarmup:
    ticket: int
    members: tuple[int, ...]
    history: int
    chunk: int


def context_warmup_cases(args, model, policy, world, longest):
    """Two query sizes and at most two positive histories per group.

    Zero context is covered by backend_warmup_plans. History is initialized in
    the native KV pool; these cases never evaluate a full synthetic prefix.
    """
    cap = getattr(args, "warmup_context_tokens", 0)
    total_limit = min(longest, model.context_length, cap or model.context_length)
    profiled = getattr(policy, "max_profiled_chunk", {})
    cases = []
    for degree in range(1, world + 1):
        maximum = min(total_limit, args.max_local_tokens * degree)
        query_limit = min(maximum - 1, args.max_batch_tokens,
                          profiled.get(degree, max(args.warmup_tokens, args.chunk_size)))
        if query_limit < 1:
            continue
        groups = tuple(dict.fromkeys(
            tuple(range(start, start + degree))
            for start in [*range(0, world - degree + 1, degree), world - degree]
        ))
        for query in sorted({min(256, query_limit), query_limit}):
            largest_history = maximum - query
            for history in sorted({min(131072, largest_history), largest_history}):
                if history <= 0:
                    continue
                for group in groups:
                    cases.append(ContextWarmup(-len(cases) - 1, group, history, query))
    return cases
