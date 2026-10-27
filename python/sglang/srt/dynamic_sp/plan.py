# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Policy-free, serializable commands for a collective prefill engine.

Ranks are ordered global worker IDs. Token p belongs to ranks[p % len(ranks)].
Changing either the degree or the rank order is an ownership change. A step's
end is an absolute, exclusive prompt offset, not a chunk length.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    request_id: str
    token_ids: tuple[int, ...]
    ttft_slo_seconds: float | None = None
    prefix_request_id: str | None = None
    prefix_length: int = 0
    retain_kv: bool = False


@dataclass(frozen=True)
class PromptStep:
    request_id: str
    ranks: tuple[int, ...]
    end: int
    use_graph: bool = False


@dataclass(frozen=True)
class PrefillPlan:
    version: int
    steps: tuple[PromptStep, ...] = ()
    admit: tuple[Prompt, ...] = ()
    release: tuple[str, ...] = ()


def local_positions(length: int, ranks: tuple[int, ...], rank: int) -> range:
    if rank not in ranks:
        return range(0)
    return range(ranks.index(rank), length, len(ranks))


def validate_ranks(ranks: tuple[int, ...], world_size: int) -> None:
    if (
        not isinstance(ranks, tuple)
        or not ranks
        or any(type(r) is not int or not 0 <= r < world_size for r in ranks)
        or len(set(ranks)) != len(ranks)
    ):
        raise ValueError(f"Expected distinct worker IDs in [0, {world_size}): {ranks}")
