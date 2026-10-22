# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""CDSP request-level SP: one whole prompt per buddy-aligned block sized by length."""

from sglang.srt.dynamic_sp.plan import PrefillPlan, PromptStep, local_positions


# Tetris Implementation
def default_sp_sizes(world_size):
    return tuple(1 << i for i in range(world_size.bit_length()) if world_size % (1 << i) == 0)


# Tetris Implementation
def default_size_thresholds(num_sizes):
    # 256 and 1024 measured best for Llama-3-8B on 4 GPUs (CDSP Phase A);
    # the x4 extension to larger blocks is arbitrary.
    return tuple(256 * 4**i for i in range(num_sizes - 1))


# Tetris Implementation
class RequestSpScheduler:
    """Port of sglang.srt.cdsp.request_planner.RequestSpPlanner.

    A block is ranks [start, start + size) with start % size == 0. A request whose
    preferred block is taken falls back to a smaller idle block, else waits while
    later requests are tried. Prompts are never chunked.
    """

    # Tetris Implementation
    def __init__(self, config, sp_sizes=None, size_thresholds=None):
        self.config = config
        self.sp_sizes = tuple(sorted(sp_sizes or default_sp_sizes(config.world_size)))
        self.size_thresholds = tuple(
            default_size_thresholds(len(self.sp_sizes))
            if size_thresholds is None
            else size_thresholds
        )
        if any(s <= 0 or s & (s - 1) or config.world_size % s for s in self.sp_sizes):
            raise ValueError(
                f"SP sizes {self.sp_sizes} must be powers of two dividing {config.world_size}"
            )
        if len(self.size_thresholds) != len(self.sp_sizes) - 1 or list(
            self.size_thresholds
        ) != sorted(self.size_thresholds):
            raise ValueError(
                f"Need ascending thresholds, one fewer than SP sizes {self.sp_sizes}"
            )
        self.last_decision = {}

    # Tetris Implementation
    def size_for(self, num_tokens):
        for size, threshold in zip(self.sp_sizes, self.size_thresholds):
            if num_tokens < threshold:
                return size
        return self.sp_sizes[-1]

    # Tetris Implementation
    def metadata(self):
        return {
            "sp_sizes": self.sp_sizes,
            "size_thresholds": self.size_thresholds,
            "last_decision": self.last_decision,
        }

    # Tetris Implementation
    def next_plan(self, state):
        busy = set(state.busy_ranks)
        release = tuple(
            r.prompt.request_id
            for r in state.requests
            if r.complete and not r.in_flight and not busy.intersection(r.ranks)
        )
        # Placed requests keep their KV and workers until released; a block
        # serves at most one request per plan.
        occupied = busy | {rank for r in state.requests for rank in r.ranks}
        kv = [0] * self.config.world_size
        for r in state.requests:
            for rank in r.ranks:
                kv[rank] += self._kv_tokens(r, r.ranks, rank)
        running = sum(1 for r in state.requests if r.ranks and not r.complete)

        steps, admit, blocks = [], [], []
        claimed = set(busy)
        for r in state.requests:
            if r.ranks and not r.complete and not r.in_flight and claimed.isdisjoint(r.ranks):
                steps.append(self._full_step(r, r.ranks))
                claimed.update(r.ranks)
        pending = sorted(
            (r for r in state.requests if not r.ranks and not r.in_flight and not r.complete),
            key=lambda r: (r.submitted_at, r.prompt.request_id),
        )
        for r in pending:
            if len(occupied) >= self.config.world_size:
                break
            if running >= self.config.max_running_requests:
                break
            length = len(r.prompt.token_ids)
            if length > self.sp_sizes[-1] * self.config.max_local_tokens:
                raise ValueError(
                    f"Prompt {r.prompt.request_id} exceeds the KV capacity of the "
                    f"largest block ({self.sp_sizes[-1]} workers)"
                )
            ranks = self._pick_block(r, occupied, kv)
            if ranks is None:
                continue
            for rank in ranks:
                kv[rank] += self._kv_tokens(r, ranks, rank)
            occupied.update(ranks)
            steps.append(self._full_step(r, ranks))
            admit.append(r.prompt)
            blocks.append([r.prompt.request_id, list(ranks), length])
            running += 1

        self.last_decision = {"version": state.version, "blocks": blocks}
        if not steps and not release:
            if busy or not pending:
                return None
            raise RuntimeError(
                "No request fits the configured KV capacity; scheduler made no progress"
            )
        return PrefillPlan(state.version, tuple(steps), tuple(admit), release)

    # Tetris Implementation
    def _pick_block(self, request, occupied, kv):
        """Largest idle aligned block not above the preferred size that fits the KV."""
        desired = self.size_for(len(request.prompt.token_ids))
        for size in sorted((s for s in self.sp_sizes if s <= desired), reverse=True):
            for start in range(0, self.config.world_size, size):
                ranks = tuple(range(start, start + size))
                if occupied.isdisjoint(ranks) and all(
                    kv[rank] + self._kv_tokens(request, ranks, rank)
                    <= self.config.max_local_tokens
                    for rank in ranks
                ):
                    return ranks
        return None

    @staticmethod
    def _kv_tokens(request, ranks, rank):
        return len(local_positions(len(request.prompt.token_ids), ranks, rank))

    @staticmethod
    def _full_step(request, ranks):
        return PromptStep(request.prompt.request_id, ranks, len(request.prompt.token_ids))
