# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Unmodified released MiniLB policy methods; HTTP transport is external."""
import logging
import random
import time

logger = logging.getLogger(__name__)

class MiniLoadBalancerPolicy:
    def _laps_classify(self, request_data: dict) -> str:
        """Classify request as 'short' or 'long' based on estimated token count."""
        tokens = _estimate_prompt_tokens(request_data)
        if tokens == 0:
            # Unknown length - route to lower-pressure group
            if self._laps_short_pending <= self._laps_long_pending:
                return "short"
            return "long"
        return "short" if tokens <= self._laps_threshold else "long"

    def _laps_maybe_rebalance(self):
        """Time-gated inline rebalance: move at most 1 instance between groups."""
        now = time.monotonic()
        if now - self._laps_last_rebalance < self._laps_rebalance_interval:
            return

        self._laps_last_rebalance = now
        sp = max(self._laps_short_pending, 1)
        lp = max(self._laps_long_pending, 1)
        ratio = self._laps_rebalance_ratio

        if sp > lp * ratio and len(self._laps_long_group) > 1:
            moved = self._laps_long_group.pop()
            self._laps_short_group.append(moved)
            logger.info(
                f"[LAPS] Rebalance: moved prefill[{moved}] long→short "
                f"(short_pending={self._laps_short_pending}, long_pending={self._laps_long_pending})"
            )
        elif lp > sp * ratio and len(self._laps_short_group) > 1:
            moved = self._laps_short_group.pop()
            self._laps_long_group.append(moved)
            logger.info(
                f"[LAPS] Rebalance: moved prefill[{moved}] short→long "
                f"(short_pending={self._laps_short_pending}, long_pending={self._laps_long_pending})"
            )

    def select_pair_laps(self, request_data: dict):
        """Select prefill/decode pair using LAPS dynamic allocation.

        Returns (prefill_url, bootstrap_port, decode_url, category).
        """
        self._laps_maybe_rebalance()
        category = self._laps_classify(request_data)

        if category == "short":
            group = self._laps_short_group if self._laps_short_group else self._laps_long_group
        else:
            group = self._laps_long_group if self._laps_long_group else self._laps_short_group

        pidx = group[random.randint(0, len(group) - 1)]
        didx = random.randint(0, len(self.decode_urls) - 1)

        if category == "short":
            self._laps_short_pending += 1
        else:
            self._laps_long_pending += 1

        return (
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            self.decode_urls[didx],
            category,
        )

    def laps_request_done(self, category: str):
        """Decrement the pending counter for the given category."""
        if category == "short":
            self._laps_short_pending = max(0, self._laps_short_pending - 1)
        else:
            self._laps_long_pending = max(0, self._laps_long_pending - 1)

def _estimate_prompt_tokens(request_data: dict) -> int:
    """Estimate the number of prompt tokens from request data.

    Uses exact count from input_ids, or chars//4 heuristic for text.
    For batch requests, returns the max length across items.
    Returns 0 if nothing recognized (caller should use random fallback).
    """
    # input_ids: exact token count
    input_ids = request_data.get("input_ids")
    if input_ids is not None:
        if isinstance(input_ids, list) and len(input_ids) > 0:
            if isinstance(input_ids[0], int):
                return len(input_ids)
            else:
                # list-of-lists (batch)
                return max(len(ids) for ids in input_ids)
        return 0

    # text: chars // 4 heuristic
    text = request_data.get("text")
    if text is not None:
        if isinstance(text, str):
            return len(text) // 4
        elif isinstance(text, list):
            return max((len(t) // 4 for t in text), default=0)
        return 0

    # messages (chat completions): sum content lengths // 4
    messages = request_data.get("messages")
    if messages is not None:
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # multimodal content parts
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
        return total_chars // 4

    return 0
