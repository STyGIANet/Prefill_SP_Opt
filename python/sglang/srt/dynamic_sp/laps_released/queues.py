# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Released short-first selection and queue-scoped admission methods."""
from __future__ import annotations
from typing import List, Optional
def trace_event_batch(event, requests):
    # Native scheduler tracing surrounds the returned batch.
    pass

class PrefillPolicy:

    def _laps_classify_requests(self: Scheduler, reqs: List[Req]) -> None:
        """Route bootstrapped requests into short/long queues by token length."""
        for req in reqs:
            if len(req.origin_input_ids) <= self._laps_threshold:
                self._laps_short_queue.append(req)
            else:
                self._laps_long_queue.append(req)

    def _build_batch_from_queue(
        self: Scheduler, queue: List[Req]
    ) -> Optional[ScheduleBatch]:
        """Build a batch from a specific sub-queue using the existing batch builder."""
        saved_queue = self.waiting_queue
        self.waiting_queue = queue
        batch = self.get_new_batch_prefill()
        # The batch builder may have consumed requests from the queue.
        # Since lists are mutable and get_new_batch_prefill modifies self.waiting_queue
        # in-place (via pop), we need to capture the updated state.
        updated_queue = self.waiting_queue
        self.waiting_queue = saved_queue
        # Update the source queue in-place
        queue[:] = updated_queue
        return batch

    def _get_next_laps_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """Decides which queue to service: short first, then long."""
        self.running_batch.batch_is_full = False
        self.process_prefill_chunk()

        batch = None
        # Short queue has priority
        if len(self._laps_short_queue) > 0:
            batch = self._build_batch_from_queue(self._laps_short_queue)

        # Fall back to long queue
        if batch is None and len(self._laps_long_queue) > 0:
            batch = self._build_batch_from_queue(self._laps_long_queue)

        batch = self.maybe_prepare_mlp_sync_batch_and_log_stats(batch)

        if batch:
            trace_event_batch("schedule", batch.reqs)

        return batch
