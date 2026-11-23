# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Bridge the released LAPS queue policy to native batch admission."""

from sglang.srt.dynamic_sp.laps_released.queues import PrefillPolicy
from sglang.srt.runtime_context import get_schedule


class _Admission(PrefillPolicy):
    def __init__(self, scheduler, running_batch, delayer):
        self.scheduler = scheduler
        self.running_batch = running_batch
        self.delayer = delayer
        self._laps_threshold = get_schedule().laps_length_threshold
        self._laps_short_queue, self._laps_long_queue = [], []
        self._laps_classify_requests(scheduler.waiting_queue)

    def _build_batch_from_queue(self, queue):
        batch, self.running_batch = self.scheduler._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=self.delayer,
            running_batch=self.running_batch,
            admission_queue=queue,
        )
        # Admission transfers ownership through the native move/retire paths.
        queue[:] = [
            req for req in queue if req.owner_list is self.scheduler.waiting_queue
        ]
        return batch

    def process_prefill_chunk(self):
        # The native caller has already cached/merged the preceding chunk.
        pass

    def maybe_prepare_mlp_sync_batch_and_log_stats(self, batch):
        # The native caller owns MLP synchronization and statistics after this
        # selection returns. Replicas do not use DP attention or PP.
        return batch


def build_laps_prefill(scheduler, running_batch, prefill_delayer_single_pass):
    if scheduler.grammar_manager.has_waiting_grammars():
        for req in scheduler.grammar_manager.get_ready_grammar_requests():
            scheduler._add_request_to_queue(req)
    admission = _Admission(scheduler, running_batch, prefill_delayer_single_pass)
    batch = admission._get_next_laps_batch_to_run()
    # Continue a resident chunk even when no new request is waiting.
    if batch is None and scheduler.chunked_req is not None:
        remaining = admission._laps_short_queue + admission._laps_long_queue
        batch = admission._build_batch_from_queue(remaining)
    return batch, admission.running_batch
