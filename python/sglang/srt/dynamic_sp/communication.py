# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Store-scoped groups whose identities do not depend on nonmember activity."""

import os
import pickle
from datetime import timedelta

import torch
import torch.distributed as dist


def rendezvous_store(control_group, timeout):
    duration = timedelta(seconds=timeout)
    host = os.environ.get("MASTER_ADDR", "127.0.0.1")
    store = None
    if dist.get_rank() == 0:
        store = dist.TCPStore(
            host, 0, dist.get_world_size(), True, duration, wait_for_workers=False
        )
    endpoint = [store.port if store is not None else None]
    dist.broadcast_object_list(endpoint, src=0, group=control_group)
    if store is None:
        store = dist.TCPStore(host, endpoint[0], dist.get_world_size(), False, duration)
    return store


class GroupCommunication:
    """Unregistered backends with deterministic, membership-specific store keys.

    Each worker creates a membership once. Worker leases serialize operations
    that overlap, including construction. Disjoint memberships need no barrier.
    Backend collectives use group-local ranks, independent of WORLD counters.
    """

    def __init__(self, store, members, device, timeout):
        self.global_ranks = tuple(members)
        self.rank = self.global_ranks.index(dist.get_rank())
        self.world_size = len(members)
        prefix = "dynamic-sp/" + ",".join(map(str, members))
        self.control = dist.ProcessGroupGloo(
            dist.PrefixStore(prefix + "/control", store),
            self.rank,
            self.world_size,
            timedelta(seconds=timeout),
        )
        self.data = self.control
        if device.type == "cuda":
            options = dist.ProcessGroupNCCL.Options()
            options._timeout = timedelta(seconds=timeout)
            self.data = dist.ProcessGroupNCCL(
                dist.PrefixStore(prefix + "/data", store),
                self.rank,
                self.world_size,
                options,
            )

    def objects(self, value):
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        size = torch.tensor([len(payload)], dtype=torch.int64)
        sizes = [torch.empty_like(size) for _ in self.global_ranks]
        self.control.allgather([sizes], [size]).wait()
        width = max(int(s.item()) for s in sizes)
        buffer = torch.zeros(width, dtype=torch.uint8)
        buffer[: len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        buffers = [torch.empty_like(buffer) for _ in sizes]
        self.control.allgather([buffers], [buffer]).wait()
        return [
            pickle.loads(bytes(b[: int(s.item())].tolist()))
            for b, s in zip(buffers, sizes)
        ]

    def status(self, tensor):
        options = dist.AllreduceOptions()
        options.reduceOp = dist.ReduceOp.MAX
        self.control.allreduce([tensor], options).wait()

    def broadcast(self, tensor, owner):
        options = dist.BroadcastOptions()
        options.rootRank = owner
        self.data.broadcast([tensor], options).wait()

    def exchange(self, output, value, receive_counts, send_counts):
        self.data.alltoall_base(output, value, receive_counts, send_counts).wait()

    def gather(self, output, value):
        # Flattening also handles packed workspaces with arbitrary trailing dims.
        chunks = list(output.view(-1).split(value.numel()))
        self.data.allgather([chunks], [value.reshape(-1)]).wait()

    def close(self):
        if self.data is not self.control:
            self.data.shutdown()
        self.control.shutdown()
