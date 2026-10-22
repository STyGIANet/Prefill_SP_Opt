# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Native SGLang slot allocation with transactional dynamic-SP ownership.

Committed prefixes keep their slots during extension. Migration writes into
reserved slots and publishes ownership only after every participant succeeds.
Scratch buffers are shared by all execution groups on one leased worker.
"""

from dataclasses import dataclass

import torch


@dataclass
class SlotKV:
    pool: object
    indices: torch.Tensor
    prefix_id: int | None = None

    def __len__(self):
        return len(self.indices)

    def materialize(self):
        """Diagnostic/reference path only, never used by model execution."""
        return torch.stack(
            [read_layer(self, i) for i in range(self.pool.layers)], dim=1
        )


def read_layer(kv, layer, rows=slice(None), out=None):
    if not isinstance(kv, SlotKV):
        value = kv[rows, layer]
        if out is None:
            return value
        out.copy_(value)
        return out
    indices = kv.indices[rows]
    if out is None:
        out = torch.empty(
            (len(indices), 2, kv.pool.heads, kv.pool.dim),
            dtype=kv.pool.dtype,
            device=kv.pool.device,
        )
    torch.index_select(kv.pool.native.get_key_buffer(layer), 0, indices, out=out[:, 0])
    torch.index_select(
        kv.pool.native.get_value_buffer(layer), 0, indices, out=out[:, 1]
    )
    return out


def write_layer(kv, layer, start, key, value):
    if isinstance(kv, SlotKV):
        indices = kv.indices[start : start + len(key)]
        kv.pool.native.get_key_buffer(layer).index_copy_(0, indices, key)
        kv.pool.native.get_value_buffer(layer).index_copy_(0, indices, value)
    else:
        kv[start : start + len(key), layer, 0].copy_(key)
        kv[start : start + len(value), layer, 1].copy_(value)


def scatter_layer(kv, layer, rows, block):
    if isinstance(kv, SlotKV):
        indices = kv.indices.index_select(0, rows)
        kv.pool.native.get_key_buffer(layer).index_copy_(0, indices, block[:, 0])
        kv.pool.native.get_value_buffer(layer).index_copy_(0, indices, block[:, 1])
    else:
        kv[:, layer].index_copy_(0, rows, block)


class NativeKVStorage:
    def __init__(
        self,
        model,
        capacity,
        world_size,
        backend,
        *,
        reserve_bytes=8 << 30,
        execution_tokens=32768,
    ):
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        self.device, self.dtype = model.device, model.dtype
        self.layers, self.heads, self.dim = (
            model.num_layers,
            model.num_kv_heads,
            model.head_dim,
        )
        self.capacity = capacity
        # A worker has at most one operation in flight. Full migration can retain
        # its entire committed capacity while constructing a new placement.
        self.slots = 2 * capacity
        row_bytes = 2 * self.heads * self.dim * model.dtype.itemsize
        self.sizes = {"migration_send": capacity, "migration_recv": capacity}
        if backend == "ring":
            self.sizes.update(ring_0=2 * capacity, ring_1=2 * capacity)
        elif backend == "gathered":
            self.sizes.update(
                gather_send=world_size * capacity, gather_recv=world_size * capacity
            )
        required = (self.slots + 1) * self.layers * row_bytes + sum(
            self.sizes.values()
        ) * row_bytes
        hidden = model.num_heads * model.head_dim
        config = getattr(getattr(model, "model", None), "config", None)
        intermediate = getattr(config, "intermediate_size", 4 * hidden)
        execution_bytes = (
            min(execution_tokens, capacity)
            * (6 * intermediate + 12 * hidden)
            * model.dtype.itemsize
        )
        if backend == "gathered":
            execution_bytes += world_size * capacity * row_bytes
        self.execution_reserve_bytes = max(reserve_bytes, execution_bytes)
        if model.device.type == "cuda":
            before = torch.cuda.memory_stats(model.device)
            free, _ = torch.cuda.mem_get_info(model.device)
            if required + self.execution_reserve_bytes > free:
                raise ValueError(
                    f"Native SP pool needs {required} bytes plus {self.execution_reserve_bytes} execution reserve; "
                    f"only {free} bytes are free. Reduce max-local-tokens or use a smaller model."
                )
        self.native = MHATokenToKVPool(
            size=self.slots,
            page_size=1,
            dtype=self.dtype,
            head_num=self.heads,
            head_dim=self.dim,
            layer_num=self.layers,
            device=str(self.device),
            enable_memory_saver=False,
            enable_alt_stream=False,
            kv_cache_layout="nhd",
        )
        # Keep slot maps on the model device, just like native SGLang. Alloc
        # slices the free list using host-known lengths; no device readback or
        # synchronous host-to-device slot-map copy is required per extension.
        self.allocator = TokenToKVPoolAllocator(
            self.slots, self.dtype, str(self.device), self.native, False
        )
        self.scratch = {
            name: torch.empty(
                (n, 2, self.heads, self.dim), device=self.device, dtype=self.dtype
            )
            for name, n in self.sizes.items()
        }
        if model.device.type == "cuda":
            after = torch.cuda.memory_stats(model.device)
            if any(
                after.get(k, 0) != before.get(k, 0)
                for k in ("num_alloc_retries", "num_ooms")
            ):
                raise RuntimeError(
                    "Allocator retry during native pool initialization; reduce capacity"
                )
        self.allocated = None
        self.reserved_bytes = required
        self.migration_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.attention_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

    def buffer(self, name, rows):
        if rows > self.sizes[name]:
            raise ValueError(
                f"{name} exceeds reserved workspace: {rows} > {self.sizes[name]}"
            )
        return self.scratch[name][:rows]

    def begin(self):
        if self.allocated is not None:
            raise RuntimeError("Nested KV transaction")
        self.allocated = []

    def allocate(self, count, previous=None):
        if self.allocated is None:
            raise RuntimeError("KV allocation outside transaction")
        prefix = len(previous) if previous is not None else 0
        if count < prefix:
            raise ValueError("Extension cannot shrink committed KV")
        new = self.allocator.alloc(count - prefix)
        if new is None:
            raise ValueError("KV slot reservation exhausted before dispatch")
        self.allocated.append(new)
        indices = torch.cat((previous.indices, new)) if previous is not None else new
        return SlotKV(self, indices, id(previous) if previous is not None else None)

    def commit(self, old, new):
        for rid, state in old.items():
            replacement = new.get(rid)
            if replacement is not None and (
                replacement.kv is state.kv or replacement.kv.prefix_id == id(state.kv)
            ):
                continue
            self.allocator.free(state.kv.indices)
        self.allocated = None

    def rollback(self):
        if self.allocated is not None:
            for indices in self.allocated:
                self.allocator.free(indices)
            self.allocated = None

    def assert_empty(self):
        if self.allocated is not None or self.allocator.available_size() != self.slots:
            raise RuntimeError("KV state leaked across benchmark phases")
