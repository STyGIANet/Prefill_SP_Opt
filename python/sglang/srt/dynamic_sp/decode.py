# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Tetris-style disaggregated decode fed by dynamic-SP prefill groups.

Workers [0, P) prefill at TP1; the remaining workers form decode instances
with a large TP degree, which minimizes time between tokens (Tetris Sec. 3.2).
A finished prompt's KV is striped over its prefill owners. Each owner streams
its rows layer by layer, one KV-head slice per decode TP rank, and frees them;
the instance reserved prompt plus output slots before the first piece arrives
(Sec. 4.2 handshake). Receivers post the transfer on a dedicated stream and
release their lease at once, so decoding overlaps the transfer. A request joins
iteration-level batching only after every piece arrived, and new requests go
to the instance with the highest freeness rate (Sec. 5.2).
"""

import time

import msgspec
import torch
import torch.nn.functional as F
from sglang.srt.dynamic_sp.kv_storage import read_layer
from sglang.srt.dynamic_sp.plan import local_positions

# Matches the native Triton backend default (triton_attention_num_kv_splits).
MAX_KV_SPLITS = 8


class KVTransfer(msgspec.Struct, frozen=True):
    ticket: int
    members: tuple[int, ...]  # sorted union of owner and instance, as leases need
    request_id: str
    prompt_length: int
    ranks: tuple[int, ...]  # prefill owners in striping order
    token: int  # first output token, sampled by prefill
    decode_tokens: int  # output tokens including the prefill-sampled one
    owner: int  # prefill worker shipping its stripe
    instance: tuple[int, ...]  # decode instance ranks, in TP order

    @property
    def local(self):
        """The owner is the decode instance itself (elastic, single-rank group)."""
        return self.instance == (self.owner,)


class DecodeStep(msgspec.Struct, frozen=True):
    ticket: int
    members: tuple[int, ...]  # decode instance ranks
    run: tuple[str, ...]


class _Sequence(msgspec.Struct):
    indices: torch.Tensor  # slot per absolute position, prompt then outputs
    length: int  # positions with KV
    token: int  # next input token
    remaining: int  # decode steps left
    missing: int  # prefill owners whose rows have not been posted
    landed: torch.cuda.Event  # every posted piece is in the pool


class _Shard:
    """One TP rank's slice of a dense Llama/Qwen3 layer stack.

    Attention heads and MLP columns split evenly; o_proj and down_proj outputs
    are partial sums that the instance all-reduces.
    """

    def __init__(self, model, tp, rank):
        from sglang.srt.models.qwen3 import Qwen3Attention

        if model.num_kv_heads % tp or model.num_heads % tp:
            raise ValueError("Decode TP must divide the query and KV head counts")
        self.model = model
        self.heads, self.kv_heads = model.num_heads // tp, model.num_kv_heads // tp
        d = model.head_dim
        q, kv = model.num_heads * d, model.num_kv_heads * d
        self.layers = []
        for layer in model.model.model.layers:
            attention, mlp = layer.self_attn, layer.mlp
            if attention.qkv_proj.bias is not None:
                raise ValueError("Decode TP does not slice attention biases")
            qkv = attention.qkv_proj.weight
            heads = slice(rank * self.heads * d, (rank + 1) * self.heads * d)
            kv_heads = slice(rank * self.kv_heads * d, (rank + 1) * self.kv_heads * d)
            inter = mlp.down_proj.weight.shape[1] // tp
            columns = slice(rank * inter, (rank + 1) * inter)
            gate_up = mlp.gate_up_proj.weight
            qk_norm = isinstance(attention, Qwen3Attention)
            if tp == 1:
                # The replica's own weights; slicing would copy the whole model.
                self.layers.append(
                    (
                        layer,
                        qkv,
                        attention.o_proj.weight,
                        gate_up,
                        mlp.down_proj.weight,
                        qk_norm,
                    )
                )
                continue
            self.layers.append(
                (
                    layer,
                    torch.cat(
                        (
                            qkv[heads],
                            qkv[q + kv_heads.start : q + kv_heads.stop],
                            qkv[q + kv + kv_heads.start : q + kv + kv_heads.stop],
                        )
                    ),
                    attention.o_proj.weight[:, heads].contiguous(),
                    torch.cat(
                        (
                            gate_up[columns],
                            gate_up[
                                inter * tp + columns.start : inter * tp + columns.stop
                            ],
                        )
                    ),
                    mlp.down_proj.weight[:, columns].contiguous(),
                    qk_norm,
                )
            )
        self.split = [self.heads * d, self.kv_heads * d, self.kv_heads * d]

    def prepare(self, layer, x, positions):
        from sglang.srt.models.utils import apply_qk_norm

        module, qkv, _, _, _, qk_norm = self.layers[layer]
        attention = module.self_attn
        q, k, v = F.linear(module.input_layernorm(x), qkv).split(self.split, dim=-1)
        if qk_norm:
            q, k = apply_qk_norm(
                q.contiguous(),
                k.contiguous(),
                attention.q_norm,
                attention.k_norm,
                self.model.head_dim,
            )
        q, k = attention.rotary_emb(positions, q, k)
        d = self.model.head_dim
        return q.reshape(-1, self.heads, d), k.reshape(-1, self.kv_heads, d), v

    def complete(self, layer, residual, attention, all_reduce):
        module, _, o_proj, gate_up, down_proj, _ = self.layers[layer]
        projected = all_reduce(F.linear(attention.flatten(1), o_proj))
        normalized, residual = module.post_attention_layernorm(projected, residual)
        hidden = module.mlp.act_fn(F.linear(normalized, gate_up))
        return residual + all_reduce(F.linear(hidden, down_proj))


class DecodeInstanceRank:
    """One TP rank of a decode instance: weight shard, KV-head slice, transfers."""

    def __init__(self, model, members, rank, kv_tokens):
        from sglang.kernels.ops.attention.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        self.model, self.members = model, tuple(members)
        self.tp_rank = self.members.index(rank)
        self.shard = _Shard(model, len(self.members), self.tp_rank)
        self.pool = MHATokenToKVPool(
            size=kv_tokens,
            page_size=1,
            dtype=model.dtype,
            head_num=self.shard.kv_heads,
            head_dim=model.head_dim,
            layer_num=model.num_layers,
            device=str(model.device),
            enable_memory_saver=False,
            enable_alt_stream=False,
            kv_cache_layout="nhd",
        )
        self.allocator = TokenToKVPoolAllocator(
            kv_tokens, model.dtype, str(model.device), self.pool, False
        )
        self.kv_tokens = kv_tokens
        # Dedicated transfer stream, as in Tetris Sec. 6; decoding never waits on it
        # except for sequences it runs.
        self.stream = torch.cuda.Stream(device=model.device)
        self.attention = decode_attention_fwd
        self.sequences = {}

    def _sequence(self, operation):
        """Sec. 4.2 handshake on the first piece: reserve prompt plus output slots."""
        sequence = self.sequences.get(operation.request_id)
        if sequence is None:
            indices = self.allocator.alloc(
                operation.prompt_length + operation.decode_tokens - 1
            )
            if indices is None:
                raise ValueError("Decode KV reservation exhausted before transfer")
            sequence = _Sequence(
                indices=indices,
                length=operation.prompt_length,
                token=operation.token,
                remaining=operation.decode_tokens - 1,
                missing=len(operation.ranks),
                landed=torch.cuda.Event(),
            )
            self.sequences[operation.request_id] = sequence
        return sequence

    def _slots(self, operation, sequence):
        stripe = operation.ranks.index(operation.owner)
        return sequence.indices[stripe : operation.prompt_length : len(operation.ranks)]

    def receive(self, operation, communication):
        """Post every layer's receive and scatter on the transfer stream; no wait."""
        m = self.model
        sequence = self._sequence(operation)
        count = len(
            local_positions(operation.prompt_length, operation.ranks, operation.owner)
        )
        with torch.cuda.stream(self.stream):
            if count:
                slots = self._slots(operation, sequence)
                for layer in range(m.num_layers):
                    block = torch.empty(
                        (count, 2, self.shard.kv_heads, m.head_dim),
                        device=m.device,
                        dtype=m.dtype,
                    )
                    # Batched like the sender; an unbatched P2P op would lazily
                    # build a separate 2-rank communicator and never match it.
                    communication.data._start_coalescing()
                    communication.data.recv(
                        [block], operation.members.index(operation.owner), layer
                    )
                    communication.data._end_coalescing().wait()
                    self.pool.get_key_buffer(layer).index_copy_(0, slots, block[:, 0])
                    self.pool.get_value_buffer(layer).index_copy_(0, slots, block[:, 1])
            sequence.landed.record(self.stream)
        sequence.missing -= 1

    def receive_local(self, operation, kv):
        """Copy this worker's own prefill stripe into its decode pool (TP1 only)."""
        if len(self.members) != 1:
            raise ValueError("Local KV handoff requires a single-rank decode instance")
        sequence = self._sequence(operation)
        if len(kv):
            slots = self._slots(operation, sequence)
            for layer in range(self.model.num_layers):
                block = read_layer(kv, layer)
                self.pool.get_key_buffer(layer).index_copy_(0, slots, block[:, 0])
                self.pool.get_value_buffer(layer).index_copy_(0, slots, block[:, 1])
        sequence.landed.record(torch.cuda.current_stream(self.model.device))
        sequence.missing -= 1

    @torch.inference_mode()
    def step(self, run, communication):
        """One greedy token for every sequence in run; frees finished sequences."""
        m = self.model
        batch = [self.sequences[rid] for rid in run]
        if any(s.missing or not s.remaining for s in batch):
            raise ValueError("Decode step includes an incomplete or finished sequence")
        stream = torch.cuda.current_stream(m.device)
        for s in batch:
            stream.wait_event(s.landed)
        device = m.device
        ids = torch.tensor([s.token for s in batch], device=device, dtype=torch.long)
        positions = torch.tensor(
            [s.length for s in batch], device=device, dtype=torch.long
        )
        slots = torch.stack([s.indices[s.length] for s in batch])
        indptr = [0]
        for s in batch:
            indptr.append(indptr[-1] + s.length + 1)
        kv_indptr = torch.tensor(indptr, device=device, dtype=torch.int32)
        kv_indices = torch.cat([s.indices[: s.length + 1] for s in batch]).to(
            torch.int64
        )
        heads = self.shard.heads
        logits_buffer = torch.empty(
            (len(batch), heads, MAX_KV_SPLITS, m.head_dim),
            device=device,
            dtype=torch.float32,
        )
        lse = torch.empty(
            (len(batch), heads, MAX_KV_SPLITS), device=device, dtype=torch.float32
        )
        splits = torch.full(
            (len(batch),), MAX_KV_SPLITS, device=device, dtype=torch.int32
        )

        def all_reduce(partial):
            if len(self.members) > 1:
                communication.data.allreduce([partial]).wait()
            return partial

        x = m.embed(ids)
        for layer in range(m.num_layers):
            q, k, v = self.shard.prepare(layer, x, positions)
            keys = self.pool.get_key_buffer(layer)
            values = self.pool.get_value_buffer(layer)
            keys.index_copy_(0, slots, k)
            values.index_copy_(0, slots, v.reshape(k.shape))
            output = torch.empty_like(q)
            self.attention(
                q,
                keys,
                values,
                output,
                kv_indptr,
                kv_indices,
                logits_buffer,
                lse,
                splits,
                MAX_KV_SPLITS,
                m.head_dim**-0.5,
                1.0,
                1.0,
            )
            x = self.shard.complete(layer, x, output, all_reduce)
        # Hidden states are identical on every TP rank after each all-reduce.
        tokens = m.logits(x).argmax(-1).tolist()
        outputs, finished = {}, []
        for rid, sequence, token in zip(run, batch, tokens):
            sequence.length += 1
            sequence.token = token
            sequence.remaining -= 1
            outputs[rid] = token
            if not sequence.remaining:
                self.allocator.free(sequence.indices)
                del self.sequences[rid]
                finished.append(rid)
        return outputs, finished


def send_rows(kv, operation, communication, layers, kv_heads):
    """Stream one owner's rows layer by layer, a KV-head slice per decode TP rank.

    Sends are not awaited per layer, so packing layer l + 1 overlaps sending l.
    """
    width = kv_heads // len(operation.instance)
    peers = [operation.members.index(rank) for rank in operation.instance]
    works, slices = [], []
    for layer in range(layers):
        block = read_layer(kv, layer)
        communication.data._start_coalescing()
        for tp_rank, peer in enumerate(peers):
            # Kept alive until every send completes.
            slices.append(
                block[:, :, tp_rank * width : (tp_rank + 1) * width].contiguous()
            )
            communication.data.send([slices[-1]], peer, layer)
        works.append(communication.data._end_coalescing())
    for work in works:
        work.wait()


class DecodeCoordinator:
    """Rank-zero handshake, transfer and batching state for decode instances.

    A finished prefill waits, in first-handshake order, for an instance with
    free slots; the reservation counts as virtual usage until decoding ends.
    The prefill policy must not release a prompt until all pieces have shipped.
    """

    def __init__(self, instances, capacity, decode_tokens, max_batch):
        self.instances = tuple(tuple(i) for i in instances)
        self.capacity = capacity
        self.decode_tokens = decode_tokens
        self.max_batch = max_batch
        self.waiting = {}
        self.shipping = {}
        self.running = {instance: {} for instance in self.instances}
        self.reserved = dict.fromkeys(self.instances, 0)
        # Instance policy override: choose(length, ranks, need) -> instance or None.
        self.choose = None
        # Decoding start per request and the mean decoding duration, which the
        # LoongServe Eq. (2) gain reads as AvgLat_d.
        self.started = {}
        self.average_seconds = 0.0
        self.finished_count = 0

    @property
    def idle(self):
        return not (self.waiting or self.shipping or any(self.running.values()))

    def holds(self, request_id):
        return request_id in self.waiting or request_id in self.shipping

    def add(self, request_id, prompt_length, ranks, token):
        need = prompt_length + self.decode_tokens - 1
        if need > self.capacity:
            raise ValueError(f"Request {request_id} exceeds decode KV capacity")
        self.waiting[request_id] = (prompt_length, tuple(ranks), token)

    def _freeness(self, instance):
        free = self.capacity - self.reserved[instance]
        return free / max(1, len(self.running[instance]))

    def operations(self, busy, next_ticket):
        """Handshakes, then transfer pieces, then one step per idle instance."""
        busy = set(busy)
        for rid, (length, ranks, token) in list(self.waiting.items()):
            need = length + self.decode_tokens - 1
            if self.choose is not None:
                target = self.choose(length, ranks, need)
                if target is None:
                    break
            else:
                fits = [
                    i
                    for i in self.instances
                    if self.reserved[i] + need <= self.capacity
                ]
                if not fits:
                    break
                # Tetris Sec. 5.2: the highest freeness rate.
                target = max(fits, key=lambda i: (self._freeness(i), -i[0]))
            self.reserved[target] += need
            self.shipping[rid] = {
                "length": length,
                "ranks": ranks,
                "token": token,
                "target": target,
                "need": need,
                "unsent": list(ranks),
                "unacked": len(ranks),
            }
            del self.waiting[rid]
        operations = []
        for rid, ship in self.shipping.items():
            for owner in list(ship["unsent"]):
                # An owner that is also the target copies its stripe locally.
                members = tuple(sorted({owner, *ship["target"]}))
                if busy.intersection(members):
                    continue
                ship["unsent"].remove(owner)
                busy.update(members)
                operations.append(
                    KVTransfer(
                        next_ticket(),
                        members,
                        rid,
                        ship["length"],
                        ship["ranks"],
                        ship["token"],
                        self.decode_tokens,
                        owner,
                        ship["target"],
                    )
                )
        for instance, sequences in self.running.items():
            if busy.intersection(instance) or not sequences:
                continue
            run = tuple(list(sequences)[: self.max_batch])
            busy.update(instance)
            operations.append(DecodeStep(next_ticket(), instance, run))
        return operations

    def completed(self, operation):
        """Return request IDs whose KV fully shipped, or which finished decoding."""
        if isinstance(operation, KVTransfer):
            ship = self.shipping[operation.request_id]
            ship["unacked"] -= 1
            if ship["unacked"]:
                return {"shipped": (), "finished": ()}
            del self.shipping[operation.request_id]
            self.started[operation.request_id] = time.perf_counter()
            self.running[ship["target"]][operation.request_id] = [
                self.decode_tokens - 1,
                ship["need"],
            ]
            return {"shipped": (operation.request_id,), "finished": ()}
        sequences = self.running[operation.members]
        finished = []
        for rid in operation.run:
            sequences[rid][0] -= 1
            if not sequences[rid][0]:
                self.reserved[operation.members] -= sequences.pop(rid)[1]
                seconds = time.perf_counter() - self.started.pop(rid)
                self.finished_count += 1
                self.average_seconds += (
                    seconds - self.average_seconds
                ) / self.finished_count
                finished.append(rid)
        return {"shipped": (), "finished": tuple(finished)}


def decode_instances(prefill_world, world_size, tp):
    ranks = range(prefill_world, world_size)
    if len(ranks) % tp:
        raise ValueError("Decode TP must divide the number of decode GPUs")
    return tuple(tuple(ranks[i : i + tp]) for i in range(0, len(ranks), tp))


def operation_response(operation, rank, started):
    ready = time.perf_counter()
    return {
        "ticket": operation.ticket,
        "rank": rank,
        "status": "committed",
        "started_at": started,
        "ready_at": ready,
        "seconds": ready - started,
    }
