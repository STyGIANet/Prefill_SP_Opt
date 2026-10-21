# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Persistent resource reservations shared by online policies and simulators.

A task depends on the preceding task on each worker it uses, and on the
preceding chunk of its request. Reservations do not count as execution. Only
observed commits satisfy dependencies; predictions affect planning alone.
"""

from dataclasses import dataclass, field


@dataclass
class Reservation:
    ticket: int
    request_id: str
    history: int
    length: int
    ranks: tuple[int, ...]
    duration: float
    dependencies: tuple[int, ...]
    status: str = "queued"
    start: float | None = None
    finish: float | None = None
    predicted_start: float = 0.0
    predicted_finish: float = 0.0

    @property
    def end(self):
        return self.history + self.length


@dataclass
class ReservationCalendar:
    world_size: int
    tasks: dict[int, Reservation] = field(default_factory=dict)
    tails: dict[int, int] = field(default_factory=dict)
    requests: dict[str, list[int]] = field(default_factory=dict)
    next_ticket: int = 0

    def reserve(self, request_id, chunks, models, now):
        if request_id in self.requests:
            raise ValueError("Request already has reservations")
        ids = []
        for chunk in chunks:
            dependencies = {self.tails[r] for r in chunk.ranks if r in self.tails}
            if ids:
                dependencies.add(ids[-1])
            task = Reservation(
                self.next_ticket,
                request_id,
                chunk.history,
                chunk.length,
                tuple(chunk.ranks),
                models[len(chunk.ranks)].seconds(chunk.history, chunk.length),
                tuple(sorted(dependencies)),
            )
            self.tasks[task.ticket] = task
            ids.append(task.ticket)
            for rank in task.ranks:
                self.tails[rank] = task.ticket
            self.next_ticket += 1
        if not ids:
            raise ValueError("Cannot reserve an empty execution plan")
        self.requests[request_id] = ids
        self.predict(now)
        return tuple(ids)

    def predict(self, now, worker_available_at=()):
        # IDs are a topological order because reservation is append-only.
        for task in self.tasks.values():
            if task.status == "completed":
                task.predicted_finish = task.finish
                continue
            if task.status == "running":
                estimated = task.start + task.duration
                if worker_available_at:
                    estimated = max(
                        estimated, *(worker_available_at[r] for r in task.ranks)
                    )
                task.predicted_finish = max(now, estimated)
                continue
            start = max(
                [now] + [self.tasks[d].predicted_finish for d in task.dependencies]
            )
            task.predicted_start = start
            task.predicted_finish = start + task.duration
        return tuple(
            max(now, self.tasks[self.tails[r]].predicted_finish)
            if r in self.tails
            else now
            for r in range(self.world_size)
        )

    def ready(self, busy=()):
        busy = set(busy)
        return [
            task
            for task in self.tasks.values()
            if task.status == "queued"
            and not busy.intersection(task.ranks)
            and all(self.tasks[d].status == "completed" for d in task.dependencies)
        ]

    def start(self, ticket, now):
        task = self.tasks[ticket]
        if task.status != "queued" or any(
            self.tasks[d].status != "completed" for d in task.dependencies
        ):
            raise ValueError("Reservation is not ready")
        task.status, task.start = "running", now

    def complete(self, ticket, now):
        task = self.tasks[ticket]
        if task.status == "completed":
            raise ValueError("Reservation already completed")
        task.status, task.finish = "completed", now

    def observe(self, state):
        views = {r.prompt.request_id: r for r in state.requests}
        for rid, ids in self.requests.items():
            request = views.get(rid)
            if request is None:
                continue
            for ticket in ids:
                task = self.tasks[ticket]
                if task.status == "completed":
                    continue
                if task.end <= request.end:
                    self.complete(ticket, state.now)
                elif task.history == request.end:
                    if request.in_flight and task.status == "queued":
                        self.start(ticket, state.now)
                    elif not request.in_flight and task.status == "running":
                        # A rejected operation reserves its same place for retry.
                        task.status, task.start = "queued", None
                    break
        return self.predict(state.now, state.worker_available_at)

    def retire(self, request_id):
        ids = self.requests.get(request_id, ())
        if any(self.tasks[i].status != "completed" for i in ids):
            raise ValueError("Cannot retire unfinished reservations")
        self.requests.pop(request_id, None)
        # Completed tasks can be removed after removing their satisfied edges.
        retired = set(ids)
        for task in self.tasks.values():
            task.dependencies = tuple(d for d in task.dependencies if d not in retired)
        self.tails = {r: t for r, t in self.tails.items() if t not in retired}
        for ticket in ids:
            self.tasks.pop(ticket)
