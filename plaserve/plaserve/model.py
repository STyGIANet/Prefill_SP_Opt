"""Requests, graph shapes, and explicitly hypothetical service-time estimates."""

from dataclasses import dataclass
import math


def finite(value: float, name: str, minimum: float = 0) -> None:
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


@dataclass(frozen=True)
class Request:
    id: str
    arrival: float
    tokens: int
    history: int = 0
    deadline: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("request id must be a nonempty string")
        finite(self.arrival, "arrival")
        for name, minimum in (("tokens", 1), ("history", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.deadline is not None:
            finite(self.deadline, "deadline")
            if self.deadline < self.arrival:
                raise ValueError("deadline must be >= arrival")


@dataclass(frozen=True)
class LatencyModel:
    # Seconds. These are illustrative coefficients, not H200 measurements.
    alpha: float = 2e-9
    beta: float = 1e-6
    gamma_write: float = 1.5e-6
    gamma_read: float = 4e-7
    eager_launch: float = 0.00015
    graph_launch: float = 0.00002
    compute_batch_exponent: float = 0.45
    memory_batch_exponent: float = 0.65

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            finite(getattr(self, name), name)
        if self.alpha == 0 or self.eager_launch == 0 or self.graph_launch == 0:
            raise ValueError("alpha and launch costs must be positive")
        if max(self.compute_batch_exponent, self.memory_batch_exponent) > 1:
            raise ValueError("batch exponents must be <= 1")

    def boundary(self, history: int = 0) -> float:
        """Positive compute/memory crossing; stable quadratic root for H > 0."""
        if history < 0:
            raise ValueError("history must be nonnegative")
        b = 2 * self.alpha * history + self.beta - self.gamma_write
        if history == 0:
            return max(0.0, -b / self.alpha)
        c = self.gamma_read * history
        root = math.sqrt(b * b + 4 * self.alpha * c)
        if b >= 0:
            return 2 * c / (b + root) if c else 0.0
        return (-b + root) / (2 * self.alpha)

    def duration(self, lengths: list[int], histories: list[int],
                 shape: "GraphShape | None" = None) -> float:
        if not lengths or len(lengths) != len(histories):
            raise ValueError("service estimate requires matching nonempty inputs")
        if shape:
            if shape.depth < len(lengths) or shape.length < max(lengths):
                raise ValueError("graph shape cannot contain this batch")
            lengths = [shape.length] * shape.depth
            histories = histories + [0] * (shape.depth - len(histories))
        depth = len(lengths)
        compute = sum(self.alpha * l * (l + 2 * h) + self.beta * l
                      for l, h in zip(lengths, histories))
        memory = sum(self.gamma_write * l + self.gamma_read * h
                     for l, h in zip(lengths, histories))
        launch = self.graph_launch if shape else self.eager_launch
        # Batch scaling is an exposed modeling assumption, absent from the paper.
        return (launch + compute / depth ** self.compute_batch_exponent
                + memory / depth ** self.memory_batch_exponent)


@dataclass(frozen=True, order=True)
class GraphShape:
    length: int
    depth: int

    def kv_tokens(self, histories: list[int]) -> int:
        # Cached history of real lanes plus padded new tokens of all lanes.
        return sum(histories) + self.length * self.depth


@dataclass(frozen=True)
class SimConfig:
    workers: int = 1
    short_threshold: int = 256
    classify_by_model: bool = False
    max_batch: int = 64
    kv_token_budget: int = 32768
    long_chunk: int = 512
    eager_padding: bool = False
    graph_lengths: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    graph_depths: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    wait_min: float = 0.0001
    wait_max: float = 0.005
    max_queue_wait: float = 0.05
    slack: float = 0.0002
    safety: float = 0.0001
    control_period: float = 0.01
    cooldown: float = 0.05
    hysteresis: float = 0.2
    min_pool: int = 1
    pressure_queue: float = 1.0
    pressure_deadline: float = 1.0
    pressure_utilization: float = 0.1

    def __post_init__(self) -> None:
        for name in ("workers", "short_threshold", "max_batch", "kv_token_budget",
                     "long_chunk", "min_pool"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("wait_min", "wait_max", "max_queue_wait", "slack", "safety",
                     "control_period", "cooldown", "hysteresis", "pressure_queue",
                     "pressure_deadline", "pressure_utilization"):
            finite(getattr(self, name), name)
        if self.wait_min > self.wait_max:
            raise ValueError("wait_min must be <= wait_max")
        if self.control_period == 0 or self.max_queue_wait == 0:
            raise ValueError("control_period and max_queue_wait must be positive")
        if self.workers > 1 and self.workers < 2 * self.min_pool:
            raise ValueError("workers cannot satisfy minimum pool allocations")
        for name in ("graph_lengths", "graph_depths"):
            values = getattr(self, name)
            if not values or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                                 for v in values):
                raise ValueError(f"{name} must contain positive integers")
            if tuple(sorted(set(values))) != tuple(values):
                raise ValueError(f"{name} must be strictly increasing")


class GraphCatalog:
    def __init__(self, config: SimConfig):
        self.config = config
        self.shapes = [GraphShape(l, d) for l in config.graph_lengths
                       for d in config.graph_depths if d <= config.max_batch]

    def bucket(self, tokens: int) -> int:
        return next((l for l in self.config.graph_lengths if l >= tokens), tokens)

    def nearest(self, lengths: list[int], histories: list[int]) -> GraphShape | None:
        if not lengths:
            return None
        feasible = [s for s in self.shapes if s.length >= max(lengths)
                    and s.depth >= len(lengths)
                    and s.kv_tokens(histories) <= self.config.kv_token_budget]
        return min(feasible, key=lambda s: (s.length * s.depth, s.length, s.depth),
                   default=None)

    def target(self, length: int, histories: list[int]) -> int:
        feasible = [s.depth for s in self.shapes if s.length == length
                    and s.kv_tokens(histories) <= self.config.kv_token_budget]
        return max(feasible, default=1)
