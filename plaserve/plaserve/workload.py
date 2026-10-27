"""Seeded Poisson arrivals and a portable, token-count-only JSONL trace."""

from dataclasses import asdict
import json
from pathlib import Path
import random

from .model import Request, finite


def generate(count: int = 1000, rate: float = 500.0, seed: int = 7,
             short_fraction: float = 0.8, slo: float | None = 0.4,
             history: bool = True) -> list[Request]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    finite(rate, "arrival rate")
    if rate == 0:
        raise ValueError("arrival rate must be positive")
    finite(short_fraction, "short_fraction")
    if short_fraction > 1:
        raise ValueError("short_fraction must be <= 1")
    if slo is not None:
        finite(slo, "slo")
    rng, now, requests = random.Random(seed), 0.0, []
    for i in range(count):
        if i:
            now += rng.expovariate(rate)
        short = rng.random() < short_fraction
        tokens = rng.choice((8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
                            if short else (512, 768, 1024, 1536, 2048))
        cached = rng.choice((0, 0, 128, 512, 1024)) if history and short else 0
        requests.append(Request(f"r{i:06d}", now, tokens, cached,
                                now + slo if slo is not None else None))
    return requests


def load_trace(path: str | Path) -> list[Request]:
    requests = []
    with Path(path).open() as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                requests.append(Request(**item))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return requests


def save_trace(requests: list[Request], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as target:
        for request in requests:
            target.write(json.dumps(asdict(request), allow_nan=False) + "\n")
