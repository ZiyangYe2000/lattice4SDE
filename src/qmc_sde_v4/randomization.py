"""Stable seeds, prime sizes, and randomized rank-one lattice data."""
from __future__ import annotations

import hashlib
import json
import math

import torch


def stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63 - 1)


def is_prime(n: object) -> bool:
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    limit = math.isqrt(n)
    return all(n % divisor for divisor in range(3, limit + 1, 2))


def next_prime(n: int) -> int:
    candidate = max(2, int(n))
    while not is_prime(candidate):
        candidate += 1
    return candidate


def previous_prime(n: int) -> int:
    candidate = max(2, int(n))
    while candidate > 2 and not is_prime(candidate):
        candidate -= 1
    return candidate if is_prime(candidate) else 2


def nearest_prime(x: float) -> int:
    target = max(2, int(round(x)))
    lower = previous_prime(target)
    upper = next_prime(target)
    return lower if target - lower <= upper - target else upper


def normal_from_uniform(u: torch.Tensor) -> torch.Tensor:
    if u.dtype != torch.float64:
        raise ValueError("uniform coordinates must be float64 before inverse normal")
    return torch.special.ndtri(u.clamp(2.0**-53, 1.0 - 2.0**-53))


def random_rule(N: int, s: int, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    vector_generator = torch.Generator().manual_seed(stable_seed(seed, "vector"))
    shift_generator = torch.Generator().manual_seed(stable_seed(seed, "shift"))
    vector = torch.randint(1, N, (s,), dtype=torch.int64, generator=vector_generator)
    shift = torch.rand(s, dtype=torch.float64, generator=shift_generator)
    return vector.to(device), shift.to(device)

