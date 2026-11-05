"""MC and randomized lattice estimators with equal-work accounting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Iterable

import torch

from .models import SDEModel
from .randomization import is_prime, nearest_prime, normal_from_uniform, random_rule


@dataclass(frozen=True)
class RuntimeOptions:
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    memory_fraction: float = 0.45
    max_chunk_points: int = 131072
    cpu_chunk_points: int = 4096
    max_rules_per_batch: int = 4


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def choose_chunk_size(model: SDEModel, m: int, rule_batch: int, options: RuntimeOptions) -> int:
    if options.device == "cpu":
        return max(1, options.cpu_chunk_points // max(1, rule_batch))
    free_bytes, _ = torch.cuda.mem_get_info()
    scalar_bytes = torch.tensor([], dtype=options.dtype).element_size()
    s = model.d_w * m
    # Residues and uniforms dominate QMC coordinate generation.  The model
    # allowance covers the bridge and interaction/Fourier temporaries.
    bytes_per_path = 20 * s + 2 * scalar_bytes * model.extra_values_per_path(m)
    target = int(free_bytes * options.memory_fraction)
    points = target // max(1, rule_batch * bytes_per_path)
    return max(1, min(options.max_chunk_points, int(points)))


@torch.inference_mode()
def lattice_rule_values(
    model: SDEModel,
    N: int,
    m: int,
    seeds: Iterable[int],
    options: RuntimeOptions,
) -> list[float]:
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("distinct rule seeds are required")
    if not is_prime(N) or N >= 2**31:
        raise ValueError("N must be prime and below 2**31")
    s = model.d_w * m
    output: list[float] = []
    for group_start in range(0, len(seeds), options.max_rules_per_batch):
        group = seeds[group_start:group_start + options.max_rules_per_batch]
        rules = [random_rule(N, s, seed, options.device) for seed in group]
        vectors = torch.stack([item[0] for item in rules])
        shifts = torch.stack([item[1] for item in rules])
        totals = torch.zeros(len(group), dtype=torch.float64, device=options.device)
        chunk = choose_chunk_size(model, m, len(group), options)
        for start in range(0, N, chunk):
            indices = torch.arange(start, min(N, start + chunk), device=options.device, dtype=torch.int64)
            residues = (indices[None, :, None] * vectors[:, None, :]) % N
            uniforms = (residues.double() / N + shifts[:, None, :]) % 1.0
            z = normal_from_uniform(uniforms).to(options.dtype).reshape(-1, s)
            values = model.evaluate(z, m).reshape(len(group), -1)
            totals += values.double().sum(dim=1)
            del residues, uniforms, z, values
        output.extend((totals / N).cpu().tolist())
    return output


@torch.inference_mode()
def mc_value(
    model: SDEModel,
    paths: int,
    m: int,
    seed: int,
    options: RuntimeOptions,
) -> tuple[float, float]:
    generator = torch.Generator(device=options.device).manual_seed(seed)
    total = torch.zeros((), dtype=torch.float64, device=options.device)
    total_sq = torch.zeros((), dtype=torch.float64, device=options.device)
    chunk = choose_chunk_size(model, m, 1, options)
    for start in range(0, paths, chunk):
        size = min(chunk, paths - start)
        z = torch.randn(size, model.d_w * m, generator=generator, device=options.device, dtype=options.dtype)
        values = model.evaluate(z, m).double()
        total += values.sum()
        total_sq += values.square().sum()
    mean = total / paths
    if paths > 1:
        variance = (total_sq - paths * mean.square()) / (paths - 1)
    else:
        variance = torch.zeros_like(mean)
    return mean.item(), max(0.0, variance.item())


def method_spec(method: str, nominal_budget: int) -> tuple[int, int]:
    rules = {"qmc1": 1, "median3": 3, "median13": 13}.get(method)
    if rules is None:
        if method == "mc":
            return nominal_budget, nominal_budget
        raise ValueError(f"unknown method {method}")
    N = nearest_prime(nominal_budget / rules)
    return N, rules * N


def aggregate_lattice(values: list[float]) -> float:
    if len(values) not in (1, 3, 13):
        raise ValueError("expected 1, 3, or 13 component rules")
    return float(median(values))

