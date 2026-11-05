"""Streaming independently scrambled Sobol estimators."""
from __future__ import annotations

from collections.abc import Iterator
import time

import torch

from .estimators import RuntimeOptions, choose_chunk_size
from .models import SDEModel
from .randomization import normal_from_uniform


MAX_SOBOL_DIMENSION = 21201


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def validate_sobol_design(dimension: int, points: int, scrambles: int) -> None:
    if not 1 <= dimension <= MAX_SOBOL_DIMENSION:
        raise ValueError(
            f"Sobol dimension must be between 1 and {MAX_SOBOL_DIMENSION}, got {dimension}"
        )
    if not _power_of_two(points):
        raise ValueError("points_per_scramble must be a positive power of two")
    if scrambles < 2:
        raise ValueError("at least two independent scrambles are required")


@torch.inference_mode()
def sobol_rule_blocks(
    model: SDEModel,
    points: int,
    m: int,
    seed: int,
    start: int,
    block_points: int,
    options: RuntimeOptions,
) -> Iterator[tuple[int, int, float, float]]:
    """Yield sums over contiguous blocks of one scrambled Sobol net."""
    dimension = model.d_w * m
    validate_sobol_design(dimension, points, 2)
    if not 0 <= start <= points:
        raise ValueError("start must lie between zero and points_per_scramble")
    if block_points < 1:
        raise ValueError("checkpoint_block_points must be positive")
    engine = torch.quasirandom.SobolEngine(
        dimension=dimension,
        scramble=True,
        seed=int(seed) % (2**31 - 1),
    )
    if start:
        engine.fast_forward(start)
    chunk = choose_chunk_size(model, m, 1, options)
    block_start = start
    while block_start < points:
        if options.device == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        block_stop = min(points, block_start + block_points)
        total = torch.zeros((), dtype=torch.float64, device=options.device)
        cursor = block_start
        while cursor < block_stop:
            size = min(chunk, block_stop - cursor)
            uniforms = engine.draw(size, dtype=torch.float64).to(options.device)
            normals = normal_from_uniform(uniforms).to(options.dtype)
            total += model.evaluate(normals, m).double().sum()
            cursor += size
            del uniforms, normals
        if options.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        yield block_start, block_stop, total.item(), elapsed
        block_start = block_stop


def sobol_rule_value(
    model: SDEModel,
    points: int,
    m: int,
    seed: int,
    options: RuntimeOptions,
) -> float:
    total = sum(
        value
        for _, _, value, _ in sobol_rule_blocks(
            model, points, m, seed, 0, points, options
        )
    )
    return total / points
