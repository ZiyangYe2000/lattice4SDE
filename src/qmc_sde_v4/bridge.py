"""Dyadic Brownian-bridge coordinates used by the manuscript."""
from __future__ import annotations

import math
import torch

Tensor = torch.Tensor


def _check_grid(z: Tensor, d_w: int, m: int) -> None:
    if m < 1 or m & (m - 1):
        raise ValueError("m must be a positive power of two")
    if z.ndim != 2 or z.shape[1] != d_w * m:
        raise ValueError(f"expected z.shape=(batch,{d_w*m}), got {tuple(z.shape)}")


def brownian_bridge_increments(z: Tensor, d_w: int, m: int, horizon: float) -> Tensor:
    """Map independent standard normals to Brownian increments.

    Coordinates are ordered endpoint-component first, followed by
    level/interval/component midpoint coordinates.
    """
    _check_grid(z, d_w, m)
    batch = z.shape[0]
    w = torch.zeros((batch, d_w, m + 1), dtype=z.dtype, device=z.device)
    w[:, :, m] = math.sqrt(horizon) * z[:, :d_w]
    offset = d_w
    for level in range(m.bit_length() - 1):
        count = 1 << level
        stride = m // count
        left = torch.arange(count, device=z.device) * stride
        right = left + stride
        midpoint = left + stride // 2
        noise = z[:, offset:offset + count * d_w].reshape(batch, count, d_w)
        scale = 0.5 * math.sqrt(horizon * 2.0 ** (-level))
        w[:, :, midpoint] = (
            0.5 * (w[:, :, left] + w[:, :, right])
            + scale * noise.transpose(1, 2)
        )
        offset += count * d_w
    return w[:, :, 1:] - w[:, :, :-1]

