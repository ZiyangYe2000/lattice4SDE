"""Time-inhomogeneous additive-noise Itô Euler models used in the paper."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol

import torch

from .bridge import brownian_bridge_increments

Tensor = torch.Tensor


class SDEModel(Protocol):
    T: float
    d_w: int

    def evaluate(self, z: Tensor, m: int) -> Tensor: ...
    def extra_values_per_path(self, m: int) -> int: ...


@dataclass(frozen=True)
class OUModel:
    T: float = 1.0
    x0: float = 0.2
    kappa: float = 1.0
    forcing: float = 0.5
    sigma: float = 0.4
    nu: float = 0.5
    payoff: str = "exponential"
    d_w: int = 1

    def extra_values_per_path(self, m: int) -> int:
        return m + 4

    def evaluate(self, z: Tensor, m: int) -> Tensor:
        h = self.T / m
        dw = brownian_bridge_increments(z, 1, m, self.T)[:, 0]
        x = torch.full((z.shape[0],), self.x0, dtype=z.dtype, device=z.device)
        for i in range(m):
            t = i * h
            x += h * (-self.kappa * x + self.forcing * math.sin(2 * math.pi * t / self.T))
            x += self.sigma * dw[:, i]
        if self.payoff == "quadratic":
            return x.square()
        if self.payoff == "exponential":
            return torch.exp(self.nu * x)
        raise ValueError(f"unknown payoff {self.payoff}")

    def exact_euler(self, m: int) -> float:
        h = self.T / m
        factor = 1.0 - self.kappa * h
        mean = self.x0
        for i in range(m):
            t = i * h
            mean = factor * mean + h * self.forcing * math.sin(2 * math.pi * t / self.T)
        variance = self.sigma**2 * h * sum(factor ** (2 * j) for j in range(m))
        if self.payoff == "quadratic":
            return mean * mean + variance
        return math.exp(self.nu * mean + 0.5 * self.nu**2 * variance)


@dataclass(frozen=True)
class KuramotoModel:
    d: int = 8
    T: float = 1.0
    coupling: float = 1.0
    asymmetry: float = 0.15
    sigma: float = 0.4

    @property
    def d_w(self) -> int:
        return self.d

    def extra_values_per_path(self, m: int) -> int:
        return self.d * m + 3 * self.d * self.d + 8 * self.d

    def evaluate(self, z: Tensor, m: int) -> Tensor:
        h = self.T / m
        dw = brownian_bridge_increments(z, self.d, m, self.T)
        idx = torch.arange(self.d, dtype=z.dtype, device=z.device)
        omega = torch.linspace(-0.5, 0.5, self.d, dtype=z.dtype, device=z.device)
        x = (2 * math.pi * idx / self.d).expand(z.shape[0], -1).clone()
        j, k = idx[:, None], idx[None, :]
        matrix = self.coupling / self.d * (1.0 + self.asymmetry * torch.sin((j + 1) * (k + 2)))
        matrix.fill_diagonal_(0.0)
        for i in range(m):
            modulation = 1.0 + 0.5 * math.sin(2 * math.pi * i * h / self.T)
            difference = x[:, None, :] - x[:, :, None]
            interaction = (torch.sin(difference) * matrix).sum(dim=2)
            x += h * (omega + modulation * interaction) + self.sigma * dw[:, :, i]
        return torch.cos(x).mean(dim=1).square() + torch.sin(x).mean(dim=1).square()


@dataclass(frozen=True)
class AnalyticLangevinModel:
    d: int = 4
    T: float = 1.0
    kappa: float = 1.0
    coupling: float = 0.5
    sigma: float = 0.4
    theta: float = 0.5
    nu: float = 0.2

    @property
    def d_w(self) -> int:
        return self.d

    def extra_values_per_path(self, m: int) -> int:
        return self.d * m + 3 * self.d * self.d + 8 * self.d

    def evaluate(self, z: Tensor, m: int) -> Tensor:
        h = self.T / m
        dw = brownian_bridge_increments(z, self.d, m, self.T)
        idx = torch.arange(self.d, dtype=z.dtype, device=z.device)
        x0 = 0.2 * torch.cos(2 * math.pi * idx / self.d)
        x = x0.expand(z.shape[0], -1).clone()
        for i in range(m):
            t = i * h
            modulation = 1.0 + 0.5 * abs(2 * t / self.T - 1.0) ** self.theta
            pair_difference = x[:, :, None] - x[:, None, :]
            interaction = torch.sin(pair_difference).sum(dim=2)
            drift = -self.kappa * x - modulation * (
                torch.sin(x) + self.coupling * interaction / self.d
            )
            x += h * drift + self.sigma * dw[:, :, i]
        return torch.exp(self.nu * x.sum(dim=1) / math.sqrt(self.d))


def normalized_fwht(values: Tensor, dim: int = 1) -> Tensor:
    """Apply the orthonormal Walsh--Hadamard transform along one axis."""
    n = values.shape[dim]
    if n < 1 or n & (n - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    # clone keeps this utility transform free of in-place changes to its input
    x = values.movedim(dim, -1).contiguous().clone()
    shape = x.shape
    x = x.reshape(-1, n)
    width = 1
    while width < n:
        y = x.reshape(-1, n // (2 * width), 2, width)
        left, right = y[:, :, 0, :].clone(), y[:, :, 1, :].clone()
        y[:, :, 0, :] = left + right
        y[:, :, 1, :] = left - right
        x = y.reshape(-1, n)
        width *= 2
    return (x / math.sqrt(n)).reshape(shape).movedim(-1, dim)


@dataclass(frozen=True)
class DimensionUniformModel:
    d: int = 4
    factorization: str = "aligned"
    T: float = 1.0
    sigma: float = 0.4

    @property
    def d_w(self) -> int:
        return self.d

    def extra_values_per_path(self, m: int) -> int:
        return 3 * self.d * m + 12 * self.d

    def evaluate(self, z: Tensor, m: int) -> Tensor:
        h = self.T / m
        dw = brownian_bridge_increments(z, self.d, m, self.T)
        if self.factorization == "rotated":
            dw = normalized_fwht(dw, dim=1)
        elif self.factorization != "aligned":
            raise ValueError(f"unknown factorization {self.factorization}")
        scales = self.sigma * torch.arange(1, self.d + 1, dtype=z.dtype, device=z.device).pow(-2)
        x = torch.zeros((z.shape[0], self.d), dtype=z.dtype, device=z.device)
        for i in range(m):
            lap = 2 * x
            lap[:, 1:] -= x[:, :-1]
            lap[:, :-1] -= x[:, 1:]
            drift = -x - 0.25 * lap + 0.5 * torch.sin(x)
            x += h * drift + scales * dw[:, :, i]
        return x.square().sum(dim=1)


def model_from_case(case: dict) -> SDEModel:
    name = case["model"]
    p = dict(case.get("parameters", {}))
    if name == "ou":
        return OUModel(**p)
    if name == "kuramoto":
        return KuramotoModel(d=case["d"], **p)
    if name == "analytic_langevin":
        return AnalyticLangevinModel(d=case["d"], **p)
    if name in ("dimension_uniform_aligned", "dimension_uniform_rotated"):
        factor = "aligned" if name.endswith("aligned") else "rotated"
        return DimensionUniformModel(d=case["d"], factorization=factor, **p)
    raise ValueError(f"unknown model {name}")


def model_parameters(model: SDEModel) -> dict:
    return asdict(model)

