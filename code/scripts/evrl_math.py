#!/usr/bin/env python3
"""Numerically stable Gaussian extreme-value utilities for EV-PPO."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import torch


@dataclass(frozen=True)
class QuadratureDiagnostics:
    accepted_order: int
    previous_order: int
    relative_residual: float
    converged: bool
    deterministic_breaks: int

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@lru_cache(maxsize=32)
def _legendre_rule(order: int) -> tuple[torch.Tensor, torch.Tensor]:
    if order < 8:
        raise ValueError("quadrature order must be at least 8")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return (
        torch.from_numpy(nodes.astype(np.float64, copy=False)),
        torch.from_numpy(weights.astype(np.float64, copy=False)),
    )


def finite_n_gaussian_calibration(best_of_n: int, order: int = 512) -> float:
    """Return kappa_N = E[max_{j <= N} Z_j] for iid standard Gaussians."""
    n = int(best_of_n)
    if n < 1:
        raise ValueError("best_of_n must be positive")
    if n == 1:
        return 0.0
    nodes, weights = _legendre_rule(int(order))
    # Map (-1, 1) to (0, infinity).  The positive-half survival identity is
    # more stable than integrating z * phi(z) * Phi(z)^(N-1) over R.
    angle = (math.pi / 4.0) * (nodes + 1.0)
    t = torch.tan(angle)
    jacobian = (math.pi / 4.0) / torch.cos(angle).square()
    log_cdf = torch.special.log_ndtr(t)
    log_sf = torch.special.log_ndtr(-t)
    integrand = (1.0 - torch.exp(n * log_cdf) - torch.exp(n * log_sf)).clamp_min(0.0)
    return float(torch.sum(weights * jacobian * integrand).item())


def _transformed_segments(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    center: float,
    scale: float,
) -> list[tuple[float, float]]:
    deterministic = mu[sigma <= 0.0]
    if deterministic.numel() == 0:
        return [(-1.0, 1.0)]
    breaks = []
    for value in torch.unique(deterministic).tolist():
        u = (2.0 / math.pi) * math.atan((float(value) - center) / scale)
        if -1.0 < u < 1.0:
            breaks.append(u)
    points = [-1.0, *sorted(set(breaks)), 1.0]
    return [(left, right) for left, right in zip(points[:-1], points[1:]) if right > left]


def gaussian_max_credit_fixed_order(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    order: int,
    scale_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute all exact marginal credits with a fixed quadrature order.

    The implementation evaluates

        integral (1 - F_j(t)) prod_{r != j} F_r(t) dt

    after a tangent transform of the real line.  Deterministic candidates are
    represented by step CDFs without adding an artificial variance floor; the
    transformed domain is partitioned at their discontinuities.
    """
    mu64 = torch.as_tensor(mu, dtype=torch.float64, device="cpu").reshape(-1)
    sigma64 = torch.as_tensor(sigma, dtype=torch.float64, device="cpu").reshape(-1)
    if mu64.shape != sigma64.shape:
        raise ValueError("mu and sigma must have the same shape")
    if mu64.numel() < 1:
        raise ValueError("at least one candidate is required")
    if not torch.isfinite(mu64).all() or not torch.isfinite(sigma64).all():
        raise ValueError("mu and sigma must be finite")
    if (sigma64 < 0.0).any():
        raise ValueError("sigma must be nonnegative")
    if mu64.numel() == 1:
        return mu64.clone()

    center = float(mu64.mean().item())
    scale = float(torch.sqrt(torch.mean((mu64 - center).square() + sigma64.square())).item())
    scale = max(scale, float(scale_epsilon))
    segments = _transformed_segments(mu64, sigma64, center, scale)
    base_nodes, base_weights = _legendre_rule(int(order))
    credit = torch.zeros_like(mu64)

    for left, right in segments:
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        u = mid + half * base_nodes
        weights = half * base_weights
        angle = (math.pi / 2.0) * u
        t = center + scale * torch.tan(angle)
        log_jacobian = math.log(scale * math.pi / 2.0) - 2.0 * torch.log(torch.cos(angle))

        positive = sigma64 > 0.0
        log_cdf = torch.empty((mu64.numel(), t.numel()), dtype=torch.float64)
        log_sf = torch.empty_like(log_cdf)
        if positive.any():
            z = (t.unsqueeze(0) - mu64[positive].unsqueeze(1)) / sigma64[positive].unsqueeze(1)
            log_cdf[positive] = torch.special.log_ndtr(z)
            log_sf[positive] = torch.special.log_ndtr(-z)
        if (~positive).any():
            threshold = mu64[~positive].unsqueeze(1)
            log_cdf[~positive] = torch.where(
                t.unsqueeze(0) >= threshold,
                torch.zeros((), dtype=torch.float64),
                torch.full((), -torch.inf, dtype=torch.float64),
            )
            log_sf[~positive] = torch.where(
                t.unsqueeze(0) < threshold,
                torch.zeros((), dtype=torch.float64),
                torch.full((), -torch.inf, dtype=torch.float64),
            )

        # Prefix/suffix products avoid inf - inf when a deterministic CDF is 0.
        prefix = torch.zeros((mu64.numel() + 1, t.numel()), dtype=torch.float64)
        suffix = torch.zeros_like(prefix)
        for idx in range(mu64.numel()):
            prefix[idx + 1] = prefix[idx] + log_cdf[idx]
        for idx in range(mu64.numel() - 1, -1, -1):
            suffix[idx] = suffix[idx + 1] + log_cdf[idx]
        log_other_cdf = prefix[:-1] + suffix[1:]
        log_terms = (
            torch.log(weights).unsqueeze(0)
            + log_jacobian.unsqueeze(0)
            + log_sf
            + log_other_cdf
        )
        credit += torch.exp(log_terms).sum(dim=1)

    return credit.clamp_min(0.0)


def gaussian_max_credit(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    initial_order: int = 64,
    max_order: int = 256,
    tolerance: float = 1e-6,
    scale_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, QuadratureDiagnostics]:
    """Adaptively evaluate exact marginal credits for one candidate group."""
    initial_order = int(initial_order)
    max_order = int(max_order)
    if initial_order > max_order:
        raise ValueError("initial_order cannot exceed max_order")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    current_order = initial_order
    current = gaussian_max_credit_fixed_order(mu, sigma, current_order, scale_epsilon)
    if current.numel() == 1:
        diag = QuadratureDiagnostics(current_order, current_order, 0.0, True, 0)
        return current.to(dtype=torch.float32), diag

    previous_order = current_order
    residual = math.inf
    converged = False
    while current_order < max_order:
        next_order = min(2 * current_order, max_order)
        refined = gaussian_max_credit_fixed_order(mu, sigma, next_order, scale_epsilon)
        residual = float(torch.max(torch.abs(refined - current) / (1.0 + torch.abs(refined))).item())
        previous_order = current_order
        current_order = next_order
        current = refined
        if residual <= tolerance:
            converged = True
            break

    deterministic_breaks = int(torch.unique(torch.as_tensor(mu)[torch.as_tensor(sigma) <= 0.0]).numel())
    diag = QuadratureDiagnostics(
        accepted_order=current_order,
        previous_order=previous_order,
        relative_residual=residual,
        converged=converged,
        deterministic_breaks=deterministic_breaks,
    )
    return current.to(dtype=torch.float32), diag


def gaussian_max_credit_batch(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    initial_order: int = 64,
    max_order: int = 256,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, list[QuadratureDiagnostics]]:
    if mu.ndim != 2 or sigma.ndim != 2 or mu.shape != sigma.shape:
        raise ValueError("mu and sigma must be equal-shaped [batch, candidates] tensors")
    credits = []
    diagnostics = []
    for prompt_mu, prompt_sigma in zip(mu, sigma):
        q, diag = gaussian_max_credit(
            prompt_mu,
            prompt_sigma,
            initial_order=initial_order,
            max_order=max_order,
            tolerance=tolerance,
        )
        credits.append(q)
        diagnostics.append(diag)
    return torch.stack(credits, dim=0), diagnostics


def gaussian_expected_improvement(mu: torch.Tensor, sigma: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    mu, sigma, threshold = torch.broadcast_tensors(mu, sigma, threshold)
    positive = sigma > 0.0
    safe_sigma = torch.where(positive, sigma, torch.ones_like(sigma))
    delta = mu - threshold
    z = delta / safe_sigma
    normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    value = delta * normal.cdf(z) + safe_sigma * torch.exp(normal.log_prob(z))
    return torch.where(positive, value, delta.clamp_min(0.0))


def monte_carlo_rb_credit(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    draws: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Rao--Blackwellized opponent-noise fallback used only for ablations."""
    mu = torch.as_tensor(mu, dtype=torch.float64, device="cpu")
    sigma = torch.as_tensor(sigma, dtype=torch.float64, device="cpu")
    if mu.ndim != 1 or sigma.shape != mu.shape or mu.numel() < 2:
        raise ValueError("fallback expects one group with at least two candidates")
    draws = int(draws)
    if draws < 1:
        raise ValueError("draws must be positive")
    noise = torch.randn((draws, mu.numel()), dtype=torch.float64, generator=generator)
    sampled = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
    top2 = torch.topk(sampled, k=2, dim=1)
    top = top2.values[:, :1]
    second = top2.values[:, 1:2]
    winner = top2.indices[:, :1]
    thresholds = top.expand(-1, mu.numel()).clone()
    thresholds.scatter_(1, winner, second)
    ei = gaussian_expected_improvement(mu.unsqueeze(0), sigma.unsqueeze(0), thresholds)
    return ei.mean(dim=0).to(dtype=torch.float32)
