#!/usr/bin/env python3
"""Exact finite-N utilities for ordinal Best-of-N objectives.

The reward model supplies a calibrated categorical distribution over ratings
0, ..., 4.  These functions contain no policy or model code and are kept
separate so the objective can be exhaustively regression-tested.
"""

from __future__ import annotations

import math

import torch


NUM_RATINGS = 5
TOP_RATING = 4


def validate_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Validate and return categorical probabilities without silently repairing them."""
    values = torch.as_tensor(probabilities)
    if values.ndim < 1 or values.shape[-1] != NUM_RATINGS:
        raise ValueError(
            f"expected probabilities with final dimension {NUM_RATINGS}, got {tuple(values.shape)}"
        )
    if not torch.isfinite(values).all():
        raise ValueError("ordinal probabilities must be finite")
    if (values < 0).any():
        raise ValueError("ordinal probabilities must be nonnegative")
    row_sums = values.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5):
        raise ValueError("ordinal probability rows must sum to one")
    return values


def _prefix_excluding(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return products over candidates before and after each candidate."""
    batch, count = values.shape[:2]
    prefix = torch.ones(
        (batch, count + 1), dtype=values.dtype, device=values.device
    )
    suffix = torch.ones_like(prefix)
    for index in range(count):
        prefix[:, index + 1] = prefix[:, index] * values[:, index]
    for index in range(count - 1, -1, -1):
        suffix[:, index] = suffix[:, index + 1] * values[:, index]
    return prefix, suffix


def exact_expected_max(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute E[max_j R_j] for independent categorical ratings."""
    probabilities = validate_probabilities(probabilities)
    cdf = probabilities.cumsum(dim=-1)
    result = torch.zeros(probabilities.shape[:-2], dtype=probabilities.dtype, device=probabilities.device)
    for threshold in range(1, NUM_RATINGS):
        result = result + 1.0 - cdf[..., threshold - 1].prod(dim=-1)
    return result


def exact_probability_any_rating(
    probabilities: torch.Tensor, rating: int = TOP_RATING
) -> torch.Tensor:
    """Compute P(any candidate receives the requested rating or higher)."""
    probabilities = validate_probabilities(probabilities)
    rating = int(rating)
    if not 0 <= rating < NUM_RATINGS:
        raise ValueError(f"rating must be in [0, {NUM_RATINGS - 1}]")
    probability_at_least = probabilities[..., rating:].sum(dim=-1)
    return 1.0 - (1.0 - probability_at_least).prod(dim=-1)


def exact_marginal_max_credit(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute q_i=E[(R_i-max_{j!=i} R_j)_+] exactly.

    The threshold identity is
    q_i = sum_{t=1}^4 P(R_i >= t) prod_{j!=i} P(R_j <= t-1).
    It is an unbiased Rao--Blackwellized policy-gradient credit for the
    expected maximum, and reduces to E[R_i] when the group has one candidate.
    """
    probabilities = validate_probabilities(probabilities)
    if probabilities.ndim != 3:
        raise ValueError(
            f"expected [batch, candidates, ratings] probabilities, got {tuple(probabilities.shape)}"
        )
    batch, count, _ = probabilities.shape
    cdf = probabilities.cumsum(dim=-1)
    credit = torch.zeros((batch, count), dtype=probabilities.dtype, device=probabilities.device)
    for threshold in range(1, NUM_RATINGS):
        opponent_cdf = cdf[..., threshold - 1]
        prefix, suffix = _prefix_excluding(opponent_cdf)
        others = prefix[:, :-1] * suffix[:, 1:]
        own_survival = probabilities[..., threshold:].sum(dim=-1)
        credit = credit + own_survival * others
    return credit


def exact_group_statistics(probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the primary ordinal objective and its predeclared tail diagnostic."""
    probabilities = validate_probabilities(probabilities)
    if probabilities.ndim != 3:
        raise ValueError(
            f"expected [batch, candidates, ratings] probabilities, got {tuple(probabilities.shape)}"
        )
    values = torch.arange(
        NUM_RATINGS, dtype=probabilities.dtype, device=probabilities.device
    )
    expected_rating = (probabilities * values).sum(dim=-1)
    variance = (
        probabilities * (values - expected_rating.unsqueeze(-1)).square()
    ).sum(dim=-1)
    return {
        "expected_max": exact_expected_max(probabilities),
        "probability_any_top_rating": exact_probability_any_rating(probabilities),
        "marginal_credit": exact_marginal_max_credit(probabilities),
        "expected_rating": expected_rating,
        "rating_variance": variance,
        "top_rating_probability": probabilities[..., TOP_RATING],
    }


def robust_lower_probabilities(
    probabilities: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Return the stochastically smallest distribution in the CDF ambiguity set.

    The one-sided Kolmogorov set permits F_q(t) <= min(1, F_p(t) + epsilon)
    for t=0,...,3.  Its joint worst case for every increasing rating
    functional is the upper CDF envelope itself.
    """
    probabilities = validate_probabilities(probabilities)
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("robust epsilon must be finite and lie in [0, 1]")
    finite_cdf = (probabilities.cumsum(dim=-1)[..., :-1] + epsilon).clamp_max(1.0)
    boundaries = torch.cat(
        (
            torch.zeros_like(finite_cdf[..., :1]),
            finite_cdf,
            torch.ones_like(finite_cdf[..., :1]),
        ),
        dim=-1,
    )
    robust = boundaries[..., 1:] - boundaries[..., :-1]
    return validate_probabilities(robust)


def robust_group_statistics(
    probabilities: torch.Tensor,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Return exact nominal and robust statistics under a frozen CDF radius."""
    probabilities = validate_probabilities(probabilities)
    robust_probabilities = robust_lower_probabilities(probabilities, epsilon)
    robust = exact_group_statistics(robust_probabilities)
    nominal = exact_group_statistics(probabilities)
    return {
        "robust_probabilities": robust_probabilities,
        "robust_expected_max": robust["expected_max"],
        "robust_probability_any_top_rating": robust["probability_any_top_rating"],
        "robust_marginal_credit": robust["marginal_credit"],
        "robust_expected_rating": robust["expected_rating"],
        "nominal_expected_max": nominal["expected_max"],
        "nominal_probability_any_top_rating": nominal["probability_any_top_rating"],
        "nominal_expected_rating": nominal["expected_rating"],
        "nominal_rating_variance": nominal["rating_variance"],
        "nominal_top_rating_probability": nominal["top_rating_probability"],
    }


def scalar_max_marginal_credit(values: torch.Tensor) -> torch.Tensor:
    """Exact sampled marginal credit for E[max_i value(Y_i)]."""
    values = torch.as_tensor(values)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("scalar max credit expects [batch, candidates] values")
    if not torch.isfinite(values).all():
        raise ValueError("scalar max values must be finite")
    if values.shape[1] == 1:
        return values.clone()
    top_two = torch.topk(values, k=2, dim=1)
    largest = top_two.values[:, :1]
    second = top_two.values[:, 1:2]
    winner = top_two.indices[:, :1]
    opponent_max = largest.expand_as(values).clone()
    opponent_max.scatter_(1, winner, second)
    return (values - opponent_max).clamp_min(0.0)
