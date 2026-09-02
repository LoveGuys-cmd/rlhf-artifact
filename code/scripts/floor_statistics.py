#!/usr/bin/env python3
"""Deterministic prompt-cluster statistics for frozen non-inferiority floors."""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

import torch


def linear_quantile(values: torch.Tensor, probability: float) -> float:
    """Return a deterministic linearly interpolated empirical quantile."""
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if tensor.numel() < 1 or not torch.isfinite(tensor).all():
        raise ValueError("quantile values must be finite and nonempty")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    ordered = tensor.sort().values
    position = (ordered.numel() - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower].item())
    weight = position - lower
    return float(
        (ordered[lower] * (1.0 - weight) + ordered[upper] * weight).item()
    )


def prompt_cluster_bootstrap_lcb(
    prompts: Sequence[str],
    scores: torch.Tensor,
    alpha: float,
    draws: int,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    """Estimate a one-sided lower bound by resampling independent prompts."""
    values = torch.as_tensor(scores, dtype=torch.float64).flatten().cpu()
    if len(prompts) != values.numel() or values.numel() < 2:
        raise ValueError("prompt-cluster bootstrap requires aligned nontrivial scores")
    if not torch.isfinite(values).all():
        raise ValueError("prompt-cluster bootstrap scores must be finite")
    alpha = float(alpha)
    draws = int(draws)
    if not 0.0 < alpha < 0.5:
        raise ValueError("one-sided floor alpha must lie in (0, 0.5)")
    if draws < 2000:
        raise ValueError("prompt-cluster bootstrap requires at least 2000 draws")

    grouped: dict[str, list[float]] = {}
    for prompt, score in zip(prompts, values.tolist()):
        key = " ".join(str(prompt).split())
        if not key:
            raise ValueError("prompt-cluster bootstrap encountered an empty prompt")
        grouped.setdefault(key, []).append(float(score))
    if len(grouped) < 2:
        raise ValueError("prompt-cluster bootstrap requires at least two prompts")
    cluster_means = torch.tensor(
        [statistics.fmean(grouped[key]) for key in sorted(grouped)],
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    estimates = []
    remaining = draws
    chunk_size = min(1024, draws)
    while remaining:
        current = min(chunk_size, remaining)
        indices = torch.randint(
            0,
            cluster_means.numel(),
            (current, cluster_means.numel()),
            generator=generator,
        )
        estimates.append(cluster_means[indices].mean(dim=1))
        remaining -= current
    bootstrap_means = torch.cat(estimates)
    lcb = linear_quantile(bootstrap_means, alpha)
    cluster_sizes = [len(grouped[key]) for key in sorted(grouped)]
    return lcb, {
        "floor_estimator": "prompt_cluster_percentile_bootstrap_lcb",
        "floor_bootstrap_unit": "prompt",
        "floor_bootstrap_alpha": alpha,
        "floor_bootstrap_confidence": 1.0 - alpha,
        "floor_bootstrap_draws": draws,
        "floor_bootstrap_seed": int(seed),
        "floor_response_count": int(values.numel()),
        "floor_prompt_clusters": int(cluster_means.numel()),
        "floor_min_responses_per_prompt": min(cluster_sizes),
        "floor_max_responses_per_prompt": max(cluster_sizes),
        "floor_response_mean": float(values.mean().item()),
        "floor_response_sd": float(values.std(unbiased=True).item()),
        "floor_prompt_mean": float(cluster_means.mean().item()),
        "floor_prompt_mean_sd": float(cluster_means.std(unbiased=True).item()),
        "floor_prompt_mean_lcb": lcb,
    }
