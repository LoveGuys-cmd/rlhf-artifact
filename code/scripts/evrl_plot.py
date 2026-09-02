#!/usr/bin/env python3
"""Create robust ordinal-v5 publication figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = (
    "ev_ppo",
    "vanilla_ppo",
    "vanilla_grpo",
    "scalar_max_ppo",
    "entropic_ppo",
    "nominal_ev_ppo",
    "ev_ppo_no_mean",
    "ev_ppo_no_quality",
    "gaussian_ev_ppo",
    "top4_ppo",
    "best_of_n",
)
COLORS = {
    "ev_ppo": "#0B6E4F",
    "vanilla_ppo": "#355C7D",
    "vanilla_grpo": "#4E7D9A",
    "scalar_max_ppo": "#A65A44",
    "entropic_ppo": "#2A9D8F",
    "nominal_ev_ppo": "#6E9F88",
    "ev_ppo_no_mean": "#A77BCA",
    "ev_ppo_no_quality": "#C17767",
    "top4_ppo": "#D39B3B",
    "gaussian_ev_ppo": "#7A6C91",
    "best_of_n": "#777777",
}


def prepare(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column not in {"method", "method_name"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
    frame["_order"] = frame["method"].map({name: index for index, name in enumerate(METHOD_ORDER)})
    return frame.sort_values("_order")


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def objective_figure(frame: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    required = {
        "robust_ordinal_expected_max_mean",
        "robust_probability_any_rating_4_mean",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"ordinal result table is missing {sorted(required - set(frame.columns))}")
    labels = frame["method_name"].fillna(frame["method"]).tolist()
    colors = [COLORS.get(method, "#777777") for method in frame["method"]]
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    metrics = (
        ("robust_ordinal_expected_max_mean", "Robust expected maximum ordinal rating", r"$\widehat{E}_{\rm rob}[\max_{j\leq N}R_j]$"),
        ("robust_probability_any_rating_4_mean", "Robust top-rating probability", r"$\widehat{P}_{\rm rob}(\exists j:R_j=4)$"),
    )
    for axis, (column, title, ylabel) in zip(axes, metrics):
        values = frame[column].to_numpy(dtype=float)
        low = frame.get(column.replace("_mean", "_ci_low"), pd.Series(values)).to_numpy(dtype=float)
        high = frame.get(column.replace("_mean", "_ci_high"), pd.Series(values)).to_numpy(dtype=float)
        errors = np.vstack((np.maximum(values - low, 0.0), np.maximum(high - values, 0.0)))
        axis.bar(x, values, color=colors, yerr=errors, capsize=3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle(f"StableMax ordinal Best-of-N evaluation {suffix}".strip(), fontsize=12)
    save(fig, output_dir, "ordinal_objective_and_upper_tail")


def quality_figure(frame: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    required = {"robust_ordinal_expected_max_mean", "candidate_quality_mean"}
    if not required.issubset(frame.columns):
        return
    plotted = frame.dropna(subset=list(required))
    if plotted.empty:
        return
    fig, axis = plt.subplots(figsize=(6.5, 5.0))
    for _, row in plotted.iterrows():
        axis.scatter(
            row["candidate_quality_mean"],
            row["robust_ordinal_expected_max_mean"],
            s=65,
            color=COLORS.get(row["method"], "#777777"),
        )
        axis.annotate(
            row.get("method_name", row["method"]),
            (
                row["candidate_quality_mean"],
                row["robust_ordinal_expected_max_mean"],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Independent Quality RM score")
    axis.set_ylabel(r"$\widehat{E}_{\rm rob}[\max_{j\leq N}R_j]$")
    axis.set_title(f"Ordinal objective and independent quality {suffix}".strip())
    axis.grid(alpha=0.22)
    save(fig, output_dir, "ordinal_objective_quality_tradeoff")


def preference_figure(path: Path, output_dir: Path, suffix: str) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if "preference_score_vs_vanilla_ppo" not in frame:
        return
    frame["preference_score_vs_vanilla_ppo"] = pd.to_numeric(
        frame["preference_score_vs_vanilla_ppo"], errors="coerce"
    )
    frame = frame.dropna(subset=["preference_score_vs_vanilla_ppo"])
    if frame.empty:
        return
    labels = frame["method_name"].fillna(frame["method"]).tolist()
    values = frame["preference_score_vs_vanilla_ppo"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(
        np.arange(len(frame)),
        values,
        color=[COLORS.get(method, "#777777") for method in frame["method"]],
    )
    axis.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(np.arange(len(frame)), labels, rotation=35, ha="right")
    axis.set_ylabel("Independent evaluator preference score")
    axis.set_title(f"Model-based preference proxies, not human labels {suffix}".strip())
    axis.grid(axis="y", alpha=0.22)
    save(fig, output_dir, "independent_preference_proxies")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_csv", required=True, type=Path)
    parser.add_argument("--preference_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--title_suffix", default="")
    args = parser.parse_args()
    frame = prepare(args.table_csv)
    objective_figure(frame, args.output_dir, args.title_suffix)
    quality_figure(frame, args.output_dir, args.title_suffix)
    preference_figure(args.preference_csv, args.output_dir, args.title_suffix)


if __name__ == "__main__":
    main()
