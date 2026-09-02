#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
SEEDS = (314, 2718, 1618)
METHODS = {
    "ev_ppo": "StableMax-PPO",
    "vanilla_ppo": "Mean-PPO",
    "vanilla_grpo": "GRPO",
    "scalar_max_ppo": "Scalar max-PPO",
}
COLORS = {
    "ev_ppo": "#0072B2",
    "vanilla_ppo": "#009E73",
    "vanilla_grpo": "#D55E00",
    "scalar_max_ppo": "#CC79A7",
}
MARKERS = {
    "ev_ppo": "o",
    "vanilla_ppo": "s",
    "vanilla_grpo": "^",
    "scalar_max_ppo": "D",
}


def load_data():
    terminal = json.loads((ROOT / "PUBLICATION_V8_TERMINAL.json").read_text())
    aggregate = {row["method"]: row for row in terminal["method_aggregate"]}
    seed_max = {}
    seed_kl = {}
    for seed in SEEDS:
        with (ROOT / f"seed{seed}_comparison_table_paper_final.csv").open() as handle:
            rows = {row["method"]: row for row in csv.DictReader(handle)}
        heldout = json.loads((ROOT / f"heldout_kl_seed{seed}.json").read_text())
        for method in METHODS:
            seed_max[method, seed] = float(rows[method]["robust_ordinal_expected_max_mean"])
            seed_kl[method, seed] = float(
                heldout["heldout_kl"][method]["common_base_trajectory_kl"]["mean"]
            )
    return aggregate, seed_max, seed_kl


def main():
    aggregate, seed_max, seed_kl = load_data()
    fig, ax = plt.subplots(figsize=(6.6, 4.1), constrained_layout=True)

    aggregate_points = {}
    for method, label in METHODS.items():
        xs = [seed_kl[method, seed] for seed in SEEDS]
        ys = [seed_max[method, seed] for seed in SEEDS]
        x = sum(xs) / len(xs)
        row = aggregate[method]
        y = row["robust_expected_max"]
        yerr = [[y - row["ci_low"]], [row["ci_high"] - y]]
        aggregate_points[method] = (x, y)

        ax.scatter(
            xs,
            ys,
            s=28,
            marker=MARKERS[method],
            facecolors="none",
            edgecolors=COLORS[method],
            linewidths=1.0,
            alpha=0.55,
            zorder=2,
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=MARKERS[method],
            markersize=8.5,
            color=COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=3,
            elinewidth=1.5,
            zorder=4,
        )
        offsets = {
            "ev_ppo": (5, -16),
            "vanilla_ppo": (5, -16),
            "vanilla_grpo": (-36, 9),
            "scalar_max_ppo": (-32, -18),
        }
        ax.annotate(
            label,
            (x, y),
            xytext=offsets[method],
            textcoords="offset points",
            fontsize=8.5,
            color=COLORS[method],
            fontweight="semibold" if method == "ev_ppo" else "normal",
        )

    cr_x, cr_y = aggregate_points["ev_ppo"]
    mean_x, mean_y = aggregate_points["vanilla_ppo"]
    grpo_x, grpo_y = aggregate_points["vanilla_grpo"]
    ax.annotate(
        "lower KL\ncompetitive max",
        xy=(cr_x, cr_y),
        xytext=(mean_x - 0.003, mean_y - 0.075),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
        fontsize=8,
        color="#444444",
        ha="center",
    )
    ax.annotate(
        "substantially lower KL\nsmall max gap",
        xy=(cr_x, cr_y),
        xytext=(grpo_x - 0.008, grpo_y - 0.075),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
        fontsize=8,
        color="#444444",
        ha="center",
    )

    ax.set_xlabel("Held-out KL on common base trajectories (lower is better)")
    ax.set_ylabel("Robust expected maximum (higher is better)")
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.12, y=0.18)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666",
                   markeredgecolor="white", markersize=7, label="Three-seed aggregate"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="#666666", markersize=5, label="Individual seed"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )

    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "max_kl_pareto.pdf", bbox_inches="tight")
    fig.savefig(out / "max_kl_pareto.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
