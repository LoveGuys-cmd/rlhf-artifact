#!/usr/bin/env python3
"""Reward-moment diagnostics for top-tier RM reporting.

This script evaluates both parts of the calibrated reward-moment scorer used by
EVRL:

* mu quality: chosen/rejected accuracy, margin, AUC-style pair ranking.
* sigma quality: whether larger uncertainty identifies harder pairs, smaller
  margins, lower confidence, and worse NLL/Brier after binning by pair sigma.

It deliberately imports ``load_rm`` from ``evrl_experiment.py`` so that the
diagnostics use the exact same scorer and mu/sigma transformation as policy
training/evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evrl_experiment import load_preference_pairs, load_rm


EPS = 1e-12


def load_preference_pair_tuples(path: str, max_pairs: int) -> list[tuple[str, str, str]]:
    rows = load_preference_pairs(Path(path), limit=max_pairs)
    return [(row["prompt"], row["chosen"], row["rejected"]) for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm-checkpoint", default="exp/strong_rm/best_ckpt")
    parser.add_argument("--id-eval-jsonl", default="dataset/paper_pairs_test.jsonl")
    parser.add_argument("--id-val-jsonl", default="dataset/paper_pairs_val.jsonl")
    parser.add_argument("--ood-jsonl", default="")
    parser.add_argument("--ood-hf-dataset", default=os.environ.get("OOD_DATASET", "Anthropic/hh-rlhf"))
    parser.add_argument("--output-dir", default="exp/strong_rm/eval/uncertainty")
    parser.add_argument("--max-id-pairs", type=int, default=int(os.environ.get("RM_UNCERTAINTY_MAX_ID_PAIRS", "8192")))
    parser.add_argument("--max-ood-pairs", type=int, default=int(os.environ.get("RM_UNCERTAINTY_MAX_OOD_PAIRS", "2048")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("RM_UNCERTAINTY_BATCH_SIZE", "1")))
    parser.add_argument("--rm-max-length", type=int, default=int(os.environ.get("RM_MAX_LENGTH", "1024")))
    parser.add_argument("--rm-temperature", type=float, default=float(os.environ.get("RM_TEMPERATURE", "1.0")))
    parser.add_argument("--num-sigma-bins", type=int, default=5)
    parser.add_argument("--num-reliability-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ood", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key, "")) for key in writer.fieldnames})


def format_cell(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.10g}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def normal_cdf(x: np.ndarray) -> np.ndarray:
    vec = np.vectorize(lambda v: 0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))))
    return vec(np.asarray(x, dtype=np.float64))


def nll_binary_positive(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0 - EPS)
    return -np.log(prob)


def brier_binary_positive(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    return (1.0 - prob) ** 2


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(np.asarray(x)), rankdata(np.asarray(y)))


def quantile_bin_indices(values: np.ndarray, num_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    bins = np.zeros(len(values), dtype=int)
    for rank, idx in enumerate(order):
        bins[idx] = min(num_bins - 1, int(rank * num_bins / max(len(values), 1)))
    return bins


def reliability_bins(prob: np.ndarray, correct: np.ndarray, num_bins: int) -> tuple[list[dict[str, Any]], float]:
    prob = np.asarray(prob, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for bin_idx in range(num_bins):
        lo = bin_idx / num_bins
        hi = (bin_idx + 1) / num_bins
        if bin_idx == num_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        n = int(np.sum(mask))
        if n == 0:
            rows.append(
                {
                    "bin": bin_idx,
                    "prob_low": lo,
                    "prob_high": hi,
                    "num_pairs": 0,
                    "mean_predicted_probability": float("nan"),
                    "empirical_accuracy": float("nan"),
                    "abs_calibration_gap": float("nan"),
                }
            )
            continue
        mean_prob = float(np.mean(prob[mask]))
        acc = float(np.mean(correct[mask]))
        gap = abs(acc - mean_prob)
        ece += (n / len(prob)) * gap
        rows.append(
            {
                "bin": bin_idx,
                "prob_low": lo,
                "prob_high": hi,
                "num_pairs": n,
                "mean_predicted_probability": mean_prob,
                "empirical_accuracy": acc,
                "abs_calibration_gap": gap,
            }
        )
    return rows, float(ece)


def load_hh_rlhf_pairs(dataset_name: str, max_pairs: int) -> list[tuple[str, str, str]]:
    from datasets import load_dataset

    loaded = load_dataset(dataset_name)
    split = "test" if "test" in loaded else "validation" if "validation" in loaded else "train"
    ds = loaded[split]
    pairs: list[tuple[str, str, str]] = []
    for row in ds:
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        if not chosen or not rejected:
            continue
        prompt = common_prefix_prompt(chosen, rejected)
        chosen_resp = chosen[len(prompt) :].strip() if prompt else chosen
        rejected_resp = rejected[len(prompt) :].strip() if prompt else rejected
        pairs.append((prompt.strip(), chosen_resp, rejected_resp))
        if len(pairs) >= max_pairs:
            break
    return pairs


def common_prefix_prompt(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    prefix = a[:i]
    markers = ["\n\nAssistant:", "\nAssistant:", "Assistant:"]
    cut = -1
    for marker in markers:
        idx = prefix.rfind(marker)
        if idx > cut:
            cut = idx
    if cut >= 0:
        return prefix[:cut]
    return prefix


def score_pairs(
    model: Any,
    pairs: list[tuple[str, str, str]],
    batch_size: int,
    rm_max_length: int,
    rm_temperature: float,
) -> dict[str, np.ndarray]:
    prompts = [p for p, _, _ in pairs]
    chosen = [c for _, c, _ in pairs]
    rejected = [r for _, _, r in pairs]
    chosen_mu_t, chosen_sigma_t = model.score_prompt_responses(
        prompts, chosen, max_length=rm_max_length, batch_size=batch_size, rm_temperature=rm_temperature
    )
    rejected_mu_t, rejected_sigma_t = model.score_prompt_responses(
        prompts, rejected, max_length=rm_max_length, batch_size=batch_size, rm_temperature=rm_temperature
    )
    chosen_mu = chosen_mu_t.detach().cpu().numpy().astype(np.float64)
    rejected_mu = rejected_mu_t.detach().cpu().numpy().astype(np.float64)
    chosen_sigma = chosen_sigma_t.detach().cpu().numpy().astype(np.float64)
    rejected_sigma = rejected_sigma_t.detach().cpu().numpy().astype(np.float64)
    margin = chosen_mu - rejected_mu
    pair_sigma = np.sqrt(np.maximum(chosen_sigma**2 + rejected_sigma**2, EPS))
    prob_gaussian = normal_cdf(margin / pair_sigma)
    prob_mu_only = sigmoid(margin)
    correct = (margin > 0.0).astype(np.float64)
    return {
        "chosen_mu": chosen_mu,
        "rejected_mu": rejected_mu,
        "chosen_sigma": chosen_sigma,
        "rejected_sigma": rejected_sigma,
        "margin_mu": margin,
        "abs_margin_mu": np.abs(margin),
        "pair_sigma": pair_sigma,
        "prob_gaussian_sigma": prob_gaussian,
        "prob_mu_only_sigmoid": prob_mu_only,
        "correct": correct,
        "nll_gaussian_sigma": nll_binary_positive(prob_gaussian),
        "brier_gaussian_sigma": brier_binary_positive(prob_gaussian),
        "nll_mu_only": nll_binary_positive(prob_mu_only),
        "brier_mu_only": brier_binary_positive(prob_mu_only),
    }


def summarize_arrays(prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    margin = arrays["margin_mu"]
    pair_sigma = arrays["pair_sigma"]
    prob = arrays["prob_gaussian_sigma"]
    correct = arrays["correct"]
    candidate_labels = np.concatenate(
        [np.ones_like(arrays["chosen_mu"], dtype=np.int64), np.zeros_like(arrays["rejected_mu"], dtype=np.int64)]
    )
    candidate_mu_scores = np.concatenate([arrays["chosen_mu"], arrays["rejected_mu"]])
    incorrect = 1.0 - correct
    reliability_rows, ece = reliability_bins(prob, correct, 10)
    return {
        f"{prefix}_num_pairs": int(len(margin)),
        f"{prefix}_accuracy": float(np.mean(correct)),
        f"{prefix}_pairwise_accuracy_half_tie": float(
            np.mean((margin > 0.0).astype(np.float64) + 0.5 * (margin == 0.0).astype(np.float64))
        ),
        f"{prefix}_candidate_mu_roc_auc": roc_auc_binary(candidate_labels, candidate_mu_scores),
        f"{prefix}_sigma_error_detection_auc": roc_auc_binary(incorrect.astype(np.int64), pair_sigma),
        f"{prefix}_mean_margin_mu": float(np.mean(margin)),
        f"{prefix}_median_margin_mu": float(np.median(margin)),
        f"{prefix}_mean_abs_margin_mu": float(np.mean(np.abs(margin))),
        f"{prefix}_chosen_mu_mean": float(np.mean(arrays["chosen_mu"])),
        f"{prefix}_rejected_mu_mean": float(np.mean(arrays["rejected_mu"])),
        f"{prefix}_chosen_sigma_mean": float(np.mean(arrays["chosen_sigma"])),
        f"{prefix}_rejected_sigma_mean": float(np.mean(arrays["rejected_sigma"])),
        f"{prefix}_pair_sigma_mean": float(np.mean(pair_sigma)),
        f"{prefix}_pair_sigma_median": float(np.median(pair_sigma)),
        f"{prefix}_mean_preference_probability_gaussian_sigma": float(np.mean(prob)),
        f"{prefix}_mean_preference_probability_mu_only": float(np.mean(arrays["prob_mu_only_sigmoid"])),
        f"{prefix}_nll_gaussian_sigma": float(np.mean(arrays["nll_gaussian_sigma"])),
        f"{prefix}_brier_gaussian_sigma": float(np.mean(arrays["brier_gaussian_sigma"])),
        f"{prefix}_ece_gaussian_sigma": ece,
        f"{prefix}_nll_mu_only": float(np.mean(arrays["nll_mu_only"])),
        f"{prefix}_brier_mu_only": float(np.mean(arrays["brier_mu_only"])),
        f"{prefix}_spearman_sigma_abs_margin": spearman(pair_sigma, np.abs(margin)),
        f"{prefix}_pearson_sigma_abs_margin": pearson(pair_sigma, np.abs(margin)),
        f"{prefix}_spearman_sigma_correct": spearman(pair_sigma, correct),
        f"{prefix}_pearson_sigma_correct": pearson(pair_sigma, correct),
        f"{prefix}_spearman_sigma_probability": spearman(pair_sigma, prob),
        f"{prefix}_pearson_sigma_probability": pearson(pair_sigma, prob),
        f"{prefix}_reliability_rows": reliability_rows,
    }


def sigma_bin_rows(arrays: dict[str, np.ndarray], num_bins: int) -> list[dict[str, Any]]:
    pair_sigma = arrays["pair_sigma"]
    bins = quantile_bin_indices(pair_sigma, num_bins)
    rows: list[dict[str, Any]] = []
    for b in range(num_bins):
        mask = bins == b
        n = int(np.sum(mask))
        if n == 0:
            continue
        rows.append(
            {
                "sigma_bin": b + 1,
                "sigma_bin_label": f"q{b + 1}_{'low' if b == 0 else 'high' if b == num_bins - 1 else 'mid'}_sigma",
                "num_pairs": n,
                "pair_sigma_min": float(np.min(pair_sigma[mask])),
                "pair_sigma_max": float(np.max(pair_sigma[mask])),
                "pair_sigma_mean": float(np.mean(pair_sigma[mask])),
                "accuracy": float(np.mean(arrays["correct"][mask])),
                "mean_abs_margin_mu": float(np.mean(arrays["abs_margin_mu"][mask])),
                "mean_margin_mu": float(np.mean(arrays["margin_mu"][mask])),
                "mean_preference_probability_gaussian_sigma": float(np.mean(arrays["prob_gaussian_sigma"][mask])),
                "mean_preference_probability_mu_only": float(np.mean(arrays["prob_mu_only_sigmoid"][mask])),
                "nll_gaussian_sigma": float(np.mean(arrays["nll_gaussian_sigma"][mask])),
                "brier_gaussian_sigma": float(np.mean(arrays["brier_gaussian_sigma"][mask])),
                "chosen_sigma_mean": float(np.mean(arrays["chosen_sigma"][mask])),
                "rejected_sigma_mean": float(np.mean(arrays["rejected_sigma"][mask])),
            }
        )
    return rows


def pair_prediction_rows(pairs: list[tuple[str, str, str]], arrays: dict[str, np.ndarray], num_bins: int) -> list[dict[str, Any]]:
    bins = quantile_bin_indices(arrays["pair_sigma"], num_bins)
    rows: list[dict[str, Any]] = []
    for idx, (prompt, chosen, rejected) in enumerate(pairs):
        rows.append(
            {
                "pair_id": idx,
                "prompt_preview": prompt[:180].replace("\n", " "),
                "chosen_preview": chosen[:120].replace("\n", " "),
                "rejected_preview": rejected[:120].replace("\n", " "),
                "sigma_bin": int(bins[idx] + 1),
                "chosen_mu": arrays["chosen_mu"][idx],
                "rejected_mu": arrays["rejected_mu"][idx],
                "margin_mu": arrays["margin_mu"][idx],
                "abs_margin_mu": arrays["abs_margin_mu"][idx],
                "chosen_sigma": arrays["chosen_sigma"][idx],
                "rejected_sigma": arrays["rejected_sigma"][idx],
                "pair_sigma": arrays["pair_sigma"][idx],
                "prob_gaussian_sigma": arrays["prob_gaussian_sigma"][idx],
                "prob_mu_only_sigmoid": arrays["prob_mu_only_sigmoid"][idx],
                "correct": bool(arrays["correct"][idx] > 0.5),
                "nll_gaussian_sigma": arrays["nll_gaussian_sigma"][idx],
                "brier_gaussian_sigma": arrays["brier_gaussian_sigma"][idx],
                "nll_mu_only": arrays["nll_mu_only"][idx],
                "brier_mu_only": arrays["brier_mu_only"][idx],
            }
        )
    return rows


def maybe_plot(output_dir: Path, bin_rows: list[dict[str, Any]], reliability_rows: list[dict[str, Any]], arrays: dict[str, np.ndarray], ood_row: dict[str, Any] | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_warning.txt").write_text(str(exc), encoding="utf-8")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(r["sigma_bin"]) for r in bin_rows]
    sigma = [float(r["pair_sigma_mean"]) for r in bin_rows]
    acc = [float(r["accuracy"]) for r in bin_rows]
    abs_margin = [float(r["mean_abs_margin_mu"]) for r in bin_rows]
    nll = [float(r["nll_gaussian_sigma"]) for r in bin_rows]

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(labels, acc, marker="o", label="Accuracy")
    plt.xlabel("Pair-sigma quantile bin (low to high)")
    plt.ylabel("Chosen/rejected accuracy")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "rm_sigma_bins_accuracy.png", dpi=240)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(sigma, abs_margin, marker="o", color="#B45F35")
    plt.xlabel("Mean pair sigma")
    plt.ylabel("Mean |mu(chosen)-mu(rejected)|")
    plt.tight_layout()
    plt.savefig(fig_dir / "rm_sigma_vs_margin_bins.png", dpi=240)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(sigma, nll, marker="o", color="#4F8F55")
    plt.xlabel("Mean pair sigma")
    plt.ylabel("Gaussian-sigma NLL")
    plt.tight_layout()
    plt.savefig(fig_dir / "rm_sigma_vs_nll_bins.png", dpi=240)
    plt.close()

    rel = [r for r in reliability_rows if int(r.get("num_pairs", 0)) > 0]
    plt.figure(figsize=(5.2, 5.0))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.scatter(
        [float(r["mean_predicted_probability"]) for r in rel],
        [float(r["empirical_accuracy"]) for r in rel],
        s=[max(18, float(r["num_pairs"]) * 3.0) for r in rel],
        color="#3B6EA8",
        alpha=0.75,
    )
    plt.xlabel("Mean predicted P(chosen > rejected)")
    plt.ylabel("Empirical accuracy")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(fig_dir / "rm_reliability_gaussian_sigma.png", dpi=240)
    plt.close()

    rng = np.random.default_rng(42)
    idx = np.arange(len(arrays["pair_sigma"]))
    if len(idx) > 1200:
        idx = rng.choice(idx, size=1200, replace=False)
    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(arrays["pair_sigma"][idx], arrays["abs_margin_mu"][idx], s=8, alpha=0.25, color="#3B6EA8")
    plt.xlabel("Pair sigma")
    plt.ylabel("|mu(chosen)-mu(rejected)|")
    plt.tight_layout()
    plt.savefig(fig_dir / "rm_sigma_vs_abs_margin_scatter.png", dpi=240)
    plt.close()

    if ood_row:
        plt.figure(figsize=(5.4, 4.2))
        plt.bar(["ID", "OOD"], [float(ood_row["id_pair_sigma_mean"]), float(ood_row["ood_pair_sigma_mean"])], color=["#3B6EA8", "#B45F35"])
        plt.ylabel("Mean pair sigma")
        plt.tight_layout()
        plt.savefig(fig_dir / "rm_id_vs_ood_pair_sigma.png", dpi=240)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_rm(args.rm_checkpoint, "unused-by-strong-rm", dtype, device)
    model.to(device)
    model.eval()

    id_pairs = load_preference_pair_tuples(args.id_eval_jsonl, args.max_id_pairs)
    if not id_pairs:
        raise SystemExit(f"No ID preference pairs found in {args.id_eval_jsonl}")
    id_arrays = score_pairs(model, id_pairs, args.batch_size, args.rm_max_length, args.rm_temperature)
    bin_rows = sigma_bin_rows(id_arrays, args.num_sigma_bins)
    pair_rows = pair_prediction_rows(id_pairs, id_arrays, args.num_sigma_bins)
    summary = summarize_arrays("id", id_arrays)
    reliability_rows = summary.pop("id_reliability_rows")
    _, ece = reliability_bins(id_arrays["prob_gaussian_sigma"], id_arrays["correct"], args.num_reliability_bins)
    summary["id_ece_gaussian_sigma"] = ece

    low_acc = float(bin_rows[0]["accuracy"]) if bin_rows else float("nan")
    high_acc = float(bin_rows[-1]["accuracy"]) if bin_rows else float("nan")
    low_margin = float(bin_rows[0]["mean_abs_margin_mu"]) if bin_rows else float("nan")
    high_margin = float(bin_rows[-1]["mean_abs_margin_mu"]) if bin_rows else float("nan")
    variance_status = str(getattr(model, "reward_variance_status", "unknown"))
    sigma_mode = str(getattr(model, "sigma_mode", "unknown"))
    summary.update(
        {
            "sigma_accuracy_drop_high_minus_low": high_acc - low_acc,
            "sigma_abs_margin_drop_high_minus_low": high_margin - low_margin,
            "sigma_quality_expectation": (
                "Useful uncertainty should usually have high-sigma accuracy <= low-sigma accuracy "
                "and high-sigma |margin| <= low-sigma |margin|."
            ),
        }
    )

    ood_row: dict[str, Any] | None = None
    ood_warning = ""
    if not args.skip_ood:
        try:
            if args.ood_jsonl:
                ood_pairs = load_preference_pair_tuples(args.ood_jsonl, args.max_ood_pairs)
                ood_source = args.ood_jsonl
            else:
                ood_pairs = load_hh_rlhf_pairs(args.ood_hf_dataset, args.max_ood_pairs)
                ood_source = args.ood_hf_dataset
            if ood_pairs:
                ood_arrays = score_pairs(model, ood_pairs, args.batch_size, args.rm_max_length, args.rm_temperature)
                ood_summary = summarize_arrays("ood", ood_arrays)
                id_pair_sigma_mean = float(summary["id_pair_sigma_mean"])
                ood_pair_sigma_mean = float(ood_summary["ood_pair_sigma_mean"])
                ood_row = {
                    "id_source": args.id_eval_jsonl,
                    "ood_source": ood_source,
                    "id_num_pairs": summary["id_num_pairs"],
                    "ood_num_pairs": ood_summary["ood_num_pairs"],
                    "id_accuracy": summary["id_accuracy"],
                    "ood_accuracy": ood_summary["ood_accuracy"],
                    "id_pair_sigma_mean": id_pair_sigma_mean,
                    "ood_pair_sigma_mean": ood_pair_sigma_mean,
                    "ood_pair_sigma_lift": ood_pair_sigma_mean / max(id_pair_sigma_mean, EPS),
                    "id_nll_gaussian_sigma": summary["id_nll_gaussian_sigma"],
                    "ood_nll_gaussian_sigma": ood_summary["ood_nll_gaussian_sigma"],
                }
                summary.update(ood_row)
            else:
                ood_warning = "No OOD pairs loaded."
        except Exception as exc:
            ood_warning = f"OOD diagnostics skipped after error: {type(exc).__name__}: {exc}"
    summary["ood_warning"] = ood_warning

    summary.update(
        {
            "rm_checkpoint": str(args.rm_checkpoint),
            "id_eval_jsonl": str(args.id_eval_jsonl),
            "rm_temperature": args.rm_temperature,
            "rm_max_length": args.rm_max_length,
            "batch_size": args.batch_size,
            "uncertainty_type": sigma_mode,
            "reward_variance_status": variance_status,
            "not_bayesian_posterior": True,
            "sigma_definition": "pair_sigma=sqrt(chosen_sigma^2+rejected_sigma^2)",
            "probability_gaussian_sigma": "Phi((mu_chosen-mu_rejected)/pair_sigma)",
            "epistemic_caveat": (
                "Aleatoric predictive scale is not an OOD detector. Epistemic uncertainty and "
                "reward-model exploitation require a separate ensemble or independent quality constraint."
            ),
        }
    )

    write_csv(output_dir / "rm_uncertainty_summary.csv", [summary])
    write_csv(output_dir / "rm_sigma_binned_diagnostics.csv", bin_rows)
    write_csv(output_dir / "rm_reliability_gaussian_sigma.csv", reliability_rows)
    write_csv(output_dir / "rm_pair_predictions.csv", pair_rows)
    if ood_row:
        write_csv(output_dir / "rm_ood_diagnostics.csv", [ood_row])
    with (output_dir / "rm_uncertainty_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    maybe_plot(output_dir, bin_rows, reliability_rows, id_arrays, ood_row)

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"[rm-uncertainty] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
