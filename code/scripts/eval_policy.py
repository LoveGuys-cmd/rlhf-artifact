#!/usr/bin/env python3
"""Evaluate exact ordinal Best-of-N metrics from cached policy responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def response_path(eval_dir: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for path in (eval_dir / name, eval_dir / "responses" / name, eval_dir.parent / name):
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot resolve response JSONL {name} under {eval_dir}")


def exact_group(
    candidates: list[dict[str, Any]],
    robust_epsilon: float,
) -> tuple[float, float, float, float, float, float]:
    probabilities = []
    for candidate in candidates:
        values = [safe_float(candidate.get(f"p{rating}")) for rating in range(5)]
        if any(value is None for value in values):
            raise ValueError("response JSONL is missing p0,...,p4; rerun ordinal evaluation")
        values = [float(value) for value in values]
        if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-5:
            raise ValueError("response JSONL contains an invalid ordinal probability row")
        probabilities.append(values)
    expected_max = 0.0
    robust_expected_max = 0.0
    for threshold in range(1, 5):
        nominal_cdfs = [sum(row[:threshold]) for row in probabilities]
        expected_max += 1.0 - math.prod(nominal_cdfs)
        robust_expected_max += 1.0 - math.prod(
            min(1.0, value + robust_epsilon) for value in nominal_cdfs
        )
    any_four = 1.0 - math.prod(1.0 - row[4] for row in probabilities)
    robust_any_four = 1.0 - math.prod(
        min(1.0, sum(row[:4]) + robust_epsilon) for row in probabilities
    )
    expected_rating = [sum(rating * row[rating] for rating in range(5)) for row in probabilities]
    selected = max(range(len(candidates)), key=lambda index: (expected_rating[index], -index))
    return (
        robust_expected_max,
        robust_any_four,
        expected_max,
        any_four,
        expected_rating[selected],
        probabilities[selected][4],
    )


def bootstrap(values: list[float], seed: int, draws: int = 2000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def paired_sign_flip_pvalue(
    values: list[float], seed: int, draws: int = 10000
) -> float:
    """One-sided paired randomization p-value for a positive mean difference."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError("paired randomization requires finite differences")
    observed = sum(finite) / len(finite)
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(draws):
        permuted = sum(value if rng.random() < 0.5 else -value for value in finite)
        exceedances += int(permuted / len(finite) >= observed)
    return (exceedances + 1.0) / (draws + 1.0)


def paired_rank_biserial(values: list[float]) -> float:
    """Matched-pairs rank-biserial effect size with zero differences as ties."""
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    nonzero = positive + negative
    return 0.0 if nonzero == 0 else (positive - negative) / nonzero


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--rm_config", type=Path, required=True)
    parser.add_argument("--robust_calibration_report", type=Path, required=True)
    parser.add_argument("--reference_method", default="vanilla_ppo")
    parser.add_argument("--best_of_n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.best_of_n < 1:
        raise SystemExit("--best-of-n must be positive")
    robust_report = json.loads(args.robust_calibration_report.read_text(encoding="utf-8"))
    if robust_report.get("protocol") != "ordinal-v5-robust-calibration-gate-before-policy-training":
        raise ValueError("wrong robust calibration report protocol")
    robust_epsilon = float(robust_report["robust_epsilon"])
    if not 0.0 <= robust_epsilon <= 1.0:
        raise ValueError("invalid robust epsilon")
    table_path = args.eval_dir / "comparison_table_paper.csv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    source_rows = read_csv(table_path)
    summaries: dict[str, dict[str, Any]] = {}
    n_scaling_rows: list[dict[str, Any]] = []
    for row in source_rows:
        method = row.get("method", "")
        name = row.get("responses_jsonl", "")
        if not method or not name:
            continue
        path = response_path(args.eval_dir, name)
        values = []
        candidate_groups: list[list[dict[str, Any]]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                candidates = record.get("responses") or []
                if len(candidates) != args.best_of_n:
                    raise ValueError(f"{method} has a non-{args.best_of_n} group in {path}")
                candidate_groups.append(candidates)
                values.append(exact_group(candidates, robust_epsilon))
        if not values:
            raise ValueError(f"no response groups found for {method}")
        robust_max_values = [value[0] for value in values]
        robust_any_four_values = [value[1] for value in values]
        max_values = [value[2] for value in values]
        any_four_values = [value[3] for value in values]
        robust_max_low, robust_max_high = bootstrap(
            robust_max_values, args.seed + 500 + len(summaries)
        )
        robust_tail_low, robust_tail_high = bootstrap(
            robust_any_four_values, args.seed + 1500 + len(summaries)
        )
        max_low, max_high = bootstrap(max_values, args.seed + len(summaries))
        tail_low, tail_high = bootstrap(any_four_values, args.seed + 1000 + len(summaries))
        summaries[method] = {
            "method": method,
            "method_name": row.get("method_name", method),
            "seed": row.get("seed", args.seed),
            "best_of_n": args.best_of_n,
            "num_eval_prompts": len(values),
            "robust_epsilon": robust_epsilon,
            "robust_ordinal_expected_max_mean": sum(robust_max_values) / len(robust_max_values),
            "robust_ordinal_expected_max_ci_low": robust_max_low,
            "robust_ordinal_expected_max_ci_high": robust_max_high,
            "robust_probability_any_rating_4_mean": sum(robust_any_four_values) / len(robust_any_four_values),
            "robust_probability_any_rating_4_ci_low": robust_tail_low,
            "robust_probability_any_rating_4_ci_high": robust_tail_high,
            "ordinal_expected_max_mean": sum(max_values) / len(max_values),
            "ordinal_expected_max_ci_low": max_low,
            "ordinal_expected_max_ci_high": max_high,
            "probability_any_rating_4_mean": sum(any_four_values) / len(any_four_values),
            "probability_any_rating_4_ci_low": tail_low,
            "probability_any_rating_4_ci_high": tail_high,
            "selected_expected_rating_mean": sum(value[4] for value in values) / len(values),
            "selected_p4_mean": sum(value[5] for value in values) / len(values),
            "responses_jsonl": name,
            "responses_jsonl_sha256": sha256(path),
            "reference_method": args.reference_method,
        }
        for candidate_count in (1, 2, 4, 8, 16, 32):
            if candidate_count > args.best_of_n:
                continue
            nested = [
                exact_group(group[:candidate_count], robust_epsilon)
                for group in candidate_groups
            ]
            nested_robust_max = [value[0] for value in nested]
            nested_robust_top = [value[1] for value in nested]
            nested_max = [value[2] for value in nested]
            nested_top = [value[3] for value in nested]
            robust_max_low, robust_max_high = bootstrap(
                nested_robust_max,
                args.seed + 2500 + candidate_count + len(n_scaling_rows),
            )
            max_low, max_high = bootstrap(
                nested_max, args.seed + 3000 + candidate_count + len(n_scaling_rows)
            )
            top_low, top_high = bootstrap(
                nested_top, args.seed + 4000 + candidate_count + len(n_scaling_rows)
            )
            n_scaling_rows.append(
                {
                    "method": method,
                    "candidate_count": candidate_count,
                    "num_eval_prompts": len(nested),
                    "robust_epsilon": robust_epsilon,
                    "robust_ordinal_expected_max_mean": sum(nested_robust_max) / len(nested_robust_max),
                    "robust_ordinal_expected_max_ci_low": robust_max_low,
                    "robust_ordinal_expected_max_ci_high": robust_max_high,
                    "robust_probability_any_rating_4_mean": sum(nested_robust_top) / len(nested_robust_top),
                    "ordinal_expected_max_mean": sum(nested_max) / len(nested_max),
                    "ordinal_expected_max_ci_low": max_low,
                    "ordinal_expected_max_ci_high": max_high,
                    "probability_any_rating_4_mean": sum(nested_top) / len(nested_top),
                    "probability_any_rating_4_ci_low": top_low,
                    "probability_any_rating_4_ci_high": top_high,
                    "protocol": "nested_prefixes_of_same_cached_32_candidate_groups",
                }
            )

    if args.reference_method not in summaries:
        raise ValueError(f"predeclared reference method is absent: {args.reference_method}")
    reference = summaries[args.reference_method]
    reference_path = response_path(args.eval_dir, reference["responses_jsonl"])
    reference_groups = []
    with reference_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                reference_groups.append(
                    exact_group(json.loads(line)["responses"], robust_epsilon)[0]
                )
    for method, summary in summaries.items():
        method_path = response_path(args.eval_dir, summary["responses_jsonl"])
        values = []
        with method_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    values.append(
                        exact_group(json.loads(line)["responses"], robust_epsilon)[0]
                    )
        margins = [value - ref for value, ref in zip(values, reference_groups)]
        low, high = bootstrap(margins, args.seed + 2000 + len(method))
        randomization_p = paired_sign_flip_pvalue(
            margins, args.seed + 5000 + len(method)
        )
        summary.update(
            {
                "paired_margin_vs_reference": sum(margins) / len(margins),
                "paired_margin_ci_low_vs_reference": low,
                "paired_margin_ci_high_vs_reference": high,
                "paired_win_rate_vs_reference": sum(value > 0.0 for value in margins) / len(margins),
                "paired_sign_flip_p_one_sided_vs_reference": randomization_p,
                "paired_rank_biserial_vs_reference": paired_rank_biserial(margins),
            }
        )

    fields = [
        "method",
        "method_name",
        "seed",
        "best_of_n",
        "num_eval_prompts",
        "robust_epsilon",
        "robust_ordinal_expected_max_mean",
        "robust_ordinal_expected_max_ci_low",
        "robust_ordinal_expected_max_ci_high",
        "robust_probability_any_rating_4_mean",
        "robust_probability_any_rating_4_ci_low",
        "robust_probability_any_rating_4_ci_high",
        "ordinal_expected_max_mean",
        "ordinal_expected_max_ci_low",
        "ordinal_expected_max_ci_high",
        "probability_any_rating_4_mean",
        "probability_any_rating_4_ci_low",
        "probability_any_rating_4_ci_high",
        "selected_expected_rating_mean",
        "selected_p4_mean",
        "paired_margin_vs_reference",
        "paired_margin_ci_low_vs_reference",
        "paired_margin_ci_high_vs_reference",
        "paired_win_rate_vs_reference",
        "paired_sign_flip_p_one_sided_vs_reference",
        "paired_rank_biserial_vs_reference",
        "reference_method",
        "responses_jsonl_sha256",
    ]
    rows = [summaries[method] for method in METHOD_ORDER if method in summaries]
    output_dir = args.eval_dir / "policy_rm_eval"
    write_csv(output_dir / "policy_ordinal_eval_by_method.csv", rows, fields)
    write_csv(
        args.eval_dir / "analysis" / "n_scaling_exact_ordinal.csv",
        n_scaling_rows,
        [
            "method",
            "candidate_count",
            "num_eval_prompts",
            "robust_epsilon",
            "robust_ordinal_expected_max_mean",
            "robust_ordinal_expected_max_ci_low",
            "robust_ordinal_expected_max_ci_high",
            "robust_probability_any_rating_4_mean",
            "ordinal_expected_max_mean",
            "ordinal_expected_max_ci_low",
            "ordinal_expected_max_ci_high",
            "probability_any_rating_4_mean",
            "probability_any_rating_4_ci_low",
            "probability_any_rating_4_ci_high",
            "protocol",
        ],
    )
    payload = {
        "protocol": "robust-exact-ordinal-finite-n-evaluation-v5",
        "metric": "validation-frozen calibrated robust E[max_{j<=N} R_j]",
        "robust_epsilon": robust_epsilon,
        "upper_tail_metric": "P(any j: R_j=4)",
        "reference_method": args.reference_method,
        "rm_config_sha256": sha256(args.rm_config),
        "methods": rows,
        "human_preference_claim": "none; independent model proxies and fresh blind human labels are separate",
    }
    (output_dir / "policy_rm_eval_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
