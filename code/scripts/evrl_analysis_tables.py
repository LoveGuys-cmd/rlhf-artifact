#!/usr/bin/env python3
"""Assemble the compact generated-policy evaluation table."""

from __future__ import annotations

import argparse
import csv
import json
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
FIELDS = (
    "method",
    "method_name",
    "seed",
    "best_of_n",
    "num_eval_prompts",
    "robust_epsilon",
    "robust_ordinal_expected_max_mean",
    "robust_ordinal_expected_max_se",
    "robust_ordinal_expected_max_ci_low",
    "robust_ordinal_expected_max_ci_high",
    "robust_probability_any_rating_4_mean",
    "robust_probability_any_rating_4_se",
    "robust_probability_any_rating_4_ci_low",
    "robust_probability_any_rating_4_ci_high",
    "ordinal_expected_max_mean",
    "ordinal_expected_max_se",
    "ordinal_expected_max_ci_low",
    "ordinal_expected_max_ci_high",
    "probability_any_rating_4_mean",
    "probability_any_rating_4_se",
    "probability_any_rating_4_ci_low",
    "probability_any_rating_4_ci_high",
    "candidate_expected_rating_mean",
    "candidate_rating_variance_mean",
    "candidate_p4_mean",
    "mean_floor",
    "candidate_mean_violation_rate",
    "selected_expected_rating_mean",
    "selected_mean_violation_rate",
    "selected_p4_mean",
    "candidate_quality_mean",
    "candidate_quality_violation_rate",
    "selected_quality_mean",
    "selected_quality_violation_rate",
    "paired_margin_vs_reference",
    "paired_margin_ci_low_vs_reference",
    "paired_margin_ci_high_vs_reference",
    "paired_win_rate_vs_reference",
    "responses_jsonl",
    "reward_distribution_status",
)


def split_methods(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def nonempty(value: Any) -> bool:
    return value not in (None, "", "nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_policy_metrics(eval_dir: Path) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    candidates = (
        eval_dir / "policy_rm_eval" / "policy_ordinal_eval_by_method.csv",
        eval_dir / "policy_rm_eval" / "policy_bon_eval_by_method.csv",
        eval_dir / "policy_bon_eval_by_method.csv",
        eval_dir / "policy_rm_eval" / "policy_rm_eval_by_method.csv",
    )
    for path in candidates:
        for row in read_csv(path):
            method = row.get("method", "")
            if method:
                output.setdefault(method, {}).update(
                    {key: value for key, value in row.items() if nonempty(value)}
                )
    return output


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_performance", action="store_true")
    args = parser.parse_args()

    policy_metrics = load_policy_metrics(args.eval_dir)
    rows: list[dict[str, Any]] = []
    for method in split_methods(args.methods):
        summary_path = args.eval_dir / f"{method}_seed{args.seed}_summary.json"
        if not summary_path.exists():
            continue
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        row.update(policy_metrics.get(method, {}))
        row.setdefault("method", method)
        row.setdefault("seed", args.seed)
        if args.require_performance and not nonempty(
            row.get("robust_ordinal_expected_max_mean")
        ):
            raise ValueError(
                f"missing robust exact ordinal expected-max metric for {method}"
            )
        response_path = row.get("responses_jsonl")
        if not response_path:
            default_response = args.eval_dir / f"{method}_seed{args.seed}_responses.jsonl"
            if default_response.exists():
                row["responses_jsonl"] = str(default_response)
        rows.append(row)

    if not rows:
        raise ValueError(f"no method summaries found under {args.eval_dir}")
    rows.sort(
        key=lambda row: METHOD_ORDER.index(row["method"])
        if row.get("method") in METHOD_ORDER
        else len(METHOD_ORDER)
    )
    for filename in ("comparison_table_paper.csv", "comparison_table.csv"):
        write_table(args.output_dir / filename, rows)
    print(
        f"[analysis-table] wrote {len(rows)} methods to "
        f"{args.output_dir / 'comparison_table_paper.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
