#!/usr/bin/env python3
"""Analyze locked blinded human labels without using model scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METHODS = ("ev_ppo", "vanilla_ppo")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join((str(seed), *map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(values: list[float], seed: int, draws: int) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    count = len(values)
    estimates = [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(draws)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_sign_flip_pvalue(values: list[float], seed: int, draws: int) -> float:
    """Two-sided paired randomization p-value for a zero mean contrast."""
    if not values:
        raise ValueError("cannot randomize an empty sample")
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(draws):
        randomized = statistics.fmean(
            value if rng.random() < 0.5 else -value for value in values
        )
        exceedances += abs(randomized) >= observed - 1e-15
    return (exceedances + 1.0) / (draws + 1.0)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Return Holm family-wise-error adjusted p-values."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def analyze_pairwise(
    labels_path: Path,
    key_path: Path,
    minimum_annotators: int,
    seed: int,
    draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    keys = {row["pair_id"]: row for row in read_csv(key_path)}
    annotations: dict[str, dict[str, str]] = defaultdict(dict)
    for row_number, row in enumerate(read_csv(labels_path), start=2):
        pair_id = row.get("pair_id", "").strip()
        annotator = row.get("annotator_id", "").strip()
        label = row.get("label", "").strip().upper()
        if pair_id not in keys or not annotator or label not in {"A", "B", "TIE"}:
            raise ValueError(f"invalid pairwise label at row {row_number}")
        if annotator in annotations[pair_id]:
            raise ValueError(f"duplicate pairwise label for {pair_id}/{annotator}")
        annotations[pair_id][annotator] = label

    detail_rows: list[dict[str, Any]] = []
    pair_scores, pair_ties = [], []
    for pair_id in sorted(keys):
        labels = annotations.get(pair_id, {})
        if len(labels) < minimum_annotators:
            raise ValueError(
                f"pair {pair_id} has {len(labels)} annotators; {minimum_annotators} required"
            )
        values, ties = [], 0
        for label in labels.values():
            if label == "TIE":
                values.append(0.5)
                ties += 1
            else:
                method = keys[pair_id]["method_a" if label == "A" else "method_b"]
                if method not in METHODS:
                    raise ValueError(f"unknown method in pairwise key: {method}")
                values.append(float(method == "ev_ppo"))
        score = statistics.fmean(values)
        tie_rate = ties / len(values)
        pair_scores.append(score)
        pair_ties.append(tie_rate)
        detail_rows.append(
            {
                "pair_id": pair_id,
                "prompt_key": keys[pair_id].get("prompt_key", ""),
                "num_annotators": len(values),
                "ev_ppo_preference_score": score,
                "tie_rate": tie_rate,
            }
        )
    low, high = bootstrap_ci(pair_scores, stable_seed(seed, "pairwise"), draws)
    return {
        "num_pairs": len(pair_scores),
        "minimum_annotators_per_pair": min(row["num_annotators"] for row in detail_rows),
        "ev_ppo_preference_score_vs_vanilla_ppo": statistics.fmean(pair_scores),
        "ci_low": low,
        "ci_high": high,
        "tie_rate": statistics.fmean(pair_ties),
        "paired_randomization_p_value": paired_sign_flip_pvalue(
            [value - 0.5 for value in pair_scores],
            stable_seed(seed, "pairwise-randomization"),
            draws,
        ),
    }, detail_rows


def analyze_groups(
    labels_path: Path,
    key_path: Path,
    minimum_annotators: int,
    best_of_n: int,
    seed: int,
    draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key_rows = read_csv(key_path)
    keys = {(row["group_pair_id"], row["group_label"]): row for row in key_rows}
    ratings: dict[tuple[str, str], dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row_number, row in enumerate(read_csv(labels_path), start=2):
        group_key = (row.get("group_pair_id", "").strip(), row.get("group_label", "").strip())
        candidate = row.get("candidate_id", "").strip()
        annotator = row.get("annotator_id", "").strip()
        raw_rating = row.get("rating_0_to_4", "").strip()
        if group_key not in keys or not candidate or not annotator:
            raise ValueError(f"invalid group label identity at row {row_number}")
        try:
            rating = int(raw_rating)
        except ValueError as exc:
            raise ValueError(f"invalid group rating at row {row_number}") from exc
        if rating not in range(5):
            raise ValueError(f"out-of-range group rating at row {row_number}")
        if candidate in ratings[group_key][annotator]:
            raise ValueError(f"duplicate group rating for {group_key}/{annotator}/{candidate}")
        ratings[group_key][annotator][candidate] = rating
        candidates[group_key].add(candidate)

    complete_by_group: dict[tuple[str, str], set[str]] = {}
    for group_key in sorted(keys):
        expected_candidates = candidates.get(group_key, set())
        if len(expected_candidates) != best_of_n:
            raise ValueError(
                f"group {group_key} has {len(expected_candidates)} candidates; {best_of_n} required"
            )
        complete = {
            annotator
            for annotator, values in ratings[group_key].items()
            if set(values) == expected_candidates
        }
        if len(complete) < minimum_annotators:
            raise ValueError(
                f"group {group_key} has {len(complete)} complete annotators; "
                f"{minimum_annotators} required"
            )
        complete_by_group[group_key] = complete

    keys_by_pair: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for group_key in keys:
        keys_by_pair[group_key[0]].append(group_key)

    group_rows: list[dict[str, Any]] = []
    grouped_by_pair: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for pair_id, pair_keys in sorted(keys_by_pair.items()):
        if len(pair_keys) != 2:
            raise ValueError(f"group pair {pair_id} must contain exactly two blinded groups")
        shared_annotators = set.intersection(*(complete_by_group[key] for key in pair_keys))
        if len(shared_annotators) < minimum_annotators:
            raise ValueError(
                f"group pair {pair_id} has {len(shared_annotators)} annotators complete for "
                f"both methods; {minimum_annotators} required"
            )
        for group_key in sorted(pair_keys):
            maxima = [
                max(ratings[group_key][annotator].values())
                for annotator in sorted(shared_annotators)
            ]
            candidate_means = [
                statistics.fmean(ratings[group_key][annotator].values())
                for annotator in sorted(shared_annotators)
            ]
            any_four = [float(value == 4) for value in maxima]
            key = keys[group_key]
            method = key.get("method", "")
            if method not in METHODS:
                raise ValueError(f"unknown method in group key: {method}")
            row = {
                "group_pair_id": group_key[0],
                "group_label": group_key[1],
                "prompt_key": key.get("prompt_key", ""),
                "method": method,
                "num_complete_annotators": len(shared_annotators),
                "realized_max_rating": statistics.fmean(maxima),
                "any_rating_4_rate": statistics.fmean(any_four),
                "candidate_mean_rating": statistics.fmean(candidate_means),
            }
            group_rows.append(row)
            grouped_by_pair[group_key[0]][method] = row

    max_deltas, top_deltas, mean_deltas = [], [], []
    for pair_id, methods in sorted(grouped_by_pair.items()):
        if set(methods) != set(METHODS):
            raise ValueError(f"group pair {pair_id} does not contain both methods")
        max_deltas.append(
            methods["ev_ppo"]["realized_max_rating"]
            - methods["vanilla_ppo"]["realized_max_rating"]
        )
        top_deltas.append(
            methods["ev_ppo"]["any_rating_4_rate"]
            - methods["vanilla_ppo"]["any_rating_4_rate"]
        )
        mean_deltas.append(
            methods["ev_ppo"]["candidate_mean_rating"]
            - methods["vanilla_ppo"]["candidate_mean_rating"]
        )
    max_low, max_high = bootstrap_ci(max_deltas, stable_seed(seed, "group-max"), draws)
    top_low, top_high = bootstrap_ci(top_deltas, stable_seed(seed, "group-top"), draws)
    mean_low, mean_high = bootstrap_ci(
        mean_deltas, stable_seed(seed, "group-candidate-mean"), draws
    )
    return {
        "num_prompt_pairs": len(max_deltas),
        "best_of_n": best_of_n,
        "minimum_complete_annotators_per_group": min(
            row["num_complete_annotators"] for row in group_rows
        ),
        "ev_ppo_minus_vanilla_realized_max": statistics.fmean(max_deltas),
        "realized_max_ci_low": max_low,
        "realized_max_ci_high": max_high,
        "ev_ppo_minus_vanilla_any_rating_4": statistics.fmean(top_deltas),
        "any_rating_4_ci_low": top_low,
        "any_rating_4_ci_high": top_high,
        "ev_ppo_minus_vanilla_candidate_mean": statistics.fmean(mean_deltas),
        "candidate_mean_ci_low": mean_low,
        "candidate_mean_ci_high": mean_high,
        "realized_max_randomization_p_value": paired_sign_flip_pvalue(
            max_deltas, stable_seed(seed, "group-max-randomization"), draws
        ),
        "any_rating_4_randomization_p_value": paired_sign_flip_pvalue(
            top_deltas, stable_seed(seed, "group-top-randomization"), draws
        ),
        "candidate_mean_randomization_p_value": paired_sign_flip_pvalue(
            mean_deltas, stable_seed(seed, "group-mean-randomization"), draws
        ),
    }, group_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise-labels", required=True, type=Path)
    parser.add_argument("--pairwise-key", required=True, type=Path)
    parser.add_argument("--group-labels", required=True, type=Path)
    parser.add_argument("--group-key", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-annotators", type=int, default=3)
    parser.add_argument("--best-of-n", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.minimum_annotators < 3 or args.best_of_n < 1 or args.bootstrap_draws < 1000:
        raise SystemExit("invalid frozen human-evaluation analysis settings")

    pairwise, pair_rows = analyze_pairwise(
        args.pairwise_labels,
        args.pairwise_key,
        args.minimum_annotators,
        args.seed,
        args.bootstrap_draws,
    )
    groups, group_rows = analyze_groups(
        args.group_labels,
        args.group_key,
        args.minimum_annotators,
        args.best_of_n,
        args.seed,
        args.bootstrap_draws,
    )
    raw_pvalues = {
        "pairwise_preference": float(pairwise["paired_randomization_p_value"]),
        "group_realized_max": float(groups["realized_max_randomization_p_value"]),
        "group_any_rating_4": float(groups["any_rating_4_randomization_p_value"]),
        "group_candidate_mean": float(groups["candidate_mean_randomization_p_value"]),
    }
    adjusted_pvalues = holm_adjust(raw_pvalues)
    pairwise["holm_adjusted_p_value"] = adjusted_pvalues["pairwise_preference"]
    groups["realized_max_holm_adjusted_p_value"] = adjusted_pvalues[
        "group_realized_max"
    ]
    groups["any_rating_4_holm_adjusted_p_value"] = adjusted_pvalues[
        "group_any_rating_4"
    ]
    groups["candidate_mean_holm_adjusted_p_value"] = adjusted_pvalues[
        "group_candidate_mean"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pairwise_by_prompt.csv", pair_rows, pair_rows[0].keys())
    write_csv(args.output_dir / "group_by_prompt_method.csv", group_rows, group_rows[0].keys())
    inputs = (
        args.pairwise_labels,
        args.pairwise_key,
        args.group_labels,
        args.group_key,
    )
    report = {
        "protocol": "locked-blinded-human-evaluation-v5",
        "model_scores_used": False,
        "annotator_effect_control": (
            "same complete annotators rate both blinded groups; contrasts are paired "
            "within prompt and annotator before prompt-level aggregation"
        ),
        "multiplicity": {
            "family": list(raw_pvalues),
            "method": "Holm family-wise error correction",
            "raw_p_values": raw_pvalues,
            "adjusted_p_values": adjusted_pvalues,
        },
        "pairwise": pairwise,
        "group_upper_tail": groups,
        "input_sha256": {str(path): sha256(path) for path in inputs},
    }
    output = args.output_dir / "human_evaluation_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "human_evaluation_summary.json.sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
