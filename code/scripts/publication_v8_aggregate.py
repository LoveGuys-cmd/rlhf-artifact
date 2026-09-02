#!/usr/bin/env python3
"""Aggregate frozen multi-seed publication evidence without outcome tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable


TRAIN_METHODS = (
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
)
EVAL_METHODS = (*TRAIN_METHODS, "best_of_n")
SCIENTIFIC_GATE_NAMES = (
    "primary_vs_scalar_max",
    "robust_noninferiority_vs_mean_ppo",
    "heldout_kl_superiority_vs_mean_ppo",
    "robust_noninferiority_vs_grpo",
    "heldout_kl_superiority_vs_grpo",
    "nominal_mean_floor",
    "quality_floor",
    "qwen_generated_proxy",
    "armorm_generated_proxy",
)
STATIC_LOCKBOX_DIAGNOSTIC_NAMES = (
    "rewardbench_policy_likelihood",
    "skywork_policy_likelihood",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def hierarchical_bootstrap(
    values: dict[int, list[float]],
    seed: int,
    draws: int,
) -> tuple[float, float, float]:
    if len(values) < 3 or any(not rows for rows in values.values()):
        raise ValueError("hierarchical bootstrap requires at least three nonempty seeds")
    point = statistics.fmean(value for rows in values.values() for value in rows)
    keys = sorted(values)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled_seeds = [keys[rng.randrange(len(keys))] for _ in keys]
        sampled_values = []
        for sampled_seed in sampled_seeds:
            rows = values[sampled_seed]
            sampled_values.extend(rows[rng.randrange(len(rows))] for _ in rows)
        estimates.append(statistics.fmean(sampled_values))
    return point, percentile(estimates, 0.025), percentile(estimates, 0.975)


def robust_group_value(record: dict[str, Any], epsilon: float) -> float:
    candidates = record.get("responses") or []
    if not candidates:
        raise ValueError("response record has no candidates")
    value = 0.0
    for threshold in range(1, 5):
        product = 1.0
        for candidate in candidates:
            probabilities = [float(candidate[f"p{rating}"]) for rating in range(5)]
            if any(not math.isfinite(item) or item < 0.0 for item in probabilities):
                raise ValueError("invalid ordinal probabilities")
            if abs(sum(probabilities) - 1.0) > 1e-5:
                raise ValueError("ordinal probabilities do not sum to one")
            product *= min(1.0, sum(probabilities[:threshold]) + epsilon)
        value += 1.0 - product
    return value


def load_response_values(path: Path, epsilon: float, expected_n: int) -> dict[str, float]:
    values = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = record.get("responses") or []
            if len(candidates) != expected_n:
                raise ValueError(f"{path}:{line_number} does not contain N={expected_n}")
            key = str(record.get("prompt_id", line_number - 1))
            if key in values:
                raise ValueError(f"duplicate prompt id in {path}: {key}")
            values[key] = robust_group_value(record, epsilon)
    if not values:
        raise ValueError(f"empty generated-response file: {path}")
    return values


def load_candidate_group_means(
    path: Path, expected_n: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Load per-prompt nominal-mean and independent-quality candidate averages."""
    nominal = {}
    quality = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = record.get("responses") or []
            if len(candidates) != expected_n:
                raise ValueError(f"{path}:{line_number} does not contain N={expected_n}")
            key = str(record.get("prompt_id", line_number - 1))
            if key in nominal:
                raise ValueError(f"duplicate prompt id in {path}: {key}")
            nominal_values = [float(candidate["mu"]) for candidate in candidates]
            quality_values = [float(candidate["quality"]) for candidate in candidates]
            if not all(math.isfinite(value) for value in nominal_values + quality_values):
                raise ValueError(f"non-finite candidate safeguard metric in {path}")
            nominal[key] = statistics.fmean(nominal_values)
            quality[key] = statistics.fmean(quality_values)
    if not nominal:
        raise ValueError(f"empty generated-response file: {path}")
    return nominal, quality


def paired_values(
    left: dict[str, float], right: dict[str, float]
) -> list[float]:
    if set(left) != set(right):
        raise ValueError("paired methods do not contain identical prompt ids")
    return [left[key] - right[key] for key in sorted(left)]


def prompt_cluster_average(values: dict[int, list[float]]) -> list[float]:
    """Average repeated-seed effects within prompts before randomization."""
    lengths = {len(rows) for rows in values.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("prompt-cluster aggregation requires aligned nonempty seeds")
    count = next(iter(lengths))
    return [
        statistics.fmean(values[seed][index] for seed in sorted(values))
        for index in range(count)
    ]


def paired_sign_flip_pvalue(values: list[float], seed: int, draws: int) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("paired randomization requires finite prompt effects")
    observed = statistics.fmean(values)
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(draws):
        permuted = statistics.fmean(
            value if rng.random() < 0.5 else -value for value in values
        )
        exceedances += int(permuted >= observed)
    return (exceedances + 1.0) / (draws + 1.0)


def paired_rank_biserial(values: list[float]) -> float:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    nonzero = positive + negative
    return 0.0 if nonzero == 0 else (positive - negative) / nonzero


def load_static_details(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    result = {row["pair_id"]: float(row["preference_correct"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate static preference pair ids: {path}")
    return result


def proxy_values(path: Path, method: str, qwen: bool) -> list[float]:
    rows = read_csv(path)
    selected = [row for row in rows if row.get("method") == method]
    if qwen:
        selected = [row for row in selected if row.get("mode") == "max_mu"]
        values = [float(row["tie_adjusted_score"]) for row in selected]
    else:
        mapping = {"method": 1.0, "reference": 0.0, "tie": 0.5}
        values = [mapping[row["outcome"]] for row in selected]
    if not values:
        raise ValueError(f"no proxy rows for {method}: {path}")
    return values


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} under {root}, got {matches}")
    return matches[0]


def finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} for {row.get('method')}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    if args.draws < 2000:
        raise ValueError("publication aggregation requires at least 2000 bootstrap draws")
    protocol = read_json(args.protocol)
    if protocol.get("protocol") != "stablemax-ppo-publication-v8-frozen-before-training":
        raise ValueError("wrong publication protocol")
    seeds = tuple(int(value) for value in protocol["seeds"])
    methods = tuple(protocol["methods"])
    if seeds != (314, 2718, 1618) or methods != TRAIN_METHODS:
        raise ValueError("publication seeds or methods drifted")
    best_of_n = int(protocol["best_of_n"])
    if best_of_n != 32:
        raise ValueError("publication best_of_n drifted")
    for section in (
        "rewardbench_calibration",
        "rewardbench_policy_lockbox",
        "skywork_policy_lockbox",
    ):
        item = protocol[section]
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"frozen input hash mismatch: {section}")

    seed_tables: dict[int, dict[str, dict[str, str]]] = {}
    response_values: dict[int, dict[str, dict[str, float]]] = {}
    candidate_nominal: dict[int, dict[str, float]] = {}
    candidate_quality: dict[int, dict[str, float]] = {}
    epsilon: float | None = None
    rewardbench_static: dict[int, list[float]] = {}
    skywork_static: dict[int, list[float]] = {}
    qwen_proxy: dict[int, list[float]] = {}
    armorm_proxy: dict[int, list[float]] = {}
    heldout_kl: dict[int, dict[str, list[float]]] = {}
    artifact_hashes: dict[str, str] = {}

    for seed in seeds:
        seed_root = args.root / f"seed{seed}"
        terminal = read_json(seed_root / "TERMINAL_VERIFICATION.json")
        if terminal.get("passed") is not True:
            raise ValueError(f"seed {seed} terminal verification did not pass")
        artifact_hashes[str(seed_root / "TERMINAL_VERIFICATION.json")] = sha256(
            seed_root / "TERMINAL_VERIFICATION.json"
        )
        table_path = seed_root / "eval" / "analysis" / "comparison_table_paper_final.csv"
        rows = read_csv(table_path)
        table = {row["method"]: row for row in rows}
        if tuple(row["method"] for row in rows) != EVAL_METHODS:
            raise ValueError(f"seed {seed} final table method order drifted")
        seed_tables[seed] = table
        row_epsilon = finite(table["ev_ppo"], "robust_epsilon")
        if epsilon is None:
            epsilon = row_epsilon
        elif abs(epsilon - row_epsilon) > 1e-12:
            raise ValueError("robust epsilon differs across seeds")
        response_values[seed] = {}
        for method in EVAL_METHODS:
            path = seed_root / "eval" / f"{method}_seed{seed}_responses.jsonl"
            response_values[seed][method] = load_response_values(path, row_epsilon, best_of_n)
            artifact_hashes[str(path)] = sha256(path)
            if method == "ev_ppo":
                candidate_nominal[seed], candidate_quality[seed] = (
                    load_candidate_group_means(path, best_of_n)
                )

        rewardbench_ev = load_static_details(
            seed_root
            / "eval"
            / "policy_pref_eval"
            / f"ev_ppo_seed{seed}"
            / "policy_preference_details.csv"
        )
        rewardbench_mean = load_static_details(
            seed_root
            / "eval"
            / "policy_pref_eval"
            / f"vanilla_ppo_seed{seed}"
            / "policy_preference_details.csv"
        )
        rewardbench_static[seed] = paired_values(rewardbench_ev, rewardbench_mean)
        skywork_ev = load_static_details(
            seed_root
            / "eval"
            / "policy_pref_lockboxes"
            / "skywork"
            / f"ev_ppo_seed{seed}"
            / "policy_preference_details.csv"
        )
        skywork_mean = load_static_details(
            seed_root
            / "eval"
            / "policy_pref_lockboxes"
            / "skywork"
            / f"vanilla_ppo_seed{seed}"
            / "policy_preference_details.csv"
        )
        skywork_static[seed] = paired_values(skywork_ev, skywork_mean)

        preference_root = seed_root / "eval" / "independent_preference"
        qwen_path = find_one(preference_root, "*/pairwise_judgments.csv")
        armorm_path = find_one(preference_root, "*/pairwise_scores.csv")
        qwen_proxy[seed] = proxy_values(qwen_path, "ev_ppo", True)
        armorm_proxy[seed] = proxy_values(armorm_path, "ev_ppo", False)
        metadata = read_json(seed_root / "eval" / "analysis" / "paper_metrics_meta.json")
        calibration = metadata.get("independent_preference_proxy") or {}
        if calibration.get("rewardbench_calibration_sha256") != protocol[
            "rewardbench_calibration"
        ]["sha256"]:
            raise ValueError(f"seed {seed} evaluator calibration used the wrong RewardBench split")
        kl_path = seed_root / "eval" / "HELDOUT_KL.json"
        kl_report = read_json(kl_path)
        if kl_report.get("protocol") != (
            "stablemax-ppo-publication-v8-confirmatory-heldout-kl-v1"
        ):
            raise ValueError(f"seed {seed} used the wrong held-out KL protocol")
        heldout_kl[seed] = {
            method: [
                float(value)
                for value in kl_report["per_prompt_heldout_kl"][method][
                    "common_base_trajectory_kl"
                ]
            ]
            for method in EVAL_METHODS
        }
        if any(len(values) != int(protocol["evaluation_prompts"]) for values in heldout_kl[seed].values()):
            raise ValueError(f"seed {seed} held-out KL prompt count drifted")
        artifact_hashes[str(kl_path)] = sha256(kl_path)

    assert epsilon is not None
    method_rows = []
    for method in EVAL_METHODS:
        by_seed = {
            seed: list(response_values[seed][method].values()) for seed in seeds
        }
        mean, low, high = hierarchical_bootstrap(
            by_seed, 1000 + EVAL_METHODS.index(method), args.draws
        )
        method_rows.append(
            {
                "method": method,
                "robust_expected_max": mean,
                "ci_low": low,
                "ci_high": high,
                "num_seeds": len(seeds),
                "prompts_per_seed": len(next(iter(by_seed.values()))),
            }
        )
    primary_deltas = {
        baseline: {
            seed: paired_values(
                response_values[seed]["ev_ppo"], response_values[seed][baseline]
            )
            for seed in seeds
        }
        for baseline in ("vanilla_ppo", "vanilla_grpo", "scalar_max_ppo")
    }
    primary = {
        baseline: {
            **dict(
                zip(
                    ("delta_mean", "ci_low", "ci_high"),
                    hierarchical_bootstrap(values, 2000 + index, args.draws),
                )
            ),
            "paired_sign_flip_p_one_sided": paired_sign_flip_pvalue(
                prompt_cluster_average(values), 2500 + index, args.draws
            ),
            "paired_rank_biserial": paired_rank_biserial(
                prompt_cluster_average(values)
            ),
            "randomization_unit": "prompt_after_averaging_three_seed_effects",
        }
        for index, (baseline, values) in enumerate(primary_deltas.items())
    }
    kl_deltas = {
        baseline: {
            seed: [
                left - right
                for left, right in zip(
                    heldout_kl[seed]["ev_ppo"], heldout_kl[seed][baseline]
                )
            ]
            for seed in seeds
        }
        for baseline in ("vanilla_ppo", "vanilla_grpo")
    }
    kl_primary = {
        baseline: dict(
            zip(
                ("delta_mean", "ci_low", "ci_high"),
                hierarchical_bootstrap(values, 2700 + index, args.draws),
            )
        )
        for index, (baseline, values) in enumerate(kl_deltas.items())
    }
    rewardbench_result = dict(
        zip(
            ("delta_mean", "ci_low", "ci_high"),
            hierarchical_bootstrap(rewardbench_static, 3001, args.draws),
        )
    )
    skywork_result = dict(
        zip(
            ("delta_mean", "ci_low", "ci_high"),
            hierarchical_bootstrap(skywork_static, 3002, args.draws),
        )
    )
    qwen_result = dict(
        zip(
            ("preference_mean", "ci_low", "ci_high"),
            hierarchical_bootstrap(qwen_proxy, 4001, args.draws),
        )
    )
    armorm_result = dict(
        zip(
            ("preference_mean", "ci_low", "ci_high"),
            hierarchical_bootstrap(armorm_proxy, 4002, args.draws),
        )
    )
    candidate_mean, candidate_mean_low, candidate_mean_high = hierarchical_bootstrap(
        {seed: list(candidate_nominal[seed].values()) for seed in seeds},
        5001,
        args.draws,
    )
    candidate_quality_mean, candidate_quality_low, candidate_quality_high = (
        hierarchical_bootstrap(
            {seed: list(candidate_quality[seed].values()) for seed in seeds},
            5002,
            args.draws,
        )
    )
    mean_cache = read_json(args.protocol.parent / "mean_floor_v7.json")
    quality_cache = read_json(args.protocol.parent / "quality_floor_v7.json")
    mean_floor = float(mean_cache.get("mean_floor", math.nan))
    quality_floor = float(quality_cache.get("quality_floor", math.nan))
    if not math.isfinite(mean_floor) or not math.isfinite(quality_floor):
        raise ValueError("frozen floor cache contains a non-finite value")
    gates = {
        "primary_vs_scalar_max": primary["scalar_max_ppo"]["ci_low"] > 0.0,
        "robust_noninferiority_vs_mean_ppo": (
            primary["vanilla_ppo"]["ci_low"] >= -0.05
        ),
        "heldout_kl_superiority_vs_mean_ppo": (
            kl_primary["vanilla_ppo"]["ci_high"] < 0.0
        ),
        "robust_noninferiority_vs_grpo": (
            primary["vanilla_grpo"]["ci_low"] >= -0.10
        ),
        "heldout_kl_superiority_vs_grpo": (
            kl_primary["vanilla_grpo"]["ci_high"] < 0.0
        ),
        "nominal_mean_floor": candidate_mean_low >= mean_floor,
        "quality_floor": candidate_quality_low >= quality_floor,
        "qwen_generated_proxy": (
            qwen_result["preference_mean"] >= 0.50 and qwen_result["ci_low"] >= 0.45
        ),
        "armorm_generated_proxy": (
            armorm_result["preference_mean"] >= 0.50
            and armorm_result["ci_low"] >= 0.45
        ),
    }
    if tuple(gates) != SCIENTIFIC_GATE_NAMES:
        raise ValueError("scientific success gates drifted from the frozen protocol")
    static_lockbox_diagnostics = {
        "selection_criterion": False,
        "best_of_n_endpoint": False,
        "generated_output_human_preference": False,
        "interpretation": (
            "length-normalized policy-likelihood retention on fixed human-labeled "
            "chosen/rejected pairs"
        ),
        "rewardbench_policy_likelihood": rewardbench_result,
        "skywork_policy_likelihood": skywork_result,
    }
    report = {
        "protocol": "stablemax-ppo-publication-v8-multiseed-terminal",
        "passed_artifact_verification": True,
        "scientific_success": all(gates.values()),
        "claim_scope": protocol["claim_scope"],
        "direct_human_labels_on_new_outputs": False,
        "absence_of_reward_hacking_proved": False,
        "model_based_anti_exploitation_battery_passed": all(
            gates[name]
            for name in (
                "nominal_mean_floor",
                "quality_floor",
                "qwen_generated_proxy",
                "armorm_generated_proxy",
            )
        ),
        "static_lockbox_diagnostics_are_selection_criteria": False,
        "seeds": list(seeds),
        "best_of_n": best_of_n,
        "robust_epsilon": epsilon,
        "primary": primary,
        "heldout_kl_primary": kl_primary,
        "static_human_label_lockbox_diagnostics": static_lockbox_diagnostics,
        "qwen_generated_output_proxy": qwen_result,
        "armorm_generated_output_proxy": armorm_result,
        "candidate_mean": candidate_mean,
        "candidate_mean_ci_low": candidate_mean_low,
        "candidate_mean_ci_high": candidate_mean_high,
        "mean_floor": mean_floor,
        "candidate_quality": candidate_quality_mean,
        "candidate_quality_ci_low": candidate_quality_low,
        "candidate_quality_ci_high": candidate_quality_high,
        "quality_floor": quality_floor,
        "gates": gates,
        "method_aggregate": method_rows,
        "artifact_sha256": artifact_hashes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "PUBLICATION_V8_TERMINAL.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_path.with_suffix(report_path.suffix + ".sha256")).write_text(
        f"{sha256(report_path)}  {report_path.name}\n", encoding="utf-8"
    )
    write_csv(
        args.output_dir / "multiseed_method_table.csv",
        method_rows,
        (
            "method",
            "robust_expected_max",
            "ci_low",
            "ci_high",
            "num_seeds",
            "prompts_per_seed",
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
