#!/usr/bin/env python3
"""Fail closed unless every frozen robust-ordinal-v5 artifact is complete."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


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
OBJECTIVES = {
    "ev_ppo": "stablemax_finite_n_ordinal_expected_max_exact_marginal_credit",
    "vanilla_ppo": "ordinal_expected_rating_mean",
    "vanilla_grpo": "ordinal_expected_rating_group_relative",
    "scalar_max_ppo": "scalar_max_at_n_exact_marginal_credit",
    "entropic_ppo": "pessimistic_entropic_ordinal_certainty_equivalent",
    "nominal_ev_ppo": "finite_n_ordinal_expected_max_exact_marginal_credit",
    "ev_ppo_no_mean": "stablemax_finite_n_ordinal_expected_max_exact_marginal_credit",
    "ev_ppo_no_quality": "stablemax_finite_n_ordinal_expected_max_exact_marginal_credit",
    "gaussian_ev_ppo": "gaussian_expected_max_quadrature_ablation",
    "top4_ppo": "finite_n_probability_any_rating_4_exact_marginal_credit",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with require_file(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}={row[key]}")
    return value


def bool_value(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid boolean value: {value}")
    return normalized == "true"


def verify_sidecar(path: Path) -> None:
    sidecar = require_file(path.with_suffix(path.suffix + ".sha256"))
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if expected != sha256(path):
        raise ValueError(f"SHA-256 sidecar mismatch: {path}")


def verify_training(
    root: Path,
    seed: int,
    steps: int,
    hard_kl: float,
    robust_epsilon: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    critical = (
        "loss",
        "adv_mean",
        "adv_std",
        "reference_kl",
        "attempted_post_update_kl",
        "gradient_norm",
        "robust_epsilon",
        "robust_ordinal_expected_max",
        "robust_probability_any_rating_4",
        "ordinal_expected_max",
        "probability_any_rating_4",
        "candidate_expected_rating_mean",
        "candidate_rating_variance_mean",
        "candidate_p4_mean",
        "lagged_global_rms_scale_used",
        "lagged_global_rms_scale",
        "kl_controller_observation",
        "kl_coefficient_used",
        "kl_coefficient_next",
        "mean_dual_state_before",
        "mean_dual_next",
        "quality_dual_state_before",
        "quality_dual_next",
    )
    for method in TRAIN_METHODS:
        method_root = root / f"{method}_seed{seed}"
        require_file(method_root / "adapter_config.json")
        require_file(method_root / "adapter_model.safetensors")
        require_file(method_root / "training_state.json")
        rows = read_csv(method_root / "train_metrics.csv")
        if len(rows) != steps or [int(row["step"]) for row in rows] != list(range(1, steps + 1)):
            raise ValueError(f"{method} does not contain exactly {steps} ordered updates")
        rejected = 0
        accepted_updates = 0
        for row in rows:
            if row["training_objective"] != OBJECTIVES[method]:
                raise ValueError(f"wrong objective for {method}: {row['training_objective']}")
            for key in critical:
                finite(row, key)
            if abs(float(row["robust_epsilon"]) - robust_epsilon) > 1e-8:
                raise ValueError(f"robust epsilon drifted for {method}")
            attempted = finite(row, "attempted_post_update_kl")
            accepted = finite(row, "reference_kl")
            was_rejected = bool_value(row["hard_kl_update_rejected"])
            accepted_transition = bool_value(row["accepted_transition"])
            if accepted_transition == was_rejected:
                raise ValueError(f"inconsistent accepted/rejected state for {method}")
            if attempted > hard_kl and not was_rejected:
                raise ValueError(f"unrejected hard-KL violation for {method}")
            if was_rejected and attempted <= hard_kl:
                raise ValueError(f"spurious hard-KL rejection for {method}")
            if was_rejected and accepted > hard_kl + 1e-6:
                raise ValueError(f"rollback did not restore feasible KL for {method}")
            rejected += int(was_rejected)
            accepted_updates += int(accepted_transition)
            if int(row["accepted_updates_cumulative"]) != accepted_updates:
                raise ValueError(f"accepted-update counter drifted for {method}")
            rollback_performed = bool_value(row["rollback_verification_performed"])
            rollback_parameters = bool_value(row["rollback_parameters_exact"])
            rollback_optimizer = bool_value(row["rollback_optimizer_exact"])
            if was_rejected:
                if not (rollback_performed and rollback_parameters and rollback_optimizer):
                    raise ValueError(f"exact rollback verification failed for {method}")
                if bool_value(row["baseline_update_applied"]):
                    raise ValueError(f"baseline advanced after rejected update for {method}")
                if abs(finite(row, "baseline_loss")) > 1e-12:
                    raise ValueError(f"baseline loss is nonzero after rejected update for {method}")
                for before, after in (
                    ("kl_coefficient_used", "kl_coefficient_next"),
                    ("lagged_global_rms_scale_used", "lagged_global_rms_scale"),
                    ("mean_dual_state_before", "mean_dual_next"),
                    ("quality_dual_state_before", "quality_dual_next"),
                ):
                    if abs(finite(row, before) - finite(row, after)) > 1e-12:
                        raise ValueError(
                            f"auxiliary state {before} advanced after rejected update for {method}"
                        )
            elif rollback_performed or rollback_parameters or rollback_optimizer:
                raise ValueError(f"rollback flags set on accepted update for {method}")
            quality_active = bool_value(row["quality_constraint_active"])
            if quality_active != (method != "ev_ppo_no_quality"):
                raise ValueError(f"wrong Quality RM constraint state for {method}")
            if quality_active:
                for key in (
                    "quality_score_mean",
                    "quality_score_min",
                    "quality_constraint_residual",
                    "quality_violation_rate",
                    "quality_floor",
                    "quality_dual_used",
                    "quality_dual_next",
                ):
                    finite(row, key)
            mean_active = bool_value(row["mean_constraint_active"])
            if mean_active != (method != "ev_ppo_no_mean"):
                raise ValueError(f"wrong mean constraint state for {method}")
            if mean_active:
                for key in (
                    "nominal_mean_score",
                    "nominal_mean_score_min",
                    "mean_constraint_residual",
                    "mean_violation_rate",
                    "mean_floor",
                    "mean_dual_used",
                    "mean_dual_next",
                ):
                    finite(row, key)
        state = read_json(method_root / "training_state.json")
        if int(state.get("accepted_updates", -1)) != accepted_updates:
            raise ValueError(f"training-state accepted update count mismatch for {method}")
        if int(state.get("hard_kl_rollbacks", -1)) != rejected:
            raise ValueError(f"training-state rollback count mismatch for {method}")
        if abs(float(state.get("robust_epsilon", math.nan)) - robust_epsilon) > 1e-8:
            raise ValueError(f"training-state robust epsilon drifted for {method}")
        output[method] = {
            "updates": len(rows),
            "accepted_updates": accepted_updates,
            "hard_kl_rollbacks": rejected,
        }
    return output


def verify_frozen_source_hashes(root: Path, protocol: dict[str, Any]) -> dict[str, str]:
    repo = root.parent.parent
    expected = protocol.get("sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("protocol manifest has no frozen source hashes")
    for relative, digest in expected.items():
        path = require_file(repo / relative)
        if sha256(path) != digest:
            raise ValueError(f"frozen source/model/data hash drifted: {relative}")
    return {str(key): str(value) for key, value in expected.items()}


def verify_floor_caches(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    source_hashes = protocol["sha256"]
    quality_path = require_file(root / "quality_floor_calibration.json")
    mean_path = require_file(root / "mean_floor_calibration.json")
    quality = read_json(quality_path)
    mean = read_json(mean_path)
    quality_spec = quality.get("spec") or {}
    mean_spec = mean.get("spec") or {}
    checks = {
        "quality_protocol": quality_spec.get("protocol")
        == "reference-policy-same-train-distribution-v3",
        "mean_protocol": mean_spec.get("protocol") == "reference-policy-ordinal-mean-floor-v1",
        "quality_floor_finite": math.isfinite(float(quality.get("quality_floor", math.nan))),
        "mean_floor_finite": math.isfinite(float(mean.get("mean_floor", math.nan))),
        "quality_data_hash": quality_spec.get("quality_calibration_jsonl_sha256")
        == source_hashes.get("dataset/quality_reference_policy_v3_floor.jsonl"),
        "mean_data_hash": mean_spec.get("calibration_jsonl_sha256")
        == source_hashes.get("dataset/quality_reference_policy_v3_floor.jsonl"),
        "quality_manifest_hash": quality_spec.get("quality_calibration_manifest_sha256")
        == source_hashes.get("dataset/quality_reference_policy_v3_manifest.json"),
        "mean_manifest_hash": mean_spec.get("calibration_manifest_sha256")
        == source_hashes.get("dataset/quality_reference_policy_v3_manifest.json"),
    }
    if not all(checks.values()):
        raise ValueError(f"frozen floor-cache verification failed: {checks}")
    return {
        "checks": checks,
        "quality_floor": float(quality["quality_floor"]),
        "mean_floor": float(mean["mean_floor"]),
        "quality_cache_sha256": sha256(quality_path),
        "mean_cache_sha256": sha256(mean_path),
    }


def verify_responses(path: Path, expected_prompts: int, best_of_n: int) -> str:
    count = 0
    with require_file(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates = row.get("responses")
            if not isinstance(candidates, list) or len(candidates) != best_of_n:
                raise ValueError(f"wrong candidate count in {path}")
            for candidate in candidates:
                probabilities = [float(candidate[f"p{index}"]) for index in range(5)]
                if not all(math.isfinite(value) and value >= 0.0 for value in probabilities):
                    raise ValueError(f"invalid ordinal probabilities in {path}")
                if abs(sum(probabilities) - 1.0) > 2e-5:
                    raise ValueError(f"ordinal probabilities do not sum to one in {path}")
                quality = float(candidate["quality"])
                if not math.isfinite(quality):
                    raise ValueError(f"non-finite quality score in {path}")
            count += 1
    if count != expected_prompts:
        raise ValueError(f"{path} has {count} prompts; {expected_prompts} required")
    return sha256(path)


def verify_calibration(path: Path, require_order: bool) -> dict[str, Any]:
    calibration = read_json(path)
    if int(calibration.get("num_pairs", 0)) < 128:
        raise ValueError(f"insufficient evaluator calibration pairs: {path}")
    if float(calibration.get("accuracy_ci_low", 0.0)) <= 0.5:
        raise ValueError(f"evaluator calibration accuracy gate failed: {path}")
    if require_order and float(calibration.get("order_consistency_ci_low", 0.0)) <= 0.5:
        raise ValueError(f"evaluator order-consistency gate failed: {path}")
    return {"sha256": sha256(path), **calibration}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--qwen-calibration", required=True, type=Path)
    parser.add_argument("--external-calibration", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--best-of-n", type=int, default=32)
    parser.add_argument("--eval-prompts", type=int, default=256)
    parser.add_argument("--hard-kl", type=float, default=0.04)
    args = parser.parse_args()

    protocol = read_json(args.root / "PROTOCOL_MANIFEST.json")
    gate = read_json(args.root / "ordinal_tail_gate.json")
    verify_sidecar(args.root / "PROTOCOL_MANIFEST.json")
    verify_sidecar(args.root / "ordinal_tail_gate.json")
    if protocol.get("protocol") != "robust-ordinal-ev-ppo-v5-frozen-before-policy-training":
        raise ValueError("wrong protocol manifest")
    if gate.get("protocol") != "ordinal-v5-robust-calibration-gate-before-policy-training":
        raise ValueError("wrong robust calibration gate protocol")
    if gate.get("gates_passed") is not True or not all(gate.get("gate_checks", {}).values()):
        raise ValueError("ordinal tail gate did not pass")
    validation_gate = gate.get("validation") or {}
    if validation_gate.get("calibration_unit") != "prompt_cluster":
        raise ValueError("robust radius was not calibrated at the prompt-cluster level")
    if int(validation_gate.get("num_prompt_clusters", 0)) < 1:
        raise ValueError("robust calibration gate has no independent prompt clusters")
    robust_epsilon = float(gate.get("robust_epsilon", math.nan))
    if not math.isfinite(robust_epsilon) or not 0.0 <= robust_epsilon <= 1.0:
        raise ValueError("robust calibration gate contains an invalid epsilon")

    frozen_hashes = verify_frozen_source_hashes(args.root, protocol)
    floor_caches = verify_floor_caches(args.root, protocol)
    training = verify_training(
        args.root, args.seed, args.steps, args.hard_kl, robust_epsilon
    )
    eval_dir = args.root / "eval"
    table = read_csv(eval_dir / "analysis" / "comparison_table_paper_final.csv")
    if [row["method"] for row in table] != list(EVAL_METHODS):
        raise ValueError("final table does not contain all predeclared rows in order")
    for row in table:
        for key in (
            "robust_epsilon",
            "robust_ordinal_expected_max_mean",
            "robust_ordinal_expected_max_ci_low",
            "robust_ordinal_expected_max_ci_high",
            "robust_probability_any_rating_4_mean",
            "ordinal_expected_max_mean",
            "probability_any_rating_4_mean",
            "candidate_mean_violation_rate",
            "candidate_quality_violation_rate",
        ):
            finite(row, key)
        if abs(float(row["robust_epsilon"]) - robust_epsilon) > 1e-8:
            raise ValueError(f"robust epsilon drifted in final table for {row['method']}")
    response_hashes = {
        method: verify_responses(
            eval_dir / f"{method}_seed{args.seed}_responses.jsonl",
            args.eval_prompts,
            args.best_of_n,
        )
        for method in EVAL_METHODS
    }

    qwen_calibration = verify_calibration(args.qwen_calibration, require_order=True)
    external_calibration = verify_calibration(args.external_calibration, require_order=False)
    proxy_rows = read_csv(eval_dir / "analysis" / "human_preference_proxy_table.csv")
    evaluators = {row["evaluator"] for row in proxy_rows}
    if len(proxy_rows) != 2 * len(EVAL_METHODS) or len(evaluators) != 2:
        raise ValueError("independent proxy table is incomplete")

    hacking_path = eval_dir / "analysis" / "reward_hacking_diagnostic" / "summary.json"
    hacking = read_json(hacking_path)
    if hacking.get("primary_diagnostic_scope") != "complete_cached_N_candidate_groups":
        raise ValueError("reward-hacking diagnostic does not use complete candidate groups")
    if abs(float(hacking.get("robust_epsilon", math.nan)) - robust_epsilon) > 1e-8:
        raise ValueError("reward-hacking diagnostic robust epsilon drifted")
    group_summary = hacking.get("robust_group_objective") or {}
    for key in (
        "paired_delta_mean",
        "paired_delta_ci_low",
        "paired_delta_ci_high",
        "ev_mean",
        "vanilla_mean",
    ):
        value = float(group_summary.get(key, math.nan))
        if not math.isfinite(value):
            raise ValueError(f"missing robust group diagnostic: {key}")
    expected_hashes = {qwen_calibration["sha256"], external_calibration["sha256"]}
    actual_hashes = {
        hacking.get("external_evaluator_calibration_sha256"),
        (hacking.get("qwen_reliability_caveat") or {}).get("calibration_sha256"),
    }
    if actual_hashes != expected_hashes:
        raise ValueError("reward-hacking diagnostics do not hash the actual calibration files")

    figures = sorted(path for path in (eval_dir / "figures").glob("*") if path.is_file())
    if not figures or any(path.stat().st_size == 0 for path in figures):
        raise ValueError("nonempty figures are required")
    pair_packet = read_csv(eval_dir / "human_eval" / "blinded_pairs.csv")
    group_packet = read_csv(eval_dir / "human_eval" / "blinded_group_ratings.csv")
    if len(pair_packet) < 64:
        raise ValueError("blinded pairwise packet is too small")
    if len(group_packet) != min(64, args.eval_prompts) * 2 * args.best_of_n:
        raise ValueError("blinded group packet is incomplete")
    require_file(eval_dir / "human_eval" / "private_key.csv")
    require_file(eval_dir / "human_eval" / "private_group_key.csv")
    pair_key = read_csv(eval_dir / "human_eval" / "private_key.csv")
    group_key = read_csv(eval_dir / "human_eval" / "private_group_key.csv")
    if {
        row[field]
        for row in pair_key
        for field in ("method_a", "method_b")
    } != {"ev_ppo", "vanilla_ppo"}:
        raise ValueError("pairwise human packet is not StableMax-PPO versus Mean-PPO")
    if {row["method"] for row in group_key} != {"ev_ppo", "vanilla_ppo"}:
        raise ValueError("group human packet is not StableMax-PPO versus Mean-PPO")

    report = {
        "protocol": "robust-ordinal-v5-terminal-verification",
        "publication_claim_complete": False,
        "remaining_requirements": [
            "predeclared additional-seed replications",
            "fresh locked blinded human labels analyzed with the frozen script",
        ],
        "training": training,
        "frozen_source_hashes": frozen_hashes,
        "floor_caches": floor_caches,
        "robust_epsilon": robust_epsilon,
        "calibration_prompt_clusters": int(validation_gate["num_prompt_clusters"]),
        "evaluation_rows": list(EVAL_METHODS),
        "response_sha256": response_hashes,
        "qwen_calibration": qwen_calibration,
        "external_calibration": external_calibration,
        "reward_hacking_summary_sha256": sha256(hacking_path),
        "figure_sha256": {path.name: sha256(path) for path in figures},
        "pairwise_packet_rows": len(pair_packet),
        "group_packet_rows": len(group_packet),
        "passed": True,
    }
    output = args.root / "TERMINAL_VERIFICATION.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.root / "TERMINAL_VERIFICATION.json.sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
