#!/usr/bin/env python3
"""Fail closed unless one publication-v8 confirmatory seed is complete."""

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


def require(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(require(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with require(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {row.get('method')}")
    return value


def boolean(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"invalid boolean: {value}")
    return lowered == "true"


def verify_transition_row(
    row: dict[str, str], method: str, hard_kl: float
) -> tuple[bool, bool]:
    attempted = finite(row, "attempted_post_update_kl")
    finite(row, "reference_kl")
    was_rejected = boolean(row["hard_kl_update_rejected"])
    was_accepted = boolean(row["accepted_transition"])
    if was_rejected == was_accepted:
        raise ValueError(f"inconsistent transition status for {method}")
    if attempted > hard_kl and not was_rejected:
        raise ValueError(f"unrejected hard-KL violation for {method}")
    if was_rejected:
        if attempted <= hard_kl:
            raise ValueError(f"invalid hard-KL rollback for {method}")
        for field in (
            "rollback_verification_performed",
            "rollback_parameters_exact",
            "rollback_optimizer_exact",
        ):
            if not boolean(row[field]):
                raise ValueError(f"failed exact rollback field {field} for {method}")
    return was_rejected, was_accepted


def verify_frozen_inputs(repo: Path, manifest: Path) -> dict[str, str]:
    hashes = {}
    with require(manifest).open(encoding="utf-8") as handle:
        for line in handle:
            expected, relative = line.strip().split(None, 1)
            path = repo / relative.strip().lstrip("*")
            require(path)
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"frozen input hash mismatch: {relative}")
            hashes[relative] = actual
    return hashes


def verify_training(root: Path, seed: int, steps: int, hard_kl: float) -> dict[str, Any]:
    report = {}
    critical = (
        "loss",
        "baseline_loss",
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
    )
    for method in TRAIN_METHODS:
        method_root = root / f"{method}_seed{seed}"
        require(method_root / "adapter_model.safetensors")
        require(method_root / "adapter_config.json")
        state = read_json(method_root / "training_state.json")
        rows = read_csv(method_root / "train_metrics.csv")
        if len(rows) != steps or [int(row["step"]) for row in rows] != list(
            range(1, steps + 1)
        ):
            raise ValueError(f"{method} does not have exactly {steps} ordered steps")
        rejected = 0
        accepted = 0
        for row in rows:
            if row["training_objective"] != OBJECTIVES[method]:
                raise ValueError(f"objective mismatch for {method}")
            if boolean(row["mean_constraint_active"]) != (
                method != "ev_ppo_no_mean"
            ):
                raise ValueError(f"mean constraint activation mismatch for {method}")
            if boolean(row["quality_constraint_active"]) != (
                method != "ev_ppo_no_quality"
            ):
                raise ValueError(f"quality constraint activation mismatch for {method}")
            for key in critical:
                finite(row, key)
            was_rejected, was_accepted = verify_transition_row(row, method, hard_kl)
            rejected += int(was_rejected)
            accepted += int(was_accepted)
        if accepted + rejected != steps:
            raise ValueError(f"transition accounting mismatch for {method}")
        if int(state.get("accepted_updates", -1)) != accepted:
            raise ValueError(f"accepted update state mismatch for {method}")
        if int(state.get("hard_kl_rollbacks", -1)) != rejected:
            raise ValueError(f"rollback state mismatch for {method}")
        quality_active = bool(state.get("quality_constraint_active"))
        if quality_active != (method != "ev_ppo_no_quality"):
            raise ValueError(f"quality constraint activation mismatch for {method}")
        report[method] = {"accepted": accepted, "rejected": rejected}
    return report


def verify_responses(path: Path, prompts: int, best_of_n: int) -> str:
    count = 0
    with require(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = record.get("responses") or []
            if len(candidates) != best_of_n:
                raise ValueError(f"{path}:{line_number} has wrong candidate count")
            for candidate in candidates:
                probabilities = [float(candidate[f"p{rating}"]) for rating in range(5)]
                if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
                    raise ValueError(f"invalid probabilities in {path}")
                if abs(sum(probabilities) - 1.0) > 1e-5:
                    raise ValueError(f"probabilities do not sum to one in {path}")
                for field in ("mu", "sigma", "quality"):
                    if not math.isfinite(float(candidate[field])):
                        raise ValueError(f"non-finite candidate {field} in {path}")
            count += 1
    if count != prompts:
        raise ValueError(f"{path} has {count} prompts, expected {prompts}")
    return sha256(path)


def verify_calibration(path: Path, require_order: bool) -> dict[str, Any]:
    value = read_json(path)
    if int(value.get("num_pairs", 0)) < 512:
        raise ValueError(f"insufficient evaluator calibration pairs: {path}")
    if float(value.get("accuracy_ci_low", -1.0)) <= 0.5:
        raise ValueError(f"evaluator calibration accuracy gate failed: {path}")
    if require_order and float(value.get("order_consistency_ci_low", -1.0)) <= 0.5:
        raise ValueError(f"evaluator order gate failed: {path}")
    return {"path": str(path), "sha256": sha256(path), **value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol_dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--best_of_n", type=int, default=32)
    parser.add_argument("--eval_prompts", type=int, default=512)
    parser.add_argument("--hard_kl", type=float, default=0.04)
    args = parser.parse_args()
    protocol = read_json(args.protocol_dir / "PUBLICATION_PROTOCOL_V8.json")
    if protocol.get("protocol") != "stablemax-ppo-publication-v8-frozen-before-training":
        raise ValueError("wrong publication protocol")
    if args.seed not in protocol["seeds"]:
        raise ValueError("seed is not predeclared")
    if int(protocol["steps"]) != args.steps or int(protocol["best_of_n"]) != args.best_of_n:
        raise ValueError("training budget drifted")
    frozen_hashes = verify_frozen_inputs(
        args.repo, args.protocol_dir / "FROZEN_INPUTS_V8.sha256"
    )
    training = verify_training(args.root, args.seed, args.steps, args.hard_kl)
    eval_dir = args.root / "eval"
    table = read_csv(eval_dir / "analysis" / "comparison_table_paper_final.csv")
    if tuple(row["method"] for row in table) != EVAL_METHODS:
        raise ValueError("final evaluation table is incomplete or out of order")
    for row in table:
        for key in (
            "robust_epsilon",
            "robust_ordinal_expected_max_mean",
            "robust_ordinal_expected_max_ci_low",
            "robust_ordinal_expected_max_ci_high",
            "ordinal_expected_max_mean",
            "probability_any_rating_4_mean",
            "candidate_mean_violation_rate",
            "candidate_quality_violation_rate",
        ):
            finite(row, key)
    response_hashes = {
        method: verify_responses(
            eval_dir / f"{method}_seed{args.seed}_responses.jsonl",
            args.eval_prompts,
            args.best_of_n,
        )
        for method in EVAL_METHODS
    }
    paired_rows = read_csv(
        eval_dir / "policy_rm_eval" / "policy_ordinal_eval_by_method.csv"
    )
    if tuple(row["method"] for row in paired_rows) != EVAL_METHODS:
        raise ValueError("paired ordinal statistics are incomplete or out of order")
    for row in paired_rows:
        p_value = finite(row, "paired_sign_flip_p_one_sided_vs_reference")
        effect = finite(row, "paired_rank_biserial_vs_reference")
        if not 0.0 <= p_value <= 1.0 or not -1.0 <= effect <= 1.0:
            raise ValueError(f"invalid paired inferential statistic for {row['method']}")
    n_scaling_path = eval_dir / "analysis" / "n_scaling_exact_ordinal.csv"
    n_scaling = read_csv(n_scaling_path)
    expected_scaling = {
        (method, str(candidate_count))
        for method in EVAL_METHODS
        for candidate_count in (1, 2, 4, 8, 16, 32)
    }
    observed_scaling = {
        (row["method"], row["candidate_count"]) for row in n_scaling
    }
    if len(n_scaling) != len(expected_scaling) or observed_scaling != expected_scaling:
        raise ValueError("nested-prefix N-sensitivity table is incomplete")
    for row in n_scaling:
        for key in (
            "robust_epsilon",
            "robust_ordinal_expected_max_mean",
            "ordinal_expected_max_mean",
            "probability_any_rating_4_mean",
        ):
            finite(row, key)
    metadata = read_json(eval_dir / "analysis" / "paper_metrics_meta.json")
    proxy_metadata = metadata.get("independent_preference_proxy") or {}
    expected_calibration_hash = protocol["rewardbench_calibration"]["sha256"]
    if proxy_metadata.get("rewardbench_calibration_sha256") != expected_calibration_hash:
        raise ValueError("independent evaluators used a non-frozen calibration split")
    preference_root = eval_dir / "independent_preference"
    qwen_paths = list(preference_root.glob("*/pairwise_judgments.csv"))
    external_paths = list(preference_root.glob("*/pairwise_scores.csv"))
    if len(qwen_paths) != 1 or len(external_paths) != 1:
        raise ValueError("independent evaluator details are incomplete")
    qwen_calibration = verify_calibration(qwen_paths[0].parent / "calibration.json", True)
    external_calibration = verify_calibration(
        external_paths[0].parent / "calibration.json", False
    )
    proxy_rows = read_csv(eval_dir / "analysis" / "human_preference_proxy_table.csv")
    if len(proxy_rows) != 2 * len(EVAL_METHODS):
        raise ValueError("proxy table does not contain both evaluators for every method")

    rewardbench_hash = protocol["rewardbench_policy_lockbox"]["sha256"]
    skywork_hash = protocol["skywork_policy_lockbox"]["sha256"]
    for method in EVAL_METHODS:
        primary = read_csv(
            eval_dir
            / "policy_pref_eval"
            / f"{method}_seed{args.seed}"
            / "policy_preference_details.csv"
        )
        secondary = read_csv(
            eval_dir
            / "policy_pref_lockboxes"
            / "skywork"
            / f"{method}_seed{args.seed}"
            / "policy_preference_details.csv"
        )
        if len(primary) != 1024 or len(secondary) != 1024:
            raise ValueError(f"static preference lockbox rows are incomplete for {method}")
    lockbox_rows = read_csv(eval_dir / "policy_preference_lockboxes.csv")
    if {row["lockbox_sha256"] for row in lockbox_rows} != {skywork_hash}:
        raise ValueError("Skywork static lockbox hash mismatch")
    performance_rows = read_csv(eval_dir / "policy_performance.csv")
    if len(performance_rows) != len(EVAL_METHODS):
        raise ValueError("RewardBench static policy table is incomplete")
    if sha256(Path(protocol["rewardbench_policy_lockbox"]["path"])) != rewardbench_hash:
        raise ValueError("RewardBench policy lockbox hash mismatch")

    hacking_path = eval_dir / "analysis" / "reward_hacking_diagnostic" / "summary.json"
    hacking = read_json(hacking_path)
    if hacking.get("primary_diagnostic_scope") != "complete_cached_N_candidate_groups":
        raise ValueError("reward-hacking diagnostic used the wrong analysis unit")
    figures = [path for path in (eval_dir / "figures").glob("*") if path.is_file()]
    if not figures or any(path.stat().st_size == 0 for path in figures):
        raise ValueError("nonempty publication figures are required")
    pair_packet = read_csv(eval_dir / "human_eval" / "blinded_pairs.csv")
    group_packet = read_csv(eval_dir / "human_eval" / "blinded_group_ratings.csv")
    if len(pair_packet) < 64 or len(group_packet) != 64 * 2 * args.best_of_n:
        raise ValueError("optional future human-evaluation packet is incomplete")
    heldout_kl_path = eval_dir / "HELDOUT_KL.json"
    heldout_kl = read_json(heldout_kl_path)
    if heldout_kl.get("protocol") != (
        "stablemax-ppo-publication-v8-confirmatory-heldout-kl-v1"
    ):
        raise ValueError("wrong held-out KL protocol")
    if int(heldout_kl.get("seed", -1)) != args.seed:
        raise ValueError("held-out KL seed mismatch")
    if set(heldout_kl.get("heldout_kl", {})) != set(EVAL_METHODS):
        raise ValueError("held-out KL methods are incomplete")
    if len(heldout_kl.get("prompt_keys", [])) != args.eval_prompts:
        raise ValueError("held-out KL prompt count mismatch")

    report = {
        "protocol": "stablemax-ppo-publication-v8-seed-terminal",
        "passed": True,
        "seed": args.seed,
        "steps": args.steps,
        "best_of_n": args.best_of_n,
        "training": training,
        "evaluation_rows": list(EVAL_METHODS),
        "frozen_input_sha256": frozen_hashes,
        "response_sha256": response_hashes,
        "paired_statistics_sha256": sha256(
            eval_dir / "policy_rm_eval" / "policy_ordinal_eval_by_method.csv"
        ),
        "n_scaling_sha256": sha256(n_scaling_path),
        "qwen_calibration": qwen_calibration,
        "external_calibration": external_calibration,
        "reward_hacking_summary_sha256": sha256(hacking_path),
        "heldout_kl_sha256": sha256(heldout_kl_path),
        "figure_sha256": {path.name: sha256(path) for path in figures},
        "existing_human_label_lockboxes": {
            "rewardbench_sha256": rewardbench_hash,
            "skywork_sha256": skywork_hash,
        },
        "fresh_human_labels_on_generated_outputs": False,
        "eligible_for_confirmatory_claims": True,
        "rollback_reference_kl_interpretation": (
            "Exact parameter and optimizer snapshots are authoritative after rejection; "
            "reference_kl on the current minibatch is diagnostic and need not fall below "
            "the hard threshold after restoration."
        ),
        "claim_scope": protocol["claim_scope"],
    }
    output = args.root / "TERMINAL_VERIFICATION.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output.with_suffix(output.suffix + ".sha256")).write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
