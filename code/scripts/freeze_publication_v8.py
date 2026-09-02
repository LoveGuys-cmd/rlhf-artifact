#!/usr/bin/env python3
"""Freeze pilot-separated publication-v8 confirmatory inputs and claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


SEEDS = (314, 2718, 1618)
METHODS = (
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
REUSED_PROTOCOL_FILES = (
    "rewardbench_judge_calibration_v7.jsonl",
    "rewardbench_policy_lockbox_v7.jsonl",
    "rewardbench_reserve_v7.jsonl",
    "reference_policy_floor_v7.jsonl",
    "reference_policy_shift_diagnostic_v7.jsonl",
    "reference_policy_manifest_v7.json",
    "reference_shift_diagnostic_v7.json",
    "reference_shift_diagnostic_v7.json.sha256",
    "mean_floor_v7.json",
    "quality_floor_v7.json",
    "floor_calibration_v7.json",
    "floor_calibration_v7.json.sha256",
    "ordinal_tail_gate_v7.json",
    "ordinal_tail_gate_v7.json.sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_manifest(repo: Path, manifest: Path) -> dict[str, str]:
    verified = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1)
        relative = relative.strip().lstrip("*")
        path = repo / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen v7 input mismatch: {relative}")
        verified[relative] = expected
    return verified


def assert_unused_seeds(experiment_root: Path) -> None:
    collisions: dict[int, list[str]] = {seed: [] for seed in SEEDS}
    if experiment_root.is_dir():
        for path in experiment_root.rglob("*"):
            lowered = path.name.casefold()
            for seed in SEEDS:
                if f"seed{seed}" in lowered:
                    collisions[seed].append(str(path))
    collisions = {seed: paths for seed, paths in collisions.items() if paths}
    if collisions:
        raise FileExistsError(f"confirmatory seed names already exist: {collisions}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--v7-protocol-dir", required=True, type=Path)
    parser.add_argument("--pilot-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = args.v7_protocol_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v8 protocol: {output}")
    verify_manifest(repo, source / "FROZEN_INPUTS_V7.sha256")
    v7 = read_json(source / "PUBLICATION_PROTOCOL_V7.json")
    if v7.get("protocol") != "stablemax-ppo-publication-v7-frozen-before-training":
        raise ValueError("wrong source protocol")
    pilot = read_json(args.pilot_audit)
    if pilot.get("scientific_status") != "development_audit_not_confirmatory_evidence":
        raise ValueError("pilot audit was not development-only")
    if int(pilot.get("seed", -1)) != 42 or int(pilot.get("num_prompts", -1)) != 128:
        raise ValueError("unexpected pilot design provenance")
    assert_unused_seeds(repo / "exp")

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        copied = {}
        for name in REUSED_PROTOCOL_FILES:
            source_path = source / name
            target = temporary / name
            shutil.copy2(source_path, target)
            copied[name] = {"sha256": sha256(target), "source": str(source_path)}
        shutil.copy2(
            source / "PUBLICATION_PROTOCOL_V7.json",
            temporary / "DEVELOPMENT_PROTOCOL_V7.json",
        )
        transition = {
            "protocol": "stablemax-ppo-publication-v8-pilot-transition-v1",
            "development_seed": 42,
            "development_evaluation_prompts": 128,
            "confirmatory_seeds": list(SEEDS),
            "development_seed_eligible_for_confirmatory_claims": False,
            "confirmatory_seeds_verified_unused_before_freeze": True,
            "pilot_audit": str(args.pilot_audit.resolve()),
            "pilot_audit_sha256": sha256(args.pilot_audit),
            "pilot_informed_design": (
                "The confirmatory estimand is a predeclared constrained Pareto claim. "
                "Seed42 is reported separately and never pooled with v8 inference."
            ),
        }
        write_json(temporary / "PILOT_TO_CONFIRMATORY_TRANSITION.json", transition)

        protocol = {
            "protocol": "stablemax-ppo-publication-v8-frozen-before-training",
            "algorithm": "StableMax-PPO",
            "claim_scope": (
                "confirmatory model-based finite-N upper-tail and constrained-Pareto "
                "evidence with nominal-mean, independent-quality, full-vocabulary KL, "
                "calibrated generated-output proxies, and report-only existing-label diagnostics"
            ),
            "development_pilot": transition,
            "primary_policy_model": v7["primary_policy_model"],
            "moment_rm": v7["moment_rm"],
            "quality_rm": v7["quality_rm"],
            "seeds": list(SEEDS),
            "methods": list(METHODS),
            "best_of_n": 32,
            "steps": 300,
            "batch_prompts": 2,
            "learning_rate": 2e-5,
            "target_kl": 0.02,
            "hard_kl": 0.04,
            "evaluation_prompts": 512,
            "evaluation_candidate_counts": [1, 2, 4, 8, 16, 32],
            "fresh_human_labels_on_generated_outputs": False,
            "direct_human_preference_claim_permitted": False,
            "absence_of_reward_hacking_claim_permitted": False,
            "superiority_is_not_guaranteed": True,
            "success_criteria": {
                "primary_vs_scalar_max": (
                    "95% hierarchical-bootstrap lower bound for CR minus Scalar-Max "
                    "robust E[max_32 R] is greater than 0"
                ),
                "robust_noninferiority_vs_mean_ppo": (
                    "95% lower bound for CR minus Mean-PPO robust E[max_32 R] is at least -0.05"
                ),
                "heldout_kl_superiority_vs_mean_ppo": (
                    "95% upper bound for paired common-trajectory full-vocabulary KL "
                    "difference CR minus Mean-PPO is below 0"
                ),
                "robust_noninferiority_vs_grpo": (
                    "95% lower bound for CR minus GRPO robust E[max_32 R] is at least -0.10; "
                    "the margin is 2.5% of the fixed 0-to-4 rating range"
                ),
                "heldout_kl_superiority_vs_grpo": (
                    "95% upper bound for paired common-trajectory full-vocabulary KL "
                    "difference CR minus GRPO is below 0"
                ),
                "nominal_mean_floor": v7["success_criteria"]["nominal_mean_floor"],
                "quality_floor": v7["success_criteria"]["quality_floor"],
                "qwen_generated_proxy": v7["success_criteria"]["qwen_generated_proxy"],
                "armorm_generated_proxy": v7["success_criteria"]["armorm_generated_proxy"],
                "claim_rule": (
                    "all gates must pass for the full constrained-Pareto success claim; "
                    "all outcomes and all ten comparators are reported regardless"
                ),
            },
            "statistics": {
                "bootstrap": "hierarchical seed-then-prompt percentile bootstrap",
                "bootstrap_draws": 10000,
                "paired_randomization": "prompt sign flip after averaging seed effects",
                "confidence_level": 0.95,
                "no_optional_stopping": True,
                "no_seed_or_comparator_removal": True,
            },
            "reused_frozen_v7_protocol_artifacts": copied,
            "rewardbench_calibration": {
                **v7["rewardbench_calibration"],
                "path": str((output / "rewardbench_judge_calibration_v7.jsonl")),
            },
            "rewardbench_policy_lockbox": {
                **v7["rewardbench_policy_lockbox"],
                "path": str((output / "rewardbench_policy_lockbox_v7.jsonl")),
            },
            "rewardbench_reserve": {
                **v7["rewardbench_reserve"],
                "path": str((output / "rewardbench_reserve_v7.jsonl")),
            },
            "skywork_policy_lockbox": v7["skywork_policy_lockbox"],
            "runtime_contract": {
                "qos": "normal",
                "no_arrays": True,
                "max_concurrent_jobs": 1,
                "max_gpus": 2,
                "training_schedule": "one seed per normal two-GPU job; five methods per GPU",
                "technical_failure_policy": (
                    "remove only incomplete technical outputs after diagnosis and regression; "
                    "never rerun or delete unfavorable completed scientific outcomes"
                ),
            },
        }
        write_json(temporary / "PUBLICATION_PROTOCOL_V8.json", protocol)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
