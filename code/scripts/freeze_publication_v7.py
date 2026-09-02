#!/usr/bin/env python3
"""Freeze contamination-audited data and success criteria for publication v7."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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
SEEDS = (42, 314, 2718)


def canonical(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def text_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    prompt = str(row.get("prompt") or "")
    chosen = str(row.get("chosen") or "")
    rejected = str(row.get("rejected") or "")
    if not canonical(prompt) or not canonical(chosen) or not canonical(rejected):
        raise ValueError("preference row lacks prompt/chosen/rejected text")
    if canonical(chosen) == canonical(rejected):
        raise ValueError("preference row has identical chosen and rejected responses")
    return text_hash(prompt), text_hash(chosen), text_hash(rejected)


def forbidden_hashes(paths: list[Path]) -> tuple[set[str], set[str]]:
    prompts: set[str] = set()
    responses: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            prompt = row.get("prompt")
            if prompt:
                prompts.add(text_hash(prompt))
            for field in ("chosen", "rejected", "response"):
                if row.get(field):
                    responses.add(text_hash(row[field]))
    return prompts, responses


def deterministic_order(row: dict[str, Any], salt: str) -> str:
    prompt_hash, chosen_hash, rejected_hash = row_identity(row)
    subset = str(row.get("subset") or "unknown")
    return hashlib.sha256(
        f"{salt}|{subset}|{prompt_hash}|{chosen_hash}|{rejected_hash}".encode("utf-8")
    ).hexdigest()


def subset_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("subset") or "unknown") for row in rows).items()))


def assert_disjoint(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    left_prompts = {row_identity(row)[0] for row in left}
    right_prompts = {row_identity(row)[0] for row in right}
    left_responses = {value for row in left for value in row_identity(row)[1:]}
    right_responses = {value for row in right for value in row_identity(row)[1:]}
    if left_prompts & right_prompts:
        raise AssertionError("frozen RewardBench partitions share prompts")
    if left_responses & right_responses:
        raise AssertionError("frozen RewardBench partitions share responses")


def remove_global_prompt_response_collisions(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Keep a deterministic maximal subsequence with unique prompts and responses."""
    retained = []
    seen_prompts: set[str] = set()
    seen_responses: set[str] = set()
    prompt_collisions = 0
    response_collisions = 0
    for row in rows:
        prompt_hash, chosen_hash, rejected_hash = row_identity(row)
        if prompt_hash in seen_prompts:
            prompt_collisions += 1
            continue
        if chosen_hash in seen_responses or rejected_hash in seen_responses:
            response_collisions += 1
            continue
        retained.append(row)
        seen_prompts.add(prompt_hash)
        seen_responses.update((chosen_hash, rejected_hash))
    return retained, prompt_collisions, response_collisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewardbench", required=True, type=Path)
    parser.add_argument("--skywork_lockbox", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--forbidden_jsonl", action="append", default=[], type=Path)
    parser.add_argument("--calibration_size", type=int, default=1024)
    parser.add_argument("--lockbox_size", type=int, default=1024)
    parser.add_argument("--salt", default="stablemax-ppo-publication-v7-frozen-20260805")
    args = parser.parse_args()
    if args.calibration_size < 512 or args.lockbox_size < 512:
        raise ValueError("publication partitions require at least 512 pairs each")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "rewardbench_judge_calibration_v7.jsonl"
    lockbox_path = args.output_dir / "rewardbench_policy_lockbox_v7.jsonl"
    reserve_path = args.output_dir / "rewardbench_reserve_v7.jsonl"
    manifest_path = args.output_dir / "PUBLICATION_PROTOCOL_V7.json"
    outputs = (calibration_path, lockbox_path, reserve_path, manifest_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite frozen v7 outputs: {existing}")

    forbidden_prompts, forbidden_responses = forbidden_hashes(args.forbidden_jsonl)
    retained: dict[tuple[str, str, str], dict[str, Any]] = {}
    excluded_prompt = 0
    excluded_response = 0
    invalid = 0
    for raw in read_jsonl(args.rewardbench):
        try:
            identity = row_identity(raw)
        except ValueError:
            invalid += 1
            continue
        if identity[0] in forbidden_prompts:
            excluded_prompt += 1
            continue
        if identity[1] in forbidden_responses or identity[2] in forbidden_responses:
            excluded_response += 1
            continue
        retained[identity] = {
            "prompt": str(raw["prompt"]),
            "chosen": str(raw["chosen"]),
            "rejected": str(raw["rejected"]),
            "subset": str(raw.get("subset") or "unknown"),
            "rewardbench_id": raw.get("rewardbench_id"),
        }
    identity_deduplicated = sorted(
        retained.values(), key=lambda row: deterministic_order(row, args.salt)
    )
    ordered, duplicate_prompt_rows, duplicate_response_rows = (
        remove_global_prompt_response_collisions(identity_deduplicated)
    )
    needed = args.calibration_size + args.lockbox_size
    if len(ordered) < needed:
        raise ValueError(f"only {len(ordered)} uncontaminated RewardBench rows; need {needed}")
    calibration = ordered[: args.calibration_size]
    lockbox = ordered[args.calibration_size : needed]
    reserve = ordered[needed:]
    assert_disjoint(calibration, lockbox)
    assert_disjoint(calibration, reserve)
    assert_disjoint(lockbox, reserve)
    skywork_rows = read_jsonl(args.skywork_lockbox)
    skywork_prompt_hashes = {row_identity(row)[0] for row in skywork_rows}
    rewardbench_prompt_hashes = {
        row_identity(row)[0] for row in calibration + lockbox + reserve
    }
    if skywork_prompt_hashes & rewardbench_prompt_hashes:
        raise AssertionError("Skywork and RewardBench lockboxes share prompts")

    write_jsonl(calibration_path, calibration)
    write_jsonl(lockbox_path, lockbox)
    write_jsonl(reserve_path, reserve)
    manifest = {
        "protocol": "stablemax-ppo-publication-v7-frozen-before-training",
        "claim_scope": (
            "model-based upper-tail improvement with nominal-mean and independent-quality "
            "safeguards plus calibrated generated-output model proxies; existing human-labeled "
            "lockboxes are preference-retention diagnostics, not success criteria or direct "
            "preference labels on newly generated outputs"
        ),
        "algorithm": "StableMax-PPO",
        "primary_policy_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "moment_rm": "Qwen/Qwen2.5-14B-Instruct ordinal Moment RM",
        "quality_rm": "independent Qwen2.5-14B scalar Quality RM",
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "best_of_n": 32,
        "steps": 300,
        "batch_prompts": 2,
        "learning_rate": 2e-5,
        "target_kl": 0.02,
        "hard_kl": 0.04,
        "entropic_beta": 1.0,
        "evaluation_prompts": 512,
        "success_criteria": {
            "primary_vs_mean_ppo": "lower endpoint of a two-sided 95% hierarchical-bootstrap interval for robust E[max_32 R] delta > 0",
            "primary_vs_scalar_max": "lower endpoint of a two-sided 95% hierarchical-bootstrap interval for robust E[max_32 R] delta > 0",
            "nominal_mean_floor": "lower endpoint of a two-sided 95% hierarchical-bootstrap interval for candidate mean is at or above the frozen base-policy floor",
            "quality_floor": "lower endpoint of a two-sided 95% hierarchical-bootstrap interval for independent Quality RM mean is at or above the frozen base-policy floor",
            "qwen_generated_proxy": "mean preference >= 0.50 and 95% lower bound >= 0.45 versus Mean-PPO",
            "armorm_generated_proxy": "mean preference >= 0.50 and 95% lower bound >= 0.45 versus Mean-PPO",
            "claim_rule": "all criteria must pass; target-RM improvement alone is insufficient",
        },
        "floor_estimation": {
            "reference_generation_seed": 20260805,
            "floor_responses": 2048,
            "floor_prompt_clusters": 512,
            "shift_diagnostic_responses": 512,
            "shift_diagnostic_prompt_clusters": 128,
            "samples_per_prompt": 4,
            "estimator": "prompt_cluster_percentile_bootstrap_lcb",
            "one_sided_alpha": 0.05,
            "bootstrap_draws": 10000,
            "bootstrap_seed": 26080517,
            "noninferiority_margin_response_sd": 0.1,
            "shift_diagnostic_is_selection_criterion": False,
        },
        "development_audit": {
            "predecessor": "publication-v6",
            "predecessor_outcome": "stopped_before_training_at_split_point_estimate_feasibility_gate",
            "redesign_scope": (
                "v7 replaces split-point feasibility selection with a predeclared "
                "prompt-cluster LCB floor and report-only shift diagnostic before any "
                "v7 policy outcomes"
            ),
        },
        "reported_diagnostics_not_success_criteria": {
            "rewardbench_policy_likelihood": (
                "length-normalized chosen/rejected log-probability accuracy and margin"
            ),
            "skywork_policy_likelihood": (
                "length-normalized chosen/rejected log-probability accuracy and margin"
            ),
            "interpretation": (
                "static preference-retention diagnostics; not Best-of-N performance, not "
                "generated-output human preference, and not a model-selection gate"
            ),
        },
        "rewardbench_source": str(args.rewardbench.resolve()),
        "rewardbench_source_sha256": file_hash(args.rewardbench),
        "rewardbench_input_rows": len(read_jsonl(args.rewardbench)),
        "rewardbench_invalid_rows": invalid,
        "rewardbench_prompt_overlaps_excluded": excluded_prompt,
        "rewardbench_response_overlaps_excluded": excluded_response,
        "rewardbench_identity_deduplicated_rows": len(identity_deduplicated),
        "rewardbench_duplicate_prompt_rows_excluded": duplicate_prompt_rows,
        "rewardbench_duplicate_response_rows_excluded": duplicate_response_rows,
        "rewardbench_deduplicated_rows": len(ordered),
        "rewardbench_calibration": {
            "path": str(calibration_path.resolve()),
            "sha256": file_hash(calibration_path),
            "rows": len(calibration),
            "subset_counts": subset_counts(calibration),
        },
        "rewardbench_policy_lockbox": {
            "path": str(lockbox_path.resolve()),
            "sha256": file_hash(lockbox_path),
            "rows": len(lockbox),
            "subset_counts": subset_counts(lockbox),
        },
        "rewardbench_reserve": {
            "path": str(reserve_path.resolve()),
            "sha256": file_hash(reserve_path),
            "rows": len(reserve),
            "subset_counts": subset_counts(reserve),
        },
        "skywork_policy_lockbox": {
            "path": str(args.skywork_lockbox.resolve()),
            "sha256": file_hash(args.skywork_lockbox),
            "rows": len(skywork_rows),
        },
        "forbidden_inputs": {
            str(path.resolve()): file_hash(path) for path in args.forbidden_jsonl
        },
    }
    write_json(manifest_path, manifest)
    (manifest_path.with_suffix(manifest_path.suffix + ".sha256")).write_text(
        f"{file_hash(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
