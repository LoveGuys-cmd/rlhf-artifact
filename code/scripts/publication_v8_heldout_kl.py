#!/usr/bin/env python3
"""Compute confirmatory per-prompt full-vocabulary KL for publication v8."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any


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
    "best_of_n",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--pilot-eval-dir", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--robust-calibration-report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--best-of-n", type=int, default=32)
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-response-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    values: list[float], seed: int, draws: int
) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("bootstrap requires nonempty finite values")
    rng = random.Random(seed)
    count = len(values)
    estimates = [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(draws)
    ]
    return {
        "count": count,
        "mean": statistics.fmean(values),
        "ci_low": percentile(estimates, 0.025),
        "ci_high": percentile(estimates, 0.975),
    }


def ordinal_group_metrics(group: dict[str, Any], epsilon: float) -> dict[str, float]:
    probabilities = [candidate["probabilities"] for candidate in group["candidates"]]
    robust_expected_max = 0.0
    nominal_expected_max = 0.0
    for threshold in range(1, 5):
        cdfs = [sum(row[:threshold]) for row in probabilities]
        nominal_expected_max += 1.0 - math.prod(cdfs)
        robust_expected_max += 1.0 - math.prod(
            min(1.0, value + epsilon) for value in cdfs
        )
    robust_any_four = 1.0 - math.prod(
        min(1.0, sum(row[:4]) + epsilon) for row in probabilities
    )
    nominal_any_four = 1.0 - math.prod(1.0 - row[4] for row in probabilities)
    selected = max(
        group["candidates"], key=lambda candidate: (float(candidate["mu"]), -candidate["index"])
    )
    return {
        "robust_expected_max": robust_expected_max,
        "nominal_expected_max": nominal_expected_max,
        "robust_probability_any_rating_4": robust_any_four,
        "nominal_probability_any_rating_4": nominal_any_four,
        "selected_mu": float(selected["mu"]),
        "candidate_mu": statistics.fmean(
            float(candidate["mu"]) for candidate in group["candidates"]
        ),
        "candidate_mu_sd": statistics.pstdev(
            float(candidate["mu"]) for candidate in group["candidates"]
        ),
        "candidate_p4": statistics.fmean(row[4] for row in probabilities),
    }


def paired_comparison(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    epsilon: float,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("paired comparison has no common prompts")
    left_metrics = {key: ordinal_group_metrics(left[key], epsilon) for key in keys}
    right_metrics = {key: ordinal_group_metrics(right[key], epsilon) for key in keys}
    fields = tuple(next(iter(left_metrics.values())))
    return {
        field: bootstrap_mean_ci(
            [left_metrics[key][field] - right_metrics[key][field] for key in keys],
            seed + index * 1009,
            draws,
        )
        for index, field in enumerate(fields)
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def add_import_paths(repo: Path) -> None:
    candidates = (
        repo / "scripts",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[2],
    )
    for candidate in candidates:
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))


def generated_samples(
    tokenizer: Any,
    groups: dict[str, dict[str, Any]],
    keys: list[str],
    max_prompt_length: int,
    max_response_tokens: int,
    selector: str,
) -> list[Any]:
    from evrl_experiment import GeneratedSample, chat_prompt

    samples = []
    for key in keys:
        group = groups[key]
        if selector == "first":
            selected = group["candidates"][0]
        elif selector == "max_mu":
            selected = max(
                group["candidates"],
                key=lambda candidate: (float(candidate["mu"]), -candidate["index"]),
            )
        else:
            raise ValueError(f"unknown trajectory selector: {selector}")
        prefix_ids = tokenizer(
            chat_prompt(tokenizer, group["prompt"]),
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        ).input_ids
        response_ids = tokenizer(
            selected["text"], add_special_tokens=False
        ).input_ids[:max_response_tokens]
        if not response_ids:
            if tokenizer.eos_token_id is None:
                raise ValueError("empty response tokenization and no EOS token")
            response_ids = [tokenizer.eos_token_id]
        samples.append(
            GeneratedSample(
                group["prompt"],
                selected["text"],
                [*prefix_ids, *response_ids],
                len(prefix_ids),
            )
        )
    return samples


def reference_log_probs_on_policy_support(
    policy_log_probs: Any,
    reference_log_probs: Any,
    max_input_token_id: int,
) -> Any:
    policy_vocab = int(policy_log_probs.shape[-1])
    reference_vocab = int(reference_log_probs.shape[-1])
    if max_input_token_id < 0 or max_input_token_id >= policy_vocab:
        raise ValueError(
            f"input token id {max_input_token_id} is outside policy vocab {policy_vocab}"
        )
    if policy_vocab > reference_vocab:
        raise ValueError(
            "policy vocabulary is not a prefix of the reference vocabulary: "
            f"policy={policy_vocab}, reference={reference_vocab}"
        )
    return reference_log_probs[..., :policy_vocab]


def reference_kl_by_sample(
    policy: Any,
    reference: Any,
    tokenizer: Any,
    samples: list[Any],
    temperature: float,
    batch_size: int,
) -> tuple[list[float], dict[str, Any]]:
    import torch
    from evrl_experiment import _policy_log_distributions, collate_trajectories

    policy.eval()
    reference.eval()
    device = str(next(policy.parameters()).device)
    values: list[float] = []
    policy_vocab_sizes: set[int] = set()
    reference_vocab_sizes: set[int] = set()
    maximum_input_token_id = -1
    with torch.no_grad():
        for start in range(0, len(samples), max(1, int(batch_size))):
            chunk = samples[start : start + batch_size]
            input_ids, attention_mask, action_mask = collate_trajectories(
                chunk, tokenizer.pad_token_id, device
            )
            policy_lp = _policy_log_distributions(
                policy, input_ids, attention_mask, temperature
            )
            reference_lp = _policy_log_distributions(
                reference, input_ids, attention_mask, temperature
            )
            maximum_input_token_id = max(
                maximum_input_token_id, int(input_ids.max().item())
            )
            policy_vocab_sizes.add(int(policy_lp.shape[-1]))
            reference_vocab_sizes.add(int(reference_lp.shape[-1]))
            reference_on_policy_support = reference_log_probs_on_policy_support(
                policy_lp, reference_lp, maximum_input_token_id
            )
            token_kl = (
                policy_lp.exp() * (policy_lp - reference_on_policy_support)
            ).sum(dim=-1)
            for row in range(len(chunk)):
                selected = token_kl[row][action_mask[row]]
                if selected.numel() == 0:
                    raise ValueError("trajectory has no action tokens")
                value = float(selected.mean().item())
                if not math.isfinite(value):
                    raise ValueError("nonfinite held-out KL")
                values.append(value)
            del (
                input_ids,
                attention_mask,
                action_mask,
                policy_lp,
                reference_lp,
                reference_on_policy_support,
                token_kl,
            )
    if len(values) != len(samples):
        raise RuntimeError("held-out KL sample count mismatch")
    return values, {
        "policy_output_vocab_sizes": sorted(policy_vocab_sizes),
        "reference_output_vocab_sizes": sorted(reference_vocab_sizes),
        "maximum_input_token_id": maximum_input_token_id,
        "support_contract": (
            "Reference log probabilities retain full-vocabulary softmax normalization; "
            "D_KL(policy||reference) sums over the aligned policy-vocabulary prefix."
        ),
    }


def training_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rejected = [
        row.get("hard_kl_update_rejected", "").strip().lower()
        in {"1", "true", "yes"}
        for row in rows
    ]
    return {
        "rows": len(rows),
        "accepted": sum(not value for value in rejected),
        "rollbacks": sum(rejected),
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    args.repo = args.repo.resolve()
    add_import_paths(args.repo)
    import torch
    from evrl_experiment import load_policy_for_eval
    from paper_metrics import load_responses

    if not torch.cuda.is_available():
        raise RuntimeError("publication-v8 audit requires one CUDA GPU")
    robust_report = json.loads(args.robust_calibration_report.read_text(encoding="utf-8"))
    epsilon = float(robust_report["robust_epsilon"])
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("invalid frozen robust epsilon")

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    response_hashes = {}
    for method in METHODS:
        path = args.pilot_eval_dir / f"{method}_seed{args.seed}_responses.jsonl"
        groups[method] = load_responses(path, args.best_of_n)
        response_hashes[method] = sha256(path)
    common = sorted(set.intersection(*(set(value) for value in groups.values())))
    common = common[: args.max_prompts]
    if len(common) != args.max_prompts:
        raise ValueError(
            f"expected {args.max_prompts} common prompts, found {len(common)}"
        )

    comparisons = {
        f"ev_ppo_minus_{method}": paired_comparison(
            groups["ev_ppo"],
            groups[method],
            epsilon,
            args.seed + index * 100003,
            args.bootstrap_draws,
        )
        for index, method in enumerate(METHODS)
        if method != "ev_ppo"
    }

    dtype = torch.bfloat16
    device = "cuda"
    reference, reference_tokenizer = load_policy_for_eval(
        args.base_model, None, dtype, device
    )
    zero_kl = [0.0] * len(common)
    kl_samples = {
        "best_of_n": {
            "common_base_trajectory_kl": zero_kl,
            "own_trajectory_kl": zero_kl,
        }
    }
    kl_results = {
        "best_of_n": {
            "common_base_trajectory_kl": bootstrap_mean_ci(
                zero_kl, args.seed + 710001, args.bootstrap_draws
            ),
            "own_trajectory_kl": bootstrap_mean_ci(
                zero_kl, args.seed + 710002, args.bootstrap_draws
            ),
        }
    }
    try:
        for method in METHODS:
            if method == "best_of_n":
                continue
            checkpoint = args.experiment_root / f"{method}_seed{args.seed}"
            if not (checkpoint / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(checkpoint / "adapter_model.safetensors")
            policy, tokenizer = load_policy_for_eval(
                args.base_model, checkpoint, dtype, device
            )
            try:
                common_samples = generated_samples(
                    reference_tokenizer,
                    groups["best_of_n"],
                    common,
                    args.max_prompt_length,
                    args.max_response_tokens,
                    "first",
                )
                own_samples = generated_samples(
                    tokenizer,
                    groups[method],
                    common,
                    args.max_prompt_length,
                    args.max_response_tokens,
                    "max_mu",
                )
                common_kl, common_vocab = reference_kl_by_sample(
                    policy,
                    reference,
                    reference_tokenizer,
                    common_samples,
                    1.0,
                    args.batch_size,
                )
                own_kl, own_vocab = reference_kl_by_sample(
                    policy,
                    reference,
                    tokenizer,
                    own_samples,
                    1.0,
                    args.batch_size,
                )
                kl_samples[method] = {
                    "common_base_trajectory_kl": common_kl,
                    "own_trajectory_kl": own_kl,
                }
                kl_results[method] = {
                    "common_base_trajectory_kl": bootstrap_mean_ci(
                        common_kl,
                        args.seed + 720000 + METHODS.index(method) * 101,
                        args.bootstrap_draws,
                    ),
                    "own_trajectory_kl": bootstrap_mean_ci(
                        own_kl,
                        args.seed + 730000 + METHODS.index(method) * 101,
                        args.bootstrap_draws,
                    ),
                    "vocabulary": {
                        "common_base_trajectory": common_vocab,
                        "own_trajectory": own_vocab,
                    },
                    "checkpoint": str(checkpoint),
                    "adapter_sha256": sha256(checkpoint / "adapter_model.safetensors"),
                    "training": training_summary(checkpoint / "train_metrics.csv"),
                }
            finally:
                del policy
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del reference
        del reference_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    paired_kl = {}
    for index, method in enumerate(METHODS):
        if method == "ev_ppo":
            continue
        paired_kl[f"ev_ppo_minus_{method}"] = {
            field: bootstrap_mean_ci(
                [
                    left - right
                    for left, right in zip(
                        kl_samples["ev_ppo"][field], kl_samples[method][field]
                    )
                ],
                args.seed + 740000 + index * 1009 + field_index,
                args.bootstrap_draws,
            )
            for field_index, field in enumerate(
                ("common_base_trajectory_kl", "own_trajectory_kl")
            )
        }

    output = {
        "protocol": "stablemax-ppo-publication-v8-confirmatory-heldout-kl-v1",
        "scientific_status": "confirmatory_frozen_protocol_evidence",
        "source_eval": str(args.pilot_eval_dir),
        "seed": args.seed,
        "best_of_n": args.best_of_n,
        "num_prompts": len(common),
        "robust_epsilon": epsilon,
        "gpu": torch.cuda.get_device_name(0),
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "response_sha256": response_hashes,
        "prompt_keys": common,
        "comparisons": comparisons,
        "heldout_kl": kl_results,
        "per_prompt_heldout_kl": kl_samples,
        "paired_heldout_kl": paired_kl,
        "interpretation_contract": {
            "common_base_trajectory_kl": (
                "prompt-mean full-vocabulary token KL on identical frozen first "
                "base-policy trajectories; no reward-model selection"
            ),
            "own_trajectory_kl": (
                "prompt-mean full-vocabulary token KL on each method's frozen "
                "max-mu-selected trajectories"
            ),
            "endpoint_limit": (
                "Final-checkpoint Pareto diagnostics are not a matched-KL training curve."
            ),
            "vocabulary_support": (
                "Reference softmax remains normalized over its full output vocabulary; "
                "KL is summed over the policy's aligned prefix support, with zero policy "
                "mass on reference-only reserved tokens."
            ),
            "claim_limit": (
                "KL is a predeclared confirmatory endpoint, not direct human preference."
            ),
        },
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
