#!/usr/bin/env python3
"""Exact ordinal finite-N Best-of-N policy optimization and evaluation.

The principal ``ev_ppo`` method optimizes the expected maximum of N bounded
ordinal human ratings under a frozen calibrated reward model.  Its marginal
credit is exact and contains neither sampled reward noise nor quadrature.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import random
import re
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from peft import (
    AutoPeftModelForCausalLM,
    AutoPeftModelForSequenceClassification,
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from floor_statistics import prompt_cluster_bootstrap_lcb
from evrl_math import (
    QuadratureDiagnostics,
    finite_n_gaussian_calibration,
    gaussian_max_credit_batch,
    monte_carlo_rb_credit,
)
from evrl_ordinal import (
    NUM_RATINGS,
    exact_group_statistics,
    exact_marginal_max_credit,
    exact_probability_any_rating,
    robust_group_statistics,
    scalar_max_marginal_credit,
    validate_probabilities,
)

try:
    from purm_eval_lib import load_purm_rm as strict_load_purm_rm
except ModuleNotFoundError:
    strict_load_purm_rm = None


EPS = 1e-8
MIN_LOG_SIGMA = -8.0
MAX_LOG_SIGMA = 8.0
METHOD_DISPLAY_NAMES = {
    "ev_ppo": "StableMax-PPO + Mean/Q",
    "nominal_ev_ppo": "Nominal Ordinal EV-PPO + Mean/Q",
    "ev_ppo_no_quality": "StableMax-PPO (no Q)",
    "ev_ppo_no_mean": "StableMax-PPO (no mean floor)",
    "vanilla_ppo": "Mean-PPO + Q",
    "vanilla_grpo": "GRPO + Q",
    "scalar_max_ppo": "Scalar Max@N-PPO + Q",
    "entropic_ppo": "Entropic-PPO + Q",
    "top4_ppo": "Top-4 PPO + Q",
    "gaussian_ev_ppo": "Gaussian EV-PPO + Q",
    "best_of_n": "Best-of-N (base policy)",
}
TRAINABLE_METHODS = {
    "ev_ppo",
    "nominal_ev_ppo",
    "ev_ppo_no_quality",
    "ev_ppo_no_mean",
    "vanilla_ppo",
    "vanilla_grpo",
    "scalar_max_ppo",
    "entropic_ppo",
    "top4_ppo",
    "gaussian_ev_ppo",
}
TABLE_METHOD_ORDER = [
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
]

MAIN_TABLE_FIELDS = [
    "method",
    "method_name",
    "seed",
    "best_of_n",
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
    "preference_accuracy",
    "preference_accuracy_metric",
    "num_preference_pairs",
    "reward_model_preference_accuracy",
    "reward_model_preference_pairs",
    "num_eval_prompts",
    "performance",
    "performance_metric",
    "best_baseline_method",
    "win_rate_vs_best_baseline",
    "win_rate_vs_best_baseline_ci_low",
    "win_rate_vs_best_baseline_ci_high",
    "bon_margin_mean_vs_best_baseline",
    "bon_margin_se_vs_best_baseline",
    "paired_ttest_p_value_vs_best_baseline",
    "wilcoxon_p_value_vs_best_baseline",
    "responses_jsonl",
    "reward_distribution_status",
]

PERFORMANCE_FIELDS = [
    "method",
    "method_name",
    "seed",
    "num_eval_pairs",
    "accuracy",
    "metric",
    "mean_margin_logprob",
    "chosen_logprob_mean",
    "rejected_logprob_mean",
    "resolved_policy_checkpoint",
]


@dataclass
class GeneratedSample:
    prompt: str
    response: str
    input_ids: list[int]
    action_start: int


class PromptBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_robust_calibration_report(path_like: str | Path) -> tuple[float, dict[str, Any]]:
    """Read the frozen validation-only robust radius and verify its sidecar."""
    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(f"Missing robust calibration report: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing robust calibration sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if expected != sha256_file(path):
        raise ValueError(f"Robust calibration SHA-256 mismatch: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != "ordinal-v5-robust-calibration-gate-before-policy-training":
        raise ValueError("Robust calibration report has the wrong protocol")
    if report.get("gates_passed") is not True or not all(
        (report.get("gate_checks") or {}).values()
    ):
        raise ValueError("Robust calibration report did not pass every frozen gate")
    epsilon = float(report.get("robust_epsilon", math.nan))
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("Robust calibration report contains an invalid epsilon")
    return epsilon, report


def clustered_one_sided_calibration_radius(
    cluster_residuals: torch.Tensor,
    alpha: float,
) -> tuple[float, list[float], float]:
    """Compute the simultaneous one-sided radius from independent prompt clusters."""
    residuals = torch.as_tensor(cluster_residuals, dtype=torch.float64)
    if residuals.ndim != 2 or residuals.shape[0] < 1 or residuals.shape[1] != NUM_RATINGS - 1:
        raise ValueError("cluster residuals must have shape [prompts, 4]")
    if not torch.isfinite(residuals).all():
        raise ValueError("cluster residuals must be finite")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("calibration alpha must lie strictly between zero and one")
    mean_residuals = residuals.mean(dim=0)
    concentration = math.sqrt(
        math.log((NUM_RATINGS - 1) / alpha) / (2.0 * residuals.shape[0])
    )
    epsilon = min(
        1.0,
        max(0.0, float(mean_residuals.max().item())) + concentration,
    )
    return epsilon, [float(value) for value in mean_residuals.tolist()], concentration


def floor_calibration_seed(args) -> int:
    value = getattr(args, "floor_calibration_seed", None)
    return int(args.seed if value is None else value)


def quality_floor_cache_spec(args) -> dict[str, Any]:
    checkpoint = Path(args.quality_rm_checkpoint)
    config_path = checkpoint / "strong_rm_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Quality RM config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = config.get("model_name") or config.get("model_name_or_path")
    if not model_name:
        raise ValueError(f"Quality RM config does not declare model_name: {config_path}")
    adapter_path = Path(model_name) / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Missing frozen Quality RM adapter: {adapter_path}")
    calibration_path = Path(args.quality_calibration_jsonl)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Missing quality calibration split: {calibration_path}")
    manifest_path = Path(args.quality_calibration_manifest) if args.quality_calibration_manifest else None
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"Missing quality calibration manifest: {manifest_path}")
    manifest_fields = {}
    protocol = "training-split-quality-floor-v1"
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = str(manifest.get("protocol") or "")
        if protocol not in {
            "reference-policy-same-train-distribution-v3",
            "reference-policy-cluster-lcb-v4",
        }:
            raise ValueError("Quality-floor calibration manifest has the wrong protocol")
        if manifest.get("calibration_output_sha256") != sha256_file(calibration_path):
            raise ValueError("Quality-floor calibration JSONL does not match its manifest")
        manifest_fields = {
            "quality_calibration_manifest": str(manifest_path.resolve()),
            "quality_calibration_manifest_sha256": sha256_file(manifest_path),
        }
    return {
        "protocol": protocol,
        "quality_rm_checkpoint": str(checkpoint.resolve()),
        "quality_rm_config_sha256": sha256_file(config_path),
        "quality_rm_adapter_sha256": sha256_file(adapter_path),
        "quality_calibration_jsonl": str(calibration_path.resolve()),
        "quality_calibration_jsonl_sha256": sha256_file(calibration_path),
        "quality_calibration_pairs": int(args.quality_calibration_pairs),
        "quality_rm_max_length": int(args.quality_rm_max_length),
        "quality_rm_batch_size": int(args.quality_rm_batch_size),
        "quality_rm_temperature": float(args.quality_rm_temperature),
        "quality_noninferiority_margin_sd": float(args.quality_noninferiority_margin_sd),
        "floor_bootstrap_alpha": float(getattr(args, "floor_bootstrap_alpha", 0.05)),
        "floor_bootstrap_draws": int(getattr(args, "floor_bootstrap_draws", 10000)),
        "floor_bootstrap_seed": int(getattr(args, "floor_bootstrap_seed", 20260805)),
        "seed": floor_calibration_seed(args),
        **manifest_fields,
    }


def read_quality_floor_cache(
    cache_path: Path | None,
    expected_spec: dict[str, Any],
) -> tuple[float, dict[str, Any]] | None:
    if cache_path is None or not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("spec") != expected_spec:
        raise ValueError(
            f"Quality floor cache does not match frozen calibration inputs: {cache_path}"
        )
    floor = float(payload.get("quality_floor", math.nan))
    if not math.isfinite(floor):
        raise ValueError(f"Quality floor cache contains a non-finite floor: {cache_path}")
    metadata = dict(payload.get("metadata") or {})
    metadata["quality_floor_cache_path"] = str(cache_path)
    metadata["quality_floor_cache_status"] = "reused"
    return floor, metadata


def write_quality_floor_cache(
    cache_path: Path,
    spec: dict[str, Any],
    floor: float,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "spec": spec,
        "quality_floor": float(floor),
        "metadata": metadata,
    }
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    write_json(temporary, payload)
    temporary.replace(cache_path)


def mean_floor_cache_spec(args) -> dict[str, Any]:
    checkpoint = Path(args.rm_checkpoint)
    config_path = checkpoint / "moment_rm_config.json"
    adapter_path = checkpoint / "adapter_model.safetensors"
    calibration_path = Path(args.mean_calibration_jsonl)
    manifest_path = Path(args.mean_calibration_manifest)
    for path in (config_path, adapter_path, calibration_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing mean-floor calibration input: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") not in {
        "reference-policy-same-train-distribution-v3",
        "reference-policy-cluster-lcb-v4",
    }:
        raise ValueError("Mean-floor calibration manifest has the wrong protocol")
    if manifest.get("calibration_output_sha256") != sha256_file(calibration_path):
        raise ValueError("Mean-floor calibration JSONL does not match its manifest")
    return {
        "protocol": "reference-policy-ordinal-mean-floor-v1",
        "rm_config_sha256": sha256_file(config_path),
        "rm_adapter_sha256": sha256_file(adapter_path),
        "calibration_jsonl_sha256": sha256_file(calibration_path),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "calibration_pairs": int(args.mean_calibration_pairs),
        "rm_max_length": int(args.rm_max_length),
        "rm_batch_size": int(args.rm_batch_size),
        "margin_sd": float(args.mean_noninferiority_margin_sd),
        "floor_bootstrap_alpha": float(getattr(args, "floor_bootstrap_alpha", 0.05)),
        "floor_bootstrap_draws": int(getattr(args, "floor_bootstrap_draws", 10000)),
        "floor_bootstrap_seed": int(getattr(args, "floor_bootstrap_seed", 20260805)),
        "seed": floor_calibration_seed(args),
    }


def read_mean_floor_cache(
    cache_path: Path,
    expected_spec: dict[str, Any],
) -> tuple[float, dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("spec") != expected_spec:
        raise ValueError(f"Mean floor cache does not match frozen inputs: {cache_path}")
    floor = float(payload.get("mean_floor", math.nan))
    if not math.isfinite(floor):
        raise ValueError(f"Mean floor cache contains a non-finite floor: {cache_path}")
    metadata = dict(payload.get("metadata") or {})
    metadata["mean_floor_cache_path"] = str(cache_path)
    metadata["mean_floor_cache_status"] = "reused"
    return floor, metadata


def write_mean_floor_cache(
    cache_path: Path,
    spec: dict[str, Any],
    floor: float,
    metadata: dict[str, Any],
) -> None:
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    write_json(
        temporary,
        {"spec": spec, "mean_floor": float(floor), "metadata": metadata},
    )
    temporary.replace(cache_path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def split_methods(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def calibration_alpha(best_of_n: int) -> float:
    return finite_n_gaussian_calibration(int(best_of_n))


def resolve_alpha(alpha: float | None, best_of_n: int, require_calibration: bool = True) -> float:
    expected = calibration_alpha(best_of_n)
    if alpha is None:
        return expected
    value = float(alpha)
    if require_calibration and abs(value - expected) > 1e-6:
        raise ValueError(
            f"alpha={value} is inconsistent with best_of_n={best_of_n}; "
            f"the finite-N Gaussian calibration is kappa_N={expected}"
        )
    return value


def gaussian_tail_probability(alpha: float) -> float:
    return 0.5 * math.erfc(float(alpha) / math.sqrt(2.0))


def latest_checkpoint(path_like: str | Path) -> Path:
    path = Path(path_like)
    if (path / "adapter_config.json").exists() or (path / "config.json").exists():
        return path
    checkpoints = []
    for child in path.glob("checkpoint-*"):
        try:
            checkpoints.append((int(child.name.rsplit("-", 1)[-1]), child))
        except ValueError:
            continue
    if not checkpoints:
        raise FileNotFoundError(f"No loadable checkpoint found under {path}")
    return max(checkpoints)[1]


def as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("content") or item.get("text") or item.get("value")
                if text:
                    role = item.get("role")
                    parts.append(f"{role}: {text}" if role else str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip() or None
    return str(value).strip() or None


_DIALOGUE_ROLE_RE = re.compile(
    r"(?im)(?:^|\n)(System|Human|User|Assistant|Question|Answer):[ \t]*"
)
_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "question": "user",
    "assistant": "assistant",
    "answer": "assistant",
}


def has_dialogue_markers(text: str) -> bool:
    return _DIALOGUE_ROLE_RE.search(text) is not None


def prompt_messages(prompt: str) -> list[dict[str, str]]:
    """Parse HelpSteer-style transcripts into canonical chat roles."""
    text = prompt.strip()
    matches = list(_DIALOGUE_ROLE_RE.finditer(text))
    if not matches:
        return [{"role": "user", "content": text}]

    messages: list[dict[str, str]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        messages.append({"role": "user", "content": prefix})

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end() : end].strip()
        if not content:
            continue
        role = _ROLE_MAP[match.group(1).lower()]
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})

    return messages or [{"role": "user", "content": text}]


def concat_prompt_response(prompt: str, response: str) -> str:
    if not prompt:
        return response
    if not response:
        return prompt
    if prompt[-1].isspace() or response[0].isspace():
        return prompt + response
    return prompt + (" " if prompt.endswith((":", "Assistant:", "Answer:")) else "\n") + response


def split_full_preference(chosen: str, rejected: str) -> tuple[str, str, str]:
    shared = 0
    while shared < min(len(chosen), len(rejected)) and chosen[shared] == rejected[shared]:
        shared += 1
    if shared < 8:
        return "", chosen, rejected
    prefix = chosen[:shared]
    candidates = []
    for marker in ("\n\nAssistant:", "\nAssistant:", "Assistant:", "\n\nAnswer:", "\nAnswer:", "Answer:"):
        idx = prefix.rfind(marker)
        if idx >= 0:
            candidates.append((idx, marker))
    if candidates:
        idx, marker = max(candidates)
        cut = idx + len(marker)
        while cut < len(prefix) and prefix[cut].isspace():
            cut += 1
    else:
        cut = prefix.rfind("\n") + 1 if "\n" in prefix else shared
    if not chosen[cut:].strip() or not rejected[cut:].strip():
        return "", chosen, rejected
    return chosen[:cut], chosen[cut:], rejected[cut:]


def normalize_preference_pair(prompt: str, chosen: str, rejected: str) -> dict[str, str]:
    if not prompt:
        recovered_prompt, recovered_chosen, recovered_rejected = split_full_preference(chosen, rejected)
        if recovered_prompt:
            prompt, chosen, rejected = recovered_prompt, recovered_chosen, recovered_rejected
    return {"prompt": prompt or "", "chosen": chosen, "rejected": rejected}


def load_preference_pairs(path: Path, limit: int | None = None, tokenizer=None) -> list[dict[str, str]]:
    if path.is_dir():
        dataset = load_from_disk(str(path))
        if {"input_ids_chosen", "input_ids_rejected"}.issubset(dataset.column_names):
            if tokenizer is None:
                raise ValueError(f"Tokenized preference dataset {path} requires a tokenizer")
            rows = []
            for index in range(min(len(dataset), limit or len(dataset))):
                item = dataset[index]
                chosen = tokenizer.decode(item["input_ids_chosen"], skip_special_tokens=True)
                rejected = tokenizer.decode(item["input_ids_rejected"], skip_special_tokens=True)
                rows.append(normalize_preference_pair("", chosen, rejected))
            return rows
        source = [dict(dataset[index]) for index in range(min(len(dataset), limit or len(dataset)))]
    else:
        source = read_jsonl(path)

    pairs = []
    for row in source:
        prompt = as_text(row.get("input") or row.get("prompt") or row.get("instruction") or row.get("question")) or ""
        chosen = as_text(row.get("chosen") or row.get("answer") or row.get("preferred") or row.get("accept"))
        rejected = as_text(row.get("rejected") or row.get("reject") or row.get("other") or row.get("dispreferred"))
        if chosen and rejected:
            pairs.append(normalize_preference_pair(prompt, chosen, rejected))
        if limit is not None and len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError(f"No prompt/chosen/rejected pairs found in {path}")
    return pairs


def load_prompt_records(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    records = []
    for row in read_jsonl(path):
        prompt = as_text(row.get("input") or row.get("prompt") or row.get("instruction") or row.get("question"))
        reference = as_text(row.get("answer") or row.get("chosen") or row.get("output") or row.get("response"))
        rejected = as_text(row.get("rejected") or row.get("reject") or row.get("other"))
        if prompt is None and reference and rejected:
            pair = normalize_preference_pair("", reference, rejected)
            prompt, reference = pair["prompt"], pair["chosen"]
        if prompt is None and reference is not None:
            prompt = ""
        if prompt is None:
            continue
        records.append({"prompt": prompt, "reference": reference or ""})
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"No prompts found in {path}")
    return records


def chat_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                prompt_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            warnings.warn(f"Falling back to raw prompt after chat-template failure: {exc}")
    return prompt


def force_pad_token(model, tokenizer) -> None:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "pad_token_id"):
            config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id


def decode_ordinal_reward_distribution(
    logits: torch.Tensor, cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode calibrated p(R=0), ..., p(R=4) and their observable moments."""
    attribute_names = list(cfg.get("attribute_names", []))
    num_attributes = len(attribute_names)
    if num_attributes <= 0 or logits.shape[-1] != 2 * num_attributes:
        raise ValueError(
            f"Moment RM expected {2 * num_attributes} logits for {num_attributes} attributes, "
            f"received shape={tuple(logits.shape)}"
        )
    reward_index = int(cfg.get("reward_attribute_index", 0))
    rating_min = float(cfg.get("rating_min", 0.0))
    rating_max = float(cfg.get("rating_max", 4.0))
    sigma_floor = float(cfg.get("sigma_floor", 0.05))
    sigma_temperature = float(cfg.get("sigma_temperature", 1.0))
    latent_mu_parameterization = str(
        cfg.get("latent_mu_parameterization", cfg.get("mu_parameterization", ""))
    )
    latent_sigma_parameterization = str(
        cfg.get("latent_sigma_parameterization", cfg.get("sigma_parameterization", ""))
    )
    if latent_mu_parameterization == "unbounded_latent_utility_raw_mu":
        latent_mu = logits[:, reward_index]
    else:
        latent_mu = rating_min + (rating_max - rating_min) * torch.sigmoid(
            logits[:, reward_index]
        )
    raw_sigma = logits[:, num_attributes + reward_index]
    if "inverse_softplus(1-sigma_floor)" in latent_sigma_parameterization:
        initial_scale = 1.0 - sigma_floor
        if initial_scale <= 0.0:
            raise ValueError("Moment RM sigma_floor must be strictly below one")
        scale_offset = initial_scale + math.log(-math.expm1(-initial_scale))
        latent_sigma = F.softplus(raw_sigma + scale_offset) + sigma_floor
    else:
        latent_sigma = F.softplus(raw_sigma) + sigma_floor
    latent_sigma = latent_sigma * sigma_temperature

    if cfg.get("reward_moment_mapping") != "ordinal_induced_observable_rating_moments":
        raise ValueError("Robust ordinal v5 requires ordinal_induced_observable_rating_moments")

    cutpoints = torch.as_tensor(
        cfg.get("ordinal_cutpoints"), dtype=logits.dtype, device=logits.device
    )
    if cutpoints.shape != (num_attributes, 4):
        raise ValueError(
            f"Moment RM ordinal_cutpoints must have shape {(num_attributes, 4)}, "
            f"got {tuple(cutpoints.shape)}"
        )
    reward_cutpoints = cutpoints[reward_index]
    finite_cdf = torch.special.ndtr(
        (reward_cutpoints.unsqueeze(0) - latent_mu.unsqueeze(-1))
        / latent_sigma.unsqueeze(-1)
    )
    cdf = torch.cat(
        (torch.zeros_like(finite_cdf[:, :1]), finite_cdf, torch.ones_like(finite_cdf[:, :1])),
        dim=-1,
    )
    probabilities = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(1e-7)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    rating_values = torch.linspace(
        rating_min, rating_max, probabilities.shape[-1],
        dtype=probabilities.dtype, device=probabilities.device,
    )
    reward_mu = (probabilities * rating_values).sum(dim=-1)
    reward_variance = (
        probabilities * (rating_values - reward_mu.unsqueeze(-1)).square()
    ).sum(dim=-1)
    return probabilities, reward_mu, reward_variance.clamp_min(1e-7).sqrt()


def decode_ordinal_reward_moments(
    logits: torch.Tensor, cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper used to verify exact observable-moment reconstruction."""
    _, reward_mu, reward_sigma = decode_ordinal_reward_distribution(logits, cfg)
    return reward_mu, reward_sigma


class StrongRewardWrapper(nn.Module):
    def __init__(self, config_path: Path, dtype: torch.dtype):
        super().__init__()
        self.config_data = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = (
            self.config_data.get("model_name")
            or self.config_data.get("model_name_or_path")
            or self.config_data.get("candidate")
        )
        if not model_name:
            raise ValueError(f"Missing model_name in {config_path}")
        self.kind = str(self.config_data.get("kind", "strong_pretrained_reward_model"))
        model_kwargs: dict[str, Any] = {}
        if self.kind == "ordinal_gaussian_moment_rm":
            attribute_names = self.config_data.get("attribute_names")
            if not isinstance(attribute_names, list) or not attribute_names:
                raise ValueError(
                    f"Moment RM config must declare a nonempty attribute_names list: {config_path}"
                )
            model_kwargs["num_labels"] = 2 * len(attribute_names)
        elif self.config_data.get("rm_type") == "scalar_quality_reward_model":
            model_kwargs["num_labels"] = 1
        model_cls = (
            AutoPeftModelForSequenceClassification
            if (Path(model_name) / "adapter_config.json").exists()
            else AutoModelForSequenceClassification
        )
        self.inner = model_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=(
                self.config_data.get("rm_type") != "scalar_quality_reward_model"
            ),
            **model_kwargs,
        )
        tokenizer_name = self.config_data.get("tokenizer_name") or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        if getattr(self.inner.config, "pad_token_id", None) is None:
            self.inner.config.pad_token_id = self.tokenizer.pad_token_id
        self.inner.to("cpu").eval()
        self._resolved_checkpoint = str(model_name)
        self.sigma_mode = str(
            self.config_data.get("sigma_mode")
            or self.config_data.get("uncertainty_type")
            or "unknown"
        )
        declared_status = self.config_data.get("reward_variance_status")
        if declared_status:
            self.reward_variance_status = str(declared_status)
        else:
            lowered = self.sigma_mode.lower()
            self.reward_variance_status = (
                "aleatoric_conditional_scale"
                if any(token in lowered for token in ("aleatoric", "conditional_variance", "heteroscedastic"))
                else "uncertainty_proxy_not_validated_as_aleatoric"
            )

    @property
    def device(self) -> torch.device:
        return next(self.inner.parameters()).device

    def to(self, *args, **kwargs):
        self.inner.to(*args, **kwargs)
        return self

    def _format(self, prompt: str, response: str) -> str:
        fmt = self.config_data.get("format", "auto")
        if fmt == "auto":
            fmt = "chat" if getattr(self.tokenizer, "chat_template", None) else "prompt_answer"
        if fmt == "chat" and getattr(self.tokenizer, "chat_template", None):
            messages = prompt_messages(prompt)
            messages.append({"role": "assistant", "content": response})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        if prompt in response or has_dialogue_markers(response):
            return response
        if has_dialogue_markers(prompt):
            return concat_prompt_response(prompt, response)
        return f"Question: {prompt.strip()}\nAnswer: {response.strip()}"

    @torch.inference_mode()
    def score_prompt_response_distributions(
        self,
        prompts: list[str],
        responses: list[str],
        max_length: int,
        batch_size: int,
        rm_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose the accepted ordinal distribution without changing RM weights."""
        if self.kind != "ordinal_gaussian_moment_rm":
            raise TypeError("Full ordinal probabilities are available only for the Moment RM")
        if abs(float(rm_temperature) - 1.0) > 1e-12:
            raise ValueError("Validated ordinal probabilities cannot be temperature-rescaled")
        texts = [self._format(prompt, response) for prompt, response in zip(prompts, responses)]
        probabilities, mus, sigmas = [], [], []
        for start in range(0, len(texts), max(1, int(batch_size))):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            logits = self.inner(**encoded).logits.float()
            prob, mu, sigma = decode_ordinal_reward_distribution(logits, self.config_data)
            probabilities.append(prob.cpu())
            mus.append(mu.cpu())
            sigmas.append(sigma.cpu())
        probability = validate_probabilities(torch.cat(probabilities).float())
        return probability, torch.cat(mus).float(), torch.cat(sigmas).float()

    @torch.inference_mode()
    def score_prompt_responses(
        self,
        prompts: list[str],
        responses: list[str],
        max_length: int,
        batch_size: int,
        rm_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "ordinal_gaussian_moment_rm":
            _, mu, sigma = self.score_prompt_response_distributions(
                prompts, responses, max_length, batch_size, rm_temperature
            )
            return mu, sigma
        texts = [self._format(prompt, response) for prompt, response in zip(prompts, responses)]
        mus, sigmas = [], []
        cfg = self.config_data
        score_index = int(cfg.get("score_index", 0))
        orientation = float(cfg.get("orientation", 1.0))
        center = float(cfg.get("score_center", 0.0))
        scale = max(float(cfg.get("score_scale", 1.0)), 1e-6)
        sigma_floor = float(cfg.get("sigma_floor", 0.05))
        sigma_proxy = max(float(cfg.get("sigma_proxy_scale", cfg.get("sigma_proxy", 1.0))), 1e-6)
        sigma_mode = str(cfg.get("sigma_mode", "legacy_proxy"))
        for start in range(0, len(texts), max(1, int(batch_size))):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            logits = self.inner(**encoded).logits.float()
            if logits.ndim == 1:
                raw = logits
            elif logits.shape[-1] == 1:
                raw = logits[:, 0]
            elif logits.shape[-1] == 2:
                raw = logits[:, 1] - logits[:, 0]
            else:
                raw = logits[:, min(max(score_index, 0), logits.shape[-1] - 1)]
            mu = (orientation * raw - center) / scale
            if sigma_mode == "none":
                sigma = torch.zeros_like(mu)
            else:
                sigma = sigma_floor + sigma_proxy / (sigma_proxy + mu.abs() + 1e-6)
                sigma = sigma / max(float(rm_temperature), 1e-6)
            mus.append(mu.cpu())
            sigmas.append(sigma.cpu())
        return torch.cat(mus), torch.cat(sigmas)


def load_rm(checkpoint: str | Path, base_model: str, dtype: torch.dtype, device: str):
    path = Path(checkpoint)
    for config_name in ("moment_rm_config.json", "strong_rm_config.json"):
        config_path = path / config_name
        if config_path.exists():
            wrapper = StrongRewardWrapper(config_path, dtype)
            return wrapper, wrapper.tokenizer
    if strict_load_purm_rm is None:
        raise FileNotFoundError(
            f"No moment_rm_config.json or strong_rm_config.json under {path}, "
            "and purm_eval_lib is unavailable"
        )
    model, tokenizer = strict_load_purm_rm(path, base_model, dtype, device)
    force_pad_token(model, tokenizer)
    if not hasattr(model, "reward_variance_status"):
        model.reward_variance_status = "unknown_fitted_reward_scale"
    return model, tokenizer


@torch.inference_mode()
def score_responses(
    rm,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_length: int,
    batch_size: int,
    rm_temperature: float,
    reward_scale_mode: str = "model",
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(rm, "score_prompt_responses"):
        mu, sigma = rm.score_prompt_responses(
            prompts, responses, max_length, batch_size, rm_temperature
        )
        mu, sigma = mu.float().cpu(), sigma.float().cpu()
        if reward_scale_mode == "zero":
            sigma = torch.zeros_like(mu)
        elif reward_scale_mode != "model":
            raise ValueError(f"Unknown reward_scale_mode={reward_scale_mode!r}")
        return mu, sigma
    texts = [concat_prompt_response(prompt, response) for prompt, response in zip(prompts, responses)]
    model_device = next(rm.parameters()).device
    mus, sigmas = [], []
    for start in range(0, len(texts), max(1, int(batch_size))):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model_device)
        logits = rm(**encoded).logits.float()
        mu = logits[:, 0]
        if logits.shape[-1] > 1:
            sigma = torch.exp(logits[:, 1].clamp(MIN_LOG_SIGMA, MAX_LOG_SIGMA))
        else:
            sigma = torch.ones_like(mu)
        mus.append(mu.cpu())
        sigmas.append(sigma.cpu())
    mu, sigma = torch.cat(mus), torch.cat(sigmas)
    if reward_scale_mode == "zero":
        sigma = torch.zeros_like(mu)
    elif reward_scale_mode != "model":
        raise ValueError(f"Unknown reward_scale_mode={reward_scale_mode!r}")
    return mu, sigma


@torch.inference_mode()
def score_ordinal_responses(
    rm,
    prompts: list[str],
    responses: list[str],
    max_length: int,
    batch_size: int,
    rm_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score responses with the frozen five-category ordinal distribution."""
    scorer = getattr(rm, "score_prompt_response_distributions", None)
    if scorer is None:
        raise TypeError("Ordinal EV-PPO requires an RM that exports p0,...,p4")
    probabilities, mu, sigma = scorer(
        prompts, responses, max_length, batch_size, rm_temperature
    )
    probabilities = validate_probabilities(probabilities.float().cpu())
    values = torch.arange(NUM_RATINGS, dtype=probabilities.dtype)
    reconstructed_mu = (probabilities * values).sum(dim=-1)
    reconstructed_sigma = (
        probabilities * (values - reconstructed_mu.unsqueeze(-1)).square()
    ).sum(dim=-1).clamp_min(1e-7).sqrt()
    if not torch.allclose(mu.float().cpu(), reconstructed_mu, atol=2e-6, rtol=2e-6):
        raise RuntimeError("ordinal probability export changed the RM expected rating")
    if not torch.allclose(sigma.float().cpu(), reconstructed_sigma, atol=2e-6, rtol=2e-6):
        raise RuntimeError("ordinal probability export changed the RM rating standard deviation")
    return probabilities, reconstructed_mu, reconstructed_sigma


def load_train_policy(base_model: str, args, dtype: torch.dtype, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    )
    force_pad_token(model, tokenizer)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config).to(device)
    return model, tokenizer


def load_policy_for_eval(base_model: str, checkpoint: Path | None, dtype: torch.dtype, device: str):
    if checkpoint is not None and checkpoint.exists():
        resolved = latest_checkpoint(checkpoint)
        model = AutoPeftModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        checkpoint_name = str(resolved)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        checkpoint_name = base_model
    tokenizer.padding_side = "left"
    force_pad_token(model, tokenizer)
    model.to(device).eval()
    model._resolved_checkpoint = checkpoint_name
    return model, tokenizer


def _trim_generated_ids(token_ids: list[int], eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    if not token_ids:
        if eos_token_id is None:
            raise RuntimeError("Generation returned no tokens and the tokenizer has no EOS token")
        return [int(eos_token_id)]
    result = []
    for token_id in token_ids:
        if pad_token_id is not None and token_id == pad_token_id and result:
            if eos_token_id is None or result[-1] == eos_token_id:
                break
        result.append(int(token_id))
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return result or [int(eos_token_id if eos_token_id is not None else token_ids[0])]


@torch.inference_mode()
def generate_group(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_batch_size: int,
) -> list[GeneratedSample]:
    if abs(float(top_p) - 1.0) > 1e-12:
        raise ValueError("Policy-gradient generation requires top_p=1.0; truncated support is not used")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    prefix = chat_prompt(tokenizer, prompt)
    encoded = tokenizer(
        prefix,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    ).to(next(model.parameters()).device)
    prefix_ids = encoded.input_ids[0].tolist()
    samples = []
    was_training = model.training
    model.eval()
    for start in range(0, int(group_size), max(1, int(generation_batch_size))):
        current = min(int(generation_batch_size), int(group_size) - start)
        input_ids = encoded.input_ids.repeat(current, 1)
        attention_mask = encoded.attention_mask.repeat(current, 1)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=float(temperature),
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = output[:, input_ids.shape[1] :].cpu().tolist()
        for token_ids in generated:
            token_ids = _trim_generated_ids(token_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
            response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            samples.append(
                GeneratedSample(
                    prompt=prompt,
                    response=response,
                    input_ids=[*prefix_ids, *token_ids],
                    action_start=len(prefix_ids),
                )
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if was_training:
        model.train()
    return samples


def collate_trajectories(
    samples: list[GeneratedSample],
    pad_token_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(sample.input_ids) for sample in samples)
    batch = len(samples)
    input_ids = torch.full((batch, max_length), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_length), dtype=torch.long, device=device)
    action_mask = torch.zeros((batch, max_length - 1), dtype=torch.bool, device=device)
    for row, sample in enumerate(samples):
        length = len(sample.input_ids)
        input_ids[row, :length] = torch.tensor(sample.input_ids, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
        start = max(int(sample.action_start) - 1, 0)
        action_mask[row, start : length - 1] = True
    return input_ids, attention_mask, action_mask


def _policy_log_distributions(model, input_ids, attention_mask, temperature: float) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1].float()
    return F.log_softmax(logits / float(temperature), dim=-1)


@torch.no_grad()
def collect_old_token_logprobs(
    model,
    tokenizer,
    samples: list[GeneratedSample],
    temperature: float,
    micro_batch_size: int,
) -> list[torch.Tensor]:
    model.eval()
    device = str(next(model.parameters()).device)
    result = []
    for start in range(0, len(samples), max(1, int(micro_batch_size))):
        chunk = samples[start : start + micro_batch_size]
        input_ids, attention_mask, action_mask = collate_trajectories(
            chunk, tokenizer.pad_token_id, device
        )
        log_probs = _policy_log_distributions(model, input_ids, attention_mask, temperature)
        chosen = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        for row in range(len(chunk)):
            result.append(chosen[row][action_mask[row]].detach().cpu())
        del input_ids, attention_mask, action_mask, log_probs, chosen
    return result


def token_ppo_terms(
    policy,
    reference,
    tokenizer,
    samples: list[GeneratedSample],
    old_logprobs: list[torch.Tensor],
    advantages: torch.Tensor,
    temperature: float,
    clip_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    device = str(next(policy.parameters()).device)
    input_ids, attention_mask, action_mask = collate_trajectories(samples, tokenizer.pad_token_id, device)
    policy_log_probs = _policy_log_distributions(policy, input_ids, attention_mask, temperature)
    with torch.no_grad():
        reference_log_probs = _policy_log_distributions(reference, input_ids, attention_mask, temperature)
    chosen_new = policy_log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    policy_sum = torch.zeros((), dtype=torch.float32, device=device)
    for row in range(len(samples)):
        mask = action_mask[row]
        new_values = chosen_new[row][mask]
        old_values = old_logprobs[row].to(device=device, dtype=new_values.dtype)
        if new_values.numel() != old_values.numel():
            raise RuntimeError("Old/new action-token counts differ")
        ratio = torch.exp((new_values - old_values).clamp(-20.0, 20.0))
        advantage = advantages[row]
        unclipped = ratio * advantage
        clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage
        policy_sum = policy_sum + torch.minimum(unclipped, clipped).sum()
    probability = policy_log_probs.exp()
    token_kl = (probability * (policy_log_probs - reference_log_probs)).sum(dim=-1)
    kl_sum = token_kl[action_mask].sum()
    token_count = int(action_mask.sum().item())
    return policy_sum, kl_sum, token_count


@torch.no_grad()
def estimate_reference_kl(
    policy,
    reference,
    tokenizer,
    samples: list[GeneratedSample],
    temperature: float,
    micro_batch_size: int,
) -> float:
    policy.eval()
    reference.eval()
    total, count = 0.0, 0
    device = str(next(policy.parameters()).device)
    for start in range(0, len(samples), max(1, int(micro_batch_size))):
        chunk = samples[start : start + micro_batch_size]
        input_ids, attention_mask, action_mask = collate_trajectories(
            chunk, tokenizer.pad_token_id, device
        )
        policy_lp = _policy_log_distributions(policy, input_ids, attention_mask, temperature)
        reference_lp = _policy_log_distributions(reference, input_ids, attention_mask, temperature)
        kl = (policy_lp.exp() * (policy_lp - reference_lp)).sum(dim=-1)
        total += float(kl[action_mask].sum().item())
        count += int(action_mask.sum().item())
        del input_ids, attention_mask, action_mask, policy_lp, reference_lp, kl
    return total / max(count, 1)


def snapshot_trainable_update(model, optimizer) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return parameters, copy.deepcopy(optimizer.state_dict())


def restore_trainable_update(
    model,
    optimizer,
    snapshot: tuple[dict[str, torch.Tensor], dict[str, Any]],
) -> None:
    parameters, optimizer_state = snapshot
    current = dict(model.named_parameters())
    missing = sorted(set(parameters) - set(current))
    if missing:
        raise KeyError(f"Cannot restore missing trainable parameters: {missing}")
    with torch.no_grad():
        for name, saved in parameters.items():
            current[name].copy_(saved)
    optimizer.load_state_dict(optimizer_state)
    optimizer.zero_grad(set_to_none=True)


def _nested_state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def verify_trainable_snapshot(
    model,
    optimizer,
    snapshot: tuple[dict[str, torch.Tensor], dict[str, Any]],
) -> tuple[bool, bool]:
    parameters, optimizer_state = snapshot
    current = dict(model.named_parameters())
    parameter_exact = all(
        name in current and torch.equal(current[name].detach(), saved)
        for name, saved in parameters.items()
    )
    optimizer_exact = _nested_state_equal(optimizer.state_dict(), optimizer_state)
    return parameter_exact, optimizer_exact


@torch.no_grad()
def prompt_features(policy, tokenizer, prompts: list[str], max_prompt_length: int) -> torch.Tensor:
    device = next(policy.parameters()).device
    prefixes = [chat_prompt(tokenizer, prompt) for prompt in prompts]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(
            prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        ).to(device)
    finally:
        tokenizer.padding_side = old_side
    embeddings = policy.get_input_embeddings()(encoded.input_ids)
    mask = encoded.attention_mask.unsqueeze(-1).to(embeddings.dtype)
    return ((embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).float().detach()


def peer_standardize(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] <= 1:
        return torch.zeros_like(values)
    peer = (values.sum(dim=1, keepdim=True) - values) / float(values.shape[1] - 1)
    scale = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
    return (values - peer) / scale


def method_credit(
    method: str,
    probabilities: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    gaussian_alpha: float,
    best_of_n: int,
    quadrature_order: int,
    quadrature_max_order: int,
    quadrature_tolerance: float,
    mc_fallback_draws: int,
    robust_epsilon: float = 0.0,
    entropic_beta: float = 1.0,
) -> tuple[torch.Tensor, float, bool, str, dict[str, Any]]:
    if method in {"ev_ppo", "ev_ppo_no_quality", "ev_ppo_no_mean"}:
        if probabilities.shape[1] != int(best_of_n):
            raise ValueError("ev_ppo requires group_size == best_of_n")
        stats = robust_group_statistics(probabilities, robust_epsilon)
        credit = stats["robust_marginal_credit"]
        extras = {
            "robust_epsilon": float(robust_epsilon),
            "robust_ordinal_expected_max": float(
                stats["robust_expected_max"].mean().item()
            ),
            "robust_probability_any_rating_4": float(
                stats["robust_probability_any_top_rating"].mean().item()
            ),
            "ordinal_expected_max": float(stats["nominal_expected_max"].mean().item()),
            "probability_any_rating_4": float(
                stats["nominal_probability_any_top_rating"].mean().item()
            ),
            "candidate_expected_rating_mean": float(
                stats["nominal_expected_rating"].mean().item()
            ),
            "candidate_p4_mean": float(
                stats["nominal_top_rating_probability"].mean().item()
            ),
            "credit_mean": float(credit.mean().item()),
            "credit_max": float(credit.max().item()),
        }
        objective = (
            "robust_ordinal_expected_rating_n1_exact_reduction"
            if int(best_of_n) == 1
            else "stablemax_finite_n_ordinal_expected_max_exact_marginal_credit"
        )
        return credit, float(best_of_n), True, objective, extras
    if method == "nominal_ev_ppo":
        if probabilities.shape[1] != int(best_of_n):
            raise ValueError("nominal_ev_ppo requires group_size == best_of_n")
        stats = exact_group_statistics(probabilities)
        credit = stats["marginal_credit"]
        return (
            credit,
            float(best_of_n),
            True,
            (
                "ordinal_expected_rating_n1_exact_reduction"
                if int(best_of_n) == 1
                else "finite_n_ordinal_expected_max_exact_marginal_credit"
            ),
            {
                "robust_epsilon": float(robust_epsilon),
                "ordinal_expected_max": float(stats["expected_max"].mean().item()),
                "probability_any_rating_4": float(
                    stats["probability_any_top_rating"].mean().item()
                ),
                "credit_mean": float(credit.mean().item()),
                "credit_max": float(credit.max().item()),
            },
        )
    if method == "top4_ppo":
        no_top = 1.0 - probabilities[..., 4]
        credit = torch.empty_like(no_top)
        for candidate in range(no_top.shape[1]):
            opponents = torch.cat(
                (no_top[:, :candidate], no_top[:, candidate + 1 :]), dim=1
            )
            credit[:, candidate] = probabilities[:, candidate, 4] * opponents.prod(dim=1)
        return (
            credit,
            float(best_of_n),
            True,
            "finite_n_probability_any_rating_4_exact_marginal_credit",
            {
                "probability_any_rating_4": float(
                    exact_probability_any_rating(probabilities).mean().item()
                ),
                "credit_mean": float(credit.mean().item()),
                "credit_max": float(credit.max().item()),
            },
        )
    if method == "gaussian_ev_ppo":
        if int(best_of_n) == 1:
            credit = mu.clone()
            diagnostics = [QuadratureDiagnostics(0, 0, 0.0, True, 0) for _ in range(mu.shape[0])]
            objective = "gaussian_mean_reward_n1_reduction"
        elif mc_fallback_draws > 0:
            credit = torch.stack(
                [monte_carlo_rb_credit(row_mu, row_sigma, mc_fallback_draws) for row_mu, row_sigma in zip(mu, sigma)]
            )
            diagnostics = []
            objective = "gaussian_expected_max_mc_ablation"
        else:
            credit, diagnostics = gaussian_max_credit_batch(
                mu,
                sigma,
                initial_order=quadrature_order,
                max_order=quadrature_max_order,
                tolerance=quadrature_tolerance,
            )
            if not all(item.converged for item in diagnostics):
                worst = max(item.relative_residual for item in diagnostics)
                raise RuntimeError(
                    f"Quadrature did not meet tolerance={quadrature_tolerance} by "
                    f"Q_max={quadrature_max_order}; worst relative residual={worst}. "
                    "Increase --quadrature_max_order rather than training on an unchecked credit."
                )
            objective = "gaussian_expected_max_quadrature_ablation"
        extras = {
            "gaussian_alpha": float(gaussian_alpha),
            "quadrature_order_max": max((item.accepted_order for item in diagnostics), default=0),
            "quadrature_residual_max": max((item.relative_residual for item in diagnostics), default=0.0),
            "quadrature_all_converged": all(item.converged for item in diagnostics) if diagnostics else mc_fallback_draws <= 0,
            "credit_mean": float(credit.mean().item()),
            "credit_max": float(credit.max().item()),
        }
        return credit, float(best_of_n), True, objective, extras
    if method == "vanilla_ppo":
        return mu.clone(), 1.0, True, "ordinal_expected_rating_mean", {}
    if method == "vanilla_grpo":
        return peer_standardize(mu), 1.0, False, "ordinal_expected_rating_group_relative", {}
    if method == "entropic_ppo":
        if not math.isfinite(entropic_beta) or entropic_beta <= 0.0:
            raise ValueError("entropic_beta must be finite and positive")
        ratings = torch.arange(
            probabilities.shape[-1],
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        certainty_equivalent = -float(entropic_beta) * torch.log(
            (
                probabilities
                * torch.exp(-ratings / float(entropic_beta)).view(1, 1, -1)
            )
            .sum(dim=-1)
            .clamp_min(EPS)
        )
        return (
            certainty_equivalent,
            1.0,
            True,
            "pessimistic_entropic_ordinal_certainty_equivalent",
            {
                "entropic_beta": float(entropic_beta),
                "entropic_reward_mean": float(certainty_equivalent.mean().item()),
            },
        )
    if method == "scalar_max_ppo":
        credit = scalar_max_marginal_credit(mu)
        return (
            credit,
            float(best_of_n),
            True,
            (
                "scalar_expected_rating_n1_reduction"
                if int(best_of_n) == 1
                else "scalar_max_at_n_exact_marginal_credit"
            ),
            {
                "scalar_group_max": float(mu.max(dim=1).values.mean().item()),
                "credit_mean": float(credit.mean().item()),
                "credit_max": float(credit.max().item()),
            },
        )
    raise ValueError(f"Unknown method: {method}")


def ordinal_stats(
    probabilities: torch.Tensor,
    robust_epsilon: float = 0.0,
) -> dict[str, float]:
    stats = exact_group_statistics(probabilities)
    robust = robust_group_statistics(probabilities, robust_epsilon)
    return {
        "robust_epsilon": float(robust_epsilon),
        "robust_ordinal_expected_max": float(
            robust["robust_expected_max"].mean().item()
        ),
        "robust_probability_any_rating_4": float(
            robust["robust_probability_any_top_rating"].mean().item()
        ),
        "ordinal_expected_max": float(stats["expected_max"].mean().item()),
        "probability_any_rating_4": float(
            stats["probability_any_top_rating"].mean().item()
        ),
        "candidate_expected_rating_mean": float(stats["expected_rating"].mean().item()),
        "candidate_rating_variance_mean": float(stats["rating_variance"].mean().item()),
        "candidate_p4_mean": float(stats["top_rating_probability"].mean().item()),
    }


def _mean_se_ci(values: torch.Tensor) -> tuple[float, float, float, float]:
    values = values.float().reshape(-1).cpu()
    mean = float(values.mean().item())
    if values.numel() <= 1:
        return mean, 0.0, mean, mean
    se = float(values.std(unbiased=True).item() / math.sqrt(values.numel()))
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values.float().reshape(-1), probability).item())


def _cvar(values: torch.Tensor, probability: float) -> float:
    flat = values.float().reshape(-1)
    threshold = torch.quantile(flat, probability)
    tail = flat[flat >= threshold]
    return float(tail.mean().item())


@torch.inference_mode()
def finite_n_ordinal_eval_metrics(
    probabilities: torch.Tensor,
    quality: torch.Tensor | None = None,
    quality_floor: float = math.nan,
    robust_epsilon: float = 0.0,
    mean_floor: float = math.nan,
) -> dict[str, float]:
    probabilities = validate_probabilities(probabilities.float().cpu())
    stats = exact_group_statistics(probabilities)
    robust = robust_group_statistics(probabilities, robust_epsilon)
    expected_max = stats["expected_max"]
    any_top = stats["probability_any_top_rating"]
    robust_expected_max_summary = _mean_se_ci(robust["robust_expected_max"])
    robust_any_top_summary = _mean_se_ci(
        robust["robust_probability_any_top_rating"]
    )
    expected_max_summary = _mean_se_ci(expected_max)
    any_top_summary = _mean_se_ci(any_top)
    expected_rating = stats["expected_rating"]
    selected_index = expected_rating.argmax(dim=1)
    row_index = torch.arange(probabilities.shape[0])
    selected_probability = probabilities[row_index, selected_index]
    selected_expected_rating = expected_rating[row_index, selected_index]
    result = {
        "robust_epsilon": float(robust_epsilon),
        "robust_ordinal_expected_max_mean": robust_expected_max_summary[0],
        "robust_ordinal_expected_max_se": robust_expected_max_summary[1],
        "robust_ordinal_expected_max_ci_low": robust_expected_max_summary[2],
        "robust_ordinal_expected_max_ci_high": robust_expected_max_summary[3],
        "robust_probability_any_rating_4_mean": robust_any_top_summary[0],
        "robust_probability_any_rating_4_se": robust_any_top_summary[1],
        "robust_probability_any_rating_4_ci_low": robust_any_top_summary[2],
        "robust_probability_any_rating_4_ci_high": robust_any_top_summary[3],
        "ordinal_expected_max_mean": expected_max_summary[0],
        "ordinal_expected_max_se": expected_max_summary[1],
        "ordinal_expected_max_ci_low": expected_max_summary[2],
        "ordinal_expected_max_ci_high": expected_max_summary[3],
        "probability_any_rating_4_mean": any_top_summary[0],
        "probability_any_rating_4_se": any_top_summary[1],
        "probability_any_rating_4_ci_low": any_top_summary[2],
        "probability_any_rating_4_ci_high": any_top_summary[3],
        "candidate_expected_rating_mean": float(expected_rating.mean().item()),
        "candidate_rating_variance_mean": float(stats["rating_variance"].mean().item()),
        "candidate_p4_mean": float(probabilities[..., 4].mean().item()),
        "mean_floor": float(mean_floor),
        "candidate_mean_violation_rate": float(
            (expected_rating < mean_floor).float().mean().item()
        ),
        "selected_expected_rating_mean": float(selected_expected_rating.mean().item()),
        "selected_mean_violation_rate": float(
            (selected_expected_rating < mean_floor).float().mean().item()
        ),
        "selected_p4_mean": float(selected_probability[..., 4].mean().item()),
    }
    if quality is not None:
        quality = quality.float().cpu()
        if quality.shape != probabilities.shape[:2]:
            raise ValueError("quality scores must have shape [prompts, N]")
        selected_quality = quality[row_index, selected_index]
        result.update(
            {
                "candidate_quality_mean": float(quality.mean().item()),
                "candidate_quality_violation_rate": float(
                    (quality < quality_floor).float().mean().item()
                ),
                "selected_quality_mean": float(selected_quality.mean().item()),
                "selected_quality_violation_rate": float(
                    (selected_quality < quality_floor).float().mean().item()
                ),
            }
        )
    return result


@torch.no_grad()
def average_response_logprob(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_prompt_length: int,
    max_response_tokens: int,
    temperature: float,
    batch_size: int,
) -> torch.Tensor:
    device = str(next(model.parameters()).device)
    values = []
    for start in range(0, len(prompts), max(1, int(batch_size))):
        samples = []
        for prompt, response in zip(prompts[start : start + batch_size], responses[start : start + batch_size]):
            prefix_ids = tokenizer(
                chat_prompt(tokenizer, prompt),
                truncation=True,
                max_length=max_prompt_length,
                add_special_tokens=False,
            ).input_ids
            response_ids = tokenizer(
                response,
                add_special_tokens=False,
            ).input_ids[:max_response_tokens]
            if not response_ids:
                response_ids = [tokenizer.eos_token_id]
            samples.append(GeneratedSample(prompt, response, [*prefix_ids, *response_ids], len(prefix_ids)))
        input_ids, attention_mask, action_mask = collate_trajectories(samples, tokenizer.pad_token_id, device)
        log_probs = _policy_log_distributions(model, input_ids, attention_mask, temperature)
        chosen = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        numerator = (chosen * action_mask).sum(dim=1)
        denominator = action_mask.sum(dim=1).clamp_min(1)
        values.append((numerator / denominator).cpu())
    return torch.cat(values)


def evaluate_policy_preference(
    model,
    tokenizer,
    eval_path: Path,
    max_pairs: int,
    max_prompt_length: int,
    max_response_tokens: int,
    temperature: float,
    batch_size: int,
    output_dir: Path,
    metric_name: str = "length_normalized_policy_logprob_accuracy_on_human_preference_pairs",
) -> dict[str, Any]:
    pairs = load_preference_pairs(eval_path, max_pairs, tokenizer)
    prompts = [pair["prompt"] for pair in pairs]
    chosen = [pair["chosen"] for pair in pairs]
    rejected = [pair["rejected"] for pair in pairs]
    chosen_lp = average_response_logprob(
        model, tokenizer, prompts, chosen, max_prompt_length, max_response_tokens, temperature, batch_size
    )
    rejected_lp = average_response_logprob(
        model, tokenizer, prompts, rejected, max_prompt_length, max_response_tokens, temperature, batch_size
    )
    margin = chosen_lp - rejected_lp
    detail_rows = []
    for index, (pair, chosen_value, rejected_value, margin_value) in enumerate(
        zip(pairs, chosen_lp.tolist(), rejected_lp.tolist(), margin.tolist())
    ):
        pair_id = hashlib.sha256(
            "\n".join(
                (
                    str(pair["prompt"]),
                    str(pair["chosen"]),
                    str(pair["rejected"]),
                )
            ).encode("utf-8")
        ).hexdigest()
        detail_rows.append(
            {
                "pair_index": index,
                "pair_id": pair_id,
                "chosen_logprob": float(chosen_value),
                "rejected_logprob": float(rejected_value),
                "margin_logprob": float(margin_value),
                "preference_correct": float(margin_value > 0.0),
            }
        )
    result = {
        "num_eval_pairs": len(pairs),
        "accuracy": float((margin > 0.0).float().mean().item()),
        "metric": metric_name,
        "mean_margin_logprob": float(margin.mean().item()),
        "chosen_logprob_mean": float(chosen_lp.mean().item()),
        "rejected_logprob_mean": float(rejected_lp.mean().item()),
        "resolved_policy_checkpoint": getattr(model, "_resolved_checkpoint", ""),
    }
    write_json(output_dir / "policy_preference_summary.json", result)
    write_csv(output_dir / "policy_preference_summary.csv", [result], list(result))
    write_csv(
        output_dir / "policy_preference_details.csv",
        detail_rows,
        (
            "pair_index",
            "pair_id",
            "chosen_logprob",
            "rejected_logprob",
            "margin_logprob",
            "preference_correct",
        ),
    )
    return result


def reward_model_metadata(checkpoint: str | Path) -> dict[str, Any]:
    root = Path(checkpoint)
    moment_path = root / "moment_rm_config.json"
    if moment_path.exists():
        config = json.loads(moment_path.read_text(encoding="utf-8"))
        test_metrics = config.get("test_metrics", {})
        return {
            "reward_model_preference_accuracy": test_metrics.get("helpfulness_pair_accuracy_non_ties", ""),
            "reward_model_preference_pairs": test_metrics.get("helpfulness_pair_non_ties", ""),
            "reward_model_ordinal_nll": test_metrics.get("ordinal_nll", ""),
            "reward_model_helpfulness_brier": test_metrics.get("helpfulness_brier", ""),
            "reward_model_helpfulness_sigma_disagreement_spearman": test_metrics.get(
                "helpfulness_sigma_disagreement_spearman", ""
            ),
            "reward_variance_status": config.get(
                "reward_variance_status", "failed_validation_not_for_policy_optimization"
            ),
            "reward_variance_validation_basis": config.get("reward_variance_validation_basis", ""),
        }
    path = root / "strong_rm_config.json"
    if not path.exists():
        return {}
    config = json.loads(path.read_text(encoding="utf-8"))
    return {
        "reward_model_preference_accuracy": config.get("heldout_test_accuracy", config.get("test_accuracy", "")),
        "reward_model_preference_pairs": config.get("heldout_test_num_pairs", config.get("num_test_pairs", "")),
        "reward_variance_status": "legacy_scalar_model_without_validated_moment_scale",
    }


def quality_constrained_credit(
    credit: torch.Tensor,
    quality_mu: torch.Tensor,
    quality_floor: float,
    dual_value: float,
    gradient_scale: float,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Apply the correctly scaled Lagrangian quality credit.

    The policy code averages candidate scores and later multiplies each
    centered credit by ``gradient_scale``.  For
    C(pi)=E[Q_eta(x,Y)] >= c, the added per-candidate credit is therefore
    lambda Q_eta/gradient_scale.  In EV-PPO, gradient_scale=N, recovering the
    exact q_j + lambda Q_eta/N expression.  The constant lambda*c is handled
    only by the dual update and must not be injected into the score credit.
    """
    if credit.shape != quality_mu.shape:
        raise ValueError(
            f"credit shape {tuple(credit.shape)} does not match quality score shape "
            f"{tuple(quality_mu.shape)}"
        )
    floor = torch.as_tensor(float(quality_floor), dtype=quality_mu.dtype, device=quality_mu.device)
    residual = floor - quality_mu
    if not math.isfinite(gradient_scale) or gradient_scale <= 0.0:
        raise ValueError("gradient_scale must be finite and positive")
    adjusted = credit + (float(dual_value) / float(gradient_scale)) * quality_mu
    return adjusted, {
        "quality_constraint_active": True,
        "quality_score_mean": float(quality_mu.mean().item()),
        "quality_score_min": float(quality_mu.min().item()),
        "quality_constraint_residual": float(residual.mean().item()),
        "quality_violation_rate": float((quality_mu < floor).float().mean().item()),
        "quality_floor": float(quality_floor),
        "quality_dual_used": float(dual_value),
    }


def mean_constrained_credit(
    credit: torch.Tensor,
    nominal_mu: torch.Tensor,
    mean_floor: float,
    dual_value: float,
    gradient_scale: float,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Add the nominal expected-rating non-inferiority Lagrangian credit."""
    if credit.shape != nominal_mu.shape:
        raise ValueError("mean constraint requires credit and nominal_mu to match")
    if not math.isfinite(gradient_scale) or gradient_scale <= 0.0:
        raise ValueError("gradient_scale must be finite and positive")
    floor = torch.as_tensor(
        float(mean_floor), dtype=nominal_mu.dtype, device=nominal_mu.device
    )
    residual = floor - nominal_mu
    adjusted = credit + (float(dual_value) / float(gradient_scale)) * nominal_mu
    return adjusted, {
        "mean_constraint_active": True,
        "nominal_mean_score": float(nominal_mu.mean().item()),
        "nominal_mean_score_min": float(nominal_mu.min().item()),
        "mean_constraint_residual": float(residual.mean().item()),
        "mean_violation_rate": float((nominal_mu < floor).float().mean().item()),
        "mean_floor": float(mean_floor),
        "mean_dual_used": float(dual_value),
    }


def projected_dual_update(
    value: float,
    residual: float,
    learning_rate: float,
    maximum: float,
    accepted_transition: bool,
) -> float:
    """Advance a projected dual only after an accepted policy transition."""
    if not accepted_transition:
        return float(value)
    return min(
        float(maximum),
        max(0.0, float(value) + float(learning_rate) * float(residual)),
    )


def adaptive_kl_update(
    beta: float,
    observed_kl: float,
    target_kl: float,
    high_multiplier: float,
    low_divisor: float,
    minimum: float,
    maximum: float,
    accepted_transition: bool,
) -> float:
    """Advance the adaptive KL coefficient only for an accepted transition."""
    if not accepted_transition:
        return float(beta)
    if observed_kl > target_kl * high_multiplier:
        return min(float(beta) * 2.0, float(maximum))
    if observed_kl < target_kl / low_divisor:
        return max(float(beta) / 2.0, float(minimum))
    return float(beta)


def calibrate_quality_floor(
    quality_rm,
    quality_tokenizer,
    calibration_jsonl: str | Path,
    max_pairs: int,
    max_length: int,
    batch_size: int,
    rm_temperature: float,
    margin_sd: float,
    seed: int,
    bootstrap_alpha: float,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[float, dict[str, float | int | str]]:
    """Calibrate a prompt-cluster lower-confidence non-inferiority floor."""
    path = Path(calibration_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"Missing quality calibration split: {path}")
    rows = read_jsonl(path)
    pairs: list[dict[str, str]] = []
    source_labels: set[str] = set()
    for row in rows:
        source_label = str(row.get("calibration_source", "")).strip()
        if source_label:
            source_labels.add(source_label)
        pair = normalize_preference_pair(
            as_text(row.get("prompt")) or "",
            as_text(row.get("chosen")) or "",
            as_text(row.get("rejected")) or "",
        )
        if pair["prompt"] and pair["chosen"]:
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"No valid chosen responses in quality calibration split: {path}")
    if len(source_labels) > 1:
        raise ValueError(f"Quality calibration mixes source labels: {sorted(source_labels)}")
    rng = random.Random(int(seed) + 104729)
    rng.shuffle(pairs)
    pairs = pairs[: min(len(pairs), max(1, int(max_pairs)))]
    score_chunks = []
    progress_chunk_size = max(1, int(batch_size)) * 32
    print(
        json.dumps(
            {
                "event": "quality_floor_calibration_start",
                "pairs": len(pairs),
                "batch_size": int(batch_size),
                "device": str(quality_rm.device),
            }
        ),
        flush=True,
    )
    for start in range(0, len(pairs), progress_chunk_size):
        chunk = pairs[start : start + progress_chunk_size]
        chunk_scores, _ = score_responses(
            quality_rm,
            quality_tokenizer,
            [pair["prompt"] for pair in chunk],
            [pair["chosen"] for pair in chunk],
            max_length,
            batch_size,
            rm_temperature,
            reward_scale_mode="zero",
        )
        score_chunks.append(chunk_scores)
        print(json.dumps({"event": "quality_floor_calibration_progress", "completed": min(start + len(chunk), len(pairs)), "total": len(pairs)}), flush=True)
    scores = torch.cat(score_chunks)
    lcb, bootstrap = prompt_cluster_bootstrap_lcb(
        [pair["prompt"] for pair in pairs],
        scores,
        bootstrap_alpha,
        bootstrap_draws,
        bootstrap_seed,
    )
    sd = float(scores.std(unbiased=True).item())
    floor = lcb - float(margin_sd) * sd
    return floor, {
        "quality_floor_source": next(iter(source_labels), "frozen_reference_policy_outputs"),
        "quality_floor_calibration_path": str(path),
        "quality_floor_calibration_pairs": int(scores.numel()),
        "quality_floor_calibration_mean": float(scores.mean().item()),
        "quality_floor_calibration_sd": sd,
        "quality_noninferiority_margin_sd": float(margin_sd),
        "quality_floor": floor,
        **bootstrap,
    }


def calibrate_mean_floor(
    ordinal_rm,
    calibration_jsonl: str | Path,
    max_pairs: int,
    max_length: int,
    batch_size: int,
    margin_sd: float,
    seed: int,
    bootstrap_alpha: float,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[float, dict[str, float | int | str]]:
    """Calibrate a prompt-cluster lower bound for nominal expected rating."""
    path = Path(calibration_jsonl)
    rows = read_jsonl(path)
    pairs: list[dict[str, str]] = []
    for row in rows:
        pair = normalize_preference_pair(
            as_text(row.get("prompt")) or "",
            as_text(row.get("chosen")) or "",
            as_text(row.get("rejected")) or "",
        )
        if pair["prompt"] and pair["chosen"]:
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"No base-policy responses for mean-floor calibration: {path}")
    rng = random.Random(int(seed) + 130363)
    rng.shuffle(pairs)
    pairs = pairs[: min(len(pairs), max(1, int(max_pairs)))]
    chunks = []
    progress_chunk_size = max(1, int(batch_size)) * 32
    for start in range(0, len(pairs), progress_chunk_size):
        chunk = pairs[start : start + progress_chunk_size]
        _, values, _ = score_ordinal_responses(
            ordinal_rm,
            [pair["prompt"] for pair in chunk],
            [pair["chosen"] for pair in chunk],
            max_length,
            batch_size,
            1.0,
        )
        chunks.append(values)
    scores = torch.cat(chunks).float()
    lcb, bootstrap = prompt_cluster_bootstrap_lcb(
        [pair["prompt"] for pair in pairs],
        scores,
        bootstrap_alpha,
        bootstrap_draws,
        bootstrap_seed,
    )
    sd = float(scores.std(unbiased=True).item())
    floor = lcb - float(margin_sd) * sd
    return floor, {
        "mean_floor_source": "frozen_base_policy_training_prompt_responses",
        "mean_floor_calibration_path": str(path),
        "mean_floor_calibration_pairs": int(scores.numel()),
        "mean_floor_calibration_mean": float(scores.mean().item()),
        "mean_floor_calibration_sd": sd,
        "mean_noninferiority_margin_sd": float(margin_sd),
        "mean_floor": floor,
        **bootstrap,
    }


def run_floor_calibration(args) -> None:
    output_path = Path(args.output_json)
    mean_cache_path = Path(args.mean_floor_cache)
    quality_cache_path = Path(args.quality_floor_cache)
    existing = [
        str(path)
        for path in (output_path, mean_cache_path, quality_cache_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite frozen floor calibration: {existing}")
    set_seed(floor_calibration_seed(args))
    device, dtype = device_name(), model_dtype()
    rm, _ = load_rm(args.rm_checkpoint, args.base_model, dtype, device)
    if getattr(rm, "reward_variance_status", "") != "aleatoric_conditional_scale":
        raise ValueError("floor calibration requires the accepted ordinal Moment RM")
    if torch.cuda.is_available():
        rm.to(device)
    try:
        mean_floor, mean_metadata = calibrate_mean_floor(
            rm,
            args.mean_calibration_jsonl,
            args.mean_calibration_pairs,
            args.rm_max_length,
            args.rm_batch_size,
            args.mean_noninferiority_margin_sd,
            floor_calibration_seed(args),
            args.floor_bootstrap_alpha,
            args.floor_bootstrap_draws,
            args.floor_bootstrap_seed,
        )
    finally:
        del rm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_spec = mean_floor_cache_spec(args)
    write_mean_floor_cache(
        mean_cache_path, mean_spec, mean_floor, {**mean_metadata, "status": "frozen"}
    )

    quality_rm, quality_tokenizer = load_rm(
        args.quality_rm_checkpoint, args.base_model, dtype, device
    )
    quality_rm.eval()
    if torch.cuda.is_available():
        quality_rm.to(device)
    try:
        quality_floor, quality_metadata = calibrate_quality_floor(
            quality_rm,
            quality_tokenizer,
            args.quality_calibration_jsonl,
            args.quality_calibration_pairs,
            args.quality_rm_max_length,
            args.quality_rm_batch_size,
            args.quality_rm_temperature,
            args.quality_noninferiority_margin_sd,
            floor_calibration_seed(args),
            args.floor_bootstrap_alpha,
            args.floor_bootstrap_draws,
            args.floor_bootstrap_seed,
        )
    finally:
        del quality_rm
        del quality_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    quality_spec = quality_floor_cache_spec(args)
    write_quality_floor_cache(
        quality_cache_path,
        quality_spec,
        quality_floor,
        {**quality_metadata, "status": "frozen"},
    )
    report = {
        "protocol": "stablemax-ppo-publication-v7-prompt-cluster-lcb-floors",
        "floor_calibration_seed": floor_calibration_seed(args),
        "mean_floor": mean_floor,
        "quality_floor": quality_floor,
        "mean_floor_cache": str(mean_cache_path.resolve()),
        "mean_floor_cache_sha256": sha256_file(mean_cache_path),
        "quality_floor_cache": str(quality_cache_path.resolve()),
        "quality_floor_cache_sha256": sha256_file(quality_cache_path),
        "mean_spec": mean_spec,
        "quality_spec": quality_spec,
    }
    write_json(output_path, report)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{sha256_file(output_path)}  {output_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def run_train(args) -> None:
    if args.method not in TRAINABLE_METHODS:
        raise ValueError(f"Method {args.method} is not trainable")
    if args.group_size != args.best_of_n:
        raise ValueError("Training group_size must equal best_of_n for compute-matched finite-N experiments")
    if args.best_of_n < 1:
        raise ValueError("best_of_n must be positive")
    if args.batch_prompts < 1 or args.ppo_epochs < 1 or args.logprob_batch_size < 1:
        raise ValueError("batch_prompts, ppo_epochs, and logprob_batch_size must be positive")
    if args.quadrature_order < 8 or args.quadrature_max_order < args.quadrature_order:
        raise ValueError("quadrature orders must satisfy 8 <= Q0 <= Qmax")
    if not 0.0 <= args.rms_decay < 1.0:
        raise ValueError("rms_decay must be in [0, 1)")
    if abs(args.top_p - 1.0) > 1e-12:
        raise ValueError("top_p must equal 1.0 during policy-gradient training")
    if args.hard_kl_limit < 0.0:
        raise ValueError("hard_kl_limit must be non-negative")
    args.gaussian_alpha = resolve_alpha(args.gaussian_alpha, args.best_of_n, True)
    robust_epsilon, robust_calibration_report = read_robust_calibration_report(
        args.robust_calibration_report
    )
    set_seed(args.seed)
    device, dtype = device_name(), model_dtype()
    records = load_prompt_records(Path(args.train_jsonl), args.max_prompt_samples)
    policy, tokenizer = load_train_policy(args.base_model, args, dtype, device)
    reference = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device).eval()
    force_pad_token(reference, tokenizer)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    rm, _ = load_rm(args.rm_checkpoint, args.base_model, dtype, device)
    fitted_variance_status = getattr(
        rm, "reward_variance_status", "unknown_fitted_reward_scale"
    )
    if fitted_variance_status != "aleatoric_conditional_scale":
        raise ValueError("Robust ordinal v5 requires the accepted repeated-rating Moment RM")
    reward_distribution_status = "calibrated_five_category_ordinal_distribution"
    mean_dual = float(args.mean_dual_init)
    mean_floor = float(args.mean_floor)
    mean_floor_metadata: dict[str, float | int | str] = {
        "mean_floor_source": "explicit_cli_value"
    }
    if args.method != "ev_ppo_no_mean":
        if math.isnan(mean_floor):
            mean_cache_path = (
                Path(args.mean_floor_cache) if args.mean_floor_cache else None
            )
            mean_spec = mean_floor_cache_spec(args)
            cached_mean = (
                read_mean_floor_cache(mean_cache_path, mean_spec)
                if mean_cache_path is not None
                else None
            )
            if cached_mean is not None:
                mean_floor, mean_floor_metadata = cached_mean
            else:
                if torch.cuda.is_available():
                    rm.to(device)
                try:
                    mean_floor, mean_floor_metadata = calibrate_mean_floor(
                        rm,
                        args.mean_calibration_jsonl,
                        args.mean_calibration_pairs,
                        args.rm_max_length,
                        args.rm_batch_size,
                        args.mean_noninferiority_margin_sd,
                        floor_calibration_seed(args),
                        args.floor_bootstrap_alpha,
                        args.floor_bootstrap_draws,
                        args.floor_bootstrap_seed,
                    )
                finally:
                    if torch.cuda.is_available():
                        rm.to("cpu")
                        gc.collect()
                        torch.cuda.empty_cache()
                mean_floor_metadata["mean_floor_cache_status"] = "computed"
                if mean_cache_path is not None:
                    mean_floor_metadata["mean_floor_cache_path"] = str(mean_cache_path)
                    write_mean_floor_cache(mean_cache_path, mean_spec, mean_floor, mean_floor_metadata)
    quality_rm = None
    quality_rm_tokenizer = None
    quality_dual = float(args.quality_dual_init)
    quality_floor = float(args.quality_floor)
    quality_floor_metadata: dict[str, float | int | str] = {
        "quality_floor_source": "explicit_cli_value"
    }
    if args.disable_quality_constraint and args.method != "ev_ppo_no_quality":
        raise ValueError("Only the declared ev_ppo_no_quality ablation may disable Q")
    if args.method != "ev_ppo_no_quality" and not args.disable_quality_constraint:
        if not args.quality_rm_checkpoint:
            raise ValueError(
                f"{args.method} requires --quality_rm_checkpoint unless "
                "--disable_quality_constraint is explicitly set"
            )
        quality_rm, quality_rm_tokenizer = load_rm(
            args.quality_rm_checkpoint, args.base_model, dtype, device
        )
        quality_rm.eval()
        if math.isnan(quality_floor):
            cache_path = Path(args.quality_floor_cache) if args.quality_floor_cache else None
            cache_spec = quality_floor_cache_spec(args)
            cached = read_quality_floor_cache(cache_path, cache_spec)
            if cached is not None:
                quality_floor, quality_floor_metadata = cached
                print(
                    json.dumps(
                        {
                            "event": "quality_floor_cache_reused",
                            "quality_floor": quality_floor,
                            "cache_path": str(cache_path),
                        }
                    ),
                    flush=True,
                )
            else:
                if torch.cuda.is_available():
                    quality_rm.to(device)
                try:
                    quality_floor, quality_floor_metadata = calibrate_quality_floor(
                        quality_rm,
                        quality_rm_tokenizer,
                        args.quality_calibration_jsonl,
                        args.quality_calibration_pairs,
                        args.quality_rm_max_length,
                        args.quality_rm_batch_size,
                        args.quality_rm_temperature,
                        args.quality_noninferiority_margin_sd,
                        floor_calibration_seed(args),
                        args.floor_bootstrap_alpha,
                        args.floor_bootstrap_draws,
                        args.floor_bootstrap_seed,
                    )
                finally:
                    if torch.cuda.is_available():
                        quality_rm.to("cpu")
                        gc.collect()
                        torch.cuda.empty_cache()
                quality_floor_metadata["quality_floor_cache_status"] = "computed"
                if cache_path is not None:
                    quality_floor_metadata["quality_floor_cache_path"] = str(cache_path)
                    write_quality_floor_cache(
                        cache_path, cache_spec, quality_floor, quality_floor_metadata
                    )

    baseline = PromptBaseline(policy.get_input_embeddings().embedding_dim, args.baseline_hidden).to(device)
    policy_optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad], lr=args.lr
    )
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=args.baseline_lr)
    beta = float(args.kl_coef)
    rms_variance = float(args.initial_rms_scale) ** 2
    lagged_scale = max(float(args.initial_rms_scale), EPS)
    metric_rows = []
    accepted_updates = 0

    for step in range(1, args.steps + 1):
        batch = [records[random.randrange(len(records))] for _ in range(args.batch_prompts)]
        prompts = [row["prompt"] for row in batch]
        features = prompt_features(policy, tokenizer, prompts, args.max_prompt_length)
        baseline.eval()
        frozen_baseline = baseline(features).detach().cpu()

        groups = [
            generate_group(
                policy,
                tokenizer,
                prompt,
                args.group_size,
                args.max_prompt_length,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.generation_batch_size,
            )
            for prompt in prompts
        ]
        flat_samples = [sample for group in groups for sample in group]
        flat_prompts = [sample.prompt for sample in flat_samples]
        flat_responses = [sample.response for sample in flat_samples]

        if torch.cuda.is_available():
            rm.to(device)
        probabilities, mu, sigma = score_ordinal_responses(
            rm,
            flat_prompts,
            flat_responses,
            args.rm_max_length,
            args.rm_batch_size,
            args.rm_temperature,
        )
        if torch.cuda.is_available():
            rm.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        mu = mu.view(args.batch_prompts, args.group_size)
        sigma = sigma.view_as(mu)
        probabilities = probabilities.view(args.batch_prompts, args.group_size, NUM_RATINGS)
        quality_mu = None
        quality_constraint_metrics: dict[str, float | bool] = {
            "quality_constraint_active": False,
            "quality_score_mean": math.nan,
            "quality_score_min": math.nan,
            "quality_constraint_residual": math.nan,
            "quality_violation_rate": math.nan,
            "quality_floor": quality_floor,
            "quality_dual_used": 0.0,
        }
        if quality_rm is not None:
            if torch.cuda.is_available():
                quality_rm.to(device)
            quality_mu_flat, _ = score_responses(
                quality_rm,
                quality_rm_tokenizer,
                flat_prompts,
                flat_responses,
                args.quality_rm_max_length,
                args.quality_rm_batch_size,
                args.quality_rm_temperature,
                reward_scale_mode="zero",
            )
            quality_mu = quality_mu_flat.view(args.batch_prompts, args.group_size)
            if torch.cuda.is_available():
                quality_rm.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
        credit, gradient_scale, use_baseline, objective, credit_extras = method_credit(
            args.method,
            probabilities,
            mu,
            sigma,
            args.gaussian_alpha,
            args.best_of_n,
            args.quadrature_order,
            args.quadrature_max_order,
            args.quadrature_tolerance,
            args.mc_fallback_draws,
            robust_epsilon,
            args.entropic_beta,
        )
        mean_constraint_metrics: dict[str, float | bool] = {
            "mean_constraint_active": False,
            "nominal_mean_score": math.nan,
            "nominal_mean_score_min": math.nan,
            "mean_constraint_residual": math.nan,
            "mean_violation_rate": math.nan,
            "mean_floor": mean_floor,
            "mean_dual_used": 0.0,
        }
        if args.method != "ev_ppo_no_mean":
            credit, mean_constraint_metrics = mean_constrained_credit(
                credit,
                mu,
                mean_floor,
                mean_dual,
                gradient_scale,
            )
        if quality_mu is not None:
            credit, quality_constraint_metrics = quality_constrained_credit(
                credit,
                quality_mu,
                quality_floor,
                quality_dual,
                gradient_scale,
            )
        if use_baseline:
            centered = credit - frozen_baseline.unsqueeze(1)
            baseline_target = credit.mean(dim=1)
        else:
            centered = credit
            baseline_target = None
        advantages = (gradient_scale * centered).reshape(-1).to(device).detach()

        old_logprobs = collect_old_token_logprobs(
            policy, tokenizer, flat_samples, args.temperature, args.logprob_batch_size
        )
        total_tokens = sum(int(values.numel()) for values in old_logprobs)
        if total_tokens == 0:
            raise RuntimeError("Generated batch contains no action tokens")

        last_loss = math.nan
        beta_used = beta
        lagged_scale_used = lagged_scale
        mean_dual_used = mean_dual
        quality_dual_used = quality_dual
        attempted_post_update_kl = math.nan
        post_update_kl = math.nan
        epochs_completed = 0
        stopped_early = False
        hard_kl_update_rejected = False
        rollback_verification_performed = False
        rollback_parameters_exact = False
        rollback_optimizer_exact = False
        transition_snapshot = (
            snapshot_trainable_update(policy, policy_optimizer)
            if args.hard_kl_limit > 0.0
            else None
        )
        for epoch in range(args.ppo_epochs):
            policy.train()
            policy_optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            for start in range(0, len(flat_samples), args.logprob_batch_size):
                stop = min(start + args.logprob_batch_size, len(flat_samples))
                policy_sum, kl_sum, _ = token_ppo_terms(
                    policy,
                    reference,
                    tokenizer,
                    flat_samples[start:stop],
                    old_logprobs[start:stop],
                    advantages[start:stop],
                    args.temperature,
                    args.clip_eps,
                )
                loss = (
                    -policy_sum / float(args.batch_prompts * args.group_size)
                    + beta * kl_sum / float(total_tokens)
                ) / (lagged_scale + EPS)
                loss.backward()
                epoch_loss += float(loss.detach().cpu().item())
                del policy_sum, kl_sum, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in policy.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            policy_optimizer.step()
            epochs_completed = epoch + 1
            last_loss = epoch_loss
            post_update_kl = estimate_reference_kl(
                policy,
                reference,
                tokenizer,
                flat_samples,
                args.temperature,
                args.logprob_batch_size,
            )
            attempted_post_update_kl = post_update_kl
            if args.hard_kl_limit > 0.0 and post_update_kl > args.hard_kl_limit:
                if transition_snapshot is None:
                    raise RuntimeError("Hard-KL transition snapshot is missing")
                restore_trainable_update(policy, policy_optimizer, transition_snapshot)
                rollback_verification_performed = True
                rollback_parameters_exact, rollback_optimizer_exact = verify_trainable_snapshot(
                    policy,
                    policy_optimizer,
                    transition_snapshot,
                )
                if not rollback_parameters_exact or not rollback_optimizer_exact:
                    raise RuntimeError(
                        "Hard-KL rollback failed exact parameter or optimizer-state verification"
                    )
                post_update_kl = estimate_reference_kl(
                    policy,
                    reference,
                    tokenizer,
                    flat_samples,
                    args.temperature,
                    args.logprob_batch_size,
                )
                hard_kl_update_rejected = True
                stopped_early = True
                break
            if post_update_kl > args.target_kl * args.kl_early_stop_multiplier:
                stopped_early = True
                break

        controller_kl = max(post_update_kl, attempted_post_update_kl) if hard_kl_update_rejected else post_update_kl
        baseline_loss = 0.0
        accepted_transition = not hard_kl_update_rejected
        beta = adaptive_kl_update(
            beta,
            controller_kl,
            args.target_kl,
            args.kl_high_multiplier,
            args.kl_low_divisor,
            args.kl_coef_min,
            args.kl_coef_max,
            accepted_transition,
        )
        if baseline_target is not None and accepted_transition:
            baseline.train()
            baseline_optimizer.zero_grad(set_to_none=True)
            prediction = baseline(features.detach())
            target = baseline_target.to(device).detach()
            value_loss = F.mse_loss(prediction, target)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), args.max_grad_norm)
            baseline_optimizer.step()
            baseline_loss = float(value_loss.detach().cpu().item())

        if args.method != "ev_ppo_no_mean":
            mean_dual = projected_dual_update(
                mean_dual,
                float(mean_constraint_metrics["mean_constraint_residual"]),
                args.mean_dual_lr,
                args.mean_dual_max,
                accepted_transition,
            )
        if quality_mu is not None:
            quality_dual = projected_dual_update(
                quality_dual,
                float(quality_constraint_metrics["quality_constraint_residual"]),
                args.quality_dual_lr,
                args.quality_dual_max,
                accepted_transition,
            )
        if accepted_transition:
            batch_rms = float(centered.square().mean().item())
            rms_variance = args.rms_decay * rms_variance + (1.0 - args.rms_decay) * batch_rms
            lagged_scale = math.sqrt(rms_variance + args.rms_epsilon)
            accepted_updates += 1
        row = {
            "step": step,
            "method": args.method,
            "best_of_n": args.best_of_n,
            "loss": last_loss,
            "baseline_loss": baseline_loss,
            "adv_mean": float(advantages.mean().item()),
            "adv_std": float(advantages.std(unbiased=False).item()),
            "accepted_transition": accepted_transition,
            "accepted_updates_cumulative": accepted_updates,
            "baseline_update_applied": baseline_target is not None and accepted_transition,
            "lagged_global_rms_scale_used": lagged_scale_used,
            "lagged_global_rms_scale": lagged_scale,
            "reference_kl": post_update_kl,
            "attempted_post_update_kl": attempted_post_update_kl,
            "kl_controller_observation": controller_kl,
            "kl_coefficient_used": beta_used,
            "kl_coefficient_next": beta,
            "ppo_epochs_completed": epochs_completed,
            "target_kl_early_stop": stopped_early,
            "hard_kl_limit": float(args.hard_kl_limit),
            "hard_kl_update_rejected": hard_kl_update_rejected,
            "rollback_verification_performed": rollback_verification_performed,
            "rollback_parameters_exact": rollback_parameters_exact,
            "rollback_optimizer_exact": rollback_optimizer_exact,
            "gradient_norm": float(grad_norm),
            "training_objective": objective,
            "robust_calibration_report": str(args.robust_calibration_report),
            "robust_calibration_report_sha256": sha256_file(
                Path(args.robust_calibration_report)
            ),
            "mean_dual_next": mean_dual,
            "mean_dual_state_before": mean_dual_used,
            "mean_floor_cache": str(args.mean_floor_cache),
            **mean_constraint_metrics,
            "reward_distribution_status": reward_distribution_status,
            "fitted_reward_variance_status": fitted_variance_status,
            "quality_dual_next": quality_dual,
            "quality_dual_state_before": quality_dual_used,
            **quality_constraint_metrics,
            **credit_extras,
            **ordinal_stats(probabilities, robust_epsilon),
        }
        metric_rows.append(row)
        if step == 1 or step % max(1, args.log_every) == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        del old_logprobs, advantages, credit, centered, probabilities, mu, sigma, flat_samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(baseline.state_dict(), output_dir / "prompt_baseline.pt")
    write_json(
        output_dir / "training_state.json",
        {
            "final_kl_coefficient": beta,
            "final_global_rms_scale": lagged_scale,
            "accepted_updates": accepted_updates,
            "hard_kl_rollbacks": sum(
                bool(row["hard_kl_update_rejected"]) for row in metric_rows
            ),
            "reward_distribution_status": reward_distribution_status,
            "fitted_reward_variance_status": fitted_variance_status,
            "quality_constraint_active": bool(quality_rm is not None),
            "quality_rm_checkpoint": args.quality_rm_checkpoint or None,
            "quality_floor": quality_floor,
            "mean_floor": mean_floor,
            "robust_epsilon": robust_epsilon,
            "robust_calibration_report": str(args.robust_calibration_report),
            **mean_floor_metadata,
            "final_mean_dual": mean_dual,
            **quality_floor_metadata,
            "final_quality_dual": quality_dual,
        },
    )
    write_json(output_dir / "train_config.json", vars(args))
    if args.metrics_csv:
        write_csv(Path(args.metrics_csv), metric_rows, list(metric_rows[0]))


def run_eval(args) -> None:
    if args.group_size != args.best_of_n:
        raise ValueError("Evaluation group_size must equal best_of_n")
    if abs(args.top_p - 1.0) > 1e-12:
        raise ValueError("Evaluation uses top_p=1.0 for policy-distribution consistency")
    if args.performance_pairs < 1:
        raise ValueError("performance_pairs must be positive")
    set_seed(args.seed)
    robust_epsilon, robust_calibration_report = read_robust_calibration_report(
        args.robust_calibration_report
    )
    device, dtype = device_name(), model_dtype()
    records = load_prompt_records(Path(args.eval_jsonl), args.max_eval_prompts)
    rm, _ = load_rm(args.rm_checkpoint, args.base_model, dtype, device)
    fitted_variance_status = getattr(
        rm, "reward_variance_status", "unknown_fitted_reward_scale"
    )
    if fitted_variance_status != "aleatoric_conditional_scale":
        raise ValueError("Robust ordinal v5 evaluation requires the accepted Moment RM")
    reward_distribution_status = "calibrated_five_category_ordinal_distribution"
    mean_spec = mean_floor_cache_spec(args)
    mean_cache = read_mean_floor_cache(Path(args.mean_floor_cache), mean_spec)
    if mean_cache is None:
        raise FileNotFoundError("Evaluation requires the frozen mean-floor cache")
    mean_floor, _ = mean_cache
    quality_spec = quality_floor_cache_spec(args)
    quality_cache = read_quality_floor_cache(Path(args.quality_floor_cache), quality_spec)
    if quality_cache is None:
        raise FileNotFoundError("Evaluation requires the frozen quality-floor cache")
    quality_floor, _ = quality_cache
    quality_rm, quality_tokenizer = load_rm(
        args.quality_rm_checkpoint, args.base_model, dtype, device
    )
    # Moment RM and Quality RM are moved onto the GPU one at a time.
    keep_rm = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rm_meta = reward_model_metadata(args.rm_checkpoint)
    summaries = []
    performance_rows = []
    lockbox_rows = []
    named_lockboxes: list[tuple[str, Path]] = []
    for value in args.preference_lockbox:
        if "=" not in value:
            raise ValueError("--preference_lockbox must use name=/path/to/file.jsonl")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name or not path.is_file():
            raise ValueError(f"invalid preference lockbox: {value}")
        named_lockboxes.append((name, path))

    for method in split_methods(args.methods):
        if method not in METHOD_DISPLAY_NAMES:
            raise ValueError(f"Unknown evaluation method: {method}")
        # Reuse the same predeclared random stream for every method so paired
        # prompt comparisons do not depend on evaluation order or resume state.
        set_seed(args.seed)
        checkpoint = None if method == "best_of_n" else Path(args.experiment_root) / f"{method}_seed{args.seed}"
        summary_path = output_dir / f"{method}_seed{args.seed}_summary.json"
        response_path = output_dir / f"{method}_seed{args.seed}_responses.jsonl"
        if args.resume_eval and summary_path.exists() and response_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        policy, tokenizer = load_policy_for_eval(args.base_model, checkpoint, dtype, device)
        preference = evaluate_policy_preference(
            policy,
            tokenizer,
            Path(args.performance_jsonl or args.eval_jsonl),
            args.performance_pairs,
            args.max_prompt_length,
            args.performance_max_response_tokens,
            args.temperature,
            args.performance_batch_size,
            output_dir / "policy_pref_eval" / f"{method}_seed{args.seed}",
            "length_normalized_policy_logprob_accuracy_on_rewardbench_policy_lockbox",
        )
        performance_rows.append(
            {"method": method, "method_name": METHOD_DISPLAY_NAMES[method], "seed": args.seed, **preference}
        )
        for lockbox_name, lockbox_path in named_lockboxes:
            result = evaluate_policy_preference(
                policy,
                tokenizer,
                lockbox_path,
                args.preference_lockbox_pairs,
                args.max_prompt_length,
                args.performance_max_response_tokens,
                args.temperature,
                args.performance_batch_size,
                output_dir
                / "policy_pref_lockboxes"
                / lockbox_name
                / f"{method}_seed{args.seed}",
                f"length_normalized_policy_logprob_accuracy_on_{lockbox_name}",
            )
            lockbox_rows.append(
                {
                    "method": method,
                    "method_name": METHOD_DISPLAY_NAMES[method],
                    "seed": args.seed,
                    "lockbox": lockbox_name,
                    "lockbox_path": str(lockbox_path),
                    "lockbox_sha256": sha256_file(lockbox_path),
                    **result,
                }
            )

        response_rows = []
        all_probabilities, all_quality = [], []
        for prompt_id, record in enumerate(records):
            samples = generate_group(
                policy,
                tokenizer,
                record["prompt"],
                args.group_size,
                args.max_prompt_length,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.generation_batch_size,
            )
            if torch.cuda.is_available() and not keep_rm:
                rm.to(device)
            probabilities, mu, sigma = score_ordinal_responses(
                rm,
                [record["prompt"]] * args.group_size,
                [sample.response for sample in samples],
                args.rm_max_length,
                args.rm_batch_size,
                args.rm_temperature,
            )
            if torch.cuda.is_available() and not keep_rm:
                rm.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
            if torch.cuda.is_available():
                quality_rm.to(device)
            quality, _ = score_responses(
                quality_rm,
                quality_tokenizer,
                [record["prompt"]] * args.group_size,
                [sample.response for sample in samples],
                args.quality_rm_max_length,
                args.quality_rm_batch_size,
                args.quality_rm_temperature,
                reward_scale_mode="zero",
            )
            if torch.cuda.is_available():
                quality_rm.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
            all_probabilities.append(probabilities)
            all_quality.append(quality)
            selected = int(mu.argmax().item())
            response_rows.append(
                {
                    "prompt_id": prompt_id,
                    "method": method,
                    "prompt": record["prompt"],
                    "selected_index": selected,
                    "selected_response": samples[selected].response,
                    "responses": [
                        {
                            "index": index,
                            "text": sample.response,
                            "mu": float(mu[index].item()),
                            "sigma": float(sigma[index].item()),
                            "p0": float(probabilities[index, 0].item()),
                            "p1": float(probabilities[index, 1].item()),
                            "p2": float(probabilities[index, 2].item()),
                            "p3": float(probabilities[index, 3].item()),
                            "p4": float(probabilities[index, 4].item()),
                            "quality": float(quality[index].item()),
                        }
                        for index, sample in enumerate(samples)
                    ],
                }
            )
        probability_matrix = torch.stack(all_probabilities)
        quality_matrix = torch.stack(all_quality)
        ordinal_metrics = finite_n_ordinal_eval_metrics(
            probability_matrix,
            quality_matrix,
            quality_floor,
            robust_epsilon,
            mean_floor,
        )
        response_name = response_path.name
        write_jsonl(response_path, response_rows)
        summary = {
            "method": method,
            "method_name": METHOD_DISPLAY_NAMES[method],
            "seed": args.seed,
            "best_of_n": args.best_of_n,
            "num_eval_prompts": len(records),
            "responses_jsonl": response_name,
            "performance": ordinal_metrics["robust_ordinal_expected_max_mean"],
            "performance_metric": "stablemax_exact_finite_n_ordinal_expected_max",
            "preference_accuracy": preference["accuracy"],
            "preference_accuracy_metric": preference["metric"],
            "num_preference_pairs": preference["num_eval_pairs"],
            "reward_distribution_status": reward_distribution_status,
            "fitted_reward_variance_status": fitted_variance_status,
            "quality_floor": quality_floor,
            "mean_floor": mean_floor,
            "robust_epsilon": robust_epsilon,
            "robust_calibration_report_sha256": sha256_file(
                Path(args.robust_calibration_report)
            ),
            **rm_meta,
            **ordinal_metrics,
        }
        write_json(summary_path, summary)
        summaries.append(summary)
        del policy, tokenizer, probability_matrix, quality_matrix
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(output_dir / "comparison_table_paper.csv", summaries, MAIN_TABLE_FIELDS)
    write_csv(output_dir / "policy_performance.csv", performance_rows, PERFORMANCE_FIELDS)
    if lockbox_rows:
        write_csv(
            output_dir / "policy_preference_lockboxes.csv",
            lockbox_rows,
            (
                "method",
                "method_name",
                "seed",
                "lockbox",
                "lockbox_path",
                "lockbox_sha256",
                "num_eval_pairs",
                "accuracy",
                "metric",
                "mean_margin_logprob",
                "chosen_logprob_mean",
                "rejected_logprob_mean",
                "resolved_policy_checkpoint",
            ),
        )


def run_reference_calibration(args) -> None:
    if args.num_calibration_responses < 1 or args.num_feasibility_responses < 1:
        raise ValueError("Reference calibration response counts must be positive")
    if args.samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")
    if abs(float(args.top_p) - 1.0) > 1e-12:
        raise ValueError("Reference calibration requires top_p=1.0")
    if args.protocol_name not in {
        "reference-policy-same-train-distribution-v3",
        "reference-policy-cluster-lcb-v4",
    }:
        raise ValueError("Unsupported reference calibration protocol")
    outputs = [
        Path(args.calibration_output_jsonl),
        Path(args.feasibility_output_jsonl),
        Path(args.manifest_json),
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite frozen reference calibration: {existing}")

    source_path = Path(args.train_jsonl)
    records = load_prompt_records(source_path, None)
    selector = random.Random(int(args.seed) + 32452843)
    selector.shuffle(records)
    calibration_prompts = math.ceil(
        int(args.num_calibration_responses) / int(args.samples_per_prompt)
    )
    feasibility_prompts = math.ceil(
        int(args.num_feasibility_responses) / int(args.samples_per_prompt)
    )
    required_prompts = calibration_prompts + feasibility_prompts
    if len(records) < required_prompts:
        raise ValueError(
            f"Need {required_prompts} distinct training prompts, found {len(records)}"
        )

    generation_seed = int(args.seed) + 49979687
    set_seed(generation_seed)
    device, dtype = device_name(), model_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    force_pad_token(model, tokenizer)

    def generate_rows(
        selected: list[dict[str, str]],
        target_responses: int,
        split: str,
    ) -> list[dict[str, Any]]:
        generated_rows: list[dict[str, Any]] = []
        for prompt_index, record in enumerate(selected):
            remaining = int(target_responses) - len(generated_rows)
            if remaining <= 0:
                break
            count = min(int(args.samples_per_prompt), remaining)
            samples = generate_group(
                model,
                tokenizer,
                record["prompt"],
                count,
                args.max_prompt_length,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.generation_batch_size,
            )
            for sample_index, sample in enumerate(samples):
                generated_rows.append(
                    {
                        "prompt": sample.prompt,
                        "chosen": sample.response,
                        "rejected": "",
                        "calibration_source": args.protocol_name,
                        "calibration_split": split,
                        "source_prompt_index": prompt_index,
                        "sample_index": sample_index,
                    }
                )
            if (prompt_index + 1) % 16 == 0 or len(generated_rows) == int(target_responses):
                print(
                    json.dumps(
                        {
                            "event": "reference_quality_calibration_progress",
                            "split": split,
                            "responses": len(generated_rows),
                            "target": int(target_responses),
                        }
                    ),
                    flush=True,
                )
        if len(generated_rows) != int(target_responses):
            raise RuntimeError(
                f"Generated {len(generated_rows)} {split} responses; expected {target_responses}"
            )
        return generated_rows

    calibration_rows = generate_rows(
        records[:calibration_prompts], args.num_calibration_responses, "floor"
    )
    feasibility_rows = generate_rows(
        records[calibration_prompts:required_prompts],
        args.num_feasibility_responses,
        "feasibility",
    )
    for path, rows in zip(outputs[:2], (calibration_rows, feasibility_rows)):
        temporary = path.with_name(path.name + ".tmp")
        write_jsonl(temporary, rows)
        temporary.replace(path)

    commit_hash = getattr(model.config, "_commit_hash", None)
    manifest = {
        "protocol": args.protocol_name,
        "base_model": args.base_model,
        "base_model_commit_hash": commit_hash,
        "train_jsonl": str(source_path.resolve()),
        "train_jsonl_sha256": sha256_file(source_path),
        "seed": int(args.seed),
        "selection_seed": int(args.seed) + 32452843,
        "generation_seed": generation_seed,
        "samples_per_prompt": int(args.samples_per_prompt),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_prompt_length": int(args.max_prompt_length),
        "max_new_tokens": int(args.max_new_tokens),
        "calibration_output_jsonl": str(outputs[0].resolve()),
        "calibration_output_sha256": sha256_file(outputs[0]),
        "calibration_responses": len(calibration_rows),
        "feasibility_output_jsonl": str(outputs[1].resolve()),
        "feasibility_output_sha256": sha256_file(outputs[1]),
        "feasibility_responses": len(feasibility_rows),
        "prompt_disjoint_between_floor_and_feasibility": True,
    }
    write_json(outputs[2], manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def run_quality_feasibility(args) -> None:
    output_path = Path(args.output_json)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite quality feasibility report: {output_path}")
    calibration_path = Path(args.calibration_jsonl)
    feasibility_path = Path(args.feasibility_jsonl)
    manifest_path = Path(args.calibration_manifest)
    for path in (calibration_path, feasibility_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing v3 feasibility input: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "reference-policy-same-train-distribution-v3":
        raise ValueError("Reference calibration manifest has the wrong protocol")
    if manifest.get("calibration_output_sha256") != sha256_file(calibration_path):
        raise ValueError("Reference calibration floor hash does not match its manifest")
    if manifest.get("feasibility_output_sha256") != sha256_file(feasibility_path):
        raise ValueError("Reference feasibility hash does not match its manifest")
    calibration_rows = read_jsonl(calibration_path)
    feasibility_rows = read_jsonl(feasibility_path)
    calibration_prompts = {as_text(row.get("prompt")) or "" for row in calibration_rows}
    feasibility_prompts = {as_text(row.get("prompt")) or "" for row in feasibility_rows}
    if calibration_prompts & feasibility_prompts:
        raise ValueError("Reference floor and feasibility prompts are not disjoint")

    device, dtype = device_name(), model_dtype()
    quality_rm, quality_tokenizer = load_rm(
        args.quality_rm_checkpoint, args.base_model, dtype, device
    )
    quality_rm.eval()
    if torch.cuda.is_available():
        quality_rm.to(device)

    def score_rows(rows: list[dict[str, Any]], label: str) -> torch.Tensor:
        prompts = [as_text(row.get("prompt")) or "" for row in rows]
        responses = [as_text(row.get("chosen")) or "" for row in rows]
        if not all(prompts) or not all(responses):
            raise ValueError(f"{label} contains an empty prompt or response")
        chunks = []
        chunk_size = max(1, int(args.batch_size)) * 32
        for start in range(0, len(rows), chunk_size):
            stop = min(start + chunk_size, len(rows))
            values, _ = score_responses(
                quality_rm,
                quality_tokenizer,
                prompts[start:stop],
                responses[start:stop],
                args.max_length,
                args.batch_size,
                args.rm_temperature,
                reward_scale_mode="zero",
            )
            chunks.append(values)
            print(
                json.dumps(
                    {
                        "event": "quality_feasibility_progress",
                        "split": label,
                        "completed": stop,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
        return torch.cat(chunks).float()

    calibration_scores = score_rows(calibration_rows, "floor")
    feasibility_scores = score_rows(feasibility_rows, "feasibility")
    calibration_mean = float(calibration_scores.mean().item())
    calibration_sd = float(calibration_scores.std(unbiased=True).item())
    floor = calibration_mean - float(args.margin_sd) * calibration_sd
    feasibility_mean = float(feasibility_scores.mean().item())
    feasibility_sd = float(feasibility_scores.std(unbiased=True).item())
    feasibility_pass_rate = float((feasibility_scores >= floor).float().mean().item())
    mean_shift_sd = abs(feasibility_mean - calibration_mean) / max(calibration_sd, EPS)
    gates = {
        "feasibility_mean_at_or_above_floor": feasibility_mean >= floor,
        "feasibility_pass_rate": feasibility_pass_rate >= float(args.min_pass_rate),
        "calibration_feasibility_mean_shift": mean_shift_sd <= float(args.max_mean_shift_sd),
    }
    quality_identity = quality_floor_cache_spec(
        argparse.Namespace(
            quality_rm_checkpoint=args.quality_rm_checkpoint,
            quality_calibration_jsonl=args.calibration_jsonl,
            quality_calibration_manifest=args.calibration_manifest,
            quality_calibration_pairs=len(calibration_rows),
            quality_rm_max_length=args.max_length,
            quality_rm_batch_size=args.batch_size,
            quality_rm_temperature=args.rm_temperature,
            quality_noninferiority_margin_sd=args.margin_sd,
            seed=args.seed,
        )
    )
    report = {
        "protocol": "reference-policy-same-train-distribution-v3",
        "gate_basis": "training-only frozen reference-policy responses",
        "calibration_jsonl": str(calibration_path.resolve()),
        "calibration_jsonl_sha256": sha256_file(calibration_path),
        "feasibility_jsonl": str(feasibility_path.resolve()),
        "feasibility_jsonl_sha256": sha256_file(feasibility_path),
        "calibration_manifest": str(manifest_path.resolve()),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "quality_rm_config_sha256": quality_identity["quality_rm_config_sha256"],
        "quality_rm_adapter_sha256": quality_identity["quality_rm_adapter_sha256"],
        "seed": int(args.seed),
        "margin_sd": float(args.margin_sd),
        "min_pass_rate": float(args.min_pass_rate),
        "max_mean_shift_sd": float(args.max_mean_shift_sd),
        "calibration_count": int(calibration_scores.numel()),
        "calibration_mean": calibration_mean,
        "calibration_sd": calibration_sd,
        "quality_floor": floor,
        "feasibility_count": int(feasibility_scores.numel()),
        "feasibility_mean": feasibility_mean,
        "feasibility_sd": feasibility_sd,
        "feasibility_pass_rate": feasibility_pass_rate,
        "calibration_feasibility_mean_shift_sd": mean_shift_sd,
        "gate_checks": gates,
        "gates_passed": all(gates.values()),
    }
    write_json(output_path, report)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{sha256_file(output_path)}  {output_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["gates_passed"]:
        raise RuntimeError(f"Reference-policy quality feasibility gates failed: {gates}")


def run_reference_shift_diagnostic(args) -> None:
    """Describe prompt-disjoint reference-policy shift without selecting a protocol."""
    output_path = Path(args.output_json)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite reference shift report: {output_path}")
    calibration_path = Path(args.calibration_jsonl)
    diagnostic_path = Path(args.diagnostic_jsonl)
    manifest_path = Path(args.calibration_manifest)
    for path in (calibration_path, diagnostic_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing reference shift input: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "reference-policy-cluster-lcb-v4":
        raise ValueError("Reference shift manifest has the wrong protocol")
    if manifest.get("calibration_output_sha256") != sha256_file(calibration_path):
        raise ValueError("Reference floor sample does not match its manifest")
    if manifest.get("feasibility_output_sha256") != sha256_file(diagnostic_path):
        raise ValueError("Reference shift sample does not match its manifest")
    calibration_rows = read_jsonl(calibration_path)
    diagnostic_rows = read_jsonl(diagnostic_path)
    calibration_prompts = {as_text(row.get("prompt")) or "" for row in calibration_rows}
    diagnostic_prompts = {as_text(row.get("prompt")) or "" for row in diagnostic_rows}
    if "" in calibration_prompts or "" in diagnostic_prompts:
        raise ValueError("Reference shift inputs contain an empty prompt")
    if calibration_prompts & diagnostic_prompts:
        raise ValueError("Reference floor and shift-diagnostic prompts are not disjoint")

    def texts(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        prompts = [as_text(row.get("prompt")) or "" for row in rows]
        responses = [as_text(row.get("chosen")) or "" for row in rows]
        if not all(prompts) or not all(responses):
            raise ValueError("Reference shift inputs contain an empty response")
        return prompts, responses

    def summarize(prompts: list[str], scores: torch.Tensor) -> dict[str, Any]:
        values = torch.as_tensor(scores, dtype=torch.float64).flatten().cpu()
        grouped: dict[str, list[float]] = {}
        for prompt, value in zip(prompts, values.tolist()):
            grouped.setdefault(" ".join(prompt.split()), []).append(float(value))
        cluster_means = torch.tensor(
            [statistics.fmean(grouped[key]) for key in sorted(grouped)],
            dtype=torch.float64,
        )
        return {
            "responses": int(values.numel()),
            "prompt_clusters": int(cluster_means.numel()),
            "response_mean": float(values.mean().item()),
            "response_sd": float(values.std(unbiased=True).item()),
            "prompt_mean": float(cluster_means.mean().item()),
            "prompt_mean_sd": float(cluster_means.std(unbiased=True).item()),
            "min_responses_per_prompt": min(len(value) for value in grouped.values()),
            "max_responses_per_prompt": max(len(value) for value in grouped.values()),
        }

    calibration_prompt_list, calibration_responses = texts(calibration_rows)
    diagnostic_prompt_list, diagnostic_responses = texts(diagnostic_rows)
    device, dtype = device_name(), model_dtype()

    ordinal_rm, _ = load_rm(args.rm_checkpoint, args.base_model, dtype, device)
    if torch.cuda.is_available():
        ordinal_rm.to(device)
    try:
        _, calibration_mean_scores, _ = score_ordinal_responses(
            ordinal_rm,
            calibration_prompt_list,
            calibration_responses,
            args.rm_max_length,
            args.rm_batch_size,
            1.0,
        )
        _, diagnostic_mean_scores, _ = score_ordinal_responses(
            ordinal_rm,
            diagnostic_prompt_list,
            diagnostic_responses,
            args.rm_max_length,
            args.rm_batch_size,
            1.0,
        )
    finally:
        del ordinal_rm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    quality_rm, quality_tokenizer = load_rm(
        args.quality_rm_checkpoint, args.base_model, dtype, device
    )
    quality_rm.eval()
    if torch.cuda.is_available():
        quality_rm.to(device)

    def quality_scores(prompts: list[str], responses: list[str]) -> torch.Tensor:
        chunks = []
        chunk_size = max(1, int(args.quality_rm_batch_size)) * 32
        for start in range(0, len(prompts), chunk_size):
            stop = min(start + chunk_size, len(prompts))
            values, _ = score_responses(
                quality_rm,
                quality_tokenizer,
                prompts[start:stop],
                responses[start:stop],
                args.quality_rm_max_length,
                args.quality_rm_batch_size,
                args.quality_rm_temperature,
                reward_scale_mode="zero",
            )
            chunks.append(values.float())
        return torch.cat(chunks)

    try:
        calibration_quality_scores = quality_scores(
            calibration_prompt_list, calibration_responses
        )
        diagnostic_quality_scores = quality_scores(
            diagnostic_prompt_list, diagnostic_responses
        )
    finally:
        del quality_rm
        del quality_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_calibration = summarize(calibration_prompt_list, calibration_mean_scores)
    mean_diagnostic = summarize(diagnostic_prompt_list, diagnostic_mean_scores)
    quality_calibration = summarize(
        calibration_prompt_list, calibration_quality_scores
    )
    quality_diagnostic = summarize(
        diagnostic_prompt_list, diagnostic_quality_scores
    )

    def shift(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
        difference = float(right["response_mean"]) - float(left["response_mean"])
        standardized = difference / max(float(left["response_sd"]), EPS)
        return {
            "diagnostic_minus_calibration_mean": difference,
            "signed_shift_in_calibration_response_sd": standardized,
            "absolute_shift_in_calibration_response_sd": abs(standardized),
        }

    report = {
        "protocol": "reference-policy-prompt-disjoint-shift-diagnostic-v1",
        "selection_criterion": False,
        "scientific_success_gate": False,
        "interpretation": (
            "report-only finite-sample distribution-shift diagnostic; it neither "
            "defines the floor nor accepts or rejects policy training"
        ),
        "calibration_jsonl": str(calibration_path.resolve()),
        "calibration_jsonl_sha256": sha256_file(calibration_path),
        "diagnostic_jsonl": str(diagnostic_path.resolve()),
        "diagnostic_jsonl_sha256": sha256_file(diagnostic_path),
        "calibration_manifest": str(manifest_path.resolve()),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "ordinal_expected_rating": {
            "calibration": mean_calibration,
            "diagnostic": mean_diagnostic,
            "shift": shift(mean_calibration, mean_diagnostic),
        },
        "independent_quality_rm": {
            "calibration": quality_calibration,
            "diagnostic": quality_diagnostic,
            "shift": shift(quality_calibration, quality_diagnostic),
        },
    }
    write_json(output_path, report)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{sha256_file(output_path)}  {output_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def run_ordinal_gate(args) -> None:
    """Validate the exported top-rating probability before policy training."""
    if not 0.0 < float(args.calibration_alpha) < 1.0:
        raise ValueError("calibration_alpha must lie strictly between zero and one")
    if not 0.0 <= float(args.max_robust_epsilon) <= 1.0:
        raise ValueError("max_robust_epsilon must lie in [0, 1]")
    output_path = Path(args.output_json)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen ordinal gate: {output_path}")
    train_rows = read_jsonl(Path(args.train_jsonl))

    def annotation_labels(rows: list[dict[str, Any]]) -> list[int]:
        labels: list[int] = []
        for row in rows:
            for side in ("chosen_annotations", "rejected_annotations"):
                annotations = row.get(side) or {}
                labels.extend(int(value) for value in annotations.get("helpfulness", []))
        return labels

    train_labels = annotation_labels(train_rows)
    if not train_labels:
        raise ValueError("Training split has no repeated helpfulness ratings")
    climatology_p4 = sum(value == 4 for value in train_labels) / len(train_labels)
    device, dtype = device_name(), model_dtype()
    rm, _ = load_rm(args.rm_checkpoint, args.base_model, dtype, device)

    def evaluate(path: Path, split: str) -> dict[str, float | int | str]:
        rows = read_jsonl(path)
        prompts: list[str] = []
        responses: list[str] = []
        labels_by_response: list[list[int]] = []
        clusters_by_response: list[str] = []
        for row in rows:
            prompt = as_text(row.get("prompt")) or ""
            for response_key, annotation_key in (
                ("chosen", "chosen_annotations"),
                ("rejected", "rejected_annotations"),
            ):
                response = as_text(row.get(response_key))
                labels = [
                    int(value)
                    for value in (row.get(annotation_key) or {}).get("helpfulness", [])
                ]
                if response and labels:
                    prompts.append(prompt)
                    responses.append(response)
                    labels_by_response.append(labels)
                    clusters_by_response.append(prompt)
        predicted: list[float] = []
        outcomes: list[float] = []
        cluster_threshold_residuals: dict[str, list[list[float]]] = {}
        if torch.cuda.is_available():
            rm.to(device)
        try:
            chunk_size = max(1, int(args.batch_size)) * 16
            for start in range(0, len(responses), chunk_size):
                stop = min(start + chunk_size, len(responses))
                probabilities, _, _ = score_ordinal_responses(
                    rm,
                    prompts[start:stop],
                    responses[start:stop],
                    args.max_length,
                    args.batch_size,
                    1.0,
                )
                cdf = probabilities.cumsum(dim=-1)
                for row_probability, row_cdf, labels, cluster in zip(
                    probabilities[:, 4].tolist(),
                    cdf[:, :-1].tolist(),
                    labels_by_response[start:stop],
                    clusters_by_response[start:stop],
                ):
                    predicted.extend([float(row_probability)] * len(labels))
                    outcomes.extend(float(label == 4) for label in labels)
                    cluster_values = cluster_threshold_residuals.setdefault(
                        cluster, [[] for _ in range(NUM_RATINGS - 1)]
                    )
                    for threshold in range(NUM_RATINGS - 1):
                        observed = [float(label <= threshold) for label in labels]
                        cluster_values[threshold].extend(
                            value - float(row_cdf[threshold]) for value in observed
                        )
        finally:
            if torch.cuda.is_available():
                rm.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
        probability = torch.tensor(predicted, dtype=torch.float64).clamp(1e-7, 1.0 - 1e-7)
        outcome = torch.tensor(outcomes, dtype=torch.float64)
        brier = float((probability - outcome).square().mean().item())
        log_loss = float(
            -(outcome * probability.log() + (1.0 - outcome) * (1.0 - probability).log())
            .mean()
            .item()
        )
        ece = 0.0
        for index in range(10):
            lower, upper = index / 10.0, (index + 1) / 10.0
            selected = (probability >= lower) & (
                probability <= upper if index == 9 else probability < upper
            )
            if selected.any():
                ece += float(selected.float().mean().item()) * abs(
                    float(probability[selected].mean().item())
                    - float(outcome[selected].mean().item())
                )
        if not cluster_threshold_residuals:
            raise ValueError(f"{split} split has no prompt clusters")
        clustered = torch.tensor(
            [
                [statistics.fmean(values) for values in threshold_values]
                for _, threshold_values in sorted(cluster_threshold_residuals.items())
            ],
            dtype=torch.float64,
        )
        robust_epsilon, threshold_residuals, concentration = (
            clustered_one_sided_calibration_radius(
                clustered,
                args.calibration_alpha,
            )
        )
        return {
            "split": split,
            "num_responses": len(responses),
            "num_individual_ratings": len(outcomes),
            "num_prompt_clusters": int(clustered.shape[0]),
            "calibration_unit": "prompt_cluster",
            "observed_p4_prevalence": float(outcome.mean().item()),
            "predicted_p4_mean": float(probability.mean().item()),
            "p4_brier": brier,
            "p4_log_loss": log_loss,
            "p4_ece_10bin": ece,
            "one_sided_cdf_mean_residual_by_threshold": threshold_residuals,
            "one_sided_cdf_concentration": concentration,
            "robust_epsilon": robust_epsilon,
        }

    validation = evaluate(Path(args.validation_jsonl), "validation")
    confirmation = evaluate(Path(args.confirmation_jsonl), "confirmation")
    climatology_brier = climatology_p4 * (1.0 - climatology_p4)
    clipped = min(max(climatology_p4, 1e-7), 1.0 - 1e-7)
    climatology_log_loss = -(
        climatology_p4 * math.log(clipped)
        + (1.0 - climatology_p4) * math.log(1.0 - clipped)
    )
    gates = {
        "validation_p4_brier_beats_training_climatology": float(validation["p4_brier"])
        < climatology_brier,
        "validation_p4_log_loss_beats_training_climatology": float(validation["p4_log_loss"])
        < climatology_log_loss,
        "validation_p4_ece": float(validation["p4_ece_10bin"]) <= float(args.max_p4_ece),
        "validation_robust_epsilon_non_degenerate": float(validation["robust_epsilon"])
        <= float(args.max_robust_epsilon),
    }
    report = {
        "protocol": "ordinal-v5-robust-calibration-gate-before-policy-training",
        "acceptance_basis": "validation only; confirmation is report-only",
        "coverage_scope": (
            "simultaneous population-average one-sided threshold calibration; "
            "not pointwise conditional coverage"
        ),
        "train_jsonl_sha256": sha256_file(Path(args.train_jsonl)),
        "validation_jsonl_sha256": sha256_file(Path(args.validation_jsonl)),
        "confirmation_jsonl_sha256": sha256_file(Path(args.confirmation_jsonl)),
        "rm_config_sha256": sha256_file(Path(args.rm_checkpoint) / "moment_rm_config.json"),
        "rm_adapter_sha256": sha256_file(Path(args.rm_checkpoint) / "adapter_model.safetensors"),
        "training_p4_climatology": climatology_p4,
        "training_climatology_p4_brier": climatology_brier,
        "training_climatology_p4_log_loss": climatology_log_loss,
        "max_p4_ece": float(args.max_p4_ece),
        "calibration_alpha": float(args.calibration_alpha),
        "max_robust_epsilon": float(args.max_robust_epsilon),
        "robust_epsilon": float(validation["robust_epsilon"]),
        "validation": validation,
        "confirmation": confirmation,
        "gate_checks": gates,
        "gates_passed": all(gates.values()),
    }
    write_json(output_path, report)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{sha256_file(output_path)}  {output_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["gates_passed"]:
        raise RuntimeError(f"Ordinal robust calibration gates failed: {gates}")


def run_table(args) -> None:
    rows = []
    input_dir = Path(args.input_dir)
    wanted = split_methods(args.methods)
    for method in wanted:
        for path in sorted(input_dir.glob(f"{method}_seed*_summary.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if args.require_performance and row.get("ordinal_expected_max_mean") in (None, ""):
                raise ValueError(f"Missing J_BoN metric in {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No summaries found under {input_dir}")
    order = {method: index for index, method in enumerate(TABLE_METHOD_ORDER)}
    rows.sort(key=lambda row: (order.get(row.get("method", ""), 999), int(row.get("seed", 0))))
    write_csv(Path(args.output_csv), rows, MAIN_TABLE_FIELDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--method", required=True, choices=sorted(TRAINABLE_METHODS))
    train.add_argument("--base_model", required=True)
    train.add_argument("--rm_checkpoint", required=True)
    train.add_argument("--train_jsonl", required=True)
    train.add_argument("--output_dir", required=True)
    train.add_argument("--metrics_csv", default="")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--floor_calibration_seed", type=int, default=None)
    train.add_argument("--floor_bootstrap_alpha", type=float, default=0.05)
    train.add_argument("--floor_bootstrap_draws", type=int, default=10000)
    train.add_argument("--floor_bootstrap_seed", type=int, default=20260805)
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--batch_prompts", type=int, default=1)
    train.add_argument("--group_size", type=int, default=32)
    train.add_argument("--best_of_n", type=int, default=32)
    train.add_argument("--max_new_tokens", type=int, default=64)
    train.add_argument("--max_prompt_length", type=int, default=384)
    train.add_argument("--rm_max_length", type=int, default=512)
    train.add_argument("--rm_temperature", type=float, default=1.0)
    train.add_argument("--rm_batch_size", type=int, default=1)
    train.add_argument("--gaussian_alpha", type=float, default=None)
    train.add_argument("--robust_calibration_report", required=True)
    train.add_argument("--mean_calibration_jsonl", default="")
    train.add_argument("--mean_calibration_manifest", default="")
    train.add_argument("--mean_calibration_pairs", type=int, default=512)
    train.add_argument("--mean_floor", type=float, default=math.nan)
    train.add_argument("--mean_floor_cache", default="")
    train.add_argument("--mean_noninferiority_margin_sd", type=float, default=0.1)
    train.add_argument("--mean_dual_init", type=float, default=1.0)
    train.add_argument("--mean_dual_lr", type=float, default=0.05)
    train.add_argument("--mean_dual_max", type=float, default=20.0)
    train.add_argument("--quality_rm_checkpoint", default="")
    train.add_argument("--quality_floor", type=float, default=math.nan)
    train.add_argument("--quality_calibration_jsonl", default="")
    train.add_argument("--quality_floor_cache", default="")
    train.add_argument("--quality_calibration_manifest", default="")
    train.add_argument("--quality_calibration_pairs", type=int, default=512)
    train.add_argument("--quality_noninferiority_margin_sd", type=float, default=0.1)
    train.add_argument("--quality_dual_init", type=float, default=1.0)
    train.add_argument("--quality_dual_lr", type=float, default=0.05)
    train.add_argument("--quality_dual_max", type=float, default=20.0)
    train.add_argument("--quality_rm_max_length", type=int, default=1024)
    train.add_argument("--quality_rm_batch_size", type=int, default=1)
    train.add_argument("--quality_rm_temperature", type=float, default=1.0)
    train.add_argument("--disable_quality_constraint", action="store_true")
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument("--lora_r", type=int, default=8)
    train.add_argument("--lora_alpha", type=int, default=32)
    train.add_argument("--ppo_epochs", type=int, default=1)
    train.add_argument("--clip_eps", type=float, default=0.2)
    train.add_argument("--max_grad_norm", type=float, default=1.0)
    train.add_argument("--kl_coef", type=float, default=0.02)
    train.add_argument("--kl_coef_min", type=float, default=1e-5)
    train.add_argument("--kl_coef_max", type=float, default=10.0)
    train.add_argument("--target_kl", type=float, default=0.02)
    train.add_argument("--kl_high_multiplier", type=float, default=1.5)
    train.add_argument("--kl_low_divisor", type=float, default=1.5)
    train.add_argument("--kl_early_stop_multiplier", type=float, default=2.0)
    train.add_argument("--hard_kl_limit", type=float, default=0.0)
    train.add_argument("--baseline_hidden", type=int, default=128)
    train.add_argument("--baseline_lr", type=float, default=1e-3)
    train.add_argument("--rms_decay", type=float, default=0.99)
    train.add_argument("--rms_epsilon", type=float, default=1e-8)
    train.add_argument("--initial_rms_scale", type=float, default=1.0)
    train.add_argument("--quadrature_order", type=int, default=64)
    train.add_argument("--quadrature_max_order", type=int, default=1024)
    train.add_argument("--quadrature_tolerance", type=float, default=1e-6)
    train.add_argument("--mc_fallback_draws", type=int, default=0)
    train.add_argument("--entropic_beta", type=float, default=1.0)
    train.add_argument("--max_prompt_samples", type=int, default=12000)
    train.add_argument("--temperature", type=float, default=1.0)
    train.add_argument("--top_p", type=float, default=1.0)
    train.add_argument("--generation_batch_size", type=int, default=2)
    train.add_argument("--logprob_batch_size", type=int, default=1)
    train.add_argument("--log_every", type=int, default=10)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--methods", required=True)
    evaluate.add_argument("--base_model", required=True)
    evaluate.add_argument("--rm_checkpoint", required=True)
    evaluate.add_argument("--eval_jsonl", required=True)
    evaluate.add_argument("--performance_jsonl", default="")
    evaluate.add_argument("--experiment_root", required=True)
    evaluate.add_argument("--output_dir", required=True)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--floor_calibration_seed", type=int, default=None)
    evaluate.add_argument("--floor_bootstrap_alpha", type=float, default=0.05)
    evaluate.add_argument("--floor_bootstrap_draws", type=int, default=10000)
    evaluate.add_argument("--floor_bootstrap_seed", type=int, default=20260805)
    evaluate.add_argument("--group_size", type=int, default=32)
    evaluate.add_argument("--best_of_n", type=int, default=32)
    evaluate.add_argument("--max_new_tokens", type=int, default=64)
    evaluate.add_argument("--max_prompt_length", type=int, default=384)
    evaluate.add_argument("--rm_max_length", type=int, default=512)
    evaluate.add_argument("--rm_temperature", type=float, default=1.0)
    evaluate.add_argument("--rm_batch_size", type=int, default=1)
    evaluate.add_argument("--robust_calibration_report", required=True)
    evaluate.add_argument("--mean_calibration_jsonl", required=True)
    evaluate.add_argument("--mean_calibration_manifest", required=True)
    evaluate.add_argument("--mean_calibration_pairs", type=int, default=512)
    evaluate.add_argument("--mean_floor_cache", required=True)
    evaluate.add_argument("--mean_noninferiority_margin_sd", type=float, default=0.1)
    evaluate.add_argument("--quality_rm_checkpoint", required=True)
    evaluate.add_argument("--quality_floor_cache", required=True)
    evaluate.add_argument("--quality_calibration_jsonl", required=True)
    evaluate.add_argument("--quality_calibration_manifest", required=True)
    evaluate.add_argument("--quality_calibration_pairs", type=int, default=512)
    evaluate.add_argument("--quality_noninferiority_margin_sd", type=float, default=0.1)
    evaluate.add_argument("--quality_rm_max_length", type=int, default=1024)
    evaluate.add_argument("--quality_rm_batch_size", type=int, default=1)
    evaluate.add_argument("--quality_rm_temperature", type=float, default=1.0)
    evaluate.add_argument("--max_eval_prompts", type=int, default=256)
    evaluate.add_argument("--temperature", type=float, default=1.0)
    evaluate.add_argument("--top_p", type=float, default=1.0)
    evaluate.add_argument("--generation_batch_size", type=int, default=2)
    evaluate.add_argument("--performance_pairs", type=int, default=1373)
    evaluate.add_argument("--performance_batch_size", type=int, default=1)
    evaluate.add_argument("--performance_max_response_tokens", type=int, default=512)
    evaluate.add_argument("--preference_lockbox", action="append", default=[])
    evaluate.add_argument("--preference_lockbox_pairs", type=int, default=1024)
    evaluate.add_argument("--resume_eval", action="store_true")

    floors = subparsers.add_parser("floor-calibration")
    floors.add_argument("--base_model", required=True)
    floors.add_argument("--rm_checkpoint", required=True)
    floors.add_argument("--quality_rm_checkpoint", required=True)
    floors.add_argument("--mean_calibration_jsonl", required=True)
    floors.add_argument("--mean_calibration_manifest", required=True)
    floors.add_argument("--quality_calibration_jsonl", required=True)
    floors.add_argument("--quality_calibration_manifest", required=True)
    floors.add_argument("--mean_floor_cache", required=True)
    floors.add_argument("--quality_floor_cache", required=True)
    floors.add_argument("--output_json", required=True)
    floors.add_argument("--seed", type=int, default=42)
    floors.add_argument("--floor_calibration_seed", type=int, default=20260804)
    floors.add_argument("--floor_bootstrap_alpha", type=float, default=0.05)
    floors.add_argument("--floor_bootstrap_draws", type=int, default=10000)
    floors.add_argument("--floor_bootstrap_seed", type=int, default=20260805)
    floors.add_argument("--mean_calibration_pairs", type=int, default=512)
    floors.add_argument("--quality_calibration_pairs", type=int, default=512)
    floors.add_argument("--rm_max_length", type=int, default=512)
    floors.add_argument("--rm_batch_size", type=int, default=1)
    floors.add_argument("--mean_noninferiority_margin_sd", type=float, default=0.1)
    floors.add_argument("--quality_rm_max_length", type=int, default=1024)
    floors.add_argument("--quality_rm_batch_size", type=int, default=1)
    floors.add_argument("--quality_rm_temperature", type=float, default=1.0)
    floors.add_argument("--quality_noninferiority_margin_sd", type=float, default=0.1)

    reference_calibration = subparsers.add_parser("reference-calibration")
    reference_calibration.add_argument("--base_model", required=True)
    reference_calibration.add_argument("--train_jsonl", required=True)
    reference_calibration.add_argument("--calibration_output_jsonl", required=True)
    reference_calibration.add_argument("--feasibility_output_jsonl", required=True)
    reference_calibration.add_argument("--manifest_json", required=True)
    reference_calibration.add_argument("--seed", type=int, default=42)
    reference_calibration.add_argument("--num_calibration_responses", type=int, default=512)
    reference_calibration.add_argument("--num_feasibility_responses", type=int, default=256)
    reference_calibration.add_argument("--samples_per_prompt", type=int, default=4)
    reference_calibration.add_argument("--max_prompt_length", type=int, default=384)
    reference_calibration.add_argument("--max_new_tokens", type=int, default=64)
    reference_calibration.add_argument("--temperature", type=float, default=1.0)
    reference_calibration.add_argument("--top_p", type=float, default=1.0)
    reference_calibration.add_argument("--generation_batch_size", type=int, default=2)
    reference_calibration.add_argument(
        "--protocol_name",
        choices=(
            "reference-policy-same-train-distribution-v3",
            "reference-policy-cluster-lcb-v4",
        ),
        default="reference-policy-same-train-distribution-v3",
    )

    feasibility = subparsers.add_parser("quality-feasibility")
    feasibility.add_argument("--base_model", required=True)
    feasibility.add_argument("--quality_rm_checkpoint", required=True)
    feasibility.add_argument("--calibration_jsonl", required=True)
    feasibility.add_argument("--feasibility_jsonl", required=True)
    feasibility.add_argument("--calibration_manifest", required=True)
    feasibility.add_argument("--output_json", required=True)
    feasibility.add_argument("--seed", type=int, default=42)
    feasibility.add_argument("--margin_sd", type=float, default=0.1)
    feasibility.add_argument("--min_pass_rate", type=float, default=0.25)
    feasibility.add_argument("--max_mean_shift_sd", type=float, default=0.25)
    feasibility.add_argument("--max_length", type=int, default=1024)
    feasibility.add_argument("--batch_size", type=int, default=1)
    feasibility.add_argument("--rm_temperature", type=float, default=1.0)

    shift = subparsers.add_parser("reference-shift-diagnostic")
    shift.add_argument("--base_model", required=True)
    shift.add_argument("--rm_checkpoint", required=True)
    shift.add_argument("--quality_rm_checkpoint", required=True)
    shift.add_argument("--calibration_jsonl", required=True)
    shift.add_argument("--diagnostic_jsonl", required=True)
    shift.add_argument("--calibration_manifest", required=True)
    shift.add_argument("--output_json", required=True)
    shift.add_argument("--rm_max_length", type=int, default=512)
    shift.add_argument("--rm_batch_size", type=int, default=1)
    shift.add_argument("--quality_rm_max_length", type=int, default=1024)
    shift.add_argument("--quality_rm_batch_size", type=int, default=1)
    shift.add_argument("--quality_rm_temperature", type=float, default=1.0)

    ordinal_gate = subparsers.add_parser("ordinal-gate")
    ordinal_gate.add_argument("--base_model", required=True)
    ordinal_gate.add_argument("--rm_checkpoint", required=True)
    ordinal_gate.add_argument("--train_jsonl", required=True)
    ordinal_gate.add_argument("--validation_jsonl", required=True)
    ordinal_gate.add_argument("--confirmation_jsonl", required=True)
    ordinal_gate.add_argument("--output_json", required=True)
    ordinal_gate.add_argument("--max_length", type=int, default=1024)
    ordinal_gate.add_argument("--batch_size", type=int, default=1)
    ordinal_gate.add_argument("--max_p4_ece", type=float, default=0.10)
    ordinal_gate.add_argument("--calibration_alpha", type=float, default=0.05)
    ordinal_gate.add_argument("--max_robust_epsilon", type=float, default=0.25)

    table = subparsers.add_parser("table")
    table.add_argument("--input_dir", required=True)
    table.add_argument("--output_csv", required=True)
    table.add_argument("--methods", default=",".join(TABLE_METHOD_ORDER))
    table.add_argument("--require_performance", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "floor-calibration":
        run_floor_calibration(args)
    elif args.cmd == "reference-calibration":
        run_reference_calibration(args)
    elif args.cmd == "quality-feasibility":
        run_quality_feasibility(args)
    elif args.cmd == "reference-shift-diagnostic":
        run_reference_shift_diagnostic(args)
    elif args.cmd == "ordinal-gate":
        run_ordinal_gate(args)
    elif args.cmd == "eval":
        run_eval(args)
    else:
        run_table(args)


if __name__ == "__main__":
    main()
