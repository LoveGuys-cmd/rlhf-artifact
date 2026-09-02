#!/usr/bin/env python3
"""Build reward and independent preference-proxy publication tables.

The reward table contains only policy/reward diagnostics. The separate
preference-proxy table evaluates deterministic reward-ranked Best-of-N
responses selected by argmax of the ordinal expected rating. Model judges are
reported as proxies and never relabeled as human preference.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


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
METHOD_NAMES = {
    "ev_ppo": "StableMax-PPO + Mean/Q",
    "vanilla_ppo": "Mean-PPO + Q",
    "vanilla_grpo": "GRPO + Q",
    "scalar_max_ppo": "Scalar Max@N-PPO + Q",
    "entropic_ppo": "Entropic-PPO + Q",
    "nominal_ev_ppo": "Nominal Ordinal EV-PPO + Mean/Q",
    "ev_ppo_no_mean": "StableMax-PPO (no mean floor)",
    "ev_ppo_no_quality": "StableMax-PPO (no Q)",
    "gaussian_ev_ppo": "Gaussian EV-PPO + Q",
    "top4_ppo": "Top-4 PPO + Q",
    "best_of_n": "Best-of-N (base policy)",
}
FINAL_COLUMNS = (
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
    "candidate_quality_mean",
    "candidate_quality_violation_rate",
    "selected_expected_rating_mean",
    "selected_mean_violation_rate",
    "selected_p4_mean",
    "selected_quality_mean",
    "selected_quality_violation_rate",
    "paired_margin_vs_reference",
    "paired_margin_ci_low_vs_reference",
    "paired_margin_ci_high_vs_reference",
    "mean_response_tokens",
    "reference_kl_final",
)

PREFERENCE_COLUMNS = (
    "method",
    "method_name",
    "seed",
    "best_of_n",
    "selection_rule",
    "evaluator",
    "evaluator_type",
    "num_eval_prompts",
    "selected_expected_rating_mean",
    "selected_expected_rating_ci_low",
    "selected_expected_rating_ci_high",
    "evaluator_score_mean",
    "evaluator_score_ci_low",
    "evaluator_score_ci_high",
    "preference_score_vs_vanilla_ppo",
    "preference_ci_low",
    "preference_ci_high",
    "wins",
    "losses",
    "ties",
    "order_consistency",
    "reward_preference_agreement",
    "reward_preference_spearman",
    "judge_calibration_accuracy",
    "judge_calibration_accuracy_ci_low",
    "judge_calibration_accuracy_ci_high",
    "judge_calibration_order_consistency",
    "judge_calibration_order_ci_low",
    "judge_calibration_order_ci_high",
    "judge_calibration_num_pairs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--preference-output", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--rm-config", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--external-rm-model", default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    )
    parser.add_argument("--judge-batch-size", type=int, default=2)
    parser.add_argument("--judge-max-length", type=int, default=2048)
    parser.add_argument("--judge-max-new-tokens", type=int, default=16)
    parser.add_argument("--max-judge-pairs", type=int, default=256)
    parser.add_argument("--rewardbench-max-pairs", type=int, default=512)
    parser.add_argument("--rewardbench-calibration-jsonl", type=Path, default=None)
    parser.add_argument("--min-calibration-pairs", type=int, default=128)
    parser.add_argument("--best-of-n", type=int, default=32)
    parser.add_argument("--rm-diagnostic-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--force-judge", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    field_list = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field, "")) for field in field_list})


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_value(row: dict[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, "", "nan"):
            return value
    return default


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(seed), *[str(part) for part in parts]])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_ci(values: list[float], seed: int, draws: int = 2000) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], finite[0]
    rng = random.Random(seed)
    n = len(finite)
    estimates = [sum(finite[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return float("nan")
    return numerator / (left_scale * right_scale)


def spearman_correlation(left: list[float], right: list[float]) -> float:
    return pearson_correlation(rank_values(left), rank_values(right))


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def prompt_text(record: dict[str, Any]) -> str:
    for name in ("prompt", "query", "instruction", "input", "question"):
        value = record.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def response_text(candidate: dict[str, Any]) -> str:
    for name in ("text", "response", "answer", "completion", "generated_text", "output", "content"):
        value = candidate.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def candidate_number(candidate: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = safe_float(candidate.get(name))
        if value is not None:
            return value
    return None


def prompt_key(record: dict[str, Any], line_number: int) -> str:
    for name in ("prompt_id", "id", "sample_id"):
        identifier = record.get(name)
        if identifier is not None:
            return str(identifier)
    digest = hashlib.sha256(prompt_text(record).encode("utf-8")).hexdigest()[:20]
    return digest or f"line-{line_number}"


def resolve_response_path(eval_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    for candidate in (
        eval_dir / path,
        eval_dir / "analysis" / path,
        eval_dir / "responses" / path,
        eval_dir.parent / path,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"response file not found: {value}")


def load_responses(path: Path, best_of_n: int) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            raw_candidates = record.get("responses") or record.get("samples") or record.get("candidates")
            if not isinstance(raw_candidates, list):
                raw_candidates = [record]
            candidates = []
            for index, raw in enumerate(raw_candidates[:best_of_n]):
                candidate = raw if isinstance(raw, dict) else {"text": str(raw)}
                mu = candidate_number(
                    candidate,
                    (
                        "mu", "reward_mu", "rm_mu", "score_mu", "calibrated_mu",
                        "trajectory_mu", "response_mu", "reward_score", "rm_score", "score", "reward",
                    ),
                )
                sigma = candidate_number(
                    candidate,
                    (
                        "sigma", "reward_sigma", "rm_sigma", "score_sigma", "calibrated_sigma",
                        "trajectory_sigma", "response_sigma", "uncertainty",
                    ),
                )
                if sigma is None:
                    variance = candidate_number(
                        candidate, ("variance", "var", "reward_variance", "rm_variance")
                    )
                    sigma = math.sqrt(max(0.0, variance)) if variance is not None else None
                text = response_text(candidate)
                if mu is None or sigma is None or not text:
                    raise ValueError(f"missing text/mu/sigma in {path}, prompt line {line_number}, candidate {index}")
                ordinal = [candidate_number(candidate, (f"p{rating}",)) for rating in range(5)]
                if any(value is None for value in ordinal):
                    raise ValueError(
                        f"missing p0,...,p4 in {path}, prompt line {line_number}, candidate {index}"
                    )
                ordinal = [float(value) for value in ordinal]
                if any(value < 0.0 for value in ordinal) or abs(sum(ordinal) - 1.0) > 1e-5:
                    raise ValueError(
                        f"invalid p0,...,p4 in {path}, prompt line {line_number}, candidate {index}"
                    )
                candidates.append(
                    {
                        "text": text,
                        "mu": float(mu),
                        "sigma": max(0.0, float(sigma)),
                        "probabilities": ordinal,
                        "index": index,
                    }
                )
            if candidates:
                key = prompt_key(record, line_number)
                output[key] = {"prompt": prompt_text(record), "candidates": candidates}
    if not output:
        raise ValueError(f"no usable generated responses in {path}")
    return output


def select_single(group: dict[str, Any], seed: int, key: str) -> dict[str, Any]:
    candidates = group["candidates"]
    index = stable_seed(seed, "single", key) % len(candidates)
    return candidates[index]


def select_max_mu(group: dict[str, Any]) -> dict[str, Any]:
    """Select the deterministic training-RM best response without reward noise."""
    return max(group["candidates"], key=lambda candidate: (float(candidate["mu"]), -candidate["index"]))


def build_pairs(
    responses: dict[str, dict[str, dict[str, Any]]],
    reference_method: str,
    seed: int,
    max_pairs: int,
) -> list[dict[str, Any]]:
    reference = responses[reference_method]
    pairs: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method == reference_method or method not in responses:
            continue
        common = sorted(set(responses[method]) & set(reference))
        stable_rng = random.Random(stable_seed(seed, "pair-order", method, "max-mu"))
        stable_rng.shuffle(common)
        for key in common[:max_pairs]:
            method_group = responses[method][key]
            reference_group = reference[key]
            method_selected = select_max_mu(method_group)
            reference_selected = select_max_mu(reference_group)
            pairs.append(
                {
                    "pair_id": f"max-mu-{method}-{len(pairs):06d}",
                    "mode": "max_mu",
                    "method": method,
                    "reference_method": reference_method,
                    "prompt_key": key,
                    "prompt": method_group["prompt"] or reference_group["prompt"],
                    "method_response": method_selected["text"],
                    "reference_response": reference_selected["text"],
                    "method_mu": float(method_selected["mu"]),
                    "reference_mu": float(reference_selected["mu"]),
                    "reward_delta": float(method_selected["mu"]) - float(reference_selected["mu"]),
                }
            )
    return pairs


JUDGE_SYSTEM = (
    "You are an impartial evaluator of assistant responses. Judge which response better satisfies the user request "
    "using correctness, relevance, helpfulness, clarity, and safety. Do not prefer a response merely because it is "
    "longer. If neither response is clearly better, choose TIE. Output exactly one label: A, B, or TIE."
)


def judge_prompt(prompt: str, response_a: str, response_b: str) -> str:
    return (
        f"User request:\n{prompt}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Which response is better? Output only A, B, or TIE."
    )


def parse_judge_label(text: str) -> str:
    normalized = text.strip().upper()
    exact = re.fullmatch(r"(?:FINAL\s*[:=-]?\s*)?(A|B|TIE)[.!]?", normalized)
    if exact:
        return exact.group(1)
    labels = re.findall(r"\b(A|B|TIE)\b", normalized)
    return labels[-1] if labels else "TIE"


def load_eval_tokenizer(auto_tokenizer: Any, model_name: str, cache_dir: Path) -> Any:
    kwargs = {"trust_remote_code": True, "cache_dir": str(cache_dir)}
    try:
        return auto_tokenizer.from_pretrained(model_name, **kwargs)
    except Exception as exc:
        print(
            f"[paper-metrics] fast tokenizer unavailable for {model_name}; "
            f"retrying slow tokenizer ({exc})",
            flush=True,
        )
        return auto_tokenizer.from_pretrained(model_name, use_fast=False, **kwargs)


def map_label(label: str, method_is_a: bool) -> str:
    if label == "TIE":
        return "tie"
    chose_a = label == "A"
    return "method" if chose_a == method_is_a else "reference"


class PairwiseJudge:
    def __init__(self, model_name: str, batch_size: int, max_length: int, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("independent model judging requires a CUDA GPU")
        self.torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        cache_dir = Path(
            os.environ.get(
                "JUDGE_CACHE_DIR",
                f"/tmp/{os.environ.get('USER', 'evrl')}/hf_judge_cache",
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = load_eval_tokenizer(AutoTokenizer, model_name, cache_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "cache_dir": str(cache_dir),
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def score(self, prompts: list[tuple[str, str, str]]) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        device = next(self.model.parameters()).device
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            texts = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": judge_prompt(*item)},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    **({"enable_thinking": False} if "qwen3" in self.model_name.lower() else {}),
                )
                for item in batch
            ]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_width = encoded["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
            results.extend({"raw": text, "label": parse_judge_label(text)} for text in decoded)
        return results

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        self.torch.cuda.empty_cache()


class ExternalRewardJudge:
    def __init__(self, model_name: str, batch_size: int, max_length: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("external reward judging requires a CUDA GPU")
        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        cache_dir = Path(
            os.environ.get(
                "JUDGE_CACHE_DIR",
                f"/tmp/{os.environ.get('USER', 'evrl')}/hf_judge_cache",
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = load_eval_tokenizer(AutoTokenizer, model_name, cache_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=str(cache_dir),
        )
        self.model.eval()

    def _format(self, prompt: str, response: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except (TypeError, ValueError):
            return f"User: {prompt}\nAssistant: {response}"

    def score(self, examples: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        device = next(self.model.parameters()).device
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start : start + self.batch_size]
            encoded = self.tokenizer(
                [self._format(*example) for example in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.float().reshape(len(batch), -1)
            scores.extend(float(value) for value in logits[:, 0].cpu())
        return scores

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        self.torch.cuda.empty_cache()


def normalized_response(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
        return "\n".join(str(message.get("content", "")) for message in value if isinstance(message, dict))
    return str(value or "")


def load_rewardbench_pairs(
    max_pairs: int,
    seed: int,
    calibration_jsonl: Path | None = None,
) -> list[tuple[str, str, str]]:
    if max_pairs <= 0:
        return []
    if calibration_jsonl is not None:
        if not calibration_jsonl.is_file():
            raise FileNotFoundError(calibration_jsonl)
        dataset = []
        with calibration_jsonl.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid RewardBench calibration JSONL "
                        f"{calibration_jsonl}:{line_number}"
                    ) from exc
        indices = list(range(len(dataset)))
    else:
        from datasets import load_dataset

        last_error: Exception | None = None
        for split in ("filtered", "test"):
            try:
                dataset = load_dataset("allenai/reward-bench", split=split)
                break
            except Exception as exc:  # Dataset revisions use different split names.
                last_error = exc
        else:
            print(f"[paper-metrics] RewardBench unavailable: {last_error}", flush=True)
            return []
        indices = list(range(len(dataset)))
        random.Random(stable_seed(seed, "rewardbench-order")).shuffle(indices)
    pairs = []
    for index in indices:
        row = dataset[index]
        prompt = prompt_text(row)
        chosen = normalized_response(row.get("chosen"))
        rejected = normalized_response(row.get("rejected"))
        if prompt and chosen and rejected:
            pairs.append((prompt, chosen, rejected))
        if len(pairs) >= max_pairs:
            break
    return pairs


def calibrate_pairwise_judge(
    judge: PairwiseJudge, pairs: list[tuple[str, str, str]], seed: int
) -> dict[str, Any]:
    if not pairs:
        return {}
    forward = judge.score(pairs)
    reverse = judge.score([(prompt, rejected, chosen) for prompt, chosen, rejected in pairs])
    values = []
    consistency = []
    for first, second in zip(forward, reverse):
        first_choice = first["label"]
        second_choice = second["label"]
        first_correct = first_choice == "A"
        second_correct = second_choice == "B"
        consistent = (first_correct and second_correct) or (
            first_choice == "B" and second_choice == "A"
        ) or (first_choice == "TIE" and second_choice == "TIE")
        consistency.append(float(consistent))
        if first_correct and second_correct:
            values.append(1.0)
        elif first_choice == "B" and second_choice == "A":
            values.append(0.0)
        else:
            values.append(0.5)
    accuracy_low, accuracy_high = bootstrap_ci(
        values, stable_seed(seed, "judge-calibration-accuracy")
    )
    order_low, order_high = bootstrap_ci(
        consistency, stable_seed(seed, "judge-calibration-order")
    )
    return {
        "accuracy": statistics.fmean(values),
        "accuracy_ci_low": accuracy_low,
        "accuracy_ci_high": accuracy_high,
        "order_consistency": statistics.fmean(consistency),
        "order_consistency_ci_low": order_low,
        "order_consistency_ci_high": order_high,
        "num_pairs": len(values),
    }


def evaluate_judge(
    pairs: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    output_dir: Path,
    seed: int,
    rewardbench_pairs: list[tuple[str, str, str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    judge = PairwiseJudge(model_name, batch_size, max_length, max_new_tokens)
    forward = [(row["prompt"], row["method_response"], row["reference_response"]) for row in pairs]
    reverse = [(row["prompt"], row["reference_response"], row["method_response"]) for row in pairs]
    forward_results = judge.score(forward)
    reverse_results = judge.score(reverse)
    calibration = calibrate_pairwise_judge(judge, rewardbench_pairs, seed)
    judge.close()

    details = []
    grouped: dict[tuple[str, str], list[float]] = {}
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row, first, second in zip(pairs, forward_results, reverse_results):
        first_outcome = map_label(first["label"], method_is_a=True)
        second_outcome = map_label(second["label"], method_is_a=False)
        outcome = first_outcome if first_outcome == second_outcome else "tie"
        value = 1.0 if outcome == "method" else (0.0 if outcome == "reference" else 0.5)
        key = (row["method"], row["mode"])
        grouped.setdefault(key, []).append(value)
        grouped_rows.setdefault(key, []).append(
            {
                "value": value,
                "outcome": outcome,
                "consistent": first_outcome == second_outcome,
                "reward_delta": float(row["reward_delta"]),
            }
        )
        details.append(
            {
                "pair_id": row["pair_id"],
                "method": row["method"],
                "reference_method": row["reference_method"],
                "mode": row["mode"],
                "prompt_key": row["prompt_key"],
                "method_mu": row["method_mu"],
                "reference_mu": row["reference_mu"],
                "reward_delta": row["reward_delta"],
                "forward_label": first["label"],
                "forward_mapped": first_outcome,
                "reverse_label": second["label"],
                "reverse_mapped": second_outcome,
                "final_outcome": outcome,
                "tie_adjusted_score": value,
                "forward_raw": first["raw"],
                "reverse_raw": second["raw"],
            }
        )
    detail_fields = (
        "pair_id", "method", "reference_method", "mode", "prompt_key",
        "method_mu", "reference_mu", "reward_delta",
        "forward_label", "forward_mapped", "reverse_label", "reverse_mapped",
        "final_outcome", "tie_adjusted_score", "forward_raw", "reverse_raw",
    )
    write_csv(output_dir / "pairwise_judgments.csv", details, detail_fields)

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    summary_rows = []
    for key, values in grouped.items():
        method, mode = key
        rows = grouped_rows[key]
        low, high = bootstrap_ci(values, stable_seed(seed, "judge-ci", method, mode))
        wins = sum(row["outcome"] == "method" for row in rows)
        losses = sum(row["outcome"] == "reference" for row in rows)
        tie_count = sum(row["outcome"] == "tie" for row in rows)
        agreements = []
        for row in rows:
            delta = row["reward_delta"]
            if delta > 0:
                agreements.append(row["value"])
            elif delta < 0:
                agreements.append(1.0 - row["value"])
            else:
                agreements.append(0.5)
        summary = {
            "method": method,
            "mode": mode,
            "reference_method": "vanilla_ppo",
            "win_rate": statistics.fmean(values),
            "ci_low": low,
            "ci_high": high,
            "wins": wins,
            "losses": losses,
            "ties": tie_count,
            "tie_rate": tie_count / len(values),
            "order_consistency": statistics.fmean(float(row["consistent"]) for row in rows),
            "reward_preference_agreement": statistics.fmean(agreements),
            "reward_preference_spearman": spearman_correlation(
                [row["reward_delta"] for row in rows], values
            ),
            "num_pairs": len(values),
        }
        summaries[key] = summary
        summary_rows.append(summary)
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        (
            "method", "mode", "reference_method", "win_rate", "ci_low", "ci_high",
            "wins", "losses", "ties", "tie_rate", "order_consistency",
            "reward_preference_agreement", "reward_preference_spearman", "num_pairs",
        ),
    )
    return summaries, calibration


def selected_max_mu_statistics(
    responses: dict[str, dict[str, dict[str, Any]]], seed: int
) -> dict[str, dict[str, Any]]:
    output = {}
    for method, prompts in responses.items():
        values = [float(select_max_mu(group)["mu"]) for group in prompts.values()]
        low, high = bootstrap_ci(values, stable_seed(seed, "max-mu-ci", method))
        output[method] = {
            "mean": statistics.fmean(values),
            "ci_low": low,
            "ci_high": high,
            "num_prompts": len(values),
        }
    return output


def evaluate_external_reward_judge(
    responses: dict[str, dict[str, dict[str, Any]]],
    reference_method: str,
    model_name: str,
    batch_size: int,
    max_length: int,
    max_pairs: int,
    rewardbench_pairs: list[tuple[str, str, str]],
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    judge = ExternalRewardJudge(model_name, batch_size, max_length)
    summaries: dict[str, dict[str, Any]] = {}
    detail_rows = []
    reference = responses[reference_method]
    reference_keys = sorted(reference)
    random.Random(stable_seed(seed, "external-rm-order", reference_method)).shuffle(reference_keys)
    reference_keys = reference_keys[:max_pairs]
    reference_candidates = [select_max_mu(reference[key]) for key in reference_keys]
    standalone_reference_scores = judge.score(
        [(reference[key]["prompt"], candidate["text"]) for key, candidate in zip(reference_keys, reference_candidates)]
    )
    reference_score_low, reference_score_high = bootstrap_ci(
        standalone_reference_scores, stable_seed(seed, "external-rm-score-ci", reference_method)
    )
    summaries[reference_method] = {
        "win_rate": 0.5,
        "ci_low": 0.5,
        "ci_high": 0.5,
        "wins": 0,
        "losses": 0,
        "ties": len(reference_keys),
        "order_consistency": "",
        "reward_preference_agreement": 0.5,
        "reward_preference_spearman": "",
        "evaluator_score_mean": statistics.fmean(standalone_reference_scores),
        "evaluator_score_ci_low": reference_score_low,
        "evaluator_score_ci_high": reference_score_high,
        "num_pairs": len(reference_keys),
    }
    for method in METHOD_ORDER:
        if method == reference_method or method not in responses:
            continue
        common = sorted(set(responses[method]) & set(reference))
        random.Random(stable_seed(seed, "external-rm-order", method)).shuffle(common)
        common = common[:max_pairs]
        method_selected = [select_max_mu(responses[method][key]) for key in common]
        reference_selected = [select_max_mu(reference[key]) for key in common]
        prompts = [responses[method][key]["prompt"] or reference[key]["prompt"] for key in common]
        method_scores = judge.score(
            [(prompt, selected["text"]) for prompt, selected in zip(prompts, method_selected)]
        )
        reference_scores = judge.score(
            [(prompt, selected["text"]) for prompt, selected in zip(prompts, reference_selected)]
        )
        values = []
        agreements = []
        score_deltas = []
        reward_deltas = []
        wins = losses = ties = 0
        for key, method_candidate, reference_candidate, method_score, reference_score in zip(
            common, method_selected, reference_selected, method_scores, reference_scores
        ):
            score_delta = method_score - reference_score
            reward_delta = float(method_candidate["mu"]) - float(reference_candidate["mu"])
            if score_delta > 1e-8:
                value = 1.0
                outcome = "method"
                wins += 1
            elif score_delta < -1e-8:
                value = 0.0
                outcome = "reference"
                losses += 1
            else:
                value = 0.5
                outcome = "tie"
                ties += 1
            values.append(value)
            score_deltas.append(score_delta)
            reward_deltas.append(reward_delta)
            agreements.append(value if reward_delta > 0 else (1.0 - value if reward_delta < 0 else 0.5))
            detail_rows.append(
                {
                    "method": method,
                    "reference_method": reference_method,
                    "prompt_key": key,
                    "method_mu": method_candidate["mu"],
                    "reference_mu": reference_candidate["mu"],
                    "reward_delta": reward_delta,
                    "method_evaluator_score": method_score,
                    "reference_evaluator_score": reference_score,
                    "evaluator_score_delta": score_delta,
                    "outcome": outcome,
                }
            )
        low, high = bootstrap_ci(values, stable_seed(seed, "external-rm-ci", method))
        score_low, score_high = bootstrap_ci(
            method_scores, stable_seed(seed, "external-rm-score-ci", method)
        )
        summaries[method] = {
            "win_rate": statistics.fmean(values),
            "ci_low": low,
            "ci_high": high,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "order_consistency": "",
            "reward_preference_agreement": statistics.fmean(agreements),
            "reward_preference_spearman": spearman_correlation(reward_deltas, score_deltas),
            "evaluator_score_mean": statistics.fmean(method_scores),
            "evaluator_score_ci_low": score_low,
            "evaluator_score_ci_high": score_high,
            "num_pairs": len(values),
        }

    calibration: dict[str, Any] = {}
    if rewardbench_pairs:
        chosen_scores = judge.score([(prompt, chosen) for prompt, chosen, _ in rewardbench_pairs])
        rejected_scores = judge.score([(prompt, rejected) for prompt, _, rejected in rewardbench_pairs])
        calibration_values = [
            1.0 if chosen > rejected else (0.0 if chosen < rejected else 0.5)
            for chosen, rejected in zip(chosen_scores, rejected_scores)
        ]
        calibration_low, calibration_high = bootstrap_ci(
            calibration_values, stable_seed(seed, "external-rm-calibration")
        )
        calibration = {
            "accuracy": statistics.fmean(calibration_values),
            "accuracy_ci_low": calibration_low,
            "accuracy_ci_high": calibration_high,
            "order_consistency": "",
            "num_pairs": len(calibration_values),
        }
    judge.close()
    write_csv(
        output_dir / "pairwise_scores.csv",
        detail_rows,
        (
            "method", "reference_method", "prompt_key", "method_mu", "reference_mu",
            "reward_delta", "method_evaluator_score", "reference_evaluator_score",
            "evaluator_score_delta", "outcome",
        ),
    )
    write_csv(
        output_dir / "summary.csv",
        [dict(method=method, **summary) for method, summary in summaries.items()],
        (
            "method", "win_rate", "ci_low", "ci_high", "wins", "losses", "ties",
            "reward_preference_agreement", "reward_preference_spearman",
            "evaluator_score_mean", "evaluator_score_ci_low", "evaluator_score_ci_high", "num_pairs",
        ),
    )
    return summaries, calibration


def preference_table_rows(
    selected_stats: dict[str, dict[str, Any]],
    generative_summary: dict[tuple[str, str], dict[str, Any]],
    generative_calibration: dict[str, Any],
    external_summary: dict[str, dict[str, Any]],
    external_calibration: dict[str, Any],
    generative_model: str,
    external_model: str,
    seed: int,
    best_of_n: int,
) -> list[dict[str, Any]]:
    rows = []
    evaluators = (
        (generative_model, "pairwise_generative_judge", generative_summary, generative_calibration),
        (external_model, "pointwise_external_reward_model", external_summary, external_calibration),
    )
    for evaluator, evaluator_type, summaries, calibration in evaluators:
        if not evaluator or not summaries:
            continue
        for method in METHOD_ORDER:
            if method not in selected_stats:
                continue
            if evaluator_type.startswith("pairwise") and method == "vanilla_ppo":
                summary = {
                    "win_rate": 0.5,
                    "ci_low": 0.5,
                    "ci_high": 0.5,
                    "wins": 0,
                    "losses": 0,
                    "ties": selected_stats[method]["num_prompts"],
                    "order_consistency": 1.0,
                    "reward_preference_agreement": 0.5,
                    "reward_preference_spearman": "",
                    "num_pairs": selected_stats[method]["num_prompts"],
                }
            else:
                summary = summaries.get((method, "max_mu"), {}) if evaluator_type.startswith("pairwise") else summaries.get(method, {})
            if not summary:
                continue
            selected = selected_stats[method]
            rows.append(
                {
                    "method": method,
                    "method_name": METHOD_NAMES.get(method, method),
                    "seed": seed,
                    "best_of_n": best_of_n,
                    "selection_rule": "argmax_training_reward_mu",
                    "evaluator": evaluator,
                    "evaluator_type": evaluator_type,
                    "num_eval_prompts": summary.get("num_pairs", selected["num_prompts"]),
                    "selected_expected_rating_mean": selected["mean"],
                    "selected_expected_rating_ci_low": selected["ci_low"],
                    "selected_expected_rating_ci_high": selected["ci_high"],
                    "evaluator_score_mean": summary.get("evaluator_score_mean", ""),
                    "evaluator_score_ci_low": summary.get("evaluator_score_ci_low", ""),
                    "evaluator_score_ci_high": summary.get("evaluator_score_ci_high", ""),
                    "preference_score_vs_vanilla_ppo": summary.get("win_rate", ""),
                    "preference_ci_low": summary.get("ci_low", ""),
                    "preference_ci_high": summary.get("ci_high", ""),
                    "wins": summary.get("wins", ""),
                    "losses": summary.get("losses", ""),
                    "ties": summary.get("ties", ""),
                    "order_consistency": summary.get("order_consistency", ""),
                    "reward_preference_agreement": summary.get("reward_preference_agreement", ""),
                    "reward_preference_spearman": summary.get("reward_preference_spearman", ""),
                    "judge_calibration_accuracy": calibration.get("accuracy", ""),
                    "judge_calibration_accuracy_ci_low": calibration.get("accuracy_ci_low", ""),
                    "judge_calibration_accuracy_ci_high": calibration.get("accuracy_ci_high", ""),
                    "judge_calibration_order_consistency": calibration.get("order_consistency", ""),
                    "judge_calibration_order_ci_low": calibration.get(
                        "order_consistency_ci_low", ""
                    ),
                    "judge_calibration_order_ci_high": calibration.get(
                        "order_consistency_ci_high", ""
                    ),
                    "judge_calibration_num_pairs": calibration.get("num_pairs", ""),
                }
            )
    return rows


def require_evaluator_calibration(
    evaluator: str,
    calibration: dict[str, Any],
    min_pairs: int,
    require_order_consistency: bool,
) -> None:
    num_pairs = int(calibration.get("num_pairs", 0) or 0)
    accuracy_low = safe_float(calibration.get("accuracy_ci_low"))
    if num_pairs < min_pairs:
        raise RuntimeError(
            f"{evaluator} calibration has {num_pairs} pairs; at least {min_pairs} are required"
        )
    if accuracy_low is None or accuracy_low <= 0.5:
        raise RuntimeError(
            f"{evaluator} fails the predeclared RewardBench gate: "
            f"95% bootstrap accuracy lower bound={accuracy_low!r} must exceed 0.5"
        )
    if require_order_consistency:
        order_low = safe_float(calibration.get("order_consistency_ci_low"))
        if order_low is None or order_low <= 0.5:
            raise RuntimeError(
                f"{evaluator} fails the predeclared order-consistency gate: "
                f"95% bootstrap lower bound={order_low!r} must exceed 0.5"
            )


def make_human_packet(
    responses: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    seed: int,
    max_pairs: int,
) -> tuple[Path, Path]:
    if "ev_ppo" not in responses or "vanilla_ppo" not in responses:
        raise ValueError("human packet requires both ev_ppo and vanilla_ppo responses")
    common = sorted(set(responses["ev_ppo"]) & set(responses["vanilla_ppo"]))
    rng = random.Random(stable_seed(seed, "human-packet-order"))
    rng.shuffle(common)
    public_rows = []
    private_rows = []
    for index, key in enumerate(common[:max_pairs]):
        ev_group = responses["ev_ppo"][key]
        vanilla_group = responses["vanilla_ppo"][key]
        ev_response = select_max_mu(ev_group)["text"]
        vanilla_response = select_max_mu(vanilla_group)["text"]
        ev_is_a = random.Random(stable_seed(seed, "human-side", key)).random() < 0.5
        pair_id = f"H{index + 1:05d}"
        public_rows.append(
            {
                "pair_id": pair_id,
                "prompt": ev_group["prompt"] or vanilla_group["prompt"],
                "response_a": ev_response if ev_is_a else vanilla_response,
                "response_b": vanilla_response if ev_is_a else ev_response,
                "annotator_id": "",
                "label": "",
            }
        )
        private_rows.append(
            {
                "pair_id": pair_id,
                "prompt_key": key,
                "method_a": "ev_ppo" if ev_is_a else "vanilla_ppo",
                "method_b": "vanilla_ppo" if ev_is_a else "ev_ppo",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "blinded_pairs.csv"
    key_path = output_dir / "private_key.csv"
    write_csv(
        packet_path,
        public_rows,
        ("pair_id", "prompt", "response_a", "response_b", "annotator_id", "label"),
    )
    write_csv(key_path, private_rows, ("pair_id", "prompt_key", "method_a", "method_b"))
    os.chmod(key_path, 0o600)
    group_rows = []
    group_key_rows = []
    for index, key in enumerate(common[: min(64, max_pairs)]):
        group_pair_id = f"G{index + 1:05d}"
        ev_is_a = random.Random(stable_seed(seed, "group-side", key)).random() < 0.5
        methods = (("A", "ev_ppo" if ev_is_a else "vanilla_ppo"), ("B", "vanilla_ppo" if ev_is_a else "ev_ppo"))
        for group_label, method in methods:
            group = responses[method][key]
            candidates = list(group["candidates"])
            random.Random(stable_seed(seed, "candidate-order", key, group_label)).shuffle(candidates)
            for candidate_index, candidate in enumerate(candidates, start=1):
                group_rows.append(
                    {
                        "group_pair_id": group_pair_id,
                        "prompt": group["prompt"],
                        "group_label": group_label,
                        "candidate_id": f"{group_label}{candidate_index:02d}",
                        "response": candidate["text"],
                        "annotator_id": "",
                        "rating_0_to_4": "",
                    }
                )
            group_key_rows.append(
                {
                    "group_pair_id": group_pair_id,
                    "prompt_key": key,
                    "group_label": group_label,
                    "method": method,
                }
            )
    group_packet_path = output_dir / "blinded_group_ratings.csv"
    group_key_path = output_dir / "private_group_key.csv"
    write_csv(
        group_packet_path,
        group_rows,
        (
            "group_pair_id",
            "prompt",
            "group_label",
            "candidate_id",
            "response",
            "annotator_id",
            "rating_0_to_4",
        ),
    )
    write_csv(
        group_key_path,
        group_key_rows,
        ("group_pair_id", "prompt_key", "group_label", "method"),
    )
    os.chmod(group_key_path, 0o600)
    protocol = (
        "# Direct human evaluation protocol\n\n"
        "Annotators are blinded to method identity. For each row, judge the two generated responses "
        "against the user prompt using correctness, relevance, helpfulness, clarity, and safety. "
        "Do not prefer length by itself. Enter an anonymized annotator ID and exactly A, B, or TIE. "
        "Use at least three independent annotators per pair, merge their rows before import, and do not "
        "give annotators access to model identities, reward scores, or the private key.\n\n"
        "The group-rating packet separately tests the paper's upper-tail claim. The same set of at "
        "least three blinded annotators must each rate every candidate in a group from 0 to 4. "
        "After locking labels, use the private group key to compare the realized per-prompt maxima "
        "and the frequency of at least one rating 4 with scripts/human_eval_analysis.py. Do not use "
        "model scores as human labels.\n"
    )
    (output_dir / "README.md").write_text(protocol, encoding="utf-8")
    return packet_path, key_path


def import_human_labels(labels_path: Path | None, key_path: Path, seed: int) -> dict[str, Any]:
    if labels_path is None or not labels_path.exists():
        return {}
    keys = {row["pair_id"]: row for row in read_csv(key_path)}
    outcomes_by_pair: dict[str, list[float]] = {}
    ties = 0
    annotations = 0
    seen: set[tuple[str, str]] = set()
    for row in read_csv(labels_path):
        pair_id = row.get("pair_id", "")
        annotator_id = row.get("annotator_id", "").strip() or f"row-{annotations}"
        label = row.get("label", "").strip().upper()
        annotation_key = (pair_id, annotator_id)
        if annotation_key in seen or pair_id not in keys or label not in {"A", "B", "TIE"}:
            continue
        seen.add(annotation_key)
        annotations += 1
        if label == "TIE":
            outcomes_by_pair.setdefault(pair_id, []).append(0.5)
            ties += 1
        else:
            selected_method = keys[pair_id]["method_a" if label == "A" else "method_b"]
            outcomes_by_pair.setdefault(pair_id, []).append(1.0 if selected_method == "ev_ppo" else 0.0)
    if not outcomes_by_pair:
        return {}
    outcomes = [statistics.fmean(values) for values in outcomes_by_pair.values()]
    low, high = bootstrap_ci(outcomes, stable_seed(seed, "human-ci"))
    return {
        "human_bon_win_rate_vs_vanilla": statistics.fmean(outcomes),
        "human_bon_ci_low": low,
        "human_bon_ci_high": high,
        "human_bon_tie_rate": ties / annotations,
        "num_human_pairs": len(outcomes),
    }


def response_lengths(
    responses: dict[str, dict[str, dict[str, Any]]], base_model: str
) -> dict[str, float]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    output = {}
    for method, prompts in responses.items():
        lengths = []
        for group in prompts.values():
            for candidate in group["candidates"]:
                lengths.append(len(tokenizer.encode(candidate["text"], add_special_tokens=False)))
        output[method] = statistics.fmean(lengths) if lengths else float("nan")
    del tokenizer
    return output


def training_diagnostics(experiment_root: Path, method: str, seed: int) -> dict[str, Any]:
    path = experiment_root / f"{method}_seed{seed}" / "train_metrics.csv"
    if not path.exists():
        return {}
    rows = read_csv(path)
    if not rows:
        return {}

    def series(names: Iterable[str]) -> list[float]:
        values = []
        for row in rows:
            value = safe_float(first_value(row, names))
            if value is not None:
                values.append(value)
        return values

    reference_kl = series(("reference_kl", "ref_kl", "kl_to_reference", "observed_kl"))
    gradient_norm = series(("gradient_norm", "grad_norm", "policy_grad_norm"))
    quadrature_residual = series(("quadrature_residual", "quadrature_error", "quad_residual"))
    converged_raw = [
        str(first_value(row, ("quadrature_converged", "quad_converged"), "")).strip().lower()
        for row in rows
        if first_value(row, ("quadrature_converged", "quad_converged"), "") != ""
    ]
    return {
        "reference_kl_final": reference_kl[-1] if reference_kl else "",
        "reference_kl_mean_last20": statistics.fmean(reference_kl[-20:]) if reference_kl else "",
        "gradient_norm_median": statistics.median(gradient_norm) if gradient_norm else "",
        "quadrature_residual_max": max(quadrature_residual) if quadrature_residual else "",
        "quadrature_all_converged": all(value in {"1", "true", "yes"} for value in converged_raw)
        if converged_raw
        else "",
    }


def export_static_appendix(eval_dir: Path) -> Path:
    rows = []
    for summary_path in sorted((eval_dir / "policy_pref_eval").glob("*/policy_preference_summary.json")):
        summary = read_json(summary_path)
        rows.append(
            {
                "method": summary_path.parent.name.rsplit("_seed", 1)[0],
                "seed": summary_path.parent.name.rsplit("_seed", 1)[-1],
                "static_policy_logprob_accuracy": first_value(summary, ("preference_accuracy", "accuracy")),
                "num_static_pairs": first_value(summary, ("num_preference_pairs", "num_pairs")),
                "metric_definition": "length-normalized policy log-probability ordering on fixed chosen/rejected pairs; not direct human evaluation",
            }
        )
    path = eval_dir / "analysis" / "appendix_static_policy_logprob_accuracy.csv"
    write_csv(
        path,
        rows,
        ("method", "seed", "static_policy_logprob_accuracy", "num_static_pairs", "metric_definition"),
    )
    return path


def reward_model_diagnostic(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    test_metrics = config.get("test_metrics", {}) if isinstance(config.get("test_metrics"), dict) else {}
    row = {
        "evaluator": first_value(config, ("model_name", "model_name_or_path", "candidate")),
        "heldout_preference_accuracy": first_value(
            test_metrics, ("helpfulness_pair_accuracy_non_ties",),
            first_value(config, ("heldout_test_accuracy", "test_accuracy", "accuracy")),
        ),
        "num_heldout_pairs": first_value(
            test_metrics, ("helpfulness_pair_non_ties",),
            first_value(config, ("heldout_test_num_pairs", "num_eval_pairs")),
        ),
        "uncertainty_type": first_value(config, ("uncertainty_type", "sigma_mode"), "unknown"),
        "ordinal_nll": first_value(test_metrics, ("ordinal_nll",)),
        "ordinal_threshold_ece": first_value(test_metrics, ("ordinal_threshold_ece",)),
        "sigma_disagreement_spearman": first_value(
            test_metrics, ("helpfulness_sigma_disagreement_spearman",)
        ),
        "diagnostic": "frozen repeated-rating reward-moment model; not policy human preference",
    }
    write_csv(
        output_path,
        [row],
        (
            "evaluator",
            "heldout_preference_accuracy",
            "num_heldout_pairs",
            "uncertainty_type",
            "ordinal_nll",
            "ordinal_threshold_ece",
            "sigma_disagreement_spearman",
            "diagnostic",
        ),
    )
    return row


def build_final_rows(
    source_rows: list[dict[str, str]],
    lengths: dict[str, float],
    experiment_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for source in source_rows:
        method = source.get("method", "")
        row: dict[str, Any] = {
            "method": method,
            "method_name": source.get("method_name") or METHOD_NAMES.get(method, method),
            "seed": source.get("seed", seed),
            "best_of_n": source.get("best_of_n", ""),
            "num_eval_prompts": source.get("num_eval_prompts", ""),
            "mean_response_tokens": lengths.get(method, ""),
        }
        for field in FINAL_COLUMNS:
            if field in source and field not in row:
                row[field] = source[field]
        row.update(training_diagnostics(experiment_root, method, seed))
        output.append(row)
    output.sort(key=lambda row: METHOD_ORDER.index(row["method"]) if row["method"] in METHOD_ORDER else 999)
    return output


def main() -> None:
    args = parse_args()
    if args.eval_dir is None:
        args.eval_dir = args.input.parent.parent
    if args.preference_output is None:
        args.preference_output = args.output.parent / "human_preference_proxy_table.csv"
    if args.best_of_n < 2:
        raise SystemExit("--best-of-n must be at least 2 for this evaluation")
    if args.max_judge_pairs <= 0 or args.judge_batch_size <= 0:
        raise SystemExit("judge pair count and batch size must be positive")
    if args.min_calibration_pairs < 1 or args.rewardbench_max_pairs < args.min_calibration_pairs:
        raise SystemExit(
            "RewardBench calibration requires rewardbench_max_pairs >= min_calibration_pairs >= 1"
        )
    source_rows = read_csv(args.input)
    if not source_rows:
        raise SystemExit(f"empty input table: {args.input}")

    responses: dict[str, dict[str, dict[str, Any]]] = {}
    for row in source_rows:
        method = row.get("method", "")
        response_name = row.get("responses_jsonl", "")
        if method and response_name:
            responses[method] = load_responses(
                resolve_response_path(args.eval_dir, response_name), args.best_of_n
            )
    if "ev_ppo" not in responses or "vanilla_ppo" not in responses:
        raise SystemExit("evaluation requires generated response files for ev_ppo and vanilla_ppo")

    lengths = response_lengths(responses, args.base_model)
    final_rows = build_final_rows(
        source_rows,
        lengths,
        args.eval_dir.parent,
        args.seed,
    )
    write_csv(args.output, final_rows, FINAL_COLUMNS)

    selected_stats = selected_max_mu_statistics(responses, args.seed)
    preference_root = args.eval_dir / "independent_preference"
    generative_summary: dict[tuple[str, str], dict[str, Any]] = {}
    generative_calibration: dict[str, Any] = {}
    external_summary: dict[str, dict[str, Any]] = {}
    external_calibration: dict[str, Any] = {}
    if not args.skip_judge:
        rewardbench_pairs = load_rewardbench_pairs(
            args.rewardbench_max_pairs,
            args.seed,
            args.rewardbench_calibration_jsonl,
        )
        pairs = build_pairs(responses, "vanilla_ppo", args.seed, args.max_judge_pairs)

        generative_dir = preference_root / model_slug(args.judge_model)
        generative_cache = generative_dir / "summary.csv"
        generative_calibration_cache = generative_dir / "calibration.json"
        if generative_cache.exists() and generative_calibration_cache.exists() and not args.force_judge:
            for row in read_csv(generative_cache):
                key = (row.get("method", ""), row.get("mode", ""))
                if all(key):
                    generative_summary[key] = {
                        name: safe_float(row.get(name)) if name not in {"wins", "losses", "ties", "num_pairs"} else row.get(name, "")
                        for name in (
                            "win_rate", "ci_low", "ci_high", "wins", "losses", "ties",
                            "order_consistency", "reward_preference_agreement",
                            "reward_preference_spearman", "num_pairs",
                        )
                    }
            generative_calibration = read_json(generative_calibration_cache)
            print(f"[paper-metrics] reused {generative_cache}", flush=True)
        else:
            generative_summary, generative_calibration = evaluate_judge(
                pairs,
                args.judge_model,
                args.judge_batch_size,
                args.judge_max_length,
                args.judge_max_new_tokens,
                generative_dir,
                args.seed,
                rewardbench_pairs,
            )
            generative_calibration_cache.parent.mkdir(parents=True, exist_ok=True)
            generative_calibration_cache.write_text(
                json.dumps(generative_calibration, indent=2), encoding="utf-8"
            )

        if args.external_rm_model:
            external_dir = preference_root / model_slug(args.external_rm_model)
            external_cache = external_dir / "summary.csv"
            external_calibration_cache = external_dir / "calibration.json"
            if external_cache.exists() and external_calibration_cache.exists() and not args.force_judge:
                for row in read_csv(external_cache):
                    method = row.get("method", "")
                    if method:
                        external_summary[method] = {
                            name: safe_float(row.get(name)) if name not in {"wins", "losses", "ties", "num_pairs"} else row.get(name, "")
                            for name in (
                                "win_rate", "ci_low", "ci_high", "wins", "losses", "ties",
                                "reward_preference_agreement", "reward_preference_spearman",
                                "evaluator_score_mean", "evaluator_score_ci_low",
                                "evaluator_score_ci_high", "num_pairs",
                            )
                        }
                external_calibration = read_json(external_calibration_cache)
                print(f"[paper-metrics] reused {external_cache}", flush=True)
            else:
                external_summary, external_calibration = evaluate_external_reward_judge(
                    responses,
                    "vanilla_ppo",
                    args.external_rm_model,
                    args.judge_batch_size,
                    args.judge_max_length,
                    args.max_judge_pairs,
                    rewardbench_pairs,
                    external_dir,
                    args.seed,
                )
                external_calibration_cache.parent.mkdir(parents=True, exist_ok=True)
                external_calibration_cache.write_text(
                    json.dumps(external_calibration, indent=2), encoding="utf-8"
                )

        require_evaluator_calibration(
            args.judge_model,
            generative_calibration,
            args.min_calibration_pairs,
            require_order_consistency=True,
        )
        if args.external_rm_model:
            require_evaluator_calibration(
                args.external_rm_model,
                external_calibration,
                args.min_calibration_pairs,
                require_order_consistency=False,
            )

        preference_rows = preference_table_rows(
            selected_stats,
            generative_summary,
            generative_calibration,
            external_summary,
            external_calibration,
            args.judge_model,
            args.external_rm_model,
            args.seed,
            args.best_of_n,
        )
        write_csv(args.preference_output, preference_rows, PREFERENCE_COLUMNS)

    human_packet_path, human_key_path = make_human_packet(
        responses, args.eval_dir / "human_eval", args.seed, args.max_judge_pairs
    )
    appendix_path = export_static_appendix(args.eval_dir)
    rm_diagnostic_path = args.eval_dir / "analysis" / "reward_model_preference_diagnostic.csv"
    rm_diagnostic = reward_model_diagnostic(args.rm_config, rm_diagnostic_path)
    metadata = {
        "optimized_proxy_endpoint": (
            "validation-frozen calibrated robust exact finite-N expected "
            "maximum ordinal rating"
        ),
        "ordinal_tail_definition": {
            "primary": "E_rob[max_{j<=N} R_j] for R_j in {0,1,2,3,4}",
            "nominal_secondary": "E_nom[max_{j<=N} R_j]",
            "upper_tail_secondary": "P_nom(any j: R_j=4)",
            "best_of_n": args.best_of_n,
            "robust_epsilon": next(
                (
                    row.get("robust_epsilon")
                    for row in final_rows
                    if row.get("method") == "ev_ppo"
                ),
                "",
            ),
        },
        "independent_preference_proxy": {
            "table": str(args.preference_output),
            "generative_model": args.judge_model,
            "external_reward_model": args.external_rm_model,
            "selection_rule": "argmax_j mu_omega(x, y_j) among exactly N generated responses",
            "selection_rule_interpretation": "mu is the expected ordinal rating from p0,...,p4",
            "protocol": "blinded generated-response pairwise judging in both A/B orders",
            "inconsistent_position_swaps": "counted conservatively as ties",
            "win_rate_definition": "wins + 0.5 * ties, divided by all valid pairs",
            "rewardbench_calibration_pairs": args.rewardbench_max_pairs,
            "rewardbench_calibration_jsonl": str(args.rewardbench_calibration_jsonl or ""),
            "rewardbench_calibration_sha256": (
                file_sha256(args.rewardbench_calibration_jsonl)
                if args.rewardbench_calibration_jsonl is not None
                else ""
            ),
            "minimum_calibration_pairs": args.min_calibration_pairs,
            "calibration_acceptance_rule": (
                "95% bootstrap lower bound for RewardBench accuracy exceeds 0.5; "
                "the generative judge additionally requires the order-consistency lower bound to exceed 0.5"
            ),
            "claim_policy": "independent model-based preference proxy; not direct human preference",
        },
        "reward_model_diagnostic": rm_diagnostic,
        "blinded_pairwise_human_packet": str(human_packet_path),
        "blinded_group_rating_packet": str(human_packet_path.parent / "blinded_group_ratings.csv"),
        "fresh_human_labels_required_for_claims": True,
        "static_policy_likelihood_appendix": str(appendix_path),
        "reward_distribution_definition": (
            "calibrated five-category conditional distribution of observable HelpSteer2 helpfulness ratings; "
            "epistemic model uncertainty is excluded"
        ),
        "removed_from_main_table": [
            "fixed q95/q99/CVaR95/CVaR99 columns",
            "all method-vs-Vanilla win-rate columns",
            "static chosen/rejected likelihood accuracy",
            "model-judge columns",
            "unpopulated direct-human column",
        ],
        "main_table_columns": list(FINAL_COLUMNS),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[paper-metrics] wrote {args.output}", flush=True)
    if not args.skip_judge:
        print(f"[paper-metrics] wrote {args.preference_output}", flush=True)
        print(f"[paper-metrics] wrote evaluator details under {preference_root}", flush=True)


if __name__ == "__main__":
    main()
