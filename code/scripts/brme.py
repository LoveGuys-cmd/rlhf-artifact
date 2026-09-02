#!/usr/bin/env python3
"""Train and validate the EV-PPO ordinal-Gaussian reward-moment model.

Each HelpSteer2 response is supervised by its released repeated annotator
ratings.  For attribute k, the model defines a latent utility

    Z_k | x,y ~ Normal(mu_k(x,y), sigma_k(x,y)^2)

and the observed 0--4 rating is obtained using attribute-specific cutpoints
fixed from smoothed training-set rating frequencies.  The policy reward uses
the conditional mean and standard deviation of the induced observable
helpfulness-rating distribution; the remaining four attributes are auxiliary
likelihood terms, not a weighted reward composite.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from peft import (
    AutoPeftModelForSequenceClassification,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest, norm, spearmanr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


ATTRIBUTE_NAMES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
NUM_ATTRIBUTES = len(ATTRIBUTE_NAMES)
NUM_RATINGS = 5
RATING_MIN = 0.0
RATING_MAX = 4.0
EPS = 1e-7


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def format_prompt_response(tokenizer, prompt: str, response: str) -> str:
    prompt = str(prompt or "").strip()
    response = str(response or "").strip()
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    return f"Question: {prompt}\nAnswer: {response}".strip()


def annotations_from_row(row: dict[str, Any], side: str) -> list[list[int]]:
    packed = row.get(f"{side}_annotations")
    if isinstance(packed, str):
        packed = json.loads(packed)
    if not isinstance(packed, dict):
        raise ValueError(f"{side}_annotations must contain repeated HelpSteer2 ratings")
    output: list[list[int]] = []
    counts = set()
    for name in ATTRIBUTE_NAMES:
        values = packed.get(name)
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"{side}_annotations[{name}] must contain at least two ratings")
        ratings = [int(value) for value in values]
        if any(value < 0 or value >= NUM_RATINGS for value in ratings):
            raise ValueError(f"{side}_annotations[{name}] contains an out-of-range rating")
        output.append(ratings)
        counts.add(len(ratings))
    if len(counts) != 1:
        raise ValueError(f"{side} attributes have inconsistent annotator counts")
    return output


def normalize_pairs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
            continue
        normalized.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_annotations": annotations_from_row(row, "chosen"),
                "rejected_annotations": annotations_from_row(row, "rejected"),
            }
        )
    if not normalized:
        raise ValueError("No repeated-rating prompt/chosen/rejected rows were found")
    return normalized


def normalize_preference_pairs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
            continue
        prompt, chosen, rejected = prompt.strip(), chosen.strip(), rejected.strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        normalized.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_annotations": [[] for _ in ATTRIBUTE_NAMES],
                "rejected_annotations": [[] for _ in ATTRIBUTE_NAMES],
                "preference_only": True,
            }
        )
    if not normalized:
        raise ValueError("No valid decontaminated preference rows were found")
    return normalized


def assert_preference_disjoint(
    preference_rows: list[dict[str, Any]],
    moment_splits: Iterable[list[dict[str, Any]]],
) -> None:
    forbidden_prompts = set()
    forbidden_responses = set()
    for rows in moment_splits:
        for row in rows:
            forbidden_prompts.add(row["prompt"].strip())
            forbidden_responses.update((row["chosen"].strip(), row["rejected"].strip()))
    for row in preference_rows:
        if row["prompt"] in forbidden_prompts:
            raise ValueError("Preference replay contains a reward-moment prompt")
        if row["chosen"] in forbidden_responses or row["rejected"] in forbidden_responses:
            raise ValueError("Preference replay contains a reward-moment response")


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse_softplus requires a positive value")
    return value + math.log(-math.expm1(-value))


def decode_moments(logits: torch.Tensor, sigma_floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode an identifiable latent utility and conditional latent scale.

    The score head is initialized at zero.  Offsetting the raw scale therefore
    makes its initial value exactly one, matching the scale used to construct
    the training-climatology cutpoints.
    """
    if not 0.0 < sigma_floor < 1.0:
        raise ValueError("sigma_floor must lie strictly between zero and one")
    logits = logits.float()
    means = logits[..., :NUM_ATTRIBUTES]
    scale_offset = inverse_softplus(1.0 - sigma_floor)
    sigmas = torch.nn.functional.softplus(
        logits[..., NUM_ATTRIBUTES:] + scale_offset
    ) + sigma_floor
    return means, sigmas


def ordinal_probabilities(
    means: torch.Tensor,
    sigmas: torch.Tensor,
    cutpoints: torch.Tensor | np.ndarray | list[list[float]],
) -> torch.Tensor:
    """Return ordinal probabilities under attribute-specific latent cutpoints."""
    cutpoints_tensor = torch.as_tensor(cutpoints, dtype=means.dtype, device=means.device)
    if cutpoints_tensor.ndim == 1:
        cutpoints_tensor = cutpoints_tensor.unsqueeze(0)
    if cutpoints_tensor.shape != (means.shape[-1], NUM_RATINGS - 1):
        raise ValueError(
            "cutpoints must have shape "
            f"({means.shape[-1]}, {NUM_RATINGS - 1}), got {tuple(cutpoints_tensor.shape)}"
        )
    cutpoint_shape = (1,) * (means.ndim - 1) + tuple(cutpoints_tensor.shape)
    finite_cdf = torch.special.ndtr(
        (cutpoints_tensor.view(cutpoint_shape) - means.unsqueeze(-1))
        / sigmas.unsqueeze(-1)
    )
    zeros = torch.zeros_like(finite_cdf[..., :1])
    ones = torch.ones_like(finite_cdf[..., :1])
    cdf = torch.cat((zeros, finite_cdf, ones), dim=-1)
    probabilities = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(EPS)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def ordinal_rating_moments(
    latent_means: torch.Tensor,
    latent_sigmas: torch.Tensor,
    cutpoints: torch.Tensor | np.ndarray | list[list[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map latent ordinal-probit parameters to observable rating moments."""
    probabilities = ordinal_probabilities(latent_means, latent_sigmas, cutpoints)
    rating_values = torch.arange(
        NUM_RATINGS, dtype=probabilities.dtype, device=probabilities.device
    )
    rating_means = (probabilities * rating_values).sum(dim=-1)
    rating_variances = (
        probabilities * (rating_values - rating_means.unsqueeze(-1)).square()
    ).sum(dim=-1)
    return rating_means, rating_variances.clamp_min(EPS).sqrt()


def ordinal_nll(
    means: torch.Tensor,
    sigmas: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    cutpoints: torch.Tensor | np.ndarray | list[list[float]],
) -> torch.Tensor:
    probabilities = ordinal_probabilities(means, sigmas, cutpoints)
    labels = labels.long().clamp(0, NUM_RATINGS - 1)
    expanded = probabilities.unsqueeze(-2).expand(*labels.shape, NUM_RATINGS)
    selected = expanded.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    weights = mask.to(selected.dtype)
    return -(selected.log() * weights).sum() / weights.sum().clamp_min(1.0)


def helpfulness_preference_loss(
    chosen_latent_mu: torch.Tensor,
    chosen_latent_sigma: torch.Tensor,
    rejected_latent_mu: torch.Tensor,
    rejected_latent_sigma: torch.Tensor,
    cutpoints: torch.Tensor | np.ndarray | list[list[float]],
) -> torch.Tensor:
    """Bradley--Terry log score on the served rating mean only.

    Preference replay improves response ranking without inventing uncertainty:
    both latent scales are detached before the ordinal distribution is mapped
    to its observable 0--4 rating mean.
    """
    helpfulness_cutpoints = torch.as_tensor(cutpoints)[:1]
    chosen_rating_mu, _ = ordinal_rating_moments(
        chosen_latent_mu[:, :1], chosen_latent_sigma[:, :1].detach(), helpfulness_cutpoints
    )
    rejected_rating_mu, _ = ordinal_rating_moments(
        rejected_latent_mu[:, :1], rejected_latent_sigma[:, :1].detach(), helpfulness_cutpoints
    )
    margin = chosen_rating_mu[:, 0] - rejected_rating_mu[:, 0]
    return torch.nn.functional.softplus(-margin).mean()


def helpfulness_mean_difference_nll(
    chosen_latent_mu: torch.Tensor,
    chosen_latent_sigma: torch.Tensor,
    rejected_latent_mu: torch.Tensor,
    rejected_latent_sigma: torch.Tensor,
    chosen_labels: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_labels: torch.Tensor,
    rejected_mask: torch.Tensor,
    cutpoints: torch.Tensor | np.ndarray | list[list[float]],
) -> torch.Tensor:
    """Gaussian log score for the repeated-rating helpfulness mean difference.

    HelpSteer2 orders each response pair by its retained annotator mean, while
    random-rating superiority can disagree with that ordering when predictive
    distributions cross.  Under the model, each empirical response mean has
    variance sigma_R^2 / n.  Scoring the observed mean difference under this
    approximation aligns location learning with the reported ranking target.
    Latent scales and the induced sampling variance are detached so this term
    cannot manufacture reward variance; ordinal likelihood and the explicit
    disagreement score remain solely responsible for dispersion.
    """
    chosen_rating_mu, chosen_rating_sigma = ordinal_rating_moments(
        chosen_latent_mu, chosen_latent_sigma.detach(), cutpoints
    )
    rejected_rating_mu, rejected_rating_sigma = ordinal_rating_moments(
        rejected_latent_mu, rejected_latent_sigma.detach(), cutpoints
    )

    chosen_weights = chosen_mask[:, 0].to(chosen_rating_mu.dtype)
    rejected_weights = rejected_mask[:, 0].to(rejected_rating_mu.dtype)
    chosen_count = chosen_weights.sum(dim=-1).clamp_min(1.0)
    rejected_count = rejected_weights.sum(dim=-1).clamp_min(1.0)
    chosen_observed_mean = (
        chosen_labels[:, 0].to(chosen_rating_mu.dtype) * chosen_weights
    ).sum(dim=-1) / chosen_count
    rejected_observed_mean = (
        rejected_labels[:, 0].to(rejected_rating_mu.dtype) * rejected_weights
    ).sum(dim=-1) / rejected_count

    predicted_difference = chosen_rating_mu[:, 0] - rejected_rating_mu[:, 0]
    observed_difference = chosen_observed_mean - rejected_observed_mean
    sampling_variance = (
        chosen_rating_sigma[:, 0].detach().square() / chosen_count
        + rejected_rating_sigma[:, 0].detach().square() / rejected_count
    ).clamp_min(1e-4)
    standardized_error = (observed_difference - predicted_difference).square() / sampling_variance
    return 0.5 * (standardized_error + sampling_variance.log()).mean()


def annotator_disagreement_loss(
    means: torch.Tensor,
    sigmas: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    cutpoints: torch.Tensor | np.ndarray | list[list[float]],
) -> torch.Tensor:
    """Proper log score for whether two conditionally iid annotators disagree.

    For ordinal probabilities p, two independent ratings disagree with
    probability 1-sum_r p_r^2.  Averaging the Bernoulli log score over all
    retained annotator pairs gives direct, response-level supervision for
    conditional dispersion.  Means are detached so this auxiliary score can
    only train sigma; the ordinal likelihood remains responsible for both
    location and the full five-category predictive distribution.
    """
    probabilities = ordinal_probabilities(means.detach(), sigmas, cutpoints)
    predicted_disagreement = (
        1.0 - probabilities.square().sum(dim=-1)
    ).clamp(EPS, 1.0 - EPS)

    annotators = labels.shape[-1]
    upper_triangle = torch.triu(
        torch.ones((annotators, annotators), dtype=torch.bool, device=labels.device),
        diagonal=1,
    )
    valid_pairs = (
        mask.unsqueeze(-1) & mask.unsqueeze(-2) & upper_triangle.view(1, 1, annotators, annotators)
    )
    outcomes = labels.unsqueeze(-1).ne(labels.unsqueeze(-2)).to(predicted_disagreement.dtype)
    pair_probabilities = predicted_disagreement.unsqueeze(-1).unsqueeze(-1)
    pair_losses = -(
        outcomes * pair_probabilities.log()
        + (1.0 - outcomes) * (1.0 - pair_probabilities).log()
    )
    pair_counts = valid_pairs.sum(dim=(-2, -1))
    response_attribute_losses = (
        (pair_losses * valid_pairs.to(pair_losses.dtype)).sum(dim=(-2, -1))
        / pair_counts.clamp_min(1).to(pair_losses.dtype)
    )
    valid_response_attributes = pair_counts > 0
    if not valid_response_attributes.any():
        return sigmas.sum() * 0.0
    return response_attribute_losses[valid_response_attributes].mean()


def pad_annotations(
    examples: list[list[list[int]]],
    max_annotators: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_annotators is None:
        max_annotators = max(
            1,
            max((len(values) for example in examples for values in example), default=0),
        )
    labels = torch.zeros((len(examples), NUM_ATTRIBUTES, max_annotators), dtype=torch.long)
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for row_index, example in enumerate(examples):
        if len(example) != NUM_ATTRIBUTES:
            raise ValueError("Every response must contain all five HelpSteer2 attributes")
        for attribute_index, values in enumerate(example):
            count = len(values)
            if count > max_annotators:
                raise ValueError("Annotation count exceeds the configured padding width")
            if count == 0:
                continue
            labels[row_index, attribute_index, :count] = torch.tensor(values, dtype=torch.long)
            mask[row_index, attribute_index, :count] = True
    return labels, mask


class PairMomentCollator:
    def __init__(self, tokenizer, max_length: int, max_annotators: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_annotators = max_annotators

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chosen_texts = [format_prompt_response(self.tokenizer, row["prompt"], row["chosen"]) for row in features]
        rejected_texts = [format_prompt_response(self.tokenizer, row["prompt"], row["rejected"]) for row in features]
        chosen = self.tokenizer(
            chosen_texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        )
        rejected = self.tokenizer(
            rejected_texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        )
        chosen_labels, chosen_mask = pad_annotations(
            [row["chosen_annotations"] for row in features], self.max_annotators
        )
        rejected_labels, rejected_mask = pad_annotations(
            [row["rejected_annotations"] for row in features], self.max_annotators
        )
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
            "labels_chosen": chosen_labels,
            "labels_chosen_mask": chosen_mask,
            "labels_rejected": rejected_labels,
            "labels_rejected_mask": rejected_mask,
            "preference_only": torch.tensor(
                [bool(row.get("preference_only", False)) for row in features],
                dtype=torch.bool,
            ),
        }


class MomentTrainer(Trainer):
    def __init__(
        self,
        *args,
        sigma_floor: float,
        cutpoints: np.ndarray,
        helpfulness_nll_weight: float,
        mean_difference_loss_weight: float,
        disagreement_loss_weight: float,
        preference_loss_weight: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sigma_floor = sigma_floor
        self.cutpoints = torch.as_tensor(cutpoints, dtype=torch.float32)
        self.helpfulness_nll_weight = helpfulness_nll_weight
        self.mean_difference_loss_weight = mean_difference_loss_weight
        self.disagreement_loss_weight = disagreement_loss_weight
        self.preference_loss_weight = preference_loss_weight

    def loss_components(self, model, inputs):
        chosen_labels = inputs.pop("labels_chosen")
        chosen_mask = inputs.pop("labels_chosen_mask")
        rejected_labels = inputs.pop("labels_rejected")
        rejected_mask = inputs.pop("labels_rejected_mask")
        preference_only = inputs.pop("preference_only")
        if bool(preference_only.any()) != bool(preference_only.all()):
            raise ValueError("Training batches cannot mix moment and preference-only rows")
        chosen_logits = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        ).logits
        rejected_logits = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        ).logits
        chosen_mu, chosen_sigma = decode_moments(chosen_logits, self.sigma_floor)
        rejected_mu, rejected_sigma = decode_moments(rejected_logits, self.sigma_floor)
        if bool(preference_only.all()):
            zero = (
                chosen_mu.sum()
                + chosen_sigma.sum()
                + rejected_mu.sum()
                + rejected_sigma.sum()
            ) * 0.0
            preference_loss = helpfulness_preference_loss(
                chosen_mu,
                chosen_sigma,
                rejected_mu,
                rejected_sigma,
                self.cutpoints,
            )
            return (
                zero,
                zero,
                zero,
                zero,
                preference_loss,
                chosen_mu,
                chosen_sigma,
                rejected_mu,
                rejected_sigma,
            )
        preference_loss = (chosen_mu[:, 0].sum() + rejected_mu[:, 0].sum()) * 0.0
        ordinal_loss = 0.5 * (
            ordinal_nll(chosen_mu, chosen_sigma, chosen_labels, chosen_mask, self.cutpoints)
            + ordinal_nll(rejected_mu, rejected_sigma, rejected_labels, rejected_mask, self.cutpoints)
        )
        helpfulness_ordinal_loss = 0.5 * (
            ordinal_nll(
                chosen_mu[:, :1], chosen_sigma[:, :1], chosen_labels[:, :1],
                chosen_mask[:, :1], self.cutpoints[:1],
            )
            + ordinal_nll(
                rejected_mu[:, :1], rejected_sigma[:, :1], rejected_labels[:, :1],
                rejected_mask[:, :1], self.cutpoints[:1],
            )
        )
        disagreement_loss = 0.5 * (
            annotator_disagreement_loss(
                chosen_mu, chosen_sigma, chosen_labels, chosen_mask, self.cutpoints
            )
            + annotator_disagreement_loss(
                rejected_mu, rejected_sigma, rejected_labels, rejected_mask, self.cutpoints
            )
        )
        mean_difference_loss = helpfulness_mean_difference_nll(
            chosen_mu,
            chosen_sigma,
            rejected_mu,
            rejected_sigma,
            chosen_labels,
            chosen_mask,
            rejected_labels,
            rejected_mask,
            self.cutpoints,
        )
        return (
            ordinal_loss,
            helpfulness_ordinal_loss,
            disagreement_loss,
            mean_difference_loss,
            preference_loss,
            chosen_mu,
            chosen_sigma,
            rejected_mu,
            rejected_sigma,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        (
            ordinal_loss,
            helpfulness_ordinal_loss,
            disagreement_loss,
            mean_difference_loss,
            preference_loss,
            chosen_mu,
            chosen_sigma,
            rejected_mu,
            rejected_sigma,
        ) = self.loss_components(model, inputs)
        loss = (
            ordinal_loss
            + self.helpfulness_nll_weight * helpfulness_ordinal_loss
            + self.disagreement_loss_weight * disagreement_loss
            + self.mean_difference_loss_weight * mean_difference_loss
            + self.preference_loss_weight * preference_loss
        )
        if return_outputs:
            return loss, {
                "ordinal_loss": ordinal_loss,
                "helpfulness_ordinal_loss": helpfulness_ordinal_loss,
                "disagreement_loss": disagreement_loss,
                "mean_difference_loss": mean_difference_loss,
                "preference_loss": preference_loss,
                "chosen_mu": chosen_mu,
                "chosen_sigma": chosen_sigma,
                "rejected_mu": rejected_mu,
                "rejected_sigma": rejected_sigma,
            }
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Use only proper predictive scores for checkpoint selection.

        Generic ``Trainer.prediction_step`` cannot infer that the custom
        ``labels_chosen``/``labels_rejected`` tensors are labels.  The
        The selected score combines ordinal likelihood, a helpfulness-focused
        likelihood, annotator disagreement, and the response-mean log score.
        """
        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                (
                    ordinal_loss,
                    helpfulness_ordinal_loss,
                    disagreement_loss,
                    mean_difference_loss,
                    _,
                    _, _, _, _,
                ) = self.loss_components(model, inputs)
                selection_loss = (
                    ordinal_loss
                    + self.helpfulness_nll_weight * helpfulness_ordinal_loss
                    + self.disagreement_loss_weight * disagreement_loss
                    + self.mean_difference_loss_weight * mean_difference_loss
                )
        return selection_loss.detach().mean(), None, None


def model_kwargs(dtype_name: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"num_labels": 2 * NUM_ATTRIBUTES, "ignore_mismatched_sizes": True}
    if dtype_name == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype_name == "float16":
        kwargs["torch_dtype"] = torch.float16
    return kwargs


def load_model(args, gradient_checkpointing: bool):
    kwargs = model_kwargs(args.torch_dtype)
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    score_head = getattr(model, "score", None)
    if not isinstance(score_head, torch.nn.Linear):
        raise TypeError("Expected a linear sequence-classification score head")
    torch.nn.init.zeros_(score_head.weight)
    if score_head.bias is not None:
        torch.nn.init.zeros_(score_head.bias)
    if gradient_checkpointing:
        model.config.use_cache = False
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[name.strip() for name in args.target_modules.split(",") if name.strip()],
            modules_to_save=["score"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()
    return model, tokenizer


def load_evaluation_model(args):
    checkpoint = Path(args.eval_checkpoint)
    if not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT evaluation checkpoint is incomplete: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    kwargs: dict[str, Any] = {}
    if args.torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif args.torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoPeftModelForSequenceClassification.from_pretrained(checkpoint, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model, tokenizer


def logits_for_rows(model, tokenizer, rows, batch_size: int, max_length: int, sigma_floor: float):
    device = next(model.parameters()).device
    model.eval()
    output = []
    max_annotators = max(
        len(values)
        for row in rows
        for response in ("chosen_annotations", "rejected_annotations")
        for values in row[response]
    )
    collator = PairMomentCollator(tokenizer, max_length, max_annotators)
    with torch.inference_mode():
        for start in range(0, len(rows), max(1, batch_size)):
            batch = collator(rows[start : start + batch_size])
            chosen_logits = model(
                input_ids=batch["input_ids_chosen"].to(device),
                attention_mask=batch["attention_mask_chosen"].to(device),
                return_dict=True,
            ).logits
            rejected_logits = model(
                input_ids=batch["input_ids_rejected"].to(device),
                attention_mask=batch["attention_mask_rejected"].to(device),
                return_dict=True,
            ).logits
            chosen_mu, chosen_sigma = decode_moments(chosen_logits, sigma_floor)
            rejected_mu, rejected_sigma = decode_moments(rejected_logits, sigma_floor)
            output.append(
                {
                    "chosen_mu": chosen_mu.cpu(),
                    "chosen_sigma": chosen_sigma.cpu(),
                    "rejected_mu": rejected_mu.cpu(),
                    "rejected_sigma": rejected_sigma.cpu(),
                    "chosen_labels": batch["labels_chosen"],
                    "chosen_mask": batch["labels_chosen_mask"],
                    "rejected_labels": batch["labels_rejected"],
                    "rejected_mask": batch["labels_rejected_mask"],
                }
            )
    if not output:
        raise ValueError("Cannot score an empty evaluation split")
    return {key: torch.cat([item[key] for item in output], dim=0) for key in output[0]}


def response_label_moments(labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.maximum(mask.sum(axis=-1), 1)
    means = (labels * mask).sum(axis=-1) / counts
    centered = (labels - means[..., None]) * mask
    denominators = np.maximum(counts - 1, 1)
    standard_deviations = np.sqrt((centered**2).sum(axis=-1) / denominators)
    return means, standard_deviations


def response_pair_disagreement_rates(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Observed fraction of unequal retained-annotator pairs per response."""
    annotator_counts = mask.sum(axis=-1)
    total_pairs = annotator_counts * (annotator_counts - 1) / 2.0
    equal_pairs = np.zeros(labels.shape[:2], dtype=np.float64)
    for rating in range(NUM_RATINGS):
        rating_counts = ((labels == rating) & mask).sum(axis=-1)
        equal_pairs += rating_counts * (rating_counts - 1) / 2.0
    return (total_pairs - equal_pairs) / np.maximum(total_pairs, 1.0)


def flatten_annotation_predictions(
    means: np.ndarray,
    sigmas: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expanded_means = np.broadcast_to(means[..., None], labels.shape)
    expanded_sigmas = np.broadcast_to(sigmas[..., None], labels.shape)
    attribute_indices = np.broadcast_to(
        np.arange(NUM_ATTRIBUTES)[None, :, None], labels.shape
    )
    return (
        expanded_means[mask],
        expanded_sigmas[mask],
        labels[mask].astype(np.int64),
        attribute_indices[mask].astype(np.int64),
    )


def ordinal_nll_numpy(
    means: np.ndarray,
    sigmas: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    cutpoints: np.ndarray,
) -> float:
    probabilities = ordinal_probabilities(
        torch.from_numpy(means), torch.from_numpy(sigmas), cutpoints
    ).numpy()
    expanded = np.broadcast_to(probabilities[..., None, :], (*labels.shape, NUM_RATINGS))
    selected = np.take_along_axis(expanded, labels[..., None].clip(0, NUM_RATINGS - 1), axis=-1).squeeze(-1)
    return float(-np.log(np.maximum(selected[mask], EPS)).mean())


def calibrate_sigma_temperatures(
    arrays: dict[str, torch.Tensor], cutpoints: np.ndarray
) -> list[float]:
    means = np.concatenate((arrays["chosen_mu"].numpy(), arrays["rejected_mu"].numpy()), axis=0)
    sigmas = np.concatenate((arrays["chosen_sigma"].numpy(), arrays["rejected_sigma"].numpy()), axis=0)
    labels = np.concatenate((arrays["chosen_labels"].numpy(), arrays["rejected_labels"].numpy()), axis=0)
    mask = np.concatenate((arrays["chosen_mask"].numpy(), arrays["rejected_mask"].numpy()), axis=0)
    temperatures = []
    for attribute in range(NUM_ATTRIBUTES):
        def objective(log_temperature: float) -> float:
            return ordinal_nll_numpy(
                means[:, attribute : attribute + 1],
                sigmas[:, attribute : attribute + 1] * math.exp(log_temperature),
                labels[:, attribute : attribute + 1],
                mask[:, attribute : attribute + 1],
                cutpoints[attribute : attribute + 1],
            )

        result = minimize_scalar(objective, bounds=(math.log(0.1), math.log(10.0)), method="bounded")
        if not result.success or not math.isfinite(float(result.fun)):
            raise RuntimeError(f"Sigma calibration failed for {ATTRIBUTE_NAMES[attribute]}: {result}")
        temperatures.append(float(math.exp(result.x)))
    return temperatures


def apply_sigma_temperatures(
    arrays: dict[str, torch.Tensor], temperatures: list[float]
) -> dict[str, torch.Tensor]:
    calibrated = dict(arrays)
    multiplier = torch.tensor(temperatures, dtype=arrays["chosen_sigma"].dtype).view(1, -1)
    calibrated["chosen_sigma"] = arrays["chosen_sigma"] * multiplier
    calibrated["rejected_sigma"] = arrays["rejected_sigma"] * multiplier
    return calibrated


def binary_ece(probability: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (probability >= lower) & (probability <= upper if index == bins - 1 else probability < upper)
        if selected.any():
            value += float(selected.mean()) * abs(float(probability[selected].mean()) - float(outcome[selected].mean()))
    return value


def ordinal_threshold_ece(probabilities: np.ndarray, labels: np.ndarray) -> float:
    cumulative = np.cumsum(probabilities[:, :-1], axis=1)
    values = []
    for threshold in range(NUM_RATINGS - 1):
        values.append(binary_ece(cumulative[:, threshold], labels <= threshold))
    return float(np.mean(values))


def ordinal_interval_coverage(probabilities: np.ndarray, labels: np.ndarray, level: float) -> float:
    lower_probability = (1.0 - level) / 2.0
    upper_probability = 1.0 - lower_probability
    cumulative = np.cumsum(probabilities, axis=1)
    lower = np.argmax(cumulative >= lower_probability, axis=1)
    upper = np.argmax(cumulative >= upper_probability, axis=1)
    return float(np.mean((labels >= lower) & (labels <= upper)))


def wilson_score_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - radius, center + radius


def metrics_for_rows(
    rows: list[dict[str, Any]],
    arrays: dict[str, torch.Tensor],
    split: str,
    train_attribute_probabilities: np.ndarray,
    cutpoints: np.ndarray,
) -> dict[str, Any]:
    chosen_mu = arrays["chosen_mu"].numpy()
    rejected_mu = arrays["rejected_mu"].numpy()
    chosen_sigma = arrays["chosen_sigma"].numpy()
    rejected_sigma = arrays["rejected_sigma"].numpy()
    chosen_labels = arrays["chosen_labels"].numpy()
    rejected_labels = arrays["rejected_labels"].numpy()
    chosen_mask = arrays["chosen_mask"].numpy()
    rejected_mask = arrays["rejected_mask"].numpy()
    all_mu = np.concatenate((chosen_mu, rejected_mu), axis=0)
    all_sigma = np.concatenate((chosen_sigma, rejected_sigma), axis=0)
    all_labels = np.concatenate((chosen_labels, rejected_labels), axis=0)
    all_mask = np.concatenate((chosen_mask, rejected_mask), axis=0)
    response_means, response_stds = response_label_moments(all_labels, all_mask)
    chosen_label_means, _ = response_label_moments(chosen_labels, chosen_mask)
    rejected_label_means, _ = response_label_moments(rejected_labels, rejected_mask)

    response_probabilities = ordinal_probabilities(
        torch.from_numpy(all_mu.astype(np.float32)),
        torch.from_numpy(all_sigma.astype(np.float32)),
        cutpoints,
    ).numpy()
    expanded_probabilities = np.broadcast_to(
        response_probabilities[:, :, None, :], (*all_labels.shape, NUM_RATINGS)
    )
    flat_probabilities = expanded_probabilities[all_mask]
    flat_labels = all_labels[all_mask].astype(np.int64)
    flat_attributes = np.broadcast_to(
        np.arange(NUM_ATTRIBUTES)[None, :, None], all_labels.shape
    )[all_mask].astype(np.int64)
    selected_probability = flat_probabilities[np.arange(len(flat_labels)), flat_labels]
    ordinal_nll_value = float(-np.log(np.maximum(selected_probability, EPS)).mean())
    climatology_probability = train_attribute_probabilities[flat_attributes, flat_labels]
    climatology_nll = float(-np.log(np.maximum(climatology_probability, EPS)).mean())

    helpfulness = flat_attributes == 0
    helpfulness_probabilities = flat_probabilities[helpfulness]
    helpfulness_labels = flat_labels[helpfulness]
    helpfulness_one_hot = np.eye(NUM_RATINGS)[helpfulness_labels]
    helpfulness_brier = float(
        np.mean(np.sum((helpfulness_probabilities - helpfulness_one_hot) ** 2, axis=1))
    )
    helpfulness_cumulative = np.cumsum(helpfulness_probabilities[:, :-1], axis=1)
    helpfulness_observed_cumulative = (
        helpfulness_labels[:, None] <= np.arange(NUM_RATINGS - 1)[None, :]
    ).astype(np.float64)
    helpfulness_rps = float(
        np.mean(np.sum((helpfulness_cumulative - helpfulness_observed_cumulative) ** 2, axis=1))
    )
    climatology_helpfulness = train_attribute_probabilities[0]
    climatology_cumulative = np.cumsum(climatology_helpfulness[:-1])
    climatology_rps = float(
        np.mean(np.sum((climatology_cumulative - helpfulness_observed_cumulative) ** 2, axis=1))
    )

    rating_values = np.arange(NUM_RATINGS, dtype=np.float64)
    expected_ratings = np.sum(response_probabilities * rating_values, axis=-1)
    rating_variances = np.sum(
        response_probabilities * (rating_values - expected_ratings[..., None]) ** 2,
        axis=-1,
    )
    rating_sigmas = np.sqrt(np.maximum(rating_variances, EPS))
    chosen_reward_mu = expected_ratings[: len(chosen_mu)]
    rejected_reward_mu = expected_ratings[len(chosen_mu) :]

    pair_label_difference = chosen_label_means[:, 0] - rejected_label_means[:, 0]
    pair_prediction_difference = chosen_reward_mu[:, 0] - rejected_reward_mu[:, 0]
    non_tie = pair_label_difference != 0.0
    pair_correct = pair_prediction_difference[non_tie] * pair_label_difference[non_tie] > 0.0
    pair_correct_count = int(pair_correct.sum())
    pair_non_tie_count = int(non_tie.sum())
    pair_accuracy = float(pair_correct.mean()) if pair_non_tie_count else 0.0
    pair_accuracy_interval = wilson_score_interval(pair_correct_count, pair_non_tie_count)
    pair_accuracy_pvalue = (
        float(binomtest(pair_correct_count, pair_non_tie_count, 0.5, alternative="greater").pvalue)
        if pair_non_tie_count
        else 1.0
    )

    mean_correlation = spearmanr(expected_ratings[:, 0], response_means[:, 0])
    sigma_correlation = spearmanr(rating_sigmas[:, 0], response_stds[:, 0])
    latent_mean_correlation = spearmanr(all_mu[:, 0], response_means[:, 0])
    latent_sigma_correlation = spearmanr(all_sigma[:, 0], response_stds[:, 0])
    rating_residual = expected_ratings[:, 0] - response_means[:, 0]
    predicted_disagreement = 1.0 - np.sum(response_probabilities**2, axis=-1)
    observed_disagreement = response_pair_disagreement_rates(all_labels, all_mask)
    disagreement_correlation = spearmanr(
        predicted_disagreement[:, 0], observed_disagreement[:, 0]
    )
    coverage_80 = ordinal_interval_coverage(helpfulness_probabilities, helpfulness_labels, 0.80)
    coverage_90 = ordinal_interval_coverage(helpfulness_probabilities, helpfulness_labels, 0.90)
    coverage_95 = ordinal_interval_coverage(helpfulness_probabilities, helpfulness_labels, 0.95)
    return {
        "split": split,
        "num_pairs": len(rows),
        "num_responses": int(len(all_mu)),
        "num_individual_ratings_all_attributes": int(all_mask.sum()),
        "num_individual_helpfulness_ratings": int(helpfulness.sum()),
        "helpfulness_pair_accuracy_non_ties": pair_accuracy,
        "helpfulness_pair_accuracy_correct": pair_correct_count,
        "helpfulness_pair_accuracy_wilson95_lower": pair_accuracy_interval[0],
        "helpfulness_pair_accuracy_wilson95_upper": pair_accuracy_interval[1],
        "helpfulness_pair_accuracy_binomial_pvalue_vs_half": pair_accuracy_pvalue,
        "helpfulness_pair_non_ties": pair_non_tie_count,
        "ordinal_nll": ordinal_nll_value,
        "ordinal_train_climatology_nll": climatology_nll,
        "ordinal_nll_minus_train_climatology": ordinal_nll_value - climatology_nll,
        "helpfulness_brier": helpfulness_brier,
        "helpfulness_ranked_probability_score": helpfulness_rps,
        "helpfulness_train_climatology_ranked_probability_score": climatology_rps,
        "helpfulness_rps_minus_train_climatology": helpfulness_rps - climatology_rps,
        "ordinal_threshold_ece": ordinal_threshold_ece(helpfulness_probabilities, helpfulness_labels),
        "helpfulness_expected_rating_rmse_against_annotator_mean": float(
            np.sqrt(np.mean(rating_residual**2))
        ),
        "helpfulness_expected_rating_mae_against_annotator_mean": float(
            np.mean(np.abs(rating_residual))
        ),
        "helpfulness_mean_spearman_against_annotator_mean": float(mean_correlation.statistic),
        "helpfulness_mean_spearman_pvalue": float(mean_correlation.pvalue),
        "helpfulness_sigma_mean": float(rating_sigmas[:, 0].mean()),
        "helpfulness_sigma_median": float(np.median(rating_sigmas[:, 0])),
        "observed_helpfulness_disagreement_sd_mean": float(response_stds[:, 0].mean()),
        "helpfulness_sigma_disagreement_spearman": float(sigma_correlation.statistic),
        "helpfulness_sigma_disagreement_spearman_pvalue": float(sigma_correlation.pvalue),
        "helpfulness_latent_mean_spearman_against_annotator_mean": float(
            latent_mean_correlation.statistic
        ),
        "helpfulness_latent_mean_spearman_pvalue": float(latent_mean_correlation.pvalue),
        "helpfulness_latent_sigma_disagreement_spearman": float(
            latent_sigma_correlation.statistic
        ),
        "helpfulness_latent_sigma_disagreement_spearman_pvalue": float(
            latent_sigma_correlation.pvalue
        ),
        "helpfulness_predicted_pair_disagreement_mean": float(
            predicted_disagreement[:, 0].mean()
        ),
        "observed_helpfulness_pair_disagreement_mean": float(
            observed_disagreement[:, 0].mean()
        ),
        "helpfulness_pair_disagreement_spearman": float(
            disagreement_correlation.statistic
        ),
        "helpfulness_pair_disagreement_spearman_pvalue": float(
            disagreement_correlation.pvalue
        ),
        "ordinal_interval_coverage_80": coverage_80,
        "ordinal_interval_coverage_90": coverage_90,
        "ordinal_interval_coverage_95": coverage_95,
        "ordinal_interval_coverage_90_abs_error": abs(coverage_90 - 0.90),
    }


def training_annotation_probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    """Laplace-smoothed annotation frequencies using training ratings only."""
    counts = np.ones((NUM_ATTRIBUTES, NUM_RATINGS), dtype=np.float64)
    for row in rows:
        for side in ("chosen_annotations", "rejected_annotations"):
            for attribute, values in enumerate(row[side]):
                counts[attribute] += np.bincount(np.asarray(values, dtype=np.int64), minlength=NUM_RATINGS)
    return counts / counts.sum(axis=1, keepdims=True)


def climatology_cutpoints(train_attribute_probabilities: np.ndarray) -> np.ndarray:
    """Anchor ordinal-probit location and scale to the training climatology."""
    probabilities = np.asarray(train_attribute_probabilities, dtype=np.float64)
    if probabilities.shape != (NUM_ATTRIBUTES, NUM_RATINGS):
        raise ValueError(
            f"training probabilities must have shape {(NUM_ATTRIBUTES, NUM_RATINGS)}"
        )
    cumulative = np.cumsum(probabilities, axis=1)[:, :-1]
    cutpoints = norm.ppf(cumulative)
    if not np.isfinite(cutpoints).all() or not np.all(np.diff(cutpoints, axis=1) > 0.0):
        raise ValueError("Training-derived ordinal cutpoints must be finite and strictly increasing")
    return cutpoints


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file_path", required=True)
    parser.add_argument("--valid_file_path", required=True)
    parser.add_argument("--test_file_path", required=True)
    parser.add_argument("--preference_train_file_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--sigma_floor", type=float, default=1e-3)
    parser.add_argument("--helpfulness_nll_weight", type=float, default=0.5)
    parser.add_argument("--mean_difference_loss_weight", type=float, default=1.0)
    parser.add_argument("--disagreement_loss_weight", type=float, default=0.5)
    parser.add_argument("--preference_loss_weight", type=float, default=1.0)
    parser.add_argument("--preference_replay_ratio", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default="")
    parser.add_argument("--eval_checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.helpfulness_nll_weight < 0.0:
        raise ValueError("helpfulness_nll_weight must be nonnegative")
    if args.mean_difference_loss_weight < 0.0:
        raise ValueError("mean_difference_loss_weight must be nonnegative")
    if args.disagreement_loss_weight < 0.0:
        raise ValueError("disagreement_loss_weight must be nonnegative")
    if args.preference_loss_weight < 0.0:
        raise ValueError("preference_loss_weight must be nonnegative")
    if args.preference_replay_ratio < 0.0:
        raise ValueError("preference_replay_ratio must be nonnegative")
    if not args.eval_checkpoint and args.preference_replay_ratio > 0.0 and not args.preference_train_file_path:
        raise ValueError("preference_train_file_path is required when preference replay is enabled")
    if args.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least one")
    set_all_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = normalize_pairs(read_jsonl(args.train_file_path))
    validation_rows = normalize_pairs(read_jsonl(args.valid_file_path))
    test_rows = normalize_pairs(read_jsonl(args.test_file_path))
    preference_rows: list[dict[str, Any]] = []
    training_rows = list(train_rows)
    if not args.eval_checkpoint and args.preference_replay_ratio > 0.0:
        preference_pool = normalize_preference_pairs(read_jsonl(args.preference_train_file_path))
        assert_preference_disjoint(preference_pool, (train_rows, validation_rows, test_rows))
        replay_count = min(
            len(preference_pool),
            max(1, int(round(len(train_rows) * args.preference_replay_ratio))),
        )
        random.Random(args.seed + 17).shuffle(preference_pool)
        preference_rows = preference_pool[:replay_count]
        training_rows.extend(preference_rows)
    train_probabilities = training_annotation_probabilities(train_rows)
    cutpoints = climatology_cutpoints(train_probabilities)
    if args.eval_checkpoint:
        model, tokenizer = load_evaluation_model(args)
        evaluation_model = model
        served_model_path = Path(args.eval_checkpoint).resolve()
        print(
            json.dumps(
                {
                    "evaluation_only": True,
                    "selected_checkpoint": str(served_model_path),
                    "train_pairs": len(train_rows),
                    "validation_pairs": len(validation_rows),
                    "test_pairs": len(test_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        model, tokenizer = load_model(args, gradient_checkpointing=True)
        training_kwargs = {
            "output_dir": str(output_dir),
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler_type": "cosine",
            "logging_steps": 20,
            "eval_steps": args.eval_steps,
            "save_strategy": "steps",
            "save_steps": args.save_steps,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "bf16": args.torch_dtype == "bfloat16",
            "fp16": args.torch_dtype == "float16",
            "gradient_checkpointing": True,
            "remove_unused_columns": False,
            "report_to": [],
            "seed": args.seed,
            "data_seed": args.seed,
        }
        evaluation_parameter = (
            "eval_strategy"
            if "eval_strategy" in inspect.signature(TrainingArguments).parameters
            else "evaluation_strategy"
        )
        training_kwargs[evaluation_parameter] = "steps"
        trainer = MomentTrainer(
            model=model,
            args=TrainingArguments(**training_kwargs),
            train_dataset=training_rows,
            eval_dataset=validation_rows,
            data_collator=PairMomentCollator(tokenizer, args.max_length),
            tokenizer=tokenizer,
            sigma_floor=args.sigma_floor,
            cutpoints=cutpoints,
            helpfulness_nll_weight=args.helpfulness_nll_weight,
            mean_difference_loss_weight=args.mean_difference_loss_weight,
            disagreement_loss_weight=args.disagreement_loss_weight,
            preference_loss_weight=args.preference_loss_weight,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_threshold=1e-4,
                )
            ],
        )
        print(
            json.dumps(
                {
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                    ),
                    "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "train_pairs": len(train_rows),
                    "preference_replay_pairs": len(preference_rows),
                    "combined_training_pairs": len(training_rows),
                    "validation_pairs": len(validation_rows),
                    "test_pairs": len(test_rows),
                    "objective": "ordinal_gaussian_nll_plus_helpfulness_nll_plus_sigma_only_disagreement_log_score_plus_mean_difference_gaussian_nll_plus_decontaminated_preference_bt_on_observable_mean",
                    "ordinal_nll_weight": 1.0,
                    "helpfulness_nll_weight": args.helpfulness_nll_weight,
                    "disagreement_loss_weight": args.disagreement_loss_weight,
                    "mean_difference_loss_weight": args.mean_difference_loss_weight,
                    "preference_loss_weight": args.preference_loss_weight,
                    "preference_replay_ratio": args.preference_replay_ratio,
                    "backbone_training_quantization": "NF4 double-quantization with bfloat16 compute" if args.load_in_4bit else "none",
                    "checkpoint_selection": "minimum validation composite of ordinal NLL, helpfulness NLL, disagreement log score, and repeated-rating mean-difference Gaussian NLL",
                    "reward_attribute": "helpfulness",
                    "ordinal_cutpoints_source": "Laplace-smoothed training annotation frequencies only",
                    "zero_head_initialization": "latent_mu=0,latent_sigma=1 reproduces training climatology",
                    "served_reward_moments": "conditional mean and standard deviation of the observable 0--4 rating distribution",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        evaluation_model = trainer.model
        served_model_path = output_dir.resolve()

    validation_arrays = logits_for_rows(
        evaluation_model, tokenizer, validation_rows, args.eval_batch_size, args.max_length, args.sigma_floor
    )
    test_arrays = logits_for_rows(
        evaluation_model, tokenizer, test_rows, args.eval_batch_size, args.max_length, args.sigma_floor
    )
    sigma_temperatures = calibrate_sigma_temperatures(validation_arrays, cutpoints)
    validation_arrays = apply_sigma_temperatures(validation_arrays, sigma_temperatures)
    test_arrays = apply_sigma_temperatures(test_arrays, sigma_temperatures)
    validation_metrics = metrics_for_rows(
        validation_rows, validation_arrays, "validation", train_probabilities, cutpoints
    )
    test_metrics = metrics_for_rows(
        test_rows, test_arrays, "confirmation", train_probabilities, cutpoints
    )

    validation_checks = {
        "mean_pair_ranking": (
            validation_metrics["helpfulness_pair_accuracy_non_ties"] > 0.50
            and validation_metrics["helpfulness_pair_accuracy_binomial_pvalue_vs_half"] < 0.001
        ),
        "proper_score_nll_beats_train_climatology": validation_metrics["ordinal_nll_minus_train_climatology"] < 0.0,
        "proper_score_rps_beats_train_climatology": validation_metrics["helpfulness_rps_minus_train_climatology"] < 0.0,
        "ordinal_calibration": validation_metrics["ordinal_threshold_ece"] <= 0.10,
        "predictive_interval_calibration": validation_metrics["ordinal_interval_coverage_90_abs_error"] <= 0.10,
        "scale_tracks_repeated_annotator_disagreement": (
            validation_metrics["helpfulness_sigma_disagreement_spearman"] > 0.0
            and validation_metrics["helpfulness_sigma_disagreement_spearman_pvalue"] < 0.05
        ),
    }
    test_checks = {
        "mean_pair_ranking": (
            test_metrics["helpfulness_pair_accuracy_non_ties"] > 0.50
            and test_metrics["helpfulness_pair_accuracy_binomial_pvalue_vs_half"] < 0.001
        ),
        "proper_score_nll_beats_train_climatology": test_metrics["ordinal_nll_minus_train_climatology"] < 0.0,
        "proper_score_rps_beats_train_climatology": test_metrics["helpfulness_rps_minus_train_climatology"] < 0.0,
        "ordinal_calibration": test_metrics["ordinal_threshold_ece"] <= 0.10,
        "predictive_interval_calibration": test_metrics["ordinal_interval_coverage_90_abs_error"] <= 0.10,
        "scale_tracks_repeated_annotator_disagreement": (
            test_metrics["helpfulness_sigma_disagreement_spearman"] > 0.0
            and test_metrics["helpfulness_sigma_disagreement_spearman_pvalue"] < 0.05
        ),
    }
    accepted = all(validation_checks.values())
    status = "aleatoric_conditional_scale" if accepted else "failed_validation_not_for_policy_optimization"
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "attributes": list(ATTRIBUTE_NAMES),
        "reward_attribute": "helpfulness",
        "rating_range": [RATING_MIN, RATING_MAX],
        "ordinal_cutpoints_by_attribute": dict(
            zip(ATTRIBUTE_NAMES, cutpoints.tolist())
        ),
        "ordinal_cutpoints_source": "Laplace-smoothed training annotation frequencies only",
        "cutpoint_smoothing_pseudocount_per_category": 1.0,
        "sigma_floor": args.sigma_floor,
        "sigma_temperature_by_attribute": dict(zip(ATTRIBUTE_NAMES, sigma_temperatures)),
        "validation": validation_metrics,
        "test": test_metrics,
        "training_objective": "ordinal Gaussian NLL + helpfulness-focused ordinal NLL + sigma-only annotator-pair disagreement log score + repeated-rating mean-difference Gaussian NLL + decontaminated preference Bradley-Terry log score on observable mean",
        "ordinal_nll_weight": 1.0,
        "helpfulness_nll_weight": args.helpfulness_nll_weight,
        "disagreement_loss_weight": args.disagreement_loss_weight,
        "mean_difference_loss_weight": args.mean_difference_loss_weight,
        "preference_loss_weight": args.preference_loss_weight,
        "preference_replay_ratio": args.preference_replay_ratio,
        "preference_train_pairs": len(preference_rows),
        "preference_training_source": args.preference_train_file_path,
        "preference_data_policy": "training-only exact prompt/response disjoint from all moment splits",
        "preference_gradient_policy": "observable helpfulness mean only; latent scale detached",
        "backbone_training_quantization": "NF4 double-quantization with bfloat16 compute" if args.load_in_4bit else "none",
        "selection_metric": "validation composite: ordinal NLL + helpfulness NLL + disagreement log score + repeated-rating mean-difference Gaussian NLL",
        "validation_acceptance_checks": validation_checks,
        "confirmation_set_checks": test_checks,
        "acceptance_basis": "fixed development validation diagnostics only",
        "confirmation_set_role": "observed post-development diagnostic; reported but not used for acceptance or tuning",
        "publication_lockbox_required": True,
        "accepted_for_ev_ppo": accepted,
        "reward_variance_status": status,
        "reward_variance_validation_basis": "repeated retained HelpSteer2 annotator ratings",
        "variance_definition": "conditional variance of the observable 0--4 human rating induced by the calibrated ordinal distribution",
        "gaussian_approximation_status": "moment-matched Gaussian approximation to a bounded ordinal rating distribution",
        "epistemic_uncertainty_status": "not included in policy reward variance",
    }
    config = {
        "kind": "ordinal_gaussian_moment_rm",
        "model_name": str(served_model_path),
        "model_name_or_path": str(served_model_path),
        "tokenizer_name": str(served_model_path),
        "base_model": args.model_name_or_path,
        "attribute_names": list(ATTRIBUTE_NAMES),
        "reward_attribute": "helpfulness",
        "reward_attribute_index": 0,
        "rating_min": RATING_MIN,
        "rating_max": RATING_MAX,
        "ordinal_cutpoints": cutpoints.tolist(),
        "ordinal_cutpoints_by_attribute": dict(
            zip(ATTRIBUTE_NAMES, cutpoints.tolist())
        ),
        "ordinal_cutpoints_source": "Laplace-smoothed training annotation frequencies only",
        "cutpoint_smoothing_pseudocount_per_category": 1.0,
        "sigma_floor": args.sigma_floor,
        "sigma_temperature": sigma_temperatures[0],
        "sigma_temperature_by_attribute": sigma_temperatures,
        "reward_moment_mapping": "ordinal_induced_observable_rating_moments",
        "latent_mu_parameterization": "unbounded_latent_utility_raw_mu",
        "latent_sigma_parameterization": "(softplus(raw_sigma+inverse_softplus(1-sigma_floor))+sigma_floor)*validation_temperature",
        "mu_parameterization": "sum_r(r*p_r) for calibrated ordinal probabilities",
        "sigma_parameterization": "sqrt(sum_r((r-mu)^2*p_r)) for calibrated ordinal probabilities",
        "sigma_mode": "ordinal_induced_aleatoric_human_rating_standard_deviation",
        "reward_variance_status": status,
        "reward_variance_validation_basis": "repeated retained HelpSteer2 annotator ratings",
        "variance_definition": "conditional variance of the observable 0--4 human rating induced by the calibrated ordinal distribution",
        "uncertainty_type": "aleatoric_conditional_human_rating_scale" if accepted else "unvalidated_scale",
        "not_bayesian_posterior": True,
        "reward_distribution_is_exactly_gaussian": False,
        "gaussian_approximation_status": "moment-matched Gaussian approximation to a bounded ordinal rating distribution",
        "epistemic_uncertainty_status": "separate_from_reward_variance",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_objective": "ordinal Gaussian NLL + helpfulness-focused ordinal NLL + sigma-only annotator-pair disagreement log score + repeated-rating mean-difference Gaussian NLL + decontaminated preference Bradley-Terry log score on observable mean",
        "ordinal_nll_weight": 1.0,
        "helpfulness_nll_weight": args.helpfulness_nll_weight,
        "disagreement_loss_weight": args.disagreement_loss_weight,
        "mean_difference_loss_weight": args.mean_difference_loss_weight,
        "preference_loss_weight": args.preference_loss_weight,
        "preference_replay_ratio": args.preference_replay_ratio,
        "preference_train_pairs": len(preference_rows),
        "preference_training_source": args.preference_train_file_path,
        "preference_data_policy": "training-only exact prompt/response disjoint from all moment splits",
        "preference_gradient_policy": "observable helpfulness mean only; latent scale detached",
        "backbone_training_quantization": "NF4 double-quantization with bfloat16 compute" if args.load_in_4bit else "none",
        "selection_metric": "validation composite: ordinal NLL + helpfulness NLL + disagreement log score + repeated-rating mean-difference Gaussian NLL",
        "acceptance_basis": "fixed development validation diagnostics only",
        "confirmation_set_role": "observed post-development diagnostic; reported but not used for acceptance or tuning",
        "publication_lockbox_required": True,
        "accepted_for_ev_ppo": accepted,
        "validation_acceptance_checks": validation_checks,
        "confirmation_set_checks": test_checks,
    }
    write_json(output_dir / "moment_rm_metrics.json", summary)
    write_json(output_dir / "moment_rm_config.json", config)
    print(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False),
        flush=True,
    )
    if not accepted:
        raise SystemExit(
            "Reward-moment model failed fixed development-validation checks; "
            "policy optimization is intentionally blocked."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
