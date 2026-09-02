#!/usr/bin/env python3
"""Prepare fixed, prompt-disjoint data for EV-PPO paper experiments.

The reward-moment data are the exact prompt-response intersection of the
released HelpSteer2 ``disagreements`` configuration and the official
HelpSteer2 splits.  Every retained response therefore has individual ordinal
ratings and an unambiguous official split.  The official HelpSteer2 validation
prompts are held out as the untouched reward-moment test split.

The anti-reward-hacking quality model is trained from a separate preference
dataset.  Its split is also prompt-disjoint and is never used to fit the main
reward-moment model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RATING_FIELDS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
BAD_MARKERS = ("<|im_start|>", "<|im_end|>", "metadata_|", "endstyff", "[INST]", "</s><s>")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("content") or "").strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role") or "").strip().capitalize()
                content = str(item.get("content") or "").strip()
                if content:
                    parts.append(f"{role}: {content}" if role else content)
            elif str(item).strip():
                parts.append(str(item).strip())
        return "\n".join(parts).strip()
    return str(value).strip()


def contaminated(*texts: str) -> bool:
    blob = "\n".join(texts)
    return len(blob) > 12000 or any(marker in blob for marker in BAD_MARKERS)


def content_hash(text: str) -> str:
    canonical = " ".join(str(text).split()).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_hf_rows(dataset: str, split: str, data_dir: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if data_dir:
        kwargs["data_dir"] = data_dir
    return [dict(row) for row in load_dataset(dataset, **kwargs)]


def annotation_lists(row: dict[str, Any]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for name in RATING_FIELDS:
        values = row.get(name)
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"HelpSteer2 disagreement row lacks repeated {name} ratings")
        ratings = [int(value) for value in values]
        if any(value < 0 or value > 4 for value in ratings):
            raise ValueError(f"Out-of-range {name} rating in HelpSteer2 disagreement row")
        output[name] = ratings
    return output


def response_record(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = text_of(row.get("prompt"))
    response = text_of(row.get("response"))
    if not prompt or not response or contaminated(prompt, response):
        return None
    annotations = annotation_lists(row)
    return {
        "prompt": prompt,
        "response": response,
        "annotations": annotations,
        "helpfulness_mean": sum(annotations["helpfulness"]) / len(annotations["helpfulness"]),
        "num_annotators": len(annotations["helpfulness"]),
    }


def build_moment_pairs(
    disagreements: list[dict[str, Any]],
    official_train: list[dict[str, Any]],
    official_validation: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_keys = {(text_of(row.get("prompt")), text_of(row.get("response"))) for row in official_train}
    test_keys = {(text_of(row.get("prompt")), text_of(row.get("response"))) for row in official_validation}
    if train_keys & test_keys:
        raise AssertionError("Official HelpSteer2 train and validation response keys overlap")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key_split: dict[tuple[str, str], str] = {}
    filtered_disagreement_rows = 0
    unmatched_disagreement_rows = 0
    matched_keys: set[tuple[str, str]] = set()
    identical_duplicate_rows = 0
    merged_annotation_batches = 0
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in disagreements:
        record = response_record(row)
        if record is None:
            filtered_disagreement_rows += 1
            continue
        key = (record["prompt"], record["response"])
        if key in train_keys:
            split = "official_train"
        elif key in test_keys:
            split = "official_validation"
        else:
            unmatched_disagreement_rows += 1
            continue
        if key in matched_keys:
            existing = records_by_key[key]
            if existing["annotations"] == record["annotations"]:
                identical_duplicate_rows += 1
            else:
                for name in RATING_FIELDS:
                    existing["annotations"][name].extend(record["annotations"][name])
                helpfulness = existing["annotations"]["helpfulness"]
                existing["helpfulness_mean"] = sum(helpfulness) / len(helpfulness)
                existing["num_annotators"] = len(helpfulness)
                merged_annotation_batches += 1
            continue
        matched_keys.add(key)
        records_by_key[key] = record
        key_split[key] = split
        grouped[record["prompt"]].append(record)

    train_pairs: list[dict[str, Any]] = []
    test_pairs: list[dict[str, Any]] = []
    incomplete_prompt_count = 0
    retained_keys: set[tuple[str, str]] = set()
    for prompt, records in grouped.items():
        unique = {record["response"]: record for record in records}
        if len(unique) != 2:
            incomplete_prompt_count += 1
            continue
        first, second = unique.values()
        first_split = key_split[(prompt, first["response"])]
        second_split = key_split[(prompt, second["response"])]
        if first_split != second_split:
            raise AssertionError("Responses for one prompt cross official HelpSteer2 splits")
        ordered = sorted(
            (first, second),
            key=lambda item: (item["helpfulness_mean"], item["response"]),
            reverse=True,
        )
        chosen, rejected = ordered
        row = {
            "prompt": prompt,
            "chosen": chosen["response"],
            "rejected": rejected["response"],
            "source": "nvidia/HelpSteer2:disagreements",
            "chosen_annotations": chosen["annotations"],
            "rejected_annotations": rejected["annotations"],
            "chosen_helpfulness_mean": chosen["helpfulness_mean"],
            "rejected_helpfulness_mean": rejected["helpfulness_mean"],
            "preference_strength": chosen["helpfulness_mean"] - rejected["helpfulness_mean"],
            "strict_helpfulness_preference": chosen["helpfulness_mean"] > rejected["helpfulness_mean"],
            "chosen_num_annotators": chosen["num_annotators"],
            "rejected_num_annotators": rejected["num_annotators"],
        }
        (train_pairs if first_split == "official_train" else test_pairs).append(row)
        retained_keys.update(((prompt, first["response"]), (prompt, second["response"])))
    if not train_pairs or not test_pairs:
        raise ValueError("Failed to construct both official-train and official-validation moment pairs")
    audit = {
        "join_key": "exact prompt and response after stripping leading/trailing whitespace",
        "disagreement_rows": len(disagreements),
        "filtered_disagreement_rows": filtered_disagreement_rows,
        "unmatched_disagreement_rows_excluded": unmatched_disagreement_rows,
        "identical_duplicate_rows_deduplicated": identical_duplicate_rows,
        "distinct_annotation_batches_merged": merged_annotation_batches,
        "matched_official_responses": len(matched_keys),
        "official_train_responses": len(train_keys),
        "official_validation_responses": len(test_keys),
        "official_responses_without_disagreement_annotations": len(
            (train_keys | test_keys) - matched_keys
        ),
        "incomplete_matched_prompts_excluded": incomplete_prompt_count,
        "retained_responses": len(retained_keys),
        "retained_train_pairs": len(train_pairs),
        "retained_test_pairs": len(test_pairs),
    }
    return train_pairs, test_pairs, audit


def split_prompt_disjoint(
    rows: list[dict[str, Any]],
    validation_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["prompt"])].append(row)
    prompts = sorted(groups)
    random.Random(seed).shuffle(prompts)
    validation: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for prompt in prompts:
        destination = validation if len(validation) < validation_size else train
        destination.extend(groups[prompt])
    if not train or not validation:
        raise ValueError("Prompt-disjoint split produced an empty partition")
    return train, validation


def split_chat_pair(chosen: Any, rejected: Any) -> tuple[str, str, str] | None:
    if not isinstance(chosen, list) or not isinstance(rejected, list) or not chosen or not rejected:
        return None

    def same_message(left: Any, right: Any) -> bool:
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.get("role") == right.get("role")
            and left.get("content") == right.get("content")
        )

    common = 0
    while common < min(len(chosen), len(rejected)) and same_message(chosen[common], rejected[common]):
        common += 1
    prompt = text_of(chosen[:common])

    def last_assistant(messages: list[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, dict) and str(message.get("role")).lower() == "assistant":
                return text_of(message.get("content"))
        return text_of(messages[-1])

    chosen_response = last_assistant(chosen)
    rejected_response = last_assistant(rejected)
    if prompt and chosen_response and rejected_response and chosen_response != rejected_response:
        return prompt, chosen_response, rejected_response
    return None


def normalize_quality_pair(row: dict[str, Any], source: str) -> dict[str, str] | None:
    split = split_chat_pair(row.get("chosen"), row.get("rejected"))
    if split is None:
        prompt = text_of(row.get("prompt") or row.get("instruction") or row.get("question"))
        chosen = text_of(row.get("chosen"))
        rejected = text_of(row.get("rejected"))
    else:
        prompt, chosen, rejected = split
    if not prompt or not chosen or not rejected or chosen == rejected or contaminated(prompt, chosen, rejected):
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": source}


def build_quality_splits(
    dataset: str,
    seed: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    forbidden_prompt_hashes: set[str],
    forbidden_response_hashes: set[str],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, int],
]:
    raw = load_hf_rows(dataset, "train")
    deduplicated: dict[tuple[str, str, str], dict[str, str]] = {}
    invalid_rows = 0
    cross_task_prompt_overlaps = 0
    cross_task_response_overlaps = 0
    for raw_row in raw:
        row = normalize_quality_pair(raw_row, dataset)
        if row is None:
            invalid_rows += 1
            continue
        if content_hash(row["prompt"]) in forbidden_prompt_hashes:
            cross_task_prompt_overlaps += 1
            continue
        if (
            content_hash(row["chosen"]) in forbidden_response_hashes
            or content_hash(row["rejected"]) in forbidden_response_hashes
        ):
            cross_task_response_overlaps += 1
            continue
        deduplicated[(row["prompt"], row["chosen"], row["rejected"])] = row
    rows = list(deduplicated.values())
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["prompt"]].append(row)
    prompts = sorted(groups)
    random.Random(seed + 1).shuffle(prompts)
    validation: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    train: list[dict[str, str]] = []
    index = 0
    for partition, limit in ((validation, validation_size), (test, test_size)):
        while index < len(prompts) and len(partition) < limit:
            partition.extend(groups[prompts[index]])
            index += 1
    remaining_prompts = prompts[index:]
    if train_size > 0:
        for prompt in remaining_prompts:
            if len(train) >= train_size:
                break
            train.extend(groups[prompt])
    else:
        for prompt in remaining_prompts:
            train.extend(groups[prompt])
    partitions = (train, validation, test)
    if any(not partition for partition in partitions):
        raise ValueError(f"Insufficient valid prompt-disjoint quality pairs from {dataset}")
    audit = {
        "raw_rows": len(raw),
        "invalid_or_contaminated_rows_excluded": invalid_rows,
        "cross_task_prompt_overlaps_excluded": cross_task_prompt_overlaps,
        "cross_task_response_overlaps_excluded": cross_task_response_overlaps,
        "deduplicated_valid_rows_after_cross_task_exclusion": len(rows),
        "unused_valid_rows_after_fixed_splits": max(0, len(rows) - sum(len(partition) for partition in partitions)),
    }
    return partitions[0], partitions[1], partitions[2], audit


def split_quality_v2(
    rows: list[dict[str, str]],
    development_size: int,
    confirmation_size: int,
    lockbox_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["prompt"]].append(row)
    prompts = sorted(groups)
    random.Random(seed + 1009).shuffle(prompts)
    partitions: dict[str, list[dict[str, str]]] = {
        "lockbox": [], "confirmation": [], "development": []
    }
    index = 0
    for name, limit in (("lockbox", lockbox_size), ("confirmation", confirmation_size), ("development", development_size)):
        destination = partitions[name]
        while index < len(prompts) and len(destination) < limit:
            destination.extend(groups[prompts[index]])
            index += 1
    train = [row for prompt in prompts[index:] for row in groups[prompt]]
    splits = (train, partitions["development"], partitions["confirmation"], partitions["lockbox"])
    if any(not split for split in splits):
        raise ValueError("Quality-v2 prompt-disjoint split produced an empty partition")
    return splits


def assert_prompt_disjoint(*splits: list[dict[str, Any]]) -> None:
    prompt_sets = [{str(row["prompt"]) for row in split} for split in splits]
    for left in range(len(prompt_sets)):
        for right in range(left + 1, len(prompt_sets)):
            if prompt_sets[left] & prompt_sets[right]:
                raise AssertionError("Prompt-disjoint split construction failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nvidia/HelpSteer2")
    parser.add_argument("--output_dir", default="dataset")
    parser.add_argument("--moment_validation_size", type=int, default=1000)
    parser.add_argument("--quality_dataset", default="Skywork/Skywork-Reward-Preference-80K-v0.2")
    parser.add_argument(
        "--quality_train_size",
        type=int,
        default=0,
        help="Maximum training pairs after fixed validation/test allocation; 0 uses all remaining pairs.",
    )
    parser.add_argument("--quality_validation_size", type=int, default=5000)
    parser.add_argument("--quality_test_size", type=int, default=5000)
    parser.add_argument("--quality_v2_development_size", type=int, default=4096)
    parser.add_argument("--quality_v2_confirmation_size", type=int, default=4096)
    parser.add_argument("--quality_v2_lockbox_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    official_train = load_hf_rows(args.dataset, "train")
    official_validation = load_hf_rows(args.dataset, "validation")
    disagreements = load_hf_rows(args.dataset, "train", data_dir="disagreements")
    moment_pool, moment_test, moment_join_audit = build_moment_pairs(
        disagreements, official_train, official_validation
    )
    moment_train, moment_validation = split_prompt_disjoint(
        moment_pool, args.moment_validation_size, args.seed
    )
    assert_prompt_disjoint(moment_train, moment_validation, moment_test)

    all_moment_rows = moment_train + moment_validation + moment_test
    moment_prompt_hashes = {content_hash(row["prompt"]) for row in all_moment_rows}
    moment_response_hashes = {
        content_hash(row[side])
        for row in all_moment_rows
        for side in ("chosen", "rejected")
    }
    quality_train, quality_validation, quality_test, quality_overlap_audit = build_quality_splits(
        args.quality_dataset,
        args.seed,
        args.quality_train_size,
        args.quality_validation_size,
        args.quality_test_size,
        moment_prompt_hashes,
        moment_response_hashes,
    )
    assert_prompt_disjoint(quality_train, quality_validation, quality_test)

    quality_v2_train, quality_v2_development, quality_v2_confirmation, quality_v2_lockbox = split_quality_v2(
        quality_train,
        args.quality_v2_development_size,
        args.quality_v2_confirmation_size,
        args.quality_v2_lockbox_size,
        args.seed,
    )
    assert_prompt_disjoint(quality_v2_train, quality_v2_development, quality_v2_confirmation, quality_v2_lockbox)

    output = Path(args.output_dir)
    write_jsonl(output / "paper_pairs_train.jsonl", moment_train)
    write_jsonl(output / "paper_pairs_val.jsonl", moment_validation)
    write_jsonl(output / "paper_pairs_test.jsonl", moment_test)
    write_jsonl(output / "quality_pairs_train.jsonl", quality_train)
    write_jsonl(output / "quality_pairs_val.jsonl", quality_validation)
    write_jsonl(output / "quality_pairs_test.jsonl", quality_test)
    write_jsonl(output / "quality_v2_train.jsonl", quality_v2_train)
    write_jsonl(output / "quality_v2_dev.jsonl", quality_v2_development)
    write_jsonl(output / "quality_v2_confirmation.jsonl", quality_v2_confirmation)
    write_jsonl(output / "quality_v2_lockbox.jsonl", quality_v2_lockbox)

    manifest = {
        "seed": args.seed,
        "moment_dataset": f"{args.dataset}:disagreements",
        "moment_labels": "individual retained annotator ratings, ordinal 0--4",
        "moment_missing_rating_policy": "attribute-wise valid ratings; annotator identities and cross-attribute alignment are unavailable",
        "moment_split_policy": "official train split into prompt-disjoint train/validation; official validation held out as test",
        "moment_join_audit": moment_join_audit,
        "moment_train_pairs": len(moment_train),
        "moment_validation_pairs": len(moment_validation),
        "moment_test_pairs": len(moment_test),
        "moment_min_annotators_per_response": min(
            row[f"{side}_num_annotators"]
            for row in moment_train + moment_validation + moment_test
            for side in ("chosen", "rejected")
        ),
        "quality_dataset": args.quality_dataset,
        "quality_train_pairs": len(quality_train),
        "quality_validation_pairs": len(quality_validation),
        "quality_test_pairs": len(quality_test),
        "quality_v2_protocol": {
            "source": "quality_pairs_train only; legacy validation/test remain report-only",
            "selection_policy": "train on train, select on development, one-shot confirmation, sealed lockbox",
            "train_pairs": len(quality_v2_train),
            "development_pairs": len(quality_v2_development),
            "confirmation_pairs": len(quality_v2_confirmation),
            "lockbox_pairs": len(quality_v2_lockbox),
            "split_seed": args.seed + 1009,
            "prompt_disjoint": True,
            "lockbox_use": "forbidden until code, hyperparameters, and model hash are frozen",
        },
        "cross_task_content_overlap_policy": (
            "exclude normalized exact prompt or response SHA-256 overlap with every reward-moment split"
        ),
        "quality_cross_task_overlap_audit": quality_overlap_audit,
        "cross_task_prompt_and_response_overlap_after_exclusion": 0,
        "prompt_disjoint_within_each_task": True,
        "main_reward_attribute": "helpfulness",
        "auxiliary_attributes": list(RATING_FIELDS[1:]),
        "no_weighted_composite_reward": True,
    }
    (output / "paper_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
