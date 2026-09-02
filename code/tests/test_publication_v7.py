import sys
import json
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from floor_statistics import prompt_cluster_bootstrap_lcb

from freeze_publication_v7 import (
    assert_disjoint,
    deterministic_order,
    remove_global_prompt_response_collisions,
)
from publication_v7_aggregate import (
    SCIENTIFIC_GATE_NAMES,
    STATIC_LOCKBOX_DIAGNOSTIC_NAMES,
    hierarchical_bootstrap,
    load_candidate_group_means,
    paired_rank_biserial,
    paired_sign_flip_pvalue,
    prompt_cluster_average,
    robust_group_value,
)


def preference(prompt: str, chosen: str, rejected: str, subset: str = "test"):
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "subset": subset,
    }


def test_rewardbench_partition_order_is_deterministic():
    row = preference("prompt", "good", "bad")
    assert deterministic_order(row, "salt") == deterministic_order(row, "salt")
    assert deterministic_order(row, "salt") != deterministic_order(row, "other")


def test_rewardbench_partitions_reject_prompt_or_response_overlap():
    assert_disjoint([preference("p1", "a", "b")], [preference("p2", "c", "d")])
    try:
        assert_disjoint([preference("p1", "a", "b")], [preference("p1", "c", "d")])
    except AssertionError:
        pass
    else:
        raise AssertionError("prompt overlap was accepted")
    try:
        assert_disjoint([preference("p1", "a", "b")], [preference("p2", "a", "d")])
    except AssertionError:
        pass
    else:
        raise AssertionError("response overlap was accepted")


def test_global_collision_filter_removes_prompt_and_response_reuse():
    rows = [
        preference("p1", "a", "b"),
        preference("p1", "c", "d"),
        preference("p2", "a", "e"),
        preference("p3", "f", "g"),
    ]
    retained, prompt_collisions, response_collisions = (
        remove_global_prompt_response_collisions(rows)
    )
    assert retained == [rows[0], rows[3]]
    assert prompt_collisions == 1
    assert response_collisions == 1


def test_hierarchical_bootstrap_preserves_constant_effect():
    mean, low, high = hierarchical_bootstrap(
        {42: [0.2] * 8, 314: [0.2] * 8, 2718: [0.2] * 8},
        seed=7,
        draws=2000,
    )
    assert abs(mean - 0.2) < 1e-12
    assert abs(low - 0.2) < 1e-12
    assert abs(high - 0.2) < 1e-12


def test_prompt_cluster_floor_bootstrap_is_deterministic_and_uses_prompts():
    prompts = ["p1", "p1", "p2", "p2", "p3", "p3"]
    scores = torch.tensor([0.0, 2.0, 1.0, 3.0, 2.0, 4.0])
    first, metadata = prompt_cluster_bootstrap_lcb(
        prompts, scores, alpha=0.05, draws=4000, seed=19
    )
    second, repeated = prompt_cluster_bootstrap_lcb(
        prompts, scores, alpha=0.05, draws=4000, seed=19
    )
    assert first == second
    assert metadata == repeated
    assert metadata["floor_prompt_clusters"] == 3
    assert metadata["floor_response_count"] == 6
    assert metadata["floor_min_responses_per_prompt"] == 2
    assert metadata["floor_max_responses_per_prompt"] == 2


def test_prompt_cluster_floor_lcb_preserves_a_constant():
    lcb, metadata = prompt_cluster_bootstrap_lcb(
        ["p1", "p1", "p2", "p2"],
        torch.tensor([1.25, 1.25, 1.25, 1.25]),
        alpha=0.05,
        draws=2000,
        seed=23,
    )
    assert abs(lcb - 1.25) < 1e-12
    assert metadata["floor_estimator"] == "prompt_cluster_percentile_bootstrap_lcb"


def test_candidate_safeguard_bootstrap_uses_complete_groups():
    record = {
        "prompt_id": "p0",
        "responses": [
            {"mu": 1.0, "quality": -1.0},
            {"mu": 3.0, "quality": 1.0},
        ],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "responses.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        nominal, quality = load_candidate_group_means(path, expected_n=2)
    assert nominal == {"p0": 2.0}
    assert quality == {"p0": 0.0}


def test_robust_group_value_matches_manual_single_candidate():
    probabilities = [0.1, 0.2, 0.3, 0.3, 0.1]
    candidate = {f"p{index}": value for index, value in enumerate(probabilities)}
    value = robust_group_value({"responses": [candidate]}, epsilon=0.0)
    expected = sum(index * value for index, value in enumerate(probabilities))
    assert abs(value - expected) < 1e-12


def test_paired_randomization_and_effect_are_deterministic():
    by_seed = {
        42: [0.2, 0.1, -0.1, 0.3],
        314: [0.1, 0.2, 0.0, 0.2],
        2718: [0.3, 0.1, -0.2, 0.4],
    }
    clustered = prompt_cluster_average(by_seed)
    expected = [0.2, 0.13333333333333333, -0.1, 0.3]
    assert all(abs(left - right) < 1e-15 for left, right in zip(clustered, expected))
    first = paired_sign_flip_pvalue(clustered, 17, 4000)
    second = paired_sign_flip_pvalue(clustered, 17, 4000)
    assert first == second
    assert 0.0 < first <= 1.0
    assert paired_rank_biserial(clustered) == 0.5


def test_static_lockboxes_are_not_scientific_success_gates():
    assert not set(SCIENTIFIC_GATE_NAMES) & set(STATIC_LOCKBOX_DIAGNOSTIC_NAMES)
    assert "qwen_generated_proxy" in SCIENTIFIC_GATE_NAMES
    assert "armorm_generated_proxy" in SCIENTIFIC_GATE_NAMES


def test_prepare_py_compile_block_contains_only_python_sources():
    repo = Path(__file__).resolve().parents[1]
    prepare_path = repo / "slurm" / "36_prepare_publication_v8.slurm"
    prepare = prepare_path.read_text(encoding="utf-8")
    block = prepare.split('"$PY" -m py_compile \\\n', 1)[1].split(
        '\n"$PY" tests/test_evrl_math.py', 1
    )[0]
    sources = [line.strip().removesuffix("\\").strip() for line in block.splitlines()]
    assert sources
    assert all(source.startswith("scripts/") and source.endswith(".py") for source in sources)


if __name__ == "__main__":
    for name in sorted(name for name in globals() if name.startswith("test_")):
        globals()[name]()
        print(f"PASS {name}")
