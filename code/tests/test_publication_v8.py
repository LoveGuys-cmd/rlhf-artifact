import sys
import tempfile
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from freeze_publication_v8 import (
    METHODS,
    REUSED_PROTOCOL_FILES,
    SEEDS,
    assert_unused_seeds,
)
from publication_v8_aggregate import hierarchical_bootstrap
from publication_v8_heldout_kl import reference_log_probs_on_policy_support
from verify_publication_v8_seed import verify_transition_row


def rollback_row(**updates):
    row = {
        "attempted_post_update_kl": "0.05",
        "reference_kl": "0.08",
        "hard_kl_update_rejected": "true",
        "accepted_transition": "false",
        "rollback_verification_performed": "true",
        "rollback_parameters_exact": "true",
        "rollback_optimizer_exact": "true",
    }
    row.update(updates)
    return row


def test_confirmatory_seeds_exclude_development_seed():
    assert SEEDS == (314, 2718, 1618)
    assert 42 not in SEEDS
    assert len(METHODS) == 10


def test_runtime_protocol_sidecars_are_copied_before_freeze():
    required = {
        "ordinal_tail_gate_v7.json.sha256",
        "floor_calibration_v7.json.sha256",
        "reference_shift_diagnostic_v7.json.sha256",
    }
    assert required <= set(REUSED_PROTOCOL_FILES)


def test_exact_rollback_is_authoritative_over_current_minibatch_kl():
    rejected, accepted = verify_transition_row(rollback_row(), "ev_ppo", 0.04)
    assert rejected is True and accepted is False


def test_rollback_requires_exact_snapshot_verification():
    try:
        verify_transition_row(
            rollback_row(rollback_parameters_exact="false"), "ev_ppo", 0.04
        )
    except ValueError as error:
        assert "rollback" in str(error)
    else:
        raise AssertionError("inexact rollback was accepted")


def test_hierarchical_bootstrap_requires_three_seeds():
    try:
        hierarchical_bootstrap({314: [1.0], 2718: [1.0]}, 1, 10)
    except ValueError as error:
        assert "three" in str(error)
    else:
        raise AssertionError("two-seed confirmatory inference was accepted")


def test_seed_collision_scan_fails_closed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "model_seed314").mkdir()
        try:
            assert_unused_seeds(root)
        except FileExistsError as error:
            assert "314" in str(error)
        else:
            raise AssertionError("used seed was accepted")


def test_reference_kl_support_rejects_policy_larger_than_reference():
    class Fake:
        def __init__(self, vocab):
            self.shape = (1, 1, vocab)

        def __getitem__(self, key):
            return self

    try:
        reference_log_probs_on_policy_support(Fake(8), Fake(7), 6)
    except ValueError as error:
        assert "not a prefix" in str(error)
    else:
        raise AssertionError("invalid vocabulary relation was accepted")


if __name__ == "__main__":
    for name in sorted(name for name in globals() if name.startswith("test_")):
        globals()[name]()
        print(f"PASS {name}")
