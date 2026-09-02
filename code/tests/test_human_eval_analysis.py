import csv
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from human_eval_analysis import analyze_groups, analyze_pairwise


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_locked_human_analysis_uses_blinded_labels_only():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pair_key = root / "pair_key.csv"
        pair_labels = root / "pair_labels.csv"
        write_rows(
            pair_key,
            [{"pair_id": "H1", "prompt_key": "p1", "method_a": "ev_ppo", "method_b": "vanilla_ppo"}],
        )
        write_rows(
            pair_labels,
            [
                {"pair_id": "H1", "annotator_id": annotator, "label": "A"}
                for annotator in ("r1", "r2", "r3")
            ],
        )
        pair_summary, _ = analyze_pairwise(pair_labels, pair_key, 3, 42, 1000)
        assert pair_summary["ev_ppo_preference_score_vs_vanilla_ppo"] == 1.0
        assert pair_summary["paired_randomization_p_value"] <= 1.0

        group_key = root / "group_key.csv"
        group_labels = root / "group_labels.csv"
        write_rows(
            group_key,
            [
                {"group_pair_id": "G1", "prompt_key": "p1", "group_label": "A", "method": "ev_ppo"},
                {"group_pair_id": "G1", "prompt_key": "p1", "group_label": "B", "method": "vanilla_ppo"},
            ],
        )
        rows = []
        for annotator in ("r1", "r2", "r3"):
            for group_label, ratings in (("A", (4, 2)), ("B", (3, 1))):
                for index, rating in enumerate(ratings, start=1):
                    rows.append(
                        {
                            "group_pair_id": "G1",
                            "group_label": group_label,
                            "candidate_id": f"{group_label}{index:02d}",
                            "annotator_id": annotator,
                            "rating_0_to_4": str(rating),
                        }
                    )
        write_rows(group_labels, rows)
        group_summary, _ = analyze_groups(group_labels, group_key, 3, 2, 42, 1000)
        assert group_summary["ev_ppo_minus_vanilla_realized_max"] == 1.0
        assert group_summary["ev_ppo_minus_vanilla_any_rating_4"] == 1.0
        assert group_summary["ev_ppo_minus_vanilla_candidate_mean"] == 1.0
        assert group_summary["candidate_mean_randomization_p_value"] <= 1.0


if __name__ == "__main__":
    test_locked_human_analysis_uses_blinded_labels_only()
    print("PASS test_locked_human_analysis_uses_blinded_labels_only")
