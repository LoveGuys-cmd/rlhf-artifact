#!/usr/bin/env bash
#SBATCH --job-name=stablemax-v8-smoke
#SBATCH --output=logs/stablemax-v8-smoke_%j.out
#SBATCH --error=logs/stablemax-v8-smoke_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ -n "$(printenv PROJECT_ROOT 2>/dev/null)" ]]; then
  ROOT="$(printenv PROJECT_ROOT)"
else
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
REPO=$(printenv EXPERIMENT_REPO || printf '%s' "$ROOT")
ENV_ROOT=$(printenv ENV_ROOT || printf '%s/.venv' "$ROOT")
PY=$(printenv PYTHON_BIN || printf '%s/bin/python' "$ENV_ROOT")
PUBLICATION=exp/stablemax_ppo_publication_v8
PROTOCOL=$PUBLICATION/protocol
OUT=exp/smoke/publication_v8_job${SLURM_JOB_ID}
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
RM=exp/strong_rm_candidates/qwen2.5-14b-clean-pref-seed42
QUALITY=exp/quality_rm_qwen14b_v2_seed42/confirmation_eval/best_ckpt
METHODS=(ev_ppo vanilla_ppo vanilla_grpo scalar_max_ppo entropic_ppo nominal_ev_ppo ev_ppo_no_mean ev_ppo_no_quality gaussian_ev_ppo top4_ppo)

cd "$REPO"
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$ROOT/hf_cache}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8

"$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("Slurm allocated no usable CUDA GPU")
print(f"[publication-v8] GPU={torch.cuda.get_device_name(0)}", flush=True)
PY

if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
sha256sum -c "$PROTOCOL/FROZEN_INPUTS_V8.sha256"
(cd "$PROTOCOL" && sha256sum -c PREPARE_VERIFICATION.json.sha256)
"$PY" tests/test_evrl_math.py
"$PY" tests/test_evrl_ordinal.py
"$PY" tests/test_evrl_reduction.py
"$PY" tests/test_publication_v7.py
"$PY" tests/test_human_eval_analysis.py
"$PY" tests/test_publication_v8.py

for METHOD in "${METHODS[@]}"; do
  EXTRA=()
  if [[ "$METHOD" == "ev_ppo_no_quality" ]]; then EXTRA+=(--disable_quality_constraint); fi
  "$PY" -u scripts/evrl_experiment.py train \
    --method "$METHOD" \
    --base_model "$BASE_MODEL" \
    --rm_checkpoint "$RM" \
    --robust_calibration_report "$PROTOCOL/ordinal_tail_gate_v7.json" \
    --mean_floor nan \
    --mean_calibration_jsonl "$PROTOCOL/reference_policy_floor_v7.jsonl" \
    --mean_calibration_manifest "$PROTOCOL/reference_policy_manifest_v7.json" \
    --mean_calibration_pairs 2048 \
    --mean_floor_cache "$PROTOCOL/mean_floor_v7.json" \
    --mean_noninferiority_margin_sd 0.1 \
    --quality_rm_checkpoint "$QUALITY" \
    --quality_floor nan \
    --quality_calibration_jsonl "$PROTOCOL/reference_policy_floor_v7.jsonl" \
    --quality_floor_cache "$PROTOCOL/quality_floor_v7.json" \
    --quality_calibration_manifest "$PROTOCOL/reference_policy_manifest_v7.json" \
    --quality_calibration_pairs 2048 \
    --quality_noninferiority_margin_sd 0.1 \
    --train_jsonl dataset/paper_pairs_train.jsonl \
    --output_dir "$OUT/${METHOD}_seed42" \
    --metrics_csv "$OUT/${METHOD}_seed42/train_metrics.csv" \
    --seed 42 \
    --floor_calibration_seed 20260805 \
    --floor_bootstrap_alpha 0.05 \
    --floor_bootstrap_draws 10000 \
    --floor_bootstrap_seed 26080517 \
    --steps 1 \
    --batch_prompts 1 \
    --group_size 32 \
    --best_of_n 32 \
    --entropic_beta 1.0 \
    --max_new_tokens 16 \
    --max_prompt_length 128 \
    --rm_max_length 512 \
    --rm_batch_size 1 \
    --lr 2e-5 \
    --lora_r 8 \
    --lora_alpha 32 \
    --ppo_epochs 1 \
    --clip_eps 0.2 \
    --kl_coef 0.02 \
    --target_kl 0.02 \
    --hard_kl_limit 0.04 \
    --max_prompt_samples 64 \
    --temperature 1.0 \
    --top_p 1.0 \
    --generation_batch_size 2 \
    --logprob_batch_size 1 \
    --log_every 1 \
    "${EXTRA[@]}"
done

"$PY" - "$OUT" "$PROTOCOL" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
protocol_root = Path(sys.argv[2])
prepare = json.loads((protocol_root / "PREPARE_VERIFICATION.json").read_text())
if prepare.get("passed") is not True:
    raise SystemExit("publication-v8 prepare verification did not pass")
objectives = {
    "ev_ppo": "stablemax_finite_n_ordinal_expected_max_exact_marginal_credit",
    "vanilla_ppo": "ordinal_expected_rating_mean",
    "vanilla_grpo": "ordinal_expected_rating_group_relative",
    "scalar_max_ppo": "scalar_max_at_n_exact_marginal_credit",
    "entropic_ppo": "pessimistic_entropic_ordinal_certainty_equivalent",
    "nominal_ev_ppo": "finite_n_ordinal_expected_max_exact_marginal_credit",
    "ev_ppo_no_mean": "stablemax_finite_n_ordinal_expected_max_exact_marginal_credit",
    "ev_ppo_no_quality": "stablemax_finite_n_ordinal_expected_max_exact_marginal_credit",
    "gaussian_ev_ppo": "gaussian_expected_max_quadrature_ablation",
    "top4_ppo": "finite_n_probability_any_rating_4_exact_marginal_credit",
}
rows = {}
live_rollbacks = 0
for method, objective in objectives.items():
    path = root / f"{method}_seed42" / "train_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    if row["training_objective"] != objective:
        raise SystemExit(f"wrong smoke objective for {method}")
    allowed_nonfinite = set()
    if method == "ev_ppo_no_mean":
        allowed_nonfinite.update(
            {
                "nominal_mean_score",
                "nominal_mean_score_min",
                "mean_constraint_residual",
                "mean_violation_rate",
                "mean_floor",
            }
        )
    if method == "ev_ppo_no_quality":
        allowed_nonfinite.update(
            {
                "quality_score_mean",
                "quality_score_min",
                "quality_constraint_residual",
                "quality_violation_rate",
                "quality_floor",
            }
        )
    for key, value in row.items():
        if value in ("", "True", "False", "true", "false"):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if not math.isfinite(number) and key not in allowed_nonfinite:
            raise SystemExit(f"non-finite smoke metric {method}.{key}")
    if row["mean_constraint_active"].lower() != str(method != "ev_ppo_no_mean").lower():
        raise SystemExit(f"mean constraint mismatch for {method}")
    if row["quality_constraint_active"].lower() != str(method != "ev_ppo_no_quality").lower():
        raise SystemExit(f"quality constraint mismatch for {method}")
    attempted = float(row["attempted_post_update_kl"])
    float(row["reference_kl"])
    rejected = row["hard_kl_update_rejected"].lower() == "true"
    accepted = row["accepted_transition"].lower() == "true"
    if rejected == accepted:
        raise SystemExit(f"inconsistent transition state for {method}")
    if attempted > 0.04:
        required = (
            "rollback_verification_performed",
            "rollback_parameters_exact",
            "rollback_optimizer_exact",
        )
        if not rejected:
            raise SystemExit(f"live hard-KL violation was not restored for {method}")
        if any(row[key].lower() != "true" for key in required):
            raise SystemExit(f"live hard-KL rollback verification failed for {method}")
        live_rollbacks += 1
    elif rejected:
        raise SystemExit(f"spurious hard-KL rejection for {method}")
    rows[method] = row
report = {
    "protocol": "stablemax-ppo-publication-v8-real-gpu-smoke",
    "job_id": int(root.name.rsplit("job", 1)[1]),
    "methods": list(objectives),
    "objectives": objectives,
    "frozen_manifest_sha256": hashlib.sha256(
        (protocol_root / "FROZEN_INPUTS_V8.sha256").read_bytes()
    ).hexdigest(),
    "prepare_verification_sha256": hashlib.sha256(
        (protocol_root / "PREPARE_VERIFICATION.json").read_bytes()
    ).hexdigest(),
    "applicable_metrics_finite": True,
    "synthetic_exact_rollback_regression_passed": True,
    "live_hard_kl_rollbacks": live_rollbacks,
    "live_hard_kl_protocol_checked": True,
    "passed": True,
}
path = root / "SMOKE_VERIFICATION.json"
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "SMOKE_VERIFICATION.json.sha256").write_text(f"{digest}  {path.name}\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

echo "[publication-v8] smoke complete"
