#!/usr/bin/env bash
#SBATCH --job-name=stablemax-v8-train
#SBATCH --output=logs/stablemax-v8-train_%j.out
#SBATCH --error=logs/stablemax-v8-train_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:2

set -euo pipefail

if [[ -n "$(printenv PROJECT_ROOT 2>/dev/null)" ]]; then
  ROOT="$(printenv PROJECT_ROOT)"
else
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
REPO=$(printenv EXPERIMENT_REPO || printf '%s' "$ROOT")
ENV_ROOT=$(printenv ENV_ROOT || printf '%s/.venv' "$ROOT")
PY=$(printenv PYTHON_BIN || printf '%s/bin/python' "$ENV_ROOT")
OUT=exp/stablemax_ppo_publication_v8
PROTOCOL=$OUT/protocol
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
RM=exp/strong_rm_candidates/qwen2.5-14b-clean-pref-seed42
QUALITY=exp/quality_rm_qwen14b_v2_seed42/confirmation_eval/best_ckpt
SEED=${V8_SEED:?submit with --export=ALL,V8_SEED=<frozen seed>}

case "$SEED" in 314|2718|1618) ;; *) echo "non-frozen seed $SEED" >&2; exit 2 ;; esac

cd "$REPO"
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$ROOT/hf_cache}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8

sha256sum -c "$PROTOCOL/FROZEN_INPUTS_V8.sha256"
"$PY" - "$PROTOCOL/PUBLICATION_PROTOCOL_V8.json" "$SEED" <<'PY'
import json, sys
protocol=json.load(open(sys.argv[1]))
seed=int(sys.argv[2])
assert protocol["protocol"] == "stablemax-ppo-publication-v8-frozen-before-training"
assert seed in protocol["seeds"] and 42 not in protocol["seeds"]
assert len(protocol["methods"]) == 10
PY

SMOKES=(exp/smoke/publication_v8_job*/SMOKE_VERIFICATION.json)
if [[ ${#SMOKES[@]} -ne 1 || ! -s "${SMOKES[0]}" ]]; then
  echo "expected exactly one successful publication-v8 smoke" >&2
  exit 2
fi
"$PY" - "${SMOKES[0]}" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
assert value["protocol"] == "stablemax-ppo-publication-v8-real-gpu-smoke"
assert value["passed"] is True and len(value["methods"]) == 10
PY

SEED_ROOT=$OUT/seed${SEED}
if [[ -e "$SEED_ROOT" ]]; then
  echo "refusing to overwrite $SEED_ROOT" >&2
  exit 2
fi
mkdir -p "$SEED_ROOT/.logs"

train_method() {
  local gpu=$1 method=$2
  local method_root=$SEED_ROOT/${method}_seed${SEED}
  local log=$SEED_ROOT/.logs/${method}.out
  local err=$SEED_ROOT/.logs/${method}.err
  local extra=()
  if [[ "$method" == ev_ppo_no_quality ]]; then extra+=(--disable_quality_constraint); fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/evrl_experiment.py train \
    --method "$method" \
    --base_model "$BASE_MODEL" \
    --rm_checkpoint "$RM" \
    --robust_calibration_report "$PROTOCOL/ordinal_tail_gate_v7.json" \
    --mean_floor nan \
    --mean_calibration_jsonl "$PROTOCOL/reference_policy_floor_v7.jsonl" \
    --mean_calibration_manifest "$PROTOCOL/reference_policy_manifest_v7.json" \
    --mean_calibration_pairs 2048 \
    --mean_floor_cache "$PROTOCOL/mean_floor_v7.json" \
    --mean_noninferiority_margin_sd 0.1 \
    --mean_dual_init 1.0 --mean_dual_lr 0.05 --mean_dual_max 20.0 \
    --quality_rm_checkpoint "$QUALITY" \
    --quality_floor nan \
    --quality_calibration_jsonl "$PROTOCOL/reference_policy_floor_v7.jsonl" \
    --quality_floor_cache "$PROTOCOL/quality_floor_v7.json" \
    --quality_calibration_manifest "$PROTOCOL/reference_policy_manifest_v7.json" \
    --quality_calibration_pairs 2048 \
    --quality_noninferiority_margin_sd 0.1 \
    --quality_dual_init 1.0 --quality_dual_lr 0.05 --quality_dual_max 20.0 \
    --train_jsonl dataset/paper_pairs_train.jsonl \
    --output_dir "$method_root" \
    --metrics_csv "$method_root/train_metrics.csv" \
    --seed "$SEED" \
    --floor_calibration_seed 20260805 \
    --floor_bootstrap_alpha 0.05 \
    --floor_bootstrap_draws 10000 \
    --floor_bootstrap_seed 26080517 \
    --steps 300 --batch_prompts 2 --group_size 32 --best_of_n 32 \
    --entropic_beta 1.0 --max_new_tokens 64 --max_prompt_length 384 \
    --rm_max_length 512 --rm_batch_size 1 --lr 2e-5 \
    --lora_r 8 --lora_alpha 32 --ppo_epochs 1 --clip_eps 0.2 \
    --kl_coef 0.02 --target_kl 0.02 --hard_kl_limit 0.04 \
    --quadrature_order 64 --quadrature_max_order 1024 \
    --quadrature_tolerance 1e-6 --max_prompt_samples 12000 \
    --temperature 1.0 --top_p 1.0 --generation_batch_size 2 \
    --logprob_batch_size 1 --log_every 10 "${extra[@]}" >"$log" 2>"$err"
}

worker() {
  local gpu=$1; shift
  local method
  for method in "$@"; do
    echo "[publication-v8] seed=$SEED gpu=$gpu start=$method"
    train_method "$gpu" "$method"
    echo "[publication-v8] seed=$SEED gpu=$gpu complete=$method"
  done
}

worker 0 ev_ppo vanilla_grpo entropic_ppo ev_ppo_no_mean gaussian_ev_ppo &
PID0=$!
worker 1 vanilla_ppo scalar_max_ppo nominal_ev_ppo ev_ppo_no_quality top4_ppo &
PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
if [[ $STATUS -ne 0 ]]; then
  echo "one or more training workers failed" >&2
  exit 1
fi

"$PY" - "$SEED_ROOT" "$SEED" <<'PY'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0,"scripts")
from verify_publication_v8_seed import TRAIN_METHODS, verify_training
root=Path(sys.argv[1]); seed=int(sys.argv[2])
report={
  "protocol":"stablemax-ppo-publication-v8-training-terminal",
  "seed":seed,
  "training":verify_training(root,seed,300,0.04),
  "methods":list(TRAIN_METHODS),
  "passed":True,
}
path=root/"TRAINING_VERIFICATION.json"
path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
digest=hashlib.sha256(path.read_bytes()).hexdigest()
(root/"TRAINING_VERIFICATION.json.sha256").write_text(f"{digest}  {path.name}\n")
print(json.dumps(report,indent=2,sort_keys=True))
PY

echo "[publication-v8] training seed=$SEED complete"
