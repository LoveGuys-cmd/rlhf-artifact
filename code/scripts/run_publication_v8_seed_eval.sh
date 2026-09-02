#!/usr/bin/env bash
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
JUDGE=Qwen/Qwen2.5-14B-Instruct
EXTERNAL=RLHFlow/ArmoRM-Llama3-8B-v0.1
METHODS=ev_ppo,vanilla_ppo,vanilla_grpo,scalar_max_ppo,entropic_ppo,nominal_ev_ppo,ev_ppo_no_mean,ev_ppo_no_quality,gaussian_ev_ppo,top4_ppo,best_of_n
SEED=${1:?usage: run_publication_v8_seed_eval.sh SEED}
case "$SEED" in 314|2718|1618) ;; *) echo "non-frozen seed $SEED" >&2; exit 2 ;; esac
SEED_ROOT=$OUT/seed${SEED}
EVAL=$SEED_ROOT/eval

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
sha256sum -c "$PROTOCOL/FROZEN_INPUTS_V8.sha256"
test -s "$SEED_ROOT/TRAINING_VERIFICATION.json"
(cd "$SEED_ROOT" && sha256sum -c TRAINING_VERIFICATION.json.sha256)
if [[ -e "$EVAL" ]]; then
  echo "refusing to overwrite $EVAL" >&2
  exit 2
fi
for METHOD in ev_ppo vanilla_ppo vanilla_grpo scalar_max_ppo entropic_ppo nominal_ev_ppo ev_ppo_no_mean ev_ppo_no_quality gaussian_ev_ppo top4_ppo; do
  test -s "$SEED_ROOT/${METHOD}_seed${SEED}/adapter_model.safetensors"
done

"$PY" -u scripts/evrl_experiment.py eval \
  --methods "$METHODS" \
  --base_model "$BASE_MODEL" \
  --rm_checkpoint "$RM" \
  --robust_calibration_report "$PROTOCOL/ordinal_tail_gate_v7.json" \
  --mean_calibration_jsonl "$PROTOCOL/reference_policy_floor_v7.jsonl" \
  --mean_calibration_manifest "$PROTOCOL/reference_policy_manifest_v7.json" \
  --mean_calibration_pairs 2048 \
  --mean_floor_cache "$PROTOCOL/mean_floor_v7.json" \
  --mean_noninferiority_margin_sd 0.1 \
  --quality_rm_checkpoint "$QUALITY" \
  --quality_floor_cache "$PROTOCOL/quality_floor_v7.json" \
  --quality_calibration_jsonl "$PROTOCOL/reference_policy_floor_v7.jsonl" \
  --quality_calibration_manifest "$PROTOCOL/reference_policy_manifest_v7.json" \
  --quality_calibration_pairs 2048 \
  --quality_noninferiority_margin_sd 0.1 \
  --eval_jsonl dataset/paper_pairs_test.jsonl \
  --performance_jsonl "$PROTOCOL/rewardbench_policy_lockbox_v7.jsonl" \
  --preference_lockbox "skywork=dataset/quality_v2_lockbox.jsonl" \
  --preference_lockbox_pairs 1024 \
  --experiment_root "$SEED_ROOT" \
  --output_dir "$EVAL" \
  --seed "$SEED" \
  --floor_calibration_seed 20260805 \
  --floor_bootstrap_alpha 0.05 \
  --floor_bootstrap_draws 10000 \
  --floor_bootstrap_seed 26080517 \
  --group_size 32 \
  --best_of_n 32 \
  --max_new_tokens 64 \
  --max_prompt_length 384 \
  --rm_max_length 512 \
  --rm_batch_size 1 \
  --quality_rm_max_length 1024 \
  --quality_rm_batch_size 1 \
  --max_eval_prompts 512 \
  --performance_pairs 1024 \
  --performance_batch_size 1 \
  --temperature 1.0 \
  --top_p 1.0 \
  --generation_batch_size 2

cp "$EVAL/comparison_table_paper.csv" "$EVAL/comparison_table_full.csv"
"$PY" -u scripts/eval_policy.py \
  --eval_dir "$EVAL" \
  --rm_config "$RM/moment_rm_config.json" \
  --robust_calibration_report "$PROTOCOL/ordinal_tail_gate_v7.json" \
  --reference_method vanilla_ppo \
  --best_of_n 32 \
  --seed "$SEED"
"$PY" -u scripts/evrl_analysis_tables.py \
  --eval_dir "$EVAL" \
  --output_dir "$EVAL/analysis" \
  --methods "$METHODS" \
  --seed "$SEED" \
  --require_performance
"$PY" -u scripts/paper_metrics.py \
  --input "$EVAL/analysis/comparison_table_paper.csv" \
  --output "$EVAL/analysis/comparison_table_paper_final.csv" \
  --metadata "$EVAL/analysis/paper_metrics_meta.json" \
  --preference-output "$EVAL/analysis/human_preference_proxy_table.csv" \
  --eval-dir "$EVAL" \
  --rm-config "$RM/moment_rm_config.json" \
  --base-model "$BASE_MODEL" \
  --judge-model "$JUDGE" \
  --external-rm-model "$EXTERNAL" \
  --judge-batch-size 2 \
  --judge-max-length 2048 \
  --judge-max-new-tokens 16 \
  --max-judge-pairs 512 \
  --rewardbench-max-pairs 1024 \
  --rewardbench-calibration-jsonl "$PROTOCOL/rewardbench_judge_calibration_v7.jsonl" \
  --min-calibration-pairs 512 \
  --best-of-n 32 \
  --seed "$SEED"
JUDGE_SLUG=$("$PY" -c 'import sys; sys.path.insert(0,"scripts"); from paper_metrics import model_slug; print(model_slug(sys.argv[1]))' "$JUDGE")
EXTERNAL_SLUG=$("$PY" -c 'import sys; sys.path.insert(0,"scripts"); from paper_metrics import model_slug; print(model_slug(sys.argv[1]))' "$EXTERNAL")
"$PY" -u scripts/reward_hacking_diagnostic.py \
  --eval-dir "$EVAL" \
  --external-details "$EVAL/independent_preference/$EXTERNAL_SLUG/pairwise_scores.csv" \
  --qwen-details "$EVAL/independent_preference/$JUDGE_SLUG/pairwise_judgments.csv" \
  --robust-calibration-report "$PROTOCOL/ordinal_tail_gate_v7.json" \
  --output-dir "$EVAL/analysis/reward_hacking_diagnostic" \
  --external-evaluator "$EXTERNAL" \
  --qwen-evaluator "$JUDGE" \
  --best-of-n 32 \
  --seed "$SEED"
"$PY" -u scripts/evrl_plot.py \
  --table_csv "$EVAL/analysis/comparison_table_paper_final.csv" \
  --preference_csv "$EVAL/analysis/human_preference_proxy_table.csv" \
  --output_dir "$EVAL/figures" \
  --title_suffix "(publication v8 confirmatory; seed=$SEED; N=32)"
"$PY" -u scripts/publication_v8_heldout_kl.py \
  --repo . \
  --pilot-eval-dir "$EVAL" \
  --experiment-root "$SEED_ROOT" \
  --base-model "$BASE_MODEL" \
  --robust-calibration-report "$PROTOCOL/ordinal_tail_gate_v7.json" \
  --seed "$SEED" \
  --best-of-n 32 \
  --max-prompts 512 \
  --max-prompt-length 384 \
  --max-response-tokens 64 \
  --batch-size 1 \
  --bootstrap-draws 10000 \
  --output "$EVAL/HELDOUT_KL.json"
sha256sum "$EVAL/HELDOUT_KL.json" > "$EVAL/HELDOUT_KL.json.sha256"
"$PY" -u scripts/verify_publication_v8_seed.py \
  --repo . \
  --root "$SEED_ROOT" \
  --protocol_dir "$PROTOCOL" \
  --seed "$SEED" \
  --steps 300 \
  --best_of_n 32 \
  --eval_prompts 512 \
  --hard_kl 0.04

echo "[publication-v8] evaluated seed=$SEED"
