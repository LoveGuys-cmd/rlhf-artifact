#!/usr/bin/env bash
#SBATCH --job-name=stablemax-v8-prepare
#SBATCH --output=logs/stablemax-v8-prepare_%j.out
#SBATCH --error=logs/stablemax-v8-prepare_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

if [[ -n "$(printenv PROJECT_ROOT 2>/dev/null)" ]]; then
  ROOT="$(printenv PROJECT_ROOT)"
else
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
REPO=$(printenv EXPERIMENT_REPO || printf '%s' "$ROOT")
PY=$(printenv PYTHON_BIN || printf '%s' python3)
OUT=exp/stablemax_ppo_publication_v8
PROTOCOL=$OUT/protocol
V7=exp/stablemax_ppo_publication_v7/protocol
PILOT=exp/stablemax_ppo_publication_v8_design/audits/PILOT_PARETO_AUDIT.json
RM=exp/strong_rm_candidates/qwen2.5-14b-clean-pref-seed42
QUALITY=exp/quality_rm_qwen14b_v2_seed42/confirmation_eval/best_ckpt

cd "$REPO"
test ! -e "$OUT"
test -s "$PILOT"
sha256sum -c "$PILOT.sha256"
sha256sum -c "$V7/FROZEN_INPUTS_V7.sha256"

"$PY" -m py_compile \
  scripts/evrl_math.py scripts/floor_statistics.py scripts/evrl_ordinal.py \
  scripts/evrl_experiment.py scripts/eval_policy.py scripts/evrl_analysis_tables.py \
  scripts/paper_metrics.py scripts/reward_hacking_diagnostic.py scripts/evrl_plot.py \
  scripts/human_eval_analysis.py scripts/freeze_publication_v8.py \
  scripts/verify_publication_v8_seed.py scripts/publication_v8_heldout_kl.py \
  scripts/publication_v8_aggregate.py
"$PY" tests/test_evrl_math.py
"$PY" tests/test_evrl_ordinal.py
"$PY" tests/test_evrl_reduction.py
"$PY" tests/test_publication_v7.py
"$PY" tests/test_human_eval_analysis.py
"$PY" tests/test_publication_v8.py
for script in slurm/36_prepare_publication_v8.slurm \
  slurm/37_smoke_publication_v8.slurm \
  slurm/38_train_publication_v8_seed.slurm \
  slurm/39_eval_publication_v8_seeds.slurm \
  slurm/40_aggregate_publication_v8.slurm \
  scripts/run_publication_v8_seed_eval.sh; do bash -n "$script"; done

"$PY" -u scripts/freeze_publication_v8.py \
  --repo . \
  --v7-protocol-dir "$V7" \
  --pilot-audit "$PILOT" \
  --output-dir "$PROTOCOL"

FROZEN_FILES=(
  scripts/evrl_math.py scripts/floor_statistics.py scripts/evrl_ordinal.py
  scripts/evrl_experiment.py scripts/eval_policy.py scripts/evrl_analysis_tables.py
  scripts/paper_metrics.py scripts/reward_hacking_diagnostic.py scripts/evrl_plot.py
  scripts/human_eval_analysis.py scripts/freeze_publication_v8.py
  scripts/verify_publication_v8_seed.py scripts/publication_v8_heldout_kl.py
  scripts/publication_v8_aggregate.py scripts/run_publication_v8_seed_eval.sh
  tests/test_evrl_math.py tests/test_evrl_ordinal.py tests/test_evrl_reduction.py
  tests/test_publication_v7.py tests/test_human_eval_analysis.py
  tests/test_publication_v8.py PUBLICATION_V8_README.md
  slurm/36_prepare_publication_v8.slurm
  slurm/37_smoke_publication_v8.slurm
  slurm/38_train_publication_v8_seed.slurm
  slurm/39_eval_publication_v8_seeds.slurm
  slurm/40_aggregate_publication_v8.slurm
  dataset/paper_pairs_train.jsonl dataset/paper_pairs_val.jsonl
  dataset/paper_pairs_test.jsonl dataset/quality_v2_lockbox.jsonl
  "$PROTOCOL/PUBLICATION_PROTOCOL_V8.json"
  "$PROTOCOL/PILOT_TO_CONFIRMATORY_TRANSITION.json"
  "$PROTOCOL/rewardbench_judge_calibration_v7.jsonl"
  "$PROTOCOL/rewardbench_policy_lockbox_v7.jsonl"
  "$PROTOCOL/rewardbench_reserve_v7.jsonl"
  "$PROTOCOL/reference_policy_floor_v7.jsonl"
  "$PROTOCOL/reference_policy_shift_diagnostic_v7.jsonl"
  "$PROTOCOL/reference_policy_manifest_v7.json"
  "$PROTOCOL/reference_shift_diagnostic_v7.json"
  "$PROTOCOL/reference_shift_diagnostic_v7.json.sha256"
  "$PROTOCOL/mean_floor_v7.json" "$PROTOCOL/quality_floor_v7.json"
  "$PROTOCOL/floor_calibration_v7.json"
  "$PROTOCOL/floor_calibration_v7.json.sha256"
  "$PROTOCOL/ordinal_tail_gate_v7.json"
  "$PROTOCOL/ordinal_tail_gate_v7.json.sha256"
  "$PILOT" "$RM/adapter_model.safetensors" "$RM/moment_rm_config.json"
  exp/quality_rm_qwen14b_v2_seed42/model/adapter_model.safetensors
  "$QUALITY/strong_rm_config.json"
  exp/quality_rm_qwen14b_v2_seed42/confirmation_eval/eval/strong_rm_test_summary.json
  exp/quality_rm_qwen14b_v2_seed42/anti_reward_hacking_audit.json
  exp/quality_rm_qwen14b_v2_seed42/data_manifest.json
)
sha256sum "${FROZEN_FILES[@]}" > "$PROTOCOL/FROZEN_INPUTS_V8.sha256"
sha256sum -c "$PROTOCOL/FROZEN_INPUTS_V8.sha256"

"$PY" - "$PROTOCOL" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root=Path(sys.argv[1])
p=json.loads((root/"PUBLICATION_PROTOCOL_V8.json").read_text())
checks={
 "protocol":p.get("protocol")=="stablemax-ppo-publication-v8-frozen-before-training",
 "pilot_excluded":p.get("seeds")==[314,2718,1618] and 42 not in p.get("seeds",[]),
 "ten_methods":len(p.get("methods",[]))==10,
 "full_budget":p.get("steps")==300 and p.get("evaluation_prompts")==512,
 "finite_n":p.get("best_of_n")==32 and p.get("evaluation_candidate_counts")==[1,2,4,8,16,32],
 "no_optional_stopping":p.get("statistics",{}).get("no_optional_stopping") is True,
 "all_gates_required":p.get("success_criteria",{}).get("claim_rule","").startswith("all gates"),
 "claim_limits":p.get("direct_human_preference_claim_permitted") is False and p.get("absence_of_reward_hacking_claim_permitted") is False,
}
if not all(checks.values()): raise SystemExit(f"v8 prepare failed: {checks}")
report={"protocol":"stablemax-ppo-publication-v8-prepare","checks":checks,"passed":True}
path=root/"PREPARE_VERIFICATION.json"
path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
with path.open("rb") as f: os.fsync(f.fileno())
digest=hashlib.sha256(path.read_bytes()).hexdigest()
(root/"PREPARE_VERIFICATION.json.sha256").write_text(f"{digest}  {path.name}\n")
print(json.dumps(report,indent=2,sort_keys=True))
PY

echo "[publication-v8] prepare complete"
