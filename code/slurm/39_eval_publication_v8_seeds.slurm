#!/usr/bin/env bash
#SBATCH --job-name=stablemax-v8-eval
#SBATCH --output=logs/stablemax-v8-eval_%j.out
#SBATCH --error=logs/stablemax-v8-eval_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:2

set -euo pipefail

if [[ -n "$(printenv PROJECT_ROOT 2>/dev/null)" ]]; then
  ROOT="$(printenv PROJECT_ROOT)"
else
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
REPO=$(printenv EXPERIMENT_REPO || printf '%s' "$ROOT")
SEED_LIST=${V8_SEEDS:?submit with --export=ALL,V8_SEEDS=seed[,seed]}
IFS=, read -r -a SEEDS <<<"$SEED_LIST"
if [[ ${#SEEDS[@]} -lt 1 || ${#SEEDS[@]} -gt 2 ]]; then
  echo "V8_SEEDS must contain one or two frozen seeds" >&2
  exit 2
fi
for seed in "${SEEDS[@]}"; do
  case "$seed" in 314|2718|1618) ;; *) echo "non-frozen seed $seed" >&2; exit 2 ;; esac
done
if [[ ${#SEEDS[@]} -eq 2 && "${SEEDS[0]}" == "${SEEDS[1]}" ]]; then
  echo "duplicate evaluation seed" >&2
  exit 2
fi

cd "$REPO"
sha256sum -c exp/stablemax_ppo_publication_v8/protocol/FROZEN_INPUTS_V8.sha256

PIDS=()
for index in "${!SEEDS[@]}"; do
  seed=${SEEDS[$index]}
  seed_root=exp/stablemax_ppo_publication_v8/seed${seed}
  test -s "$seed_root/TRAINING_VERIFICATION.json"
  test ! -e "$seed_root/eval"
  CUDA_VISIBLE_DEVICES=$index bash scripts/run_publication_v8_seed_eval.sh "$seed" \
    >"$seed_root/EVALUATION_PIPELINE.out" \
    2>"$seed_root/EVALUATION_PIPELINE.err" &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
if [[ $STATUS -ne 0 ]]; then
  echo "one or more evaluation workers failed" >&2
  exit 1
fi
for seed in "${SEEDS[@]}"; do
  test -s "exp/stablemax_ppo_publication_v8/seed${seed}/TERMINAL_VERIFICATION.json"
done
echo "[publication-v8] evaluation complete seeds=$SEED_LIST"
