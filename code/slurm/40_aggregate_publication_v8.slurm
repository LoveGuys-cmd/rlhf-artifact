#!/usr/bin/env bash
#SBATCH --job-name=stablemax-v8-aggregate
#SBATCH --output=logs/stablemax-v8-aggregate_%j.out
#SBATCH --error=logs/stablemax-v8-aggregate_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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
cd "$REPO"
export PYTHONUNBUFFERED=1
sha256sum -c "$PROTOCOL/FROZEN_INPUTS_V8.sha256"
if [[ -e "$OUT/aggregate" ]]; then
  echo "refusing to overwrite publication-v8 aggregate" >&2
  exit 2
fi
for seed in 314 2718 1618; do
  test -s "$OUT/seed${seed}/TERMINAL_VERIFICATION.json"
  (cd "$OUT/seed${seed}" && sha256sum -c TERMINAL_VERIFICATION.json.sha256)
done
"$PY" -u scripts/publication_v8_aggregate.py \
  --root "$OUT" \
  --protocol "$PROTOCOL/PUBLICATION_PROTOCOL_V8.json" \
  --output_dir "$OUT/aggregate" \
  --draws 10000
test -s "$OUT/aggregate/PUBLICATION_V8_TERMINAL.json"
echo "[publication-v8] aggregate complete"
