# StableMax-PPO v8 experiment code

This directory contains the executable training, evaluation, held-out-KL,
diagnostic, aggregation, and verification code used by the publication-v8
protocol. The core policy implementation is `scripts/evrl_experiment.py`.
The v8 orchestration scripts are under `scripts/` and the Slurm entry points
are under `slurm/`.

## Reproduction scope

The repository intentionally excludes model weights, private or licensed
datasets, credentials, caches, and generated checkpoints. A full rerun
therefore requires access to the public base models named in the paper, the
permitted HelpSteer2/Skywork data, the frozen reward-model adapters, and the
protocol assets referenced by the Slurm files. The original run content-hashed
these assets; the hashes and expected paths are recorded by the frozen
protocol, but the external assets are not redistributed here.

The result CSV/JSON files at the repository root are the exact machine-readable
artifacts used in the paper and can be inspected without model downloads.
The aggregate report and per-seed summary tables are released; raw generated
response JSONL, training logs, and checkpoint blobs are not redistributed.
The terminal report records SHA-256 digests for those unreleased run files.

## Environment and layout

The Slurm files treat this `code/` directory as the experiment root. Set
`PROJECT_ROOT` to the absolute path of `code/` (or set `EXPERIMENT_REPO` to a
directory with the same `scripts/`, `tests/`, `slurm/`, `dataset/`, and
`exp/` layout). Set `ENV_ROOT` to a virtual environment and optionally set
`PYTHON_BIN` to its Python executable. The defaults are `PROJECT_ROOT/.venv`
and `python3` where appropriate.

## Tests

From this directory, install `requirements.txt` and run:

    python3 -m pytest tests/test_evrl_math.py tests/test_evrl_ordinal.py \
      tests/test_evrl_reduction.py tests/test_human_eval_analysis.py \
      tests/test_publication_v7.py tests/test_publication_v8.py

The tests cover exact finite-support reductions, ordinal probabilities,
rollback invariants, protocol freezing, aggregate bookkeeping, and blinded
human-evaluation parsing. They do not download models or claim to reproduce
GPU training.

## Frozen v8 entry points

The five Slurm files implement the fixed sequence:

1. `36_prepare_publication_v8.slurm`
2. `37_smoke_publication_v8.slurm`
3. `38_train_publication_v8_seed.slurm`
4. `39_eval_publication_v8_seeds.slurm`
5. `40_aggregate_publication_v8.slurm`

The scripts refuse to overwrite existing protocol outputs, restrict
confirmatory seeds to 314, 2718, and 1618, and retain failed scientific gates
in the aggregate report. They contain no path, username, host name, or
credential from the original compute environment.
