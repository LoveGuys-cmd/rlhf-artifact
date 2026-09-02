# StableMax-PPO Reproducibility Artifact

This repository contains the anonymous paper source, ICLR 2027 style files,
aggregate evaluation summaries, and terminal verification metadata for the
finite-budget RLHF experiment. Large model caches, virtual environments,
credentials, and full checkpoint blobs are intentionally excluded.

The executable experiment implementation is under `code/`. It includes the
StableMax-PPO and comparison-method training loop, frozen-protocol evaluation and
diagnostic scripts, unit tests, and the v8 Slurm entry points. The code is
portable and does not contain the original compute host paths.

The confirmatory experiment uses seeds 314, 2718, and 1618. The aggregate
terminal file reports all gates without modifying or hiding failed criteria.

The frozen Moment RM is trained from HelpSteer2 repeated helpfulness ratings
and preference pairs. Its predictive mean is used as a model-implied expected
ordinal rating, and its conditional variance is reported as a model-based
signal of annotator disagreement under that repeated-annotation protocol. The
variance is not treated as a reward bonus or as a direct human-preference
measurement for newly generated outputs.

## Build

Run `pdflatex paper.tex`, `bibtex paper`, then `pdflatex paper.tex` twice.
The source uses the bundled official ICLR 2027 `.sty` and `.bst` files.
`paper.tex` contains the anonymous nine-page main paper followed by references
and appendices. Machine-readable tables are the source of all reported values.

Regenerate the robust-max--KL Pareto figure with
`python3 plot_max_kl_pareto.py`. The script reads the checked-in per-seed CSV
tables and held-out KL JSON files and writes PDF/PNG outputs under `figures/`.

This repository is an anonymous review artifact. Model weights, credentials,
private datasets, local paths, and author-identifying metadata are excluded.
The supplementary archive contains the same paper artifact plus `code/`; the
main PDF is uploaded separately in the conference submission system. Set
`PROJECT_ROOT` to the `code/` directory when using the Slurm entry points; the
external `exp/` and `dataset/` trees described in `code/README.md` are not
distributed.
