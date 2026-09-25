# StableMax-PPO Reproducibility Artifact

This repository contains the anonymous paper source, ICLR 2027 style files,
aggregate evaluation summaries, and terminal verification metadata for the
Best-of-\(N\) RLHF experiment. Large model caches, virtual environments,
credentials, and full checkpoint blobs are intentionally excluded.

The recorded experiment implementation is preserved under `code/`. It includes the
StableMax-PPO and comparison-method training loop, frozen-protocol evaluation and
diagnostic scripts, unit tests, and the v8 Slurm entry points. The code is
portable and does not contain the original compute host paths.

## Interpretation of the recorded calibration radius

All tables and figures retain the recorded radius
`epsilon_run = 0.04680826120821986`; no training or measurement was rerun.
The paper corrects the independent-cluster population bound to
`sqrt(2 * log(4 / alpha_cal) / G_cal)`. The historical calibration routine in
`code/scripts/evrl_experiment.py` uses the smaller width-one term and is
retained to reproduce the recorded protocol. Its output is not a
certified population calibration radius. Freezing the evaluator after fitting
it on the same calibration labels would also fail the independence premise.
The original cluster report and raw calibration data remain external.
The exact robust value and credit are valid for any fixed radius in [0,1],
while the true-rating lower bound additionally requires pointwise coverage.

The confirmatory experiment uses seeds 314, 2718, and 1618. The aggregate
terminal file reports all gates without modifying or hiding failed criteria.
All confirmatory training and evaluation runs use \(N=32\); this artifact does
not report sensitivity across different values of \(N\).

The frozen ordinal Moment RM is trained on HelpSteer2 repeated helpfulness
ratings, with the decontaminated preference replay recorded in its training
manifest. Its predictive mean is used as a model-implied expected ordinal
rating, and its conditional variance is reported as a model-based signal of
annotator disagreement under that repeated-annotation protocol. The
independent scalar Quality RM is trained separately on preference pairs and is
used only as a quality safeguard. Neither variance nor either model-based
score is presented as a direct human-preference measurement for newly
generated outputs.

## Build

Run `latexmk -pdf paper.tex`, or run `pdflatex paper.tex`, `bibtex paper`,
then `pdflatex paper.tex` twice. Set the Overleaf compiler to pdfLaTeX and
the main document to `paper.tex`. `main.tex` is an identical alternative entry
point. The included `paper.pdf` was rebuilt from these revised sources;
`paper.bbl` is included for convenience.
The source uses the bundled official ICLR 2027 `.sty` and `.bst` files.
`paper.tex` contains nine pages of anonymous main text and submission
statements, followed by references and the complete appendices. The ICLR 2027
style and bibliography files match the official distribution byte for byte. Machine-readable tables are the source
of all reported values.

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
