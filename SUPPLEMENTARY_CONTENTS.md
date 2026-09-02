# Supplementary Artifact Contents

This anonymous archive contains the source and machine-readable artifacts
needed to inspect the submitted paper and reproduce the analysis pipeline.

## Included

- Anonymous ICLR paper source (`main.tex`, appendix sources, and
  `references.bib`).
- Official ICLR 2027 style files (`iclr2027_conference.sty` and
  `iclr2027_conference.bst`).
- The robust-max/KL figure, plotting script, per-seed tables, held-out KL
  reports, and `PUBLICATION_V8_TERMINAL.json`.
- `code/`, containing the actual v8 training loop, evaluation and held-out KL
  scripts, diagnostics, protocol freezing, aggregate computation, tests, and
  Slurm entry points.
- `code/requirements.txt` and READMEs describing commands and data/model
  assets that are intentionally external.

## External by design

Model checkpoints, private or licensed datasets, raw generated-response
JSONL, training logs, local caches, and credentials are not redistributed.
The paper and code document the expected model identifiers, file layout,
frozen hashes, and command-line arguments. The checked-in aggregate report,
per-seed summary tables, held-out KL reports, and figures are the outputs used
for the reported claims; the terminal manifest records hashes for the raw run
files that remain external. Absence of those files is not a claim that the
experiment can be rerun without access to the corresponding external assets.

## Submission distinction

The conference system receives the anonymous paper PDF as the main
submission and this archive as supplementary material. OpenReview metadata
(title, authors, abstract, subject areas, keywords, policy confirmations, and
reviewer registration) is entered in OpenReview and is not represented by a
ZIP file. The PDF is included in the local convenience bundle when available,
but it must still be uploaded separately as the main submission.
