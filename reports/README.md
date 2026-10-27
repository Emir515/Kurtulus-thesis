# Reports

Material built from the notebook outputs for meetings and for reading. Run the notebooks first (NB01 to NB05); these scripts only read their outputs.

| What | How to build | Output |
|---|---|---|
| Expert review for Wiener Linien: 13-page PDF, 3-page workflow/metrics PDF, copy of the verification sheet | `python reports/generate_expert_review.py` | `outputs/expert_review/` |
| "Thesis Notebooks Explained": every notebook in plain language with all outputs and figures | edit `explainer/notebooks_explained.html`, then `python reports/explainer/build_explainer.py` | `reports/explainer/notebooks_explained_standalone.html` (figures embedded, about 10 MB) |
| NB04 in five steps (no figures) | open `explainer/nb04_workflow.html` | |

The generated files (`outputs/expert_review/`, `explainer/figs/`, the standalone page) are not in git.

When the experts return the sheet, save it as `outputs/nb05_artifacts/expert_verification_sheet_FILLED.csv`; NB05 reads it from there.
