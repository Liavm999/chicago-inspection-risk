# Project walkthrough

Follow this route to understand the reasoning and code in about 30 minutes.

1. Read the first half of `README.md` for the question, real result, and leakage controls.
2. Open `src/inspection_risk/data.py`. The downloader asks Socrata only for ten needed columns and deterministic chronological pages. The loader validates the schema and parses dates.
3. Spend most of your attention on `src/inspection_risk/features.py`. `prepare_model_frame` is the crucial logic. It aggregates each license/date before calculating cumulative history, so same-day outcomes are unavailable to peer inspections. It then adds known context and cyclical month fields.
4. Read `chronological_split` in the same file. Entire dates are assigned to one partition. The final test set is genuinely later than validation.
5. Open `src/inspection_risk/modeling.py`. Both real models include their preprocessing. `fit_and_compare` touches validation; the test set stays untouched. `expanding_window_scores` repeats the comparison across earlier time blocks.
6. Read `src/inspection_risk/training.py`. This is the experimental protocol in executable form: EDA, compare, select, choose a threshold on validation, refit, evaluate test once, then interpret.
7. Finish with `src/inspection_risk/evaluation.py` and the files in `artifacts/reports/`. They show calibration, permutation importance, and where threshold errors occur.

## Trace one feature

For each license and inspection date, the code first counts that day's inspections and failures. Cumulative totals subtract the current day before merging back to inspection rows. `smoothed_prior_fail_rate = (prior_failures + 1) / (prior_inspections + 4)`. The first visit starts at 0.25 instead of an unjustified extreme, and every later value uses only earlier dates.

## Run small experiments

- Remove history features and compare validation AP to quantify their contribution.
- Replace the chronological split with a random one and observe why that estimate is less deployment-like.
- Change the operational queue from the top 10% to top 20% and explain the precision/recall tradeoff.
- Inspect `segment_error_analysis.csv` and decide which segment deserves deeper review before deployment.

