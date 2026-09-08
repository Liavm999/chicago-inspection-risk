# Interview notes

## The 30-second explanation

I used nearly 60,000 Chicago food inspections to ask whether upcoming failures can be prioritized from pre-inspection information. The hard part was not the classifier; it was creating establishment history without leaking current or same-day outcomes and evaluating on future time periods. I compared a prevalence baseline, logistic regression, and gradient boosting. Boosting reached 0.395 average precision on an untouched 2026 test set versus a 0.244 failure-rate reference, but the calibration and error analysis show it is only a prioritization aid.

## Methodology in plain language

I sorted records by date, calculated each license's history from prior calendar dates, and created contextual features known before an inspection. Training ends April 11, 2025; validation ends December 22, 2025; testing covers later records. Validation selected the model and F1 threshold. I refit that model on development data, touched the test set once, then generated calibration, error, and permutation-importance analyses.

## Hardest decisions

**Defining the prediction moment.** The model acts immediately before an inspection. Violation text, the current outcome, and any field learned during the visit are forbidden.

**Preventing same-day leakage.** A simple row-level `groupby().shift()` can treat another record from the same day as history. I aggregate at license/date, subtract the entire current date, then merge the prior totals back.

**Choosing a metric.** Failures are the minority and the use case is prioritizing a queue, so average precision and precision in the top-risk 10% are more informative than accuracy. ROC AUC and Brier score answer complementary ranking and probability questions.

**Respecting time.** A random split would mix policy regimes and let older rows be predicted using patterns fitted on later records. Whole-date chronological splits better approximate deployment and expose declining fold performance.

## Results without exaggeration

- Test AP: 0.395; test failure prevalence: 0.244.
- Test ROC AUC: 0.671; useful but far from deterministic.
- Top-risk 10% precision: 47.8%; recall: 19.6%.
- The validation-selected F1 threshold flags 50.7% of test rows, so it is not a sensible scarce-capacity policy by itself.
- High predicted probabilities are somewhat overconfident. Recalibration and monitoring would be required.

## Important tradeoffs and mistakes to avoid

ZIP and coordinates may improve ranking while encoding neighborhood inequality. Predictive importance is not causality. The `Fail` label reflects enforcement and administrative practice. The model should never automatically punish a business, and fairness cannot be declared from the available fields alone.

I grouped rare facility labels rather than treating 171 inconsistent strings as equally stable categories. I chose modest model complexity because the data signal and deployment question do not justify deep learning.

## What I would improve in production

First, define the decision with inspection staff: outreach, scheduling, or resource preparation lead to different costs. Then create a point-in-time feature store from authoritative history, test identifier changes, collect delayed labels, assess subgroup impacts, calibrate on rolling windows, monitor feature/outcome drift, and require human review. I would version data and models and use a model registry only when multiple deployed versions create that need.

## Likely questions and strong answers

**Why average precision?**  
It summarizes precision/recall ranking across thresholds and responds to minority-class performance. Accuracy could look good by predicting mostly passes, while ROC AUC can appear comfortable even when the actionable top of the ranking is weak.

**Why not use violations?**  
Violations are observed during the inspection. They would make the score look much better while making pre-inspection prediction impossible—that is target leakage.

**How did you choose the final model?**  
I selected by validation AP, supported by three date-grouped expanding-window folds. The test period did not participate in selection. Boosting beat logistic regression, but logistic remains a valuable interpretable benchmark.

**Why does the model care about inspection type?**  
Complaint, canvass, license, and re-inspection visits are scheduled under different conditions and have different base rates. The field is known beforehand and predictive, but it partly reflects policy, so I would monitor it rather than give it a causal interpretation.

**What does top-decile precision mean?**  
If capacity covers 10% of upcoming inspections, about 47.8% of that ranked test queue failed. It captures only 19.6% of all failures, so widening the queue trades precision for recall.

**Is the model calibrated?**  
Only moderately. The Brier score is 0.173, and the reliability plot shows overprediction in the highest bin. I would recalibrate on a recent held-out window and monitor calibration drift before using probabilities as absolute risk.

**What is the biggest validity threat?**  
The label and inspection schedule are administrative processes. The model predicts recorded outcomes under that process, not latent food safety. Policy changes or selective inspection can shift both features and targets.

