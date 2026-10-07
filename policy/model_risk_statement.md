# Model Risk Statement

**Status:** Educational, non-production demonstration. This does not constitute
legal or regulatory compliance and is not a lending recommendation.

## Known limitations

- Public datasets may be stale, unrepresentative, noisy, or missing relevant context.
- OOT validation is simulated when application dates are absent and cannot prove future performance.
- ROC-AUC can hide poor minority-class performance; PR-AUC depends on prevalence.
- Calibration, Brier score, and ECE are sensitive to sample size and binning choices.
- Selection-rate and error-rate gaps describe observed outcomes; they do not establish causality.
- PSI/CSI are alerts for distribution change, not proof of model deterioration.

## Manual review routing

Route an application to manual review when required fields are missing, the score is
outside the validated range, the predicted probability is within 0.05 of the policy
cutoff, an input is out of distribution, or a monitoring alert is active. A trained
reviewer must verify the data, record the reason, and follow applicable law and policy.

## Governance

Record data lineage, code version, model version, calibration method, threshold,
reason codes, fairness slices, monitoring results, and reviewer decisions. Obtain
independent legal, compliance, privacy, and fairness review before any real use.