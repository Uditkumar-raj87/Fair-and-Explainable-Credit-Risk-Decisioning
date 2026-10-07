# Educational Credit-Risk Governance Model Card

## Model details

This repository demonstrates an interpretable WoE-transformed logistic scorecard
and a LightGBM challenger. Probabilities can be calibrated with Platt scaling or
isotonic regression. The scorecard maps good-to-bad log odds to a target score of
600 at 50:1 odds with PDO 20 and emits feature-level reason codes.

## Intended use and users

Intended users are risk analysts, data scientists, and responsible-AI learners
evaluating validation, calibration, decision economics, subgroup outcomes, and
stability monitoring. It is not intended for real lending, underwriting, pricing,
credit limit management, or adverse-action decisions.

## Data lineage

Use a public Home Credit Default Risk or German Credit dataset, or synthetic data.
The loader accepts a reviewed local CSV or injected DataFrame and logs the required
educational-use warning. Record dataset URL, license, snapshot date, feature policy,
preprocessing version, and any simulated application dates in a project run log.

## Evaluation

Use chronological out-of-time validation where possible. Report ROC-AUC/Gini,
PR-AUC, Brier score, ECE, expected value at the selected threshold, approval rates,
FPR/FNR gaps, PSI for scores, and CSI for input features. Interpret every metric
with its limitations in `policy/model_risk_statement.md`.

## Fairness and human oversight

Protected-attribute slices are evaluation diagnostics, not decision rules. Small
groups, missing attributes, proxy effects, label bias, and historical inequity can
make estimates unreliable. Route missing data, out-of-distribution cases, active
stability alerts, and probabilities near the threshold to documented manual review.

## Limitations and risk controls

Public data may not represent a lender's population, economic environment, or legal
requirements. Calibration and fairness results can shift under prevalence changes.
No output should be used without independent validation, privacy review, legal and
compliance review, documented monitoring ownership, and a governed change process.

**Strict limitation:** EDUCATIONAL USE ONLY: Not for production lending. Does not
constitute legal or regulatory compliance.