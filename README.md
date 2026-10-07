# credit-risk-governance

An educational, non-production project for examining fair and explainable credit-risk
decisioning. It demonstrates an interpretable WoE/logistic scorecard, a LightGBM
challenger, probability calibration, expected-value threshold selection, subgroup
fairness slices, and PSI/CSI stability monitoring.

> **EDUCATIONAL USE ONLY:** Not for production lending. Does not constitute legal or
> regulatory compliance. Never use this project to approve, deny, price, or limit real
> credit. Use only public or synthetic data and obtain independent legal, privacy,
> compliance, and fairness review for any real system.

## Preview

The API starts with a deterministic synthetic scorecard so the local preview has a
working endpoint. The dashboard starts with synthetic governance artifacts so all
three tabs show charts immediately. Both are explicitly labeled as preview data.

Install dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

Start the API in one terminal:

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive API preview. Check the service:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/score \
	-H 'Content-Type: application/json' \
	-d '{"features":{"income":65000,"debt_to_income":0.32,"credit_history_months":84,"prior_delinquencies":0}}'
```

Start the governance dashboard in another terminal:

```bash
streamlit run app/dashboard.py --server.port 8501
```

Open `http://localhost:8501`. The dashboard contains:

- **Performance & Calibration:** synthetic ROC, precision-recall, and reliability diagrams.
- **Fairness & Subgroups:** approval-rate slices by group.
- **Stability:** CSI values and stable/warning/alert classifications.

Upload a CSV with matching governance columns to replace the preview artifacts. The
dashboard recognizes `model`, `fpr`, `tpr`, `recall`, `precision`, `mean_predicted`,
`observed_rate`, `group`, `approval_rate`, `feature`, and `csi` columns.

## Repository guide

```text
policy/                         Feature rules and model-risk limitations
data/raw/                       User-managed public or synthetic source data
data/processed/                 Reproducible derived data artifacts
src/data_prep/                  Loading and chronological OOT validation
src/scorecard/                  WoE, IV, logistic model, scores, reason codes
src/challenger/                 LightGBM and calibration utilities
src/fairness/                   Expected value and subgroup analysis
src/monitoring/                 PSI, CSI, and stability alert bands
app/main.py                     FastAPI health and scoring endpoints
app/dashboard.py                Streamlit governance dashboard
model_card/                     Intended use, lineage, metrics, and limitations
tests/                          Data, fairness, and monitoring tests
```

## Data and modeling workflow

1. Put a reviewed local CSV in `data/raw/` or pass a DataFrame to the loader.
2. Record source, license, snapshot date, missingness, and feature-policy decisions.
3. Add an application date or use the documented simulated date column.
4. Use `out_of_time_split` so training dates are strictly earlier than test dates.
5. Fit the WoE transformer only on training data and apply the same mapping to test data.
6. Compare scorecard and challenger discrimination, calibration, economics, fairness,
	 and stability metrics before any governance decision.
7. Route missing, out-of-distribution, near-threshold, and monitoring-alert cases to
	 manual review as described in `policy/model_risk_statement.md`.

## Docker

Build and run both local services:

```bash
docker build -t credit-risk-governance .
docker run --rm -p 8000:8000 -p 8501:8501 credit-risk-governance
```

The API is on port `8000` and Streamlit is on port `8501`.

## Governance limitations

Public datasets can be stale, biased, incomplete, or unrepresentative. ROC-AUC may
hide poor minority-class performance; PR-AUC depends on prevalence; calibration and
ECE depend on sample size and binning; fairness gaps do not establish causality; and
PSI/CSI indicate distribution change rather than proving model degradation. Read the
full limitations and manual-review rules in `policy/model_risk_statement.md` and the
intended-use statement in `model_card/model_card.md`.