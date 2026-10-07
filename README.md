# Credit Risk Governance

> **A transparent, educational workspace for credit-risk modeling, fairness review,
> calibration, and stability monitoring.**

| Preview | Link |
| Governance dashboard | [Open Streamlit](http://localhost:8501) |
| Interactive API docs | [Open Swagger UI](http://localhost:8000/docs) |
| API status | [Check health](http://localhost:8000/health) |
![Status](https://img.shields.io/badge/status-educational%20preview-1f6feb)
![Tests](https://img.shields.io/badge/tests-5%20passing-2da44e)
![Use](https://img.shields.io/badge/use-non--production-d1242f)
## The idea

This project demonstrates how a responsible-AI risk workflow can connect model
development with governance. It compares an interpretable Weight of Evidence (WoE)
logistic scorecard with a LightGBM challenger, then evaluates calibration, decision
economics, subgroup outcomes, and distribution stability.
The local preview uses deterministic synthetic data. It is designed for learning,
experimentation, and review of modeling choices, not for making credit decisions.

> **Strict limitation**
> **EDUCATIONAL USE ONLY: Not for production lending. Does not constitute legal or
> regulatory compliance.** Never use this repository to approve, deny, price, or
> limit real credit. Use only public or synthetic data and obtain independent legal,
> privacy, compliance, and fairness review before any real-world use.

## See it running
The services are currently available locally:

- **Dashboard:** [localhost:8501](http://localhost:8501) shows Performance & Calibration,
  Fairness & Subgroups, and Stability tabs.
- **API documentation:** [localhost:8000/docs](http://localhost:8000/docs) provides
	interactive `GET /health` and `POST /score` requests.
- **API landing page:** [localhost:8000](http://localhost:8000) lists the available
	routes and confirms that the service is running.

The API preview trains a small deterministic scorecard at startup. The dashboard
renders synthetic ROC, precision-recall, reliability, subgroup, and CSI artifacts on
first load. Upload a governance CSV in the dashboard to replace those preview values.

## Quick start
### 1. Install and test

```bash
python -m pip install -r requirements.txt
python -m pytest
```

### 2. Start the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Try a preview score:
```bash
curl -X POST http://localhost:8000/score \
	-H 'Content-Type: application/json' \
	-d '{"features":{"income":65000,"debt_to_income":0.32,"credit_history_months":84,"prior_delinquencies":0}}'
```

The response contains a traditional score, probability of default, three reason
codes, and an `educational_use_only` flag.

### 3. Start the dashboard
In a second terminal:

```bash
streamlit run app/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) and switch between the three tabs.
## What is included

| Area | Implementation | Purpose |
| --- | --- | --- |
| Data preparation | `src/data_prep/` | Local data loading, provenance warning, and strict out-of-time splitting |
| Interpretable baseline | `src/scorecard/` | WoE, IV filtering, logistic model, score scaling, and reason codes |
| Challenger | `src/challenger/` | LightGBM wrapper, Platt scaling, isotonic regression, Brier, ECE, ROC-AUC, and PR-AUC |
| Fairness review | `src/fairness/` | Expected-value threshold selection, approval rates, FPR/FNR gaps |
| Monitoring | `src/monitoring/` | PSI, CSI, and stable/warning/alert bands |
| API | `app/main.py` | Health, OpenAPI documentation, and scoring endpoint |
| Dashboard | `app/dashboard.py` | Performance, calibration, fairness, and stability views |
| Governance | `policy/` and `model_card/` | Feature policy, risk limitations, lineage, and manual-review rules |
## Repository map

```text
credit-risk-governance/
├── app/                  FastAPI service and Streamlit dashboard
├── data/                 raw/ and processed/ data boundaries
├── model_card/           intended use, lineage, metrics, and limitations
├── policy/               feature policy and model-risk statement
├── src/
│   ├── challenger/       LightGBM and calibration
│   ├── data_prep/       loading and out-of-time validation
│   ├── fairness/        economics and subgroup analysis
│   ├── monitoring/      PSI and CSI
│   └── scorecard/       WoE and logistic scorecard
├── tests/                focused data, fairness, and monitoring tests
├── Dockerfile            runs API and dashboard together
├── pyproject.toml        package and pytest configuration
└── requirements.txt      runtime dependencies
```

## Responsible modeling workflow

1. Place a reviewed public or synthetic CSV in `data/raw/`, or pass a DataFrame to
	`load_credit_data`.
2. Record source, license, snapshot date, missingness, feature owner, and policy
	decision for every input.
3. Add an application date, or use the documented simulated date column only for
	educational demonstrations.
4. Split chronologically with `out_of_time_split`; every training row must precede
	every validation row.
5. Fit WoE mappings on training data only, then apply the same mappings to validation
	data. Do not leak target or post-outcome information into features.
6. Compare discrimination, calibration, expected value, subgroup gaps, PSI, and CSI.
7. Route missing, out-of-distribution, near-threshold, and active-alert cases to
	manual review according to `policy/model_risk_statement.md`.

## Dashboard upload format

The default preview includes the columns below. A replacement CSV can provide the
same fields to render the corresponding views:

`model`, `fpr`, `tpr`, `recall`, `precision`, `mean_predicted`, `observed_rate`,
`group`, `approval_rate`, `feature`, and `csi`.

## Docker

Run both services in one container:

```bash
docker build -t credit-risk-governance .
docker run --rm -p 8000:8000 -p 8501:8501 credit-risk-governance
```

Then open [the dashboard](http://localhost:8501) or [the API docs](http://localhost:8000/docs).

## Governance notes

Public datasets may be stale, biased, incomplete, or unrepresentative. ROC-AUC can
hide poor minority-class performance; PR-AUC depends on prevalence; calibration and
ECE depend on sample size and binning; fairness gaps describe observed outcomes but
do not establish causality; and PSI/CSI signal distribution change rather than prove
model deterioration.

Read [the feature policy](policy/feature_policy.md),
[the model-risk statement](policy/model_risk_statement.md), and
[the model card](model_card/model_card.md) before extending the project.
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