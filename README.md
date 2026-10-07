# credit-risk-governance

Educational, non-production demonstration of fair and explainable credit-risk
validation. It compares a WoE/logistic scorecard with a LightGBM challenger,
calibrates probabilities, evaluates decision economics and subgroup slices, and
monitors PSI/CSI stability.

## Run

```bash
python -m pip install -r requirements.txt
pytest
uvicorn app.main:app --reload
streamlit run app/dashboard.py
```

The FastAPI service exposes `GET /health` and `POST /score`; a trained model must
be explicitly loaded into `app.main.scorecard`. The dashboard accepts a CSV of
governance artifacts for review.

**EDUCATIONAL USE ONLY: Not for production lending. Does not constitute legal or
regulatory compliance.** Review `policy/` and `model_card/` before using the code.