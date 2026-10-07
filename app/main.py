"""FastAPI scoring surface for the educational demo."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.scorecard.logistic_model import LogisticScorecard

app = FastAPI(title="Credit Risk Governance Demo", version="0.1.0")


def build_demo_scorecard() -> LogisticScorecard:
    """Train a deterministic toy model so the local preview has a working score endpoint."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "income": rng.normal(60000, 15000, 160).clip(18000),
            "debt_to_income": rng.uniform(0.1, 0.8, 160),
            "credit_history_months": rng.integers(6, 240, 160),
            "prior_delinquencies": rng.poisson(0.7, 160),
        }
    )
    risk = 2.8 * frame["debt_to_income"] - frame["income"] / 90000 - frame["credit_history_months"] / 400 + frame["prior_delinquencies"] * 0.35
    probability = 1 / (1 + np.exp(-risk))
    target = (rng.random(len(frame)) < probability).astype(int)
    return LogisticScorecard(iv_threshold=0.0).fit(frame, pd.Series(target))


scorecard: LogisticScorecard | None = build_demo_scorecard()


class Applicant(BaseModel):
    features: dict[str, Any] = Field(description="Feature values matching the trained educational model")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Credit Risk Governance Demo",
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "score": "/score",
        "educational_use_only": True,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "use": "educational only"}


@app.post("/score")
def score(applicant: Applicant) -> dict[str, Any]:
    if scorecard is None:
        raise HTTPException(status_code=503, detail="No model loaded. This demo requires an explicitly trained model.")
    frame = pd.DataFrame([applicant.features])
    probability = float(scorecard.predict_proba(frame)[0])
    return {
        "score": float(scorecard.predict_score(frame)[0]),
        "probability_of_default": probability,
        "reason_codes": scorecard.reason_codes(frame, top_n=3)[0],
        "educational_use_only": True,
    }