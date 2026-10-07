"""FastAPI scoring surface for the educational demo."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.scorecard.logistic_model import LogisticScorecard

app = FastAPI(title="Credit Risk Governance Demo", version="0.1.0")
scorecard: LogisticScorecard | None = None


class Applicant(BaseModel):
    features: dict[str, Any] = Field(description="Feature values matching the trained educational model")


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