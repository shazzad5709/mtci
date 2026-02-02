from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from mtci.adapters import load_entrypoint
from mtci.models.simple import SimpleSentimentModel


class PredictRequest(BaseModel):
    inputs: List[str]


class PredictResponse(BaseModel):
    scores: List[float]


def _load_model():
    entrypoint = os.getenv("MTCI_MODEL_ENTRYPOINT")
    if entrypoint:
        return load_entrypoint(entrypoint)
    if os.getenv("MTCI_LIGHT_MODEL") == "1":
        return SimpleSentimentModel()
    try:
        from transformers import pipeline
    except Exception:
        return SimpleSentimentModel()

    pipe = pipeline("sentiment-analysis")

    class HFModel:
        def predict(self, xs: List[str]) -> List[float]:
            outputs = pipe(xs)
            scores: List[float] = []
            for out in outputs:
                label = out.get("label", "").upper()
                score = float(out.get("score", 0.0))
                scores.append(score if label == "POSITIVE" else 1.0 - score)
            return scores

    return HFModel()


def create_app() -> FastAPI:
    app = FastAPI()
    model = _load_model()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        scores = model.predict(request.inputs)
        return PredictResponse(scores=[float(x) for x in scores])

    return app
