import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
MAPPING_PATH = ROOT / "configs" / "urgency_mapping.json"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    state["mapping"] = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    yield
    state.clear()


app = FastAPI(title="Medical Triage API", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=10000)


class PredictResponse(BaseModel):
    category: str
    urgency: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = state["model"]
    probabilities = model.predict_proba([request.text])[0]
    best = int(probabilities.argmax())
    label = int(model.classes_[best])
    info = state["mapping"][str(label)]
    return PredictResponse(
        category=info["category"],
        urgency=info["urgency"],
        confidence=round(float(probabilities[best]), 4),
    )
