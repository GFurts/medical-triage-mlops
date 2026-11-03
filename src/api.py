import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
MAPPING_PATH = ROOT / "configs" / "urgency_mapping.json"

# Buckets mais finos na faixa de ~5 a 100 ms, onde nossa latência (~28 ms) vive
LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.035,
    0.04,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    1,
)

REQUESTS = Counter(
    "http_requests_total", "Total de requisições HTTP", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Duração das requisições HTTP",
    ["method", "path"],
    buckets=LATENCY_BUCKETS,
)
INFERENCE_LATENCY = Histogram(
    "model_inference_seconds",
    "Tempo gasto apenas na inferência (TF-IDF + Random Forest)",
    buckets=LATENCY_BUCKETS,
)
PREDICTIONS = Counter(
    "predictions_total", "Previsões por nível de urgência", ["urgency"]
)

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    state["mapping"] = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    yield
    state.clear()


app = FastAPI(title="Medical Triage API", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    # /health e /metrics são chamados por máquinas, não por usuários: ficam fora da contagem
    if request.url.path in {"/health", "/metrics"}:
        return await call_next(request)

    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        REQUESTS.labels(request.method, path, str(status)).inc()
        REQUEST_LATENCY.labels(request.method, path).observe(
            time.perf_counter() - start
        )


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


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = state["model"]
    with INFERENCE_LATENCY.time():
        probabilities = model.predict_proba([request.text])[0]
    best = int(probabilities.argmax())
    label = int(model.classes_[best])
    info = state["mapping"][str(label)]
    PREDICTIONS.labels(info["urgency"]).inc()
    return PredictResponse(
        category=info["category"],
        urgency=info["urgency"],
        confidence=round(float(probabilities[best]), 4),
    )
