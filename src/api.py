import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from src.predictors import OnnxPredictor, SklearnPredictor

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
MAPPING_PATH = ROOT / "configs" / "urgency_mapping.json"
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "onnx")  # "onnx" ou "sklearn"

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]


def _int_env(name):
    value = os.getenv(name)
    return int(value) if value else None


SKLEARN_N_JOBS = _int_env("SKLEARN_N_JOBS")  # só vale para o backend sklearn
ONNX_THREADS = _int_env("ONNX_THREADS")  # só vale para o backend onnx

# Buckets finos também abaixo de 5 ms, porque o ONNX responde em poucos milissegundos
LATENCY_BUCKETS = (
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
)

REQUESTS = Counter(
    "http_requests_total",
    "Total de requisições HTTP",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Duração das requisições HTTP",
    ["method", "path"],
    buckets=LATENCY_BUCKETS,
)

INFERENCE_LATENCY = Histogram(
    "model_inference_seconds",
    "Tempo gasto apenas na inferência (TF-IDF + floresta)",
    buckets=LATENCY_BUCKETS,
)

PREDICTIONS = Counter(
    "predictions_total",
    "Previsões por nível de urgência",
    ["urgency"],
)

state = {}


def build_predictor():
    if MODEL_BACKEND == "onnx":
        return OnnxPredictor(MODELS_DIR, threads=ONNX_THREADS)

    if MODEL_BACKEND == "sklearn":
        return SklearnPredictor(MODELS_DIR, n_jobs=SKLEARN_N_JOBS)

    raise ValueError(
        f"MODEL_BACKEND inválido: {MODEL_BACKEND!r} (use 'onnx' ou 'sklearn')"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = build_predictor()
    state["mapping"] = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

    yield

    state.clear()


app = FastAPI(
    title="Medical Triage API",
    version="0.3.0",
    lifespan=lifespan,
)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.middleware("http")
async def track_requests(request: Request, call_next):
    # /health e /metrics são chamados por máquinas, não por usuários:
    # ficam fora da contagem.
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

        REQUESTS.labels(
            request.method,
            path,
            str(status),
        ).inc()

        REQUEST_LATENCY.labels(
            request.method,
            path,
        ).observe(time.perf_counter() - start)


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        min_length=1,
        max_length=10000,
    )


class PredictResponse(BaseModel):
    category: str
    urgency: str
    confidence: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": MODEL_BACKEND,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    with INFERENCE_LATENCY.time():
        label, confidence = state["predictor"].predict(request.text)

    info = state["mapping"][str(label)]

    PREDICTIONS.labels(info["urgency"]).inc()

    return PredictResponse(
        category=info["category"],
        urgency=info["urgency"],
        confidence=round(confidence, 4),
    )
