import joblib
import pytest
from fastapi.testclient import TestClient

from src import api
from src.onnx_export import export_model


@pytest.fixture(params=["sklearn", "onnx"])
def client(request, tiny_model, tmp_path, monkeypatch):
    joblib.dump(tiny_model, tmp_path / "model.joblib")
    export_model(tiny_model, tmp_path)
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(api, "MODEL_BACKEND", request.param)

    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json={"text": "heart coronary disease"})
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] in {"urgent", "attention", "normal"}
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json={"text": "heart coronary disease"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "predictions_total" in response.text
