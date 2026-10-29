import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src import api


@pytest.fixture
def client(tmp_path, monkeypatch):
    texts = [
        "tumor cancer chemotherapy",
        "liver stomach intestine",
        "brain neuron stroke seizure",
        "heart coronary myocardial",
        "infection inflammation general",
    ]
    labels = [1, 2, 3, 4, 5]
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]
    ).fit(texts, labels)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setattr(api, "MODEL_PATH", model_path)

    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json={"text": "heart coronary disease"})
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] in {"urgente", "atencao", "normal"}
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
