import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

TEXTS = [
    "tumor cancer chemotherapy",
    "liver stomach intestine",
    "brain neuron stroke seizure",
    "heart coronary myocardial",
    "infection inflammation general",
]


@pytest.fixture
def tiny_model():
    """Pipeline minúsculo, só para os testes não dependerem do modelo real."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]
    ).fit(TEXTS, [1, 2, 3, 4, 5])
