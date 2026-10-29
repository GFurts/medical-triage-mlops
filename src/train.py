import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_cleaning import remove_ambiguous_texts

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MAPPING_PATH = ROOT / "configs" / "urgency_mapping.json"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
RANDOM_STATE = 42


def load_clean_data():
    train_df = pd.read_csv(RAW_DIR / "medical_tc_train.csv")
    test_df = pd.read_csv(RAW_DIR / "medical_tc_test.csv")
    train_clean, test_clean = remove_ambiguous_texts(train_df, test_df)
    train_clean = train_clean.drop_duplicates("medical_abstract")
    test_clean = test_clean.drop_duplicates("medical_abstract")
    return train_clean, test_clean


def load_urgency_mapping():
    with open(MAPPING_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(label): info["urgency"] for label, info in raw.items()}


def build_pipeline(class_weight=None):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(model, texts, labels, mapping):
    pred = model.predict(texts)
    true_urgency = labels.map(mapping)
    pred_urgency = pd.Series(pred, index=labels.index).map(mapping)
    return {
        "f1_macro (5 categorias)": f1_score(labels, pred, average="macro"),
        "recall da classe urgente": recall_score(
            true_urgency == "urgente", pred_urgency == "urgente"
        ),
    }


def compare_class_weights():
    """Experimento: com e sem class_weight, avaliado numa validação tirada do treino."""
    train_clean, _ = load_clean_data()  # o teste fica intocado
    mapping = load_urgency_mapping()

    X_train, X_val, y_train, y_val = train_test_split(
        train_clean["medical_abstract"],
        train_clean["condition_label"],
        test_size=0.2,
        stratify=train_clean["condition_label"],
        random_state=RANDOM_STATE,
    )

    for name, class_weight in [
        ("sem class_weight", None),
        ("class_weight=balanced", "balanced"),
    ]:
        model = build_pipeline(class_weight)
        model.fit(X_train, y_train)
        print(f"\n== {name} ==")
        for metric, value in evaluate(model, X_val, y_val, mapping).items():
            print(f"{metric}: {value:.3f}")


def train_final():
    """Treina o modelo escolhido com todo o treino limpo, avalia no teste e salva."""
    train_clean, test_clean = load_clean_data()
    mapping = load_urgency_mapping()

    model = build_pipeline(class_weight="balanced")
    model.fit(train_clean["medical_abstract"], train_clean["condition_label"])

    metrics = evaluate(
        model, test_clean["medical_abstract"], test_clean["condition_label"], mapping
    )
    metrics["linhas_treino"] = len(train_clean)
    metrics["linhas_teste"] = len(test_clean)

    print("\n== Modelo final (avaliado no teste) ==")
    for metric, value in metrics.items():
        print(
            f"{metric}: {value:.3f}"
            if isinstance(value, float)
            else f"{metric}: {value}"
        )

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nModelo salvo em {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare", action="store_true", help="roda o experimento de class_weight"
    )
    args = parser.parse_args()

    if args.compare:
        compare_class_weights()
    else:
        train_final()


if __name__ == "__main__":
    main()
