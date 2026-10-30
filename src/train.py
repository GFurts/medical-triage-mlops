import argparse
import json
import os
import shutil
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
PROCESSED_DIR = ROOT / "data" / "processed"
MAPPING_PATH = ROOT / "configs" / "urgency_mapping.json"
MODELS_DIR = ROOT / "models"
CANDIDATE_PATH = MODELS_DIR / "candidate" / "model.joblib"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
RANDOM_STATE = 42
MIN_URGENT_RECALL = float(os.getenv("MIN_URGENT_RECALL", "0.80"))


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
        "f1_macro": f1_score(labels, pred, average="macro"),
        "urgent_recall": recall_score(
            true_urgency == "urgente", pred_urgency == "urgente"
        ),
    }


def passes_quality_gate(metrics, min_recall=MIN_URGENT_RECALL):
    """Só deixa um modelo novo substituir o atual se o recall urgente for bom o bastante."""
    return metrics["urgent_recall"] >= min_recall


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


def prepare_data():
    """Etapa 1: lê os CSVs brutos, limpa e grava em data/processed/."""
    train_clean, test_clean = load_clean_data()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_clean.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_clean.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(
        f"Dados preparados: {len(train_clean)} linhas de treino, {len(test_clean)} de teste"
    )


def train_model():
    """Etapa 2: treina o modelo e salva como CANDIDATO (não mexe no modelo em uso)."""
    train_clean = pd.read_csv(PROCESSED_DIR / "train.csv")
    model = build_pipeline(class_weight="balanced")
    model.fit(train_clean["medical_abstract"], train_clean["condition_label"])
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CANDIDATE_PATH)
    print(f"Modelo candidato salvo em {CANDIDATE_PATH}")


def evaluate_and_promote():
    """Etapa 3: avalia o candidato no teste e só o promove se passar no portão de qualidade."""
    train_clean = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_clean = pd.read_csv(PROCESSED_DIR / "test.csv")
    mapping = load_urgency_mapping()
    model = joblib.load(CANDIDATE_PATH)

    metrics = evaluate(
        model, test_clean["medical_abstract"], test_clean["condition_label"], mapping
    )
    metrics["train_rows"] = len(train_clean)
    metrics["test_rows"] = len(test_clean)

    print("\n== Modelo candidato (avaliado no teste) ==")
    for name, value in metrics.items():
        print(
            f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}"
        )

    if not passes_quality_gate(metrics):
        raise SystemExit(
            f"Modelo REPROVADO: urgent_recall {metrics['urgent_recall']:.3f} "
            f"< mínimo {MIN_URGENT_RECALL}. O modelo em uso não foi alterado."
        )

    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copyfile(CANDIDATE_PATH, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nModelo aprovado e promovido para {MODEL_PATH}")


STEPS = {
    "prepare": prepare_data,
    "train": train_model,
    "evaluate": evaluate_and_promote,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare", action="store_true", help="roda o experimento de class_weight"
    )
    parser.add_argument(
        "--step", choices=STEPS, help="roda só uma etapa (sem isso, roda todas)"
    )
    args = parser.parse_args()

    if args.compare:
        compare_class_weights()
    elif args.step:
        STEPS[args.step]()
    else:
        for step in STEPS.values():
            step()


if __name__ == "__main__":
    main()
