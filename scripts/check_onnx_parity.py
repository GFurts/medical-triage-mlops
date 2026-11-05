"""Compara scikit-learn e ONNX em todo o conjunto de teste limpo (um texto por vez)."""

from pathlib import Path

import numpy as np

from src.predictors import OnnxPredictor, SklearnPredictor
from src.train import load_clean_data

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main():
    _, test = load_clean_data()
    sklearn_predictor = SklearnPredictor(MODELS_DIR, n_jobs=1)
    onnx_predictor = OnnxPredictor(MODELS_DIR)

    different = 0
    max_confidence_gap = 0.0
    for text in test["medical_abstract"]:
        sklearn_label, sklearn_conf = sklearn_predictor.predict(text)
        onnx_label, onnx_conf = onnx_predictor.predict(text)
        max_confidence_gap = max(max_confidence_gap, abs(sklearn_conf - onnx_conf))
        if sklearn_label != onnx_label:
            different += 1
            top_two = np.sort(sklearn_predictor.pipeline.predict_proba([text])[0])[-2:]
            print(
                f"Classe diferente (sklearn={sklearn_label}, onnx={onnx_label}); "
                f"duas maiores probabilidades no sklearn: {top_two}"
            )

    total = len(test)
    print(f"\nTextos: {total}")
    print(f"Classes diferentes: {different} ({different / total:.2%})")
    print(f"Maior diferença de confiança: {max_confidence_gap:.6f}")


if __name__ == "__main__":
    main()
