"""Exporta o modelo treinado para o formato de serviço: TF-IDF (joblib) + floresta (ONNX)."""

from pathlib import Path

import joblib
import numpy as np
from skl2onnx import to_onnx

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def export_model(pipeline, models_dir=MODELS_DIR):
    """Grava vectorizer.joblib e forest.onnx em models_dir, a partir do pipeline treinado."""
    vectorizer = pipeline.named_steps["tfidf"]
    forest = pipeline.named_steps["clf"]

    onnx_model = to_onnx(
        forest,
        np.zeros((1, forest.n_features_in_), dtype=np.float32),
        options={id(forest): {"zipmap": False}},
        target_opset={"": 17, "ai.onnx.ml": 3},
    )

    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    joblib.dump(vectorizer, models_dir / "vectorizer.joblib")
    (models_dir / "forest.onnx").write_bytes(onnx_model.SerializeToString())


if __name__ == "__main__":
    export_model(joblib.load(MODELS_DIR / "model.joblib"))
    print(f"Arquivos exportados para {MODELS_DIR}")
