"""Exporta o modelo treinado para o formato de serviço: TF-IDF (joblib) + floresta (ONNX)."""

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
CHUNK_SIZE = 500


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


def max_probability_gap(pipeline, models_dir, texts):
    """Maior diferença de probabilidade entre o pipeline original e o ONNX exportado."""
    models_dir = Path(models_dir)
    vectorizer = joblib.load(models_dir / "vectorizer.joblib")
    session = ort.InferenceSession(
        str(models_dir / "forest.onnx"), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    gaps = []
    for start in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[start : start + CHUNK_SIZE]
        features = vectorizer.transform(chunk).toarray().astype(np.float32)
        onnx_probabilities = session.run(None, {input_name: features})[1]
        gaps.append(np.abs(pipeline.predict_proba(chunk) - onnx_probabilities).max())
    return float(max(gaps))


if __name__ == "__main__":
    export_model(joblib.load(MODELS_DIR / "model.joblib"))
    print(f"Arquivos exportados para {MODELS_DIR}")
