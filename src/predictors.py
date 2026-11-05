"""Backends de inferência: o pipeline original (scikit-learn) e a versão com ONNX Runtime."""

from pathlib import Path

import joblib
import numpy as np


class SklearnPredictor:
    """Pipeline completo (TF-IDF + floresta) em scikit-learn: o modelo original."""

    def __init__(self, models_dir: Path, n_jobs: int | None = None):
        self.pipeline = joblib.load(models_dir / "model.joblib")
        if n_jobs is not None:
            self.pipeline.named_steps["clf"].n_jobs = n_jobs

    def predict(self, text: str) -> tuple[int, float]:
        probabilities = self.pipeline.predict_proba([text])[0]
        best = int(probabilities.argmax())
        return int(self.pipeline.classes_[best]), float(probabilities[best])


class OnnxPredictor:
    """TF-IDF em scikit-learn + floresta em ONNX Runtime."""

    def __init__(self, models_dir: Path, threads: int | None = None):
        import onnxruntime as ort  # aqui dentro, para o backend sklearn não exigir a biblioteca

        self.vectorizer = joblib.load(models_dir / "vectorizer.joblib")
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(models_dir / "forest.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, text: str) -> tuple[int, float]:
        features = self.vectorizer.transform([text]).toarray().astype(np.float32)
        labels, probabilities = self.session.run(None, {self.input_name: features})
        return int(labels[0]), float(probabilities[0].max())
