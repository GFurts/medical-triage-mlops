"""Mede onde o tempo da inferência é gasto e compara a floresta em scikit-learn vs ONNX."""

import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
ONNX_PATH = ROOT / "models" / "forest.onnx"
TEST_CSV = ROOT / "data" / "raw" / "medical_tc_test.csv"
N_TEXTS = 300
WARMUP = 20


def time_each(fn, items):
    """Chama fn(item) uma vez por item e devolve os tempos em ms, depois do aquecimento."""
    for item in items[:WARMUP]:
        fn(item)
    times = []
    for item in items[WARMUP:]:
        start = time.perf_counter()
        fn(item)
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def report(name, times):
    p50, p95 = np.percentile(times, [50, 95])
    print(
        f"{name:<40} p50 {p50:7.2f} ms | p95 {p95:7.2f} ms | média {times.mean():7.2f} ms"
    )


def main():
    pipeline = joblib.load(MODEL_PATH)
    tfidf, forest = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    texts = (
        pd.read_csv(TEST_CSV)["medical_abstract"]
        .sample(N_TEXTS + WARMUP, random_state=42)
        .tolist()
    )

    # Converte SÓ a floresta para ONNX; o TF-IDF continua em scikit-learn
    n_features = len(tfidf.vocabulary_)
    onnx_model = to_onnx(
        forest,
        np.zeros((1, n_features), dtype=np.float32),
        options={id(forest): {"zipmap": False}},
        target_opset={"": 17, "ai.onnx.ml": 3},  # versões que o onnxruntime aceita
    )
    ONNX_PATH.write_bytes(onnx_model.SerializeToString())

    default_session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    one_thread = ort.SessionOptions()
    one_thread.intra_op_num_threads = 1
    single_session = ort.InferenceSession(
        str(ONNX_PATH), sess_options=one_thread, providers=["CPUExecutionProvider"]
    )
    input_name = default_session.get_inputs()[0].name

    vectors = [tfidf.transform([text]) for text in texts]  # entradas já vetorizadas

    def onnx_predict(session):
        return lambda x: session.run(None, {input_name: x.toarray().astype(np.float32)})

    print("\n== Tempo por etapa, uma requisição por vez ==")
    report("TF-IDF (scikit-learn)", time_each(lambda t: tfidf.transform([t]), texts))
    report(
        "Floresta scikit-learn, n_jobs=-1 (atual)",
        time_each(forest.predict_proba, vectors),
    )
    forest.n_jobs = 1
    report("Floresta scikit-learn, n_jobs=1", time_each(forest.predict_proba, vectors))
    report(
        "Floresta ONNX, threads padrão",
        time_each(onnx_predict(default_session), vectors),
    )
    report("Floresta ONNX, 1 thread", time_each(onnx_predict(single_session), vectors))

    # Paridade: o ONNX dá as mesmas probabilidades que o scikit-learn?
    batch = tfidf.transform(texts)
    probs_sklearn = forest.predict_proba(batch)
    probs_onnx = default_session.run(
        None, {input_name: batch.toarray().astype(np.float32)}
    )[1]
    same_class = (probs_sklearn.argmax(axis=1) == probs_onnx.argmax(axis=1)).mean()

    print("\n== Paridade scikit-learn vs ONNX ==")
    print(
        f"Maior diferença de probabilidade: {np.abs(probs_sklearn - probs_onnx).max():.6f}"
    )
    print(f"Mesma classe prevista: {same_class:.1%}")

    print("\n== Tamanho dos arquivos ==")
    print(f"model.joblib (pipeline completo): {MODEL_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"forest.onnx (só a floresta):      {ONNX_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
