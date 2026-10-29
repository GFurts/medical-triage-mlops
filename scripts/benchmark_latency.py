"""Mede a latência do endpoint /predict (requisições sequenciais, após aquecimento)."""

import argparse
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "raw" / "medical_tc_test.csv"
DOCS_DIR = ROOT / "docs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    texts = (
        pd.read_csv(TEST_CSV)["medical_abstract"]
        .sample(args.requests + args.warmup, random_state=42)
        .tolist()
    )

    latencies_ms = []
    with httpx.Client(timeout=30) as client:
        for i, text in enumerate(texts):
            start = time.perf_counter()
            response = client.post(args.url, json={"text": text})
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            if i >= args.warmup:
                latencies_ms.append(elapsed_ms)

    p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
    results = {
        "label": args.label,
        "url": args.url,
        "requests": len(latencies_ms),
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "max_ms": round(float(np.max(latencies_ms)), 2),
    }

    for key, value in results.items():
        print(f"{key}: {value}")

    DOCS_DIR.mkdir(exist_ok=True)
    out_path = DOCS_DIR / f"latency_{args.label}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
