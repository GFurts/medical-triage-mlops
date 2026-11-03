"""Gera tráfego para popular o dashboard: requisições válidas e algumas inválidas."""

import argparse
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "raw" / "medical_tc_test.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--invalid-every", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    texts = pd.read_csv(TEST_CSV)["medical_abstract"].sample(
        args.requests, random_state=1
    )

    with httpx.Client(timeout=30) as client:
        for i, text in enumerate(texts, start=1):
            payload = {"text": "   "} if i % args.invalid_every == 0 else {"text": text}
            client.post(f"{args.url}/predict", json=payload)
            time.sleep(args.sleep)

    print(
        f"{args.requests} requisições enviadas (1 a cada {args.invalid_every} inválida)"
    )


if __name__ == "__main__":
    main()
