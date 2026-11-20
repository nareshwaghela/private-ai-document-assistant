#!/usr/bin/env python3
"""Real fixed-question benchmark. Never invent model or GPU measurements."""
from __future__ import annotations
import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from rag_local import ask, encoder, load_store


def run(cases_path: Path, store: Path, base: str, repeats: int = 5, warmup=True,
        top_k: int = 4):
    if not 1 <= repeats <= 100:
        raise ValueError("--runs must be between 1 and 100")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    index, records = load_store(store)
    model = encoder()
    api_key = os.environ.get("LLAMA_SERVER_API_KEY")
    if warmup:
        _ = ask(cases[0]["question"], model, index, records, base, top_k=top_k, api_key=api_key)
    results = []
    for run_index in range(repeats):
        for case in cases:
            result = ask(case["question"], model, index, records, base,
                         top_k=top_k, api_key=api_key)
            result["case_id"] = case["id"]
            result["run"] = run_index + 1
            result["review_expectation"] = case["review_expectation"]
            results.append(result)
            print(f"Run {run_index + 1}/{repeats}: {case['id']} ({result['total_s']:.2f}s)")
    medians = {
        name: statistics.median(row[name] for row in results)
        for name in ("retrieval_s", "generation_s", "total_s")
    }
    return {"utc": datetime.now(timezone.utc).isoformat(), "base_url_redacted": "llama-server",
            "runs": repeats, "cases": len(cases), "top_k": top_k,
            "median_seconds": medians, "raw_results": results,
            "note": "Actual wall-clock timing for this run only; manually review answers and cited evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("eval_cases.json"))
    parser.add_argument("--store", type=Path, default=Path("store"))
    parser.add_argument("--base", default=os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("reports/baseline.json"))
    args = parser.parse_args()
    report = run(args.cases, args.store, args.base, args.runs, top_k=args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Recorded actual measurements in {args.out}; check raw answers manually.")


if __name__ == "__main__":
    main()
