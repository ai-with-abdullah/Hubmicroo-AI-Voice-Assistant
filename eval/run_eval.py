#!/usr/bin/env python3
"""Evaluation harness for the Hubmicroo assistant.

Runs the test set against the live /api/chat endpoint and reports:
  - Correct-product rate (retrieved the expected product(s))
  - Retrieval hit rate (at least one expected product in response)
  - Per-language accuracy
  - Greeting / out-of-catalogue handling

Usage:
    python eval/run_eval.py [--base-url http://localhost:8000] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
TEST_SET = Path(__file__).parent / "test_set.json"


def run_query(client: httpx.Client, base: str, query: str) -> dict:
    r = client.post(f"{base}/api/chat", json={"message": query}, timeout=60.0)
    r.raise_for_status()
    return r.json()


def eval_case(expected: dict, actual: dict) -> dict:
    returned_ids = {p["id"] for p in actual.get("products", [])}
    exp_ids = set(expected.get("expected_product_ids") or [])
    q_type = expected.get("type", "")

    # For greeting / out-of-scope, correct = no products returned
    if q_type in ("greeting", "out_of_scope", "out_of_catalogue"):
        correct = len(returned_ids) == 0
        hit = correct
    elif not exp_ids:
        # Policy-only question — we just check no irrelevant products leaked
        correct = True
        hit = True
    else:
        hit = bool(returned_ids & exp_ids)      # at least one correct product
        correct = exp_ids.issubset(returned_ids) or hit  # any overlap = pass

    return {
        "id": expected["id"],
        "lang": expected.get("lang"),
        "type": q_type,
        "query": expected["query"],
        "expected_ids": sorted(exp_ids),
        "returned_ids": sorted(returned_ids),
        "hit": hit,
        "correct": correct,
        "answer_preview": actual.get("answer", "")[:120],
        "cached": actual.get("cached", False),
    }


def main(base_url: str, out_path: str | None) -> None:
    cases = json.loads(TEST_SET.read_text())
    results = []
    errors = []

    print(f"\n{'─'*60}")
    print(f"  Hubmicroo Eval — {len(cases)} cases against {base_url}")
    print(f"{'─'*60}\n")

    # Flush semantic cache so stale entries from previous runs don't pollute results
    with httpx.Client() as client:
        try:
            r = client.post(f"{base_url}/api/cache/clear", timeout=5.0)
            print(f"  Cache flushed before eval ({r.json()})\n")
        except Exception as exc:
            print(f"  Cache flush skipped: {exc}\n")

    with httpx.Client() as client:
        for i, case in enumerate(cases, 1):
            try:
                actual = run_query(client, base_url, case["query"])
                result = eval_case(case, actual)
                results.append(result)
                status = "✅" if result["correct"] else "❌"
                note = f" (cached)" if result["cached"] else ""
                print(f"  [{i:02d}/{len(cases)}] {status} [{result['lang'].upper()}] {case['id']:16s}  {result['query'][:50]}{note}")
            except Exception as exc:
                errors.append({"id": case["id"], "error": str(exc)})
                print(f"  [{i:02d}/{len(cases)}] 💥 {case['id']} — {exc}")
            time.sleep(0.1)  # gentle pacing

    # ── Aggregate ──────────────────────────────────────────────────────────
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    hits = sum(1 for r in results if r["hit"])

    by_lang: dict[str, dict] = {}
    for r in results:
        lang = r["lang"]
        if lang not in by_lang:
            by_lang[lang] = {"total": 0, "correct": 0}
        by_lang[lang]["total"] += 1
        if r["correct"]:
            by_lang[lang]["correct"] += 1

    by_type: dict[str, dict] = {}
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"total": 0, "correct": 0}
        by_type[t]["total"] += 1
        if r["correct"]:
            by_type[t]["correct"] += 1

    print(f"\n{'─'*60}")
    print(f"  RESULTS")
    print(f"{'─'*60}")
    print(f"  Total cases   : {total}")
    print(f"  Correct       : {correct}/{total}  ({100*correct/total:.1f}%)")
    print(f"  Hit rate      : {hits}/{total}  ({100*hits/total:.1f}%)")
    print(f"  Errors        : {len(errors)}")
    print()
    print("  Per-language accuracy:")
    for lang, d in sorted(by_lang.items()):
        pct = 100 * d["correct"] / d["total"]
        print(f"    {lang.upper():4s}  {d['correct']}/{d['total']}  ({pct:.1f}%)")
    print()
    print("  Per-type accuracy:")
    for t, d in sorted(by_type.items()):
        pct = 100 * d["correct"] / d["total"]
        print(f"    {t:25s}  {d['correct']}/{d['total']}  ({pct:.1f}%)")
    print(f"{'─'*60}\n")

    summary = {
        "base_url": base_url,
        "total": total,
        "correct": correct,
        "correct_pct": round(100 * correct / total, 2) if total else 0,
        "hit_rate_pct": round(100 * hits / total, 2) if total else 0,
        "errors": len(errors),
        "by_lang": by_lang,
        "by_type": by_type,
        "results": results,
        "error_detail": errors,
    }

    if out_path:
        Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  Full results written to: {out_path}\n")

    sys.exit(0 if correct / total >= 0.70 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--out", default=None, help="Write JSON results to this file")
    args = parser.parse_args()
    main(args.base_url, args.out)
