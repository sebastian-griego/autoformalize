#!/usr/bin/env python3
"""Fast consensus k-sweep using cached token similarities."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|[^\s]")


def _extract_theorem_block(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    idx = lower.rfind("theorem")
    if idx == -1:
        return text.strip()
    snippet = text[idx:].strip()
    head, sep, _ = snippet.partition(":=")
    if sep:
        snippet = head.strip()
    return snippet


def _strip_theorem_name(text: str) -> str:
    if not text:
        return ""
    parts = text.split(None, 2)
    if len(parts) < 2:
        return text.strip()
    if parts[0] != "theorem":
        return text.strip()
    if len(parts) == 2:
        return "theorem"
    return "theorem " + parts[2]


def _normalize_statement(text: str) -> str:
    block = _extract_theorem_block(text)
    return _strip_theorem_name(block)


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def _counter(tokens: Iterable[str]) -> Counter[str]:
    return Counter(tokens)


def _f1_score(pred: Counter[str], gold: Counter[str]) -> float:
    if not pred or not gold:
        return 0.0
    common = sum((pred & gold).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(gold.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _statement_f1(pred_text: str, gold_text: str) -> float:
    pred = _counter(_tokenize(_normalize_statement(pred_text)))
    gold = _counter(_tokenize(_normalize_statement(gold_text)))
    return _f1_score(pred, gold)


def _prepare_record(record: dict) -> dict | None:
    candidates = record.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return None
    ground_truth = record.get("ground_truth", "")

    texts: List[str] = []
    typechecks: List[bool] = []
    tokens: List[Counter[str]] = []
    f1s: List[float] = []

    for entry in candidates:
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        texts.append(candidate)
        typechecks.append(bool(entry.get("typecheck", False)))
        tok = _counter(_tokenize(_normalize_statement(candidate)))
        tokens.append(tok)
        f1s.append(_statement_f1(candidate, ground_truth))

    if not texts:
        return None

    n = len(texts)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            score = _f1_score(tokens[i], tokens[j])
            sim[i, j] = score
            sim[j, i] = score

    return {
        "sim": sim,
        "f1s": np.asarray(f1s, dtype=np.float32),
        "typechecks": np.asarray(typechecks, dtype=bool),
    }


def _pick_consensus(sim: np.ndarray, subset: np.ndarray) -> int:
    sub = sim[np.ix_(subset, subset)]
    scores = (sub.sum(axis=1) - np.diag(sub)) / max(1, sub.shape[0] - 1)
    best_local = int(np.argmax(scores))
    return int(subset[best_local])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast consensus F1 vs subset size.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    prepared = []
    for record in records:
        prepped = _prepare_record(record)
        if prepped is not None:
            prepared.append(prepped)

    summary = {}
    for require_tc in (False, True):
        rng = random.Random(args.seed + (1 if require_tc else 0))
        key = "tc_only" if require_tc else "all"
        results = {}
        for k in args.ks:
            count = 0
            f1_sum = 0.0
            tc_sum = 0.0
            for record in prepared:
                sim = record["sim"]
                f1s = record["f1s"]
                typechecks = record["typechecks"]
                if require_tc:
                    indices = np.flatnonzero(typechecks)
                else:
                    indices = np.arange(typechecks.shape[0])
                if indices.shape[0] < k:
                    continue
                indices_list = indices.tolist()
                for _ in range(args.trials):
                    subset = np.array(rng.sample(indices_list, k), dtype=int)
                    best_idx = _pick_consensus(sim, subset)
                    f1_sum += float(f1s[best_idx])
                    tc_sum += int(bool(typechecks[best_idx]))
                    count += 1
            results[f"k_{k}"] = {
                "count": count,
                "f1_avg": f1_sum / count if count else 0.0,
                "typecheck_rate": tc_sum / count if count else 0.0,
            }
        summary[key] = results

    if args.output is not None:
        payload = {
            "meta": {
                "results_path": str(args.results),
                "ks": args.ks,
                "trials": args.trials,
                "seed": args.seed,
                "prepared_records": len(prepared),
            },
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for group, results in summary.items():
        print(group + ":")
        for k_key, metrics in results.items():
            print(
                f"  {k_key} f1={metrics['f1_avg']:.3f} "
                f"tc={metrics['typecheck_rate']:.3f} "
                f"count={metrics['count']}"
            )


if __name__ == "__main__":
    main()
