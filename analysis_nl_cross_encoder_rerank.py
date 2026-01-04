#!/usr/bin/env python3
"""Rerank candidates by NL↔Lean cross-encoder scores."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def _score_pairs(
    tokenizer,
    model,
    device: torch.device,
    queries: List[str],
    passages: List[str],
    batch_size: int,
    max_length: int,
) -> List[float]:
    scores: List[float] = []
    for start in range(0, len(queries), batch_size):
        batch_q = queries[start : start + batch_size]
        batch_p = passages[start : start + batch_size]
        encoded = tokenizer(
            batch_q,
            batch_p,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
        if logits.shape[-1] == 1:
            batch_scores = logits.squeeze(-1)
        else:
            batch_scores = logits[:, -1]
        scores.extend(batch_scores.detach().cpu().tolist())
    return scores


def _maybe_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text
    return prefix + text


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank candidates by NL↔Lean cross-encoder scores.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--model-id", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--query-prefix", type=str, default="")
    parser.add_argument("--passage-prefix", type=str, default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id).to(device)
    model.eval()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    metrics = {
        "count": 0,
        "baseline_f1_sum": 0.0,
        "cross_f1_sum": 0.0,
        "cross_tc_f1_sum": 0.0,
        "cross_typecheck_sum": 0.0,
        "cross_tc_typecheck_sum": 0.0,
        "cross_improve_sum": 0,
        "cross_tc_improve_sum": 0,
    }

    for record in records:
        candidates = record.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            continue
        nl_statement = record.get("nl_statement", "")
        if not isinstance(nl_statement, str) or not nl_statement.strip():
            continue
        ground_truth = record.get("ground_truth", "")
        baseline_candidate = record.get("baseline_candidate", "")
        baseline_typecheck = bool(record.get("baseline_typecheck", False))

        entries = list(candidates)
        if baseline_candidate and not any(
            isinstance(entry, dict) and entry.get("candidate") == baseline_candidate
            for entry in entries
        ):
            entries.append({"candidate": baseline_candidate, "typecheck": baseline_typecheck})

        texts: List[str] = []
        normalized: List[str] = []
        typechecks: List[bool] = []
        for entry in entries:
            candidate = entry.get("candidate")
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            norm = _normalize_statement(candidate)
            if not norm:
                continue
            texts.append(candidate)
            normalized.append(norm)
            typechecks.append(bool(entry.get("typecheck", False)))

        if not texts:
            continue

        query = _maybe_prefix(nl_statement, args.query_prefix)
        passage_texts = [_maybe_prefix(text, args.passage_prefix) for text in normalized]
        queries = [query for _ in passage_texts]
        scores = _score_pairs(
            tokenizer,
            model,
            device,
            queries,
            passage_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        best_idx = int(torch.tensor(scores).argmax().item())
        tc_indices = [i for i, tc in enumerate(typechecks) if tc]
        if tc_indices:
            tc_scores = [scores[i] for i in tc_indices]
            best_tc_idx = tc_indices[int(torch.tensor(tc_scores).argmax().item())]
        else:
            best_tc_idx = best_idx

        best_candidate = texts[best_idx]
        best_tc_candidate = texts[best_tc_idx]
        best_f1 = _statement_f1(best_candidate, ground_truth)
        best_tc_f1 = _statement_f1(best_tc_candidate, ground_truth)
        baseline_f1 = _statement_f1(baseline_candidate, ground_truth)

        record["cross_best_candidate"] = best_candidate
        record["cross_best_typecheck"] = bool(typechecks[best_idx])
        record["cross_best_tc_candidate"] = best_tc_candidate
        record["cross_best_tc_typecheck"] = bool(typechecks[best_tc_idx])
        record["cross_best_stmt_f1"] = best_f1
        record["cross_best_tc_stmt_f1"] = best_tc_f1

        metrics["count"] += 1
        metrics["baseline_f1_sum"] += baseline_f1
        metrics["cross_f1_sum"] += best_f1
        metrics["cross_tc_f1_sum"] += best_tc_f1
        metrics["cross_typecheck_sum"] += int(bool(typechecks[best_idx]))
        metrics["cross_tc_typecheck_sum"] += int(bool(typechecks[best_tc_idx]))
        metrics["cross_improve_sum"] += int(best_f1 > baseline_f1)
        metrics["cross_tc_improve_sum"] += int(best_tc_f1 > baseline_f1)

    count = metrics["count"]
    summary = {
        "count": count,
        "baseline_f1_avg": metrics["baseline_f1_sum"] / count if count else 0.0,
        "cross_f1_avg": metrics["cross_f1_sum"] / count if count else 0.0,
        "cross_tc_f1_avg": metrics["cross_tc_f1_sum"] / count if count else 0.0,
        "cross_typecheck_rate": metrics["cross_typecheck_sum"] / count if count else 0.0,
        "cross_tc_typecheck_rate": metrics["cross_tc_typecheck_sum"] / count if count else 0.0,
        "cross_improve_rate": metrics["cross_improve_sum"] / count if count else 0.0,
        "cross_tc_improve_rate": metrics["cross_tc_improve_sum"] / count if count else 0.0,
    }

    payload = {
        "meta": {
            "results_path": str(args.results),
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": str(device),
            "query_prefix": args.query_prefix,
            "passage_prefix": args.passage_prefix,
        },
        "summary": summary,
        "records": records,
    }

    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
