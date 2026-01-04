#!/usr/bin/env python3
"""Evaluate embedding-based consensus reranking for candidate sets."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


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


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def _embed_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    cache: Dict[str, torch.Tensor],
) -> List[torch.Tensor]:
    missing = [text for text in texts if text not in cache]
    if missing:
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for text, emb in zip(batch, pooled.cpu()):
                cache[text] = emb
    return [cache[text] for text in texts]


def _select_embedding_consensus(
    candidates: List[Dict[str, object]],
    *,
    require_typecheck: bool,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    cache: Dict[str, torch.Tensor],
) -> Dict[str, object] | None:
    filtered = [c for c in candidates if not require_typecheck or c.get("typecheck", False)]
    if not filtered:
        return None
    normalized: List[str] = []
    prepared: List[Dict[str, object]] = []
    for entry in filtered:
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        norm = _normalize_statement(candidate)
        if not norm:
            continue
        prepared.append(entry)
        normalized.append(norm)
    if not prepared:
        return None
    if len(prepared) == 1:
        return prepared[0]

    embeddings = _embed_texts(
        normalized,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        cache=cache,
    )
    emb_tensor = torch.stack(embeddings, dim=0)
    sim = emb_tensor @ emb_tensor.T
    n = sim.size(0)
    if n <= 1:
        return prepared[0]
    avg = (sim.sum(dim=1) - 1.0) / max(1, n - 1)
    best_idx = int(torch.argmax(avg).item())

    best = prepared[best_idx]
    best_score = float(avg[best_idx].item())
    for idx, entry in enumerate(prepared):
        if idx == best_idx:
            continue
        score = float(avg[idx].item())
        if score < best_score:
            continue
        if score > best_score:
            best = entry
            best_score = score
            continue
        cycle_score = entry.get("cycle_score")
        best_cycle = best.get("cycle_score")
        if isinstance(cycle_score, (int, float)) and isinstance(best_cycle, (int, float)):
            if math.isfinite(float(cycle_score)) and math.isfinite(float(best_cycle)) and cycle_score > best_cycle:
                best = entry
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embedding consensus reranking.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device)
    model.eval()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    metrics = {
        "count": 0,
        "baseline_f1_sum": 0.0,
        "embed_consensus_f1_sum": 0.0,
        "embed_consensus_tc_f1_sum": 0.0,
        "embed_consensus_rate_sum": 0.0,
        "embed_consensus_tc_rate_sum": 0.0,
        "embed_consensus_improve_sum": 0,
        "embed_consensus_tc_improve_sum": 0,
    }

    cache: Dict[str, torch.Tensor] = {}

    for record in records:
        candidates = record.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
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

        embed_consensus = _select_embedding_consensus(
            entries,
            require_typecheck=False,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            cache=cache,
        )
        embed_consensus_tc = _select_embedding_consensus(
            entries,
            require_typecheck=True,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            cache=cache,
        )
        if embed_consensus is None:
            embed_consensus = {"candidate": baseline_candidate, "typecheck": baseline_typecheck}
        if embed_consensus_tc is None:
            embed_consensus_tc = embed_consensus

        baseline_f1 = _statement_f1(str(baseline_candidate), ground_truth)
        embed_f1 = _statement_f1(str(embed_consensus.get("candidate", "")), ground_truth)
        embed_tc_f1 = _statement_f1(str(embed_consensus_tc.get("candidate", "")), ground_truth)

        record["embed_consensus_candidate"] = embed_consensus.get("candidate", "")
        record["embed_consensus_typecheck"] = bool(embed_consensus.get("typecheck", False))
        record["embed_consensus_tc_candidate"] = embed_consensus_tc.get("candidate", "")
        record["embed_consensus_tc_typecheck"] = bool(embed_consensus_tc.get("typecheck", False))
        record["embed_consensus_stmt_f1"] = embed_f1
        record["embed_consensus_tc_stmt_f1"] = embed_tc_f1

        metrics["count"] += 1
        metrics["baseline_f1_sum"] += baseline_f1
        metrics["embed_consensus_f1_sum"] += embed_f1
        metrics["embed_consensus_tc_f1_sum"] += embed_tc_f1
        metrics["embed_consensus_rate_sum"] += int(bool(embed_consensus.get("typecheck", False)))
        metrics["embed_consensus_tc_rate_sum"] += int(bool(embed_consensus_tc.get("typecheck", False)))
        metrics["embed_consensus_improve_sum"] += int(embed_f1 > baseline_f1)
        metrics["embed_consensus_tc_improve_sum"] += int(embed_tc_f1 > baseline_f1)

    count = metrics["count"]
    summary = {
        "count": count,
        "baseline_f1_avg": metrics["baseline_f1_sum"] / count if count else 0.0,
        "embed_consensus_f1_avg": metrics["embed_consensus_f1_sum"] / count if count else 0.0,
        "embed_consensus_tc_f1_avg": metrics["embed_consensus_tc_f1_sum"] / count if count else 0.0,
        "embed_consensus_rate": metrics["embed_consensus_rate_sum"] / count if count else 0.0,
        "embed_consensus_tc_rate": metrics["embed_consensus_tc_rate_sum"] / count if count else 0.0,
        "embed_consensus_improve_rate": metrics["embed_consensus_improve_sum"] / count if count else 0.0,
        "embed_consensus_tc_improve_rate": metrics["embed_consensus_tc_improve_sum"] / count if count else 0.0,
    }

    payload: Dict[str, object] = {
        "meta": {
            "results_path": str(args.results),
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": str(device),
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
