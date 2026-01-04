#!/usr/bin/env python3
"""Rerank candidates by NL↔Lean embedding similarity."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

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


class _Embedder:
    def __init__(self, model_id: str, device: str | None, batch_size: int, max_length: int) -> None:
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.cache: Dict[str, torch.Tensor] = {}

    def embed(self, texts: List[str]) -> List[torch.Tensor]:
        missing = [text for text in texts if text not in self.cache]
        if missing:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                for text, emb in zip(batch, pooled.cpu()):
                    self.cache[text] = emb
        return [self.cache[text] for text in texts]


def _maybe_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text
    return prefix + text


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank candidates by NL↔Lean embedding similarity.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--model-id", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--query-prefix", type=str, default="")
    parser.add_argument("--passage-prefix", type=str, default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.query_prefix and "e5" in args.model_id:
        args.query_prefix = "query: "
        args.passage_prefix = "passage: "

    embedder = _Embedder(
        args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    metrics = {
        "count": 0,
        "baseline_f1_sum": 0.0,
        "embed_f1_sum": 0.0,
        "embed_tc_f1_sum": 0.0,
        "embed_typecheck_sum": 0.0,
        "embed_tc_typecheck_sum": 0.0,
        "embed_improve_sum": 0,
        "embed_tc_improve_sum": 0,
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

        query_text = _maybe_prefix(nl_statement, args.query_prefix)
        passage_texts = [_maybe_prefix(text, args.passage_prefix) for text in normalized]

        query_emb = embedder.embed([query_text])[0]
        cand_embs = embedder.embed(passage_texts)
        sim = torch.stack(cand_embs, dim=0) @ query_emb
        best_idx = int(torch.argmax(sim).item())

        tc_indices = [i for i, tc in enumerate(typechecks) if tc]
        if tc_indices:
            tc_scores = [float(sim[i].item()) for i in tc_indices]
            best_tc_idx = tc_indices[int(torch.argmax(torch.tensor(tc_scores)).item())]
        else:
            best_tc_idx = best_idx

        best_candidate = texts[best_idx]
        best_tc_candidate = texts[best_tc_idx]
        best_f1 = _statement_f1(best_candidate, ground_truth)
        best_tc_f1 = _statement_f1(best_tc_candidate, ground_truth)
        baseline_f1 = _statement_f1(baseline_candidate, ground_truth)

        record["embed_best_candidate"] = best_candidate
        record["embed_best_typecheck"] = bool(typechecks[best_idx])
        record["embed_best_tc_candidate"] = best_tc_candidate
        record["embed_best_tc_typecheck"] = bool(typechecks[best_tc_idx])
        record["embed_best_stmt_f1"] = best_f1
        record["embed_best_tc_stmt_f1"] = best_tc_f1

        metrics["count"] += 1
        metrics["baseline_f1_sum"] += baseline_f1
        metrics["embed_f1_sum"] += best_f1
        metrics["embed_tc_f1_sum"] += best_tc_f1
        metrics["embed_typecheck_sum"] += int(bool(typechecks[best_idx]))
        metrics["embed_tc_typecheck_sum"] += int(bool(typechecks[best_tc_idx]))
        metrics["embed_improve_sum"] += int(best_f1 > baseline_f1)
        metrics["embed_tc_improve_sum"] += int(best_tc_f1 > baseline_f1)

    count = metrics["count"]
    summary = {
        "count": count,
        "baseline_f1_avg": metrics["baseline_f1_sum"] / count if count else 0.0,
        "embed_f1_avg": metrics["embed_f1_sum"] / count if count else 0.0,
        "embed_tc_f1_avg": metrics["embed_tc_f1_sum"] / count if count else 0.0,
        "embed_typecheck_rate": metrics["embed_typecheck_sum"] / count if count else 0.0,
        "embed_tc_typecheck_rate": metrics["embed_tc_typecheck_sum"] / count if count else 0.0,
        "embed_improve_rate": metrics["embed_improve_sum"] / count if count else 0.0,
        "embed_tc_improve_rate": metrics["embed_tc_improve_sum"] / count if count else 0.0,
    }

    payload: Dict[str, object] = {
        "meta": {
            "results_path": str(args.results),
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": str(embedder.device),
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
