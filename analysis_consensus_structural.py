#!/usr/bin/env python3
"""Evaluate a structural consensus reranker with identifier normalization."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


STRUCT_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|\d+|[^\s]")
STD_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|[^\s]")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")

KEYWORDS = {
    "theorem",
    "lemma",
    "def",
    "by",
    "have",
    "let",
    "calc",
    "simp",
    "simp_all",
    "simp_rw",
    "match",
    "if",
    "then",
    "else",
    "fun",
    "forall",
    "exists",
    "where",
    "open",
    "namespace",
    "section",
    "variable",
    "variables",
    "local",
    "attribute",
    "instance",
    "structure",
    "class",
    "extends",
    "deriving",
    "inductive",
    "coinductive",
    "abbrev",
    "protected",
    "private",
    "macro",
    "syntax",
    "macro_rules",
    "termination_by",
    "partial",
    "unsafe",
    "mutual",
}


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


def _tokenize_with(text: str, token_re: re.Pattern[str]) -> List[str]:
    return token_re.findall(text)


def _tokenize_structural(text: str) -> List[str]:
    return _tokenize_with(text, STRUCT_TOKEN_RE)


def _tokenize_standard(text: str) -> List[str]:
    return _tokenize_with(text, STD_TOKEN_RE)


def _normalize_tokens(
    tokens: Iterable[str],
    *,
    mask_strategy: str,
    min_len: int,
    mask_numbers: bool,
) -> List[str]:
    mapping: Dict[str, str] = {}
    next_id = 0
    normalized: List[str] = []
    for tok in tokens:
        if mask_numbers and tok.isdigit():
            normalized.append("NUM")
            continue
        if not IDENT_RE.fullmatch(tok):
            normalized.append(tok)
            continue
        if tok in KEYWORDS or tok[0].isupper():
            normalized.append(tok)
            continue
        mask = False
        if mask_strategy == "all":
            mask = True
        elif mask_strategy == "short":
            if len(tok) <= min_len or any(ch.isdigit() for ch in tok):
                mask = True
        elif mask_strategy == "none":
            mask = False
        else:
            raise ValueError(f"Unknown mask strategy: {mask_strategy}")
        if mask:
            if tok not in mapping:
                mapping[tok] = f"id{next_id}"
                next_id += 1
            normalized.append(mapping[tok])
        else:
            normalized.append(tok)
    return normalized


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


def _statement_f1(pred_text: str, gold_text: str, *, token_re: re.Pattern[str]) -> float:
    pred = _counter(_tokenize_with(_normalize_statement(pred_text), token_re))
    gold = _counter(_tokenize_with(_normalize_statement(gold_text), token_re))
    return _f1_score(pred, gold)


def _build_candidate_tokens(
    candidates: List[Dict[str, object]],
    *,
    mask_strategy: str,
    min_len: int,
    mask_numbers: bool,
) -> List[Tuple[Dict[str, object], Counter[str]]]:
    prepared: List[Tuple[Dict[str, object], Counter[str]]] = []
    for entry in candidates:
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        tokens = _tokenize_structural(_normalize_statement(candidate))
        tokens = _normalize_tokens(
            tokens,
            mask_strategy=mask_strategy,
            min_len=min_len,
            mask_numbers=mask_numbers,
        )
        prepared.append((entry, _counter(tokens)))
    return prepared


def _consensus_candidate(
    candidates: List[Dict[str, object]],
    *,
    require_typecheck: bool,
    mask_strategy: str,
    min_len: int,
    mask_numbers: bool,
) -> Dict[str, object] | None:
    filtered = [c for c in candidates if not require_typecheck or c.get("typecheck", False)]
    if not filtered:
        return None
    prepared = _build_candidate_tokens(
        filtered,
        mask_strategy=mask_strategy,
        min_len=min_len,
        mask_numbers=mask_numbers,
    )
    if not prepared:
        return None
    if len(prepared) == 1:
        return prepared[0][0]

    best = None
    best_score = float("-inf")
    for idx, (entry, tokens) in enumerate(prepared):
        score = 0.0
        for jdx, (_, other_tokens) in enumerate(prepared):
            if idx == jdx:
                continue
            score += _f1_score(tokens, other_tokens)
        score /= max(1, len(prepared) - 1)
        if score > best_score:
            best_score = score
            best = entry
        elif score == best_score and best is not None:
            cycle_score = entry.get("cycle_score")
            best_cycle = best.get("cycle_score")
            if isinstance(cycle_score, (int, float)) and isinstance(best_cycle, (int, float)):
                if math.isfinite(cycle_score) and math.isfinite(best_cycle) and cycle_score > best_cycle:
                    best = entry
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate consensus reranking with identifier normalization.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument(
        "--mask-identifiers",
        choices=["none", "short", "all"],
        default="short",
        help="Identifier masking strategy",
    )
    parser.add_argument("--mask-numbers", action="store_true", help="Replace digit tokens with NUM")
    parser.add_argument(
        "--eval-tokenizer",
        choices=["structural", "standard"],
        default="structural",
        help="Tokenizer for evaluation F1",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Min identifier length to keep when using short masking",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])
    eval_token_re = STRUCT_TOKEN_RE if args.eval_tokenizer == "structural" else STD_TOKEN_RE

    metrics = {
        "count": 0,
        "baseline_f1_sum": 0.0,
        "consensus_f1_sum": 0.0,
        "consensus_tc_f1_sum": 0.0,
        "consensus_tc_rate_sum": 0.0,
        "consensus_rate_sum": 0.0,
    }

    for record in records:
        candidates = record.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            continue
        ground_truth = record.get("ground_truth", "")
        baseline_candidate = record.get("baseline_candidate", "")

        consensus = _consensus_candidate(
            candidates,
            require_typecheck=False,
            mask_strategy=args.mask_identifiers,
            min_len=args.min_len,
            mask_numbers=args.mask_numbers,
        )
        consensus_tc = _consensus_candidate(
            candidates,
            require_typecheck=True,
            mask_strategy=args.mask_identifiers,
            min_len=args.min_len,
            mask_numbers=args.mask_numbers,
        )
        if consensus is None:
            consensus = {"candidate": baseline_candidate, "typecheck": record.get("baseline_typecheck", False)}
        if consensus_tc is None:
            consensus_tc = consensus

        metrics["count"] += 1
        metrics["baseline_f1_sum"] += _statement_f1(str(baseline_candidate), ground_truth, token_re=eval_token_re)
        metrics["consensus_f1_sum"] += _statement_f1(
            str(consensus.get("candidate", "")), ground_truth, token_re=eval_token_re
        )
        metrics["consensus_tc_f1_sum"] += _statement_f1(
            str(consensus_tc.get("candidate", "")), ground_truth, token_re=eval_token_re
        )
        metrics["consensus_rate_sum"] += int(bool(consensus.get("typecheck", False)))
        metrics["consensus_tc_rate_sum"] += int(bool(consensus_tc.get("typecheck", False)))

    summary = {
        "count": metrics["count"],
        "baseline_f1_avg": metrics["baseline_f1_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "consensus_f1_avg": metrics["consensus_f1_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "consensus_tc_f1_avg": metrics["consensus_tc_f1_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "consensus_typecheck_rate": metrics["consensus_rate_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "consensus_tc_typecheck_rate": metrics["consensus_tc_rate_sum"] / metrics["count"] if metrics["count"] else 0.0,
    }

    if args.output is not None:
        payload = {
            "meta": {
                "results_path": str(args.results),
                "mask_identifiers": args.mask_identifiers,
                "min_len": args.min_len,
                "mask_numbers": args.mask_numbers,
                "eval_tokenizer": args.eval_tokenizer,
            },
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
