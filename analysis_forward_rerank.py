#!/usr/bin/env python3
"""Score candidates with a forward LM and sweep forward(+cycle) reranking."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from run_kimina_genlm_full_pipeline import build_formalization_prompt, load_hf_causal_lm


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


def _candidate_len(candidate: str) -> int:
    return len(_tokenize(_normalize_statement(candidate)))


def _as_float(value: object) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _theorem_name(record: dict) -> str:
    idx = record.get("index")
    if isinstance(idx, int):
        return f"autoformalized_theorem_{idx}"
    for key in ("baseline_candidate",):
        cand = record.get(key, "")
        if isinstance(cand, str):
            match = re.search(r"theorem\\s+([A-Za-z0-9_']+)", cand)
            if match:
                return match.group(1)
    for entry in record.get("candidates", []) or []:
        cand = entry.get("candidate")
        if isinstance(cand, str):
            match = re.search(r"theorem\\s+([A-Za-z0-9_']+)", cand)
            if match:
                return match.group(1)
    return "autoformalized_theorem"


def _build_prompt_ids(tokenizer, nl_statement: str, theorem_name: str) -> List[int]:
    user_prompt = build_formalization_prompt(nl_statement, theorem_name)
    user_prompt += "\nOkay, I'm done thinking.\n```lean4\n"
    messages = [
        {
            "role": "system",
            "content": "You are an expert mathematician and Lean 4 autoformalizer.",
        },
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        prompt_text = "\n".join(msg["content"] for msg in messages)
        return tokenizer(prompt_text, add_special_tokens=True).input_ids


def _score_forward(
    candidate: str,
    prompt_ids: List[int],
    model,
    tokenizer,
) -> float:
    if not candidate.strip():
        return float("-inf")
    target_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not prompt_ids or not target_ids:
        return float("-inf")
    device = model.device
    full_ids = torch.tensor([prompt_ids + target_ids], device=device)
    attention_mask = torch.ones_like(full_ids)
    with torch.no_grad():
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
    prefix_len = len(prompt_ids)
    if prefix_len == 0:
        return float("-inf")
    logp = 0.0
    for idx, token_id in enumerate(target_ids):
        pos = prefix_len - 1 + idx
        if pos >= log_probs.size(0):
            return float("-inf")
        logp += float(log_probs[pos, token_id].item())
    return logp / max(1.0, float(len(target_ids)))


def _sort_key(entry: Dict[str, object], *, by: str) -> float:
    key = "cycle_score" if by == "cycle" else "length_penalized_score"
    value = entry.get(key)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return float("-inf")


def _pick_candidate(
    entries: List[Dict[str, object]],
    *,
    alpha: float,
    cycle_weight: float,
    require_typecheck: bool,
) -> Dict[str, object] | None:
    best = None
    best_score = float("-inf")
    for entry in entries:
        if require_typecheck and not entry.get("typecheck", False):
            continue
        candidate = entry.get("candidate")
        if not isinstance(candidate, str):
            continue
        forward = _as_float(entry.get("forward_score"))
        if forward is None:
            continue
        cycle = _as_float(entry.get("cycle_score")) or 0.0
        score = forward + cycle_weight * cycle - alpha * float(_candidate_len(candidate))
        if score > best_score:
            best_score = score
            best = entry
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-score candidates and sweep reranking.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model for forward scoring",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force model on CPU float32")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0],
        help="Length-penalty weights to sweep",
    )
    parser.add_argument(
        "--cycle-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Cycle-score weights to sweep (0.0 = forward-only)",
    )
    parser.add_argument(
        "--score-top-k",
        type=int,
        default=None,
        help="Score only top-k candidates (by cycle or length-penalized score)",
    )
    parser.add_argument(
        "--score-top-k-by",
        choices=["cycle", "lenpen"],
        default="cycle",
        help="Sort key for top-k candidate scoring",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on records")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])
    if args.max_records is not None:
        records = records[: args.max_records]
        data["records"] = records

    model, tokenizer = load_hf_causal_lm(
        args.model_id,
        load_in_4bit=False,
        force_cpu=args.force_cpu,
    )

    prompt_cache: Dict[Tuple[str, str], List[int]] = {}
    for idx, record in enumerate(records, start=1):
        nl_statement = record.get("nl_statement", "") or ""
        theorem_name = _theorem_name(record)
        cache_key = (nl_statement, theorem_name)
        if cache_key in prompt_cache:
            prompt_ids = prompt_cache[cache_key]
        else:
            prompt_ids = _build_prompt_ids(tokenizer, nl_statement, theorem_name)
            prompt_cache[cache_key] = prompt_ids

        baseline_candidate = record.get("baseline_candidate", "") or ""
        baseline_forward = _score_forward(baseline_candidate, prompt_ids, model, tokenizer)
        record["baseline_forward_score"] = baseline_forward

        candidates = record.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
            record["candidates"] = candidates
        score_indices = set(range(len(candidates)))
        if args.score_top_k is not None and len(candidates) > args.score_top_k:
            indexed = list(enumerate(candidates))
            indexed.sort(
                key=lambda item: _sort_key(item[1], by=args.score_top_k_by),
                reverse=True,
            )
            score_indices = {idx for idx, _entry in indexed[: args.score_top_k]}
        for idx, entry in enumerate(candidates):
            candidate = entry.get("candidate", "")
            if idx not in score_indices:
                entry["forward_score"] = float("-inf")
                continue
            if not isinstance(candidate, str):
                entry["forward_score"] = float("-inf")
                continue
            entry["forward_score"] = _score_forward(candidate, prompt_ids, model, tokenizer)

        if idx % 5 == 0 or idx == len(records):
            print(f"[{idx}/{len(records)}] forward scored")

    summary: Dict[str, Dict[str, float]] = {}
    for alpha in args.alphas:
        for cycle_weight in args.cycle_weights:
            metrics = {
                "count": 0,
                "all_f1_sum": 0.0,
                "all_typecheck_sum": 0.0,
                "tc_f1_sum": 0.0,
                "tc_typecheck_sum": 0.0,
            }
            for record in records:
                candidates = record.get("candidates", [])
                if not isinstance(candidates, list) or not candidates:
                    continue
                ground_truth = record.get("ground_truth", "") or ""
                baseline_candidate = record.get("baseline_candidate", "") or ""
                baseline_typecheck = bool(record.get("baseline_typecheck", False))
                baseline_cycle = record.get("baseline_cycle_score", None)
                baseline_forward = record.get("baseline_forward_score", None)

                entries = list(candidates)
                if baseline_candidate and not any(
                    isinstance(entry, dict) and entry.get("candidate") == baseline_candidate
                    for entry in entries
                ):
                    entries = list(entries) + [
                        {
                            "candidate": baseline_candidate,
                            "typecheck": baseline_typecheck,
                            "cycle_score": baseline_cycle,
                            "forward_score": baseline_forward,
                        }
                    ]

                pick_all = _pick_candidate(
                    entries,
                    alpha=alpha,
                    cycle_weight=cycle_weight,
                    require_typecheck=False,
                )
                pick_tc = _pick_candidate(
                    entries,
                    alpha=alpha,
                    cycle_weight=cycle_weight,
                    require_typecheck=True,
                )
                if pick_all is None:
                    pick_all = {"candidate": baseline_candidate, "typecheck": baseline_typecheck}
                    pick_tc = pick_all
                if pick_tc is None:
                    pick_tc = pick_all

                cand_all = pick_all.get("candidate", "")
                cand_tc = pick_tc.get("candidate", "")
                metrics["count"] += 1
                metrics["all_f1_sum"] += _statement_f1(str(cand_all), ground_truth)
                metrics["all_typecheck_sum"] += int(bool(pick_all.get("typecheck", False)))
                metrics["tc_f1_sum"] += _statement_f1(str(cand_tc), ground_truth)
                metrics["tc_typecheck_sum"] += int(bool(pick_tc.get("typecheck", False)))

            key = f"alpha_{alpha}_beta_{cycle_weight}"
            if metrics["count"] == 0:
                summary[key] = {
                    "count": 0,
                    "all_f1_avg": 0.0,
                    "all_typecheck_rate": 0.0,
                    "tc_f1_avg": 0.0,
                    "tc_typecheck_rate": 0.0,
                }
            else:
                summary[key] = {
                    "count": metrics["count"],
                    "all_f1_avg": metrics["all_f1_sum"] / metrics["count"],
                    "all_typecheck_rate": metrics["all_typecheck_sum"] / metrics["count"],
                    "tc_f1_avg": metrics["tc_f1_sum"] / metrics["count"],
                    "tc_typecheck_rate": metrics["tc_typecheck_sum"] / metrics["count"],
                }

    meta = data.get("meta", {})
    meta["forward_model"] = args.model_id
    data["meta"] = meta
    data["forward_rerank_summary"] = summary

    output_path = args.output
    if output_path is None:
        output_path = args.results.with_name(args.results.stem + "_forward.json")
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote forward scores to {output_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Summary (top 10 by all-F1):")
    ranked = sorted(summary.items(), key=lambda item: item[1]["all_f1_avg"], reverse=True)
    for key, metrics in ranked[:10]:
        print(
            f"{key} all_f1={metrics['all_f1_avg']:.3f} "
            f"all_tc={metrics['all_typecheck_rate']:.3f} "
            f"tc_f1={metrics['tc_f1_avg']:.3f} "
            f"tc_tc={metrics['tc_typecheck_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
