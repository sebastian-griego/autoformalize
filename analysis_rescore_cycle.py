#!/usr/bin/env python3
"""Rescore cycle-consistency values for an existing best-of-cycle run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch

from run_kimina_bestof_cycle import SummaryStats, _statement_len, _update_stats_from_record, score_cycle
from run_kimina_genlm_full_pipeline import load_hf_causal_lm


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore cycle-consistency for existing results.")
    parser.add_argument("--results", type=Path, required=True, help="Existing results JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (defaults to *_rescored.json)")
    parser.add_argument("--reference-model-id", type=str, default=None, help="Override reference model id")
    parser.add_argument("--length-penalty", type=float, default=None, help="Override length penalty")
    parser.add_argument("--force-cpu", action="store_true", help="Force reference model on CPU float32")
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])
    meta = data.get("meta", {})

    reference_model_id = args.reference_model_id or meta.get("reference_model")
    if not reference_model_id:
        raise SystemExit("Missing reference model id; pass --reference-model-id.")
    length_penalty = args.length_penalty
    if length_penalty is None:
        length_penalty = _as_float(meta.get("length_penalty")) or 0.0

    original_reference = meta.get("reference_model")
    if original_reference and original_reference != reference_model_id:
        meta["reference_model_original"] = original_reference
    meta["reference_model"] = reference_model_id
    meta["length_penalty"] = length_penalty

    reference_model, reference_tokenizer = load_hf_causal_lm(
        reference_model_id,
        load_in_4bit=False,
        force_cpu=args.force_cpu,
    )

    stats = SummaryStats()
    for idx, record in enumerate(records, start=1):
        nl_statement = record.get("nl_statement", "")
        baseline_candidate = record.get("baseline_candidate", "")
        baseline_cycle = score_cycle(
            baseline_candidate,
            nl_statement,
            reference_model,
            reference_tokenizer,
        )
        if not math.isfinite(baseline_cycle):
            baseline_cycle = float("-inf")
        baseline_len = _statement_len(baseline_candidate)
        baseline_lenpen = baseline_cycle - (length_penalty * baseline_len)
        record["baseline_cycle_score"] = baseline_cycle
        record["baseline_length_penalized_score"] = baseline_lenpen
        record["baseline_statement_len"] = baseline_len

        best_cycle_candidate = ""
        best_cycle_score = float("-inf")
        best_cycle_typecheck = False
        best_cycle_tc_candidate = ""
        best_cycle_tc_score = float("-inf")
        best_lenpen_candidate = ""
        best_lenpen_score = float("-inf")
        best_lenpen_typecheck = False
        best_lenpen_tc_candidate = ""
        best_lenpen_tc_score = float("-inf")

        candidates = record.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
            record["candidates"] = candidates
        typechecked_count = 0

        for entry in candidates:
            candidate = entry.get("candidate", "")
            if not isinstance(candidate, str):
                candidate = ""
            candidate_len = _statement_len(candidate)
            cycle_score = score_cycle(
                candidate,
                nl_statement,
                reference_model,
                reference_tokenizer,
            )
            if not math.isfinite(cycle_score):
                cycle_score = float("-inf")
            lenpen_score = cycle_score - (length_penalty * candidate_len)
            entry["cycle_score"] = cycle_score
            entry["length_penalized_score"] = lenpen_score
            entry["statement_len"] = candidate_len
            candidate_typecheck = bool(entry.get("typecheck", False))
            if candidate_typecheck:
                typechecked_count += 1
            if cycle_score > best_cycle_score:
                best_cycle_score = cycle_score
                best_cycle_candidate = candidate
                best_cycle_typecheck = candidate_typecheck
            if candidate_typecheck and cycle_score > best_cycle_tc_score:
                best_cycle_tc_score = cycle_score
                best_cycle_tc_candidate = candidate
            if lenpen_score > best_lenpen_score:
                best_lenpen_score = lenpen_score
                best_lenpen_candidate = candidate
                best_lenpen_typecheck = candidate_typecheck
            if candidate_typecheck and lenpen_score > best_lenpen_tc_score:
                best_lenpen_tc_score = lenpen_score
                best_lenpen_tc_candidate = candidate

        baseline_in_candidates = baseline_candidate in {
            c.get("candidate") for c in candidates if isinstance(c, dict)
        }
        record["baseline_in_candidates"] = baseline_in_candidates
        record["num_candidates"] = len(candidates)
        record["num_candidates_with_baseline"] = len(candidates)
        record["num_typechecked_candidates"] = typechecked_count

        if not best_cycle_candidate:
            best_cycle_candidate = baseline_candidate
            best_cycle_score = baseline_cycle
            best_cycle_typecheck = bool(record.get("baseline_typecheck", False))
        if not best_cycle_tc_candidate:
            best_cycle_tc_candidate = best_cycle_candidate
            best_cycle_tc_score = best_cycle_score
            best_cycle_tc_typecheck = best_cycle_typecheck
        else:
            best_cycle_tc_typecheck = True
        if not best_lenpen_candidate:
            best_lenpen_candidate = baseline_candidate
            best_lenpen_score = baseline_lenpen
            best_lenpen_typecheck = bool(record.get("baseline_typecheck", False))
        if not best_lenpen_tc_candidate:
            best_lenpen_tc_candidate = best_lenpen_candidate
            best_lenpen_tc_score = best_lenpen_score
            best_lenpen_tc_typecheck = best_lenpen_typecheck
        else:
            best_lenpen_tc_typecheck = True

        record.update(
            {
                "best_cycle_candidate": best_cycle_candidate,
                "best_cycle_score": best_cycle_score,
                "best_cycle_typecheck": best_cycle_typecheck,
                "best_cycle_tc_candidate": best_cycle_tc_candidate,
                "best_cycle_tc_score": best_cycle_tc_score,
                "best_cycle_tc_typecheck": best_cycle_tc_typecheck,
                "best_lenpen_candidate": best_lenpen_candidate,
                "best_lenpen_score": best_lenpen_score,
                "best_lenpen_typecheck": best_lenpen_typecheck,
                "best_lenpen_tc_candidate": best_lenpen_tc_candidate,
                "best_lenpen_tc_score": best_lenpen_tc_score,
                "best_lenpen_tc_typecheck": best_lenpen_tc_typecheck,
            }
        )

        _update_stats_from_record(stats, record)
        if idx % 5 == 0:
            print(f"[{idx}/{len(records)}] rescored")

    data["summary"] = stats.as_dict()
    output_path = args.output
    if output_path is None:
        output_path = args.results.with_name(args.results.stem + "_rescored.json")
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote rescored results to {output_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
