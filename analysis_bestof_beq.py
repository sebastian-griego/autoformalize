#!/usr/bin/env python3
"""Evaluate BEq+ for best-of-cycle runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent / "LeanInteract"
if PROJECT_ROOT.exists():
    sys.path.append(str(PROJECT_ROOT))

from examples.beq_plus import DEFAULT_TIMEOUT, beq_plus
from lean_interact import AutoLeanServer, LeanREPLConfig
from lean_interact.project import TempRequireProject


def _eval_candidate(
    ground_truth: str,
    candidate: str,
    lean_header: str,
    server: AutoLeanServer,
    timeout: int,
) -> bool:
    if not candidate.strip():
        return False
    return bool(
        beq_plus(
            ground_truth,
            candidate,
            lean_header,
            server,
            timeout_per_proof=timeout,
            verbose=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEq+ on best-of-cycle outputs.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--lean-version", type=str, default="v4.8.0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    repl_config = LeanREPLConfig(
        project=TempRequireProject(lean_version=args.lean_version, require="mathlib"),
        verbose=False,
    )
    server = AutoLeanServer(config=repl_config)

    summary = {
        "count": 0,
        "baseline_beq_true": 0,
        "best_cycle_beq_true": 0,
        "best_cycle_tc_beq_true": 0,
        "best_cycle_improve": 0,
        "best_cycle_degrade": 0,
        "best_cycle_tc_improve": 0,
        "best_cycle_tc_degrade": 0,
    }

    for idx, record in enumerate(records, start=1):
        ground_truth = record.get("ground_truth", "")
        lean_header = record.get("lean_header", "")
        baseline = record.get("baseline_candidate", "")
        best_cycle = record.get("best_cycle_candidate", "")
        best_cycle_tc = record.get("best_cycle_tc_candidate", best_cycle)

        baseline_beq = False
        best_cycle_beq = False
        best_cycle_tc_beq = False
        try:
            baseline_beq = _eval_candidate(ground_truth, baseline, lean_header, server, args.timeout)
        except Exception:
            baseline_beq = False
        try:
            best_cycle_beq = _eval_candidate(ground_truth, best_cycle, lean_header, server, args.timeout)
        except Exception:
            best_cycle_beq = False
        try:
            best_cycle_tc_beq = _eval_candidate(ground_truth, best_cycle_tc, lean_header, server, args.timeout)
        except Exception:
            best_cycle_tc_beq = False

        record["baseline_beq_plus"] = baseline_beq
        record["best_cycle_beq_plus"] = best_cycle_beq
        record["best_cycle_tc_beq_plus"] = best_cycle_tc_beq

        summary["count"] += 1
        summary["baseline_beq_true"] += int(baseline_beq)
        summary["best_cycle_beq_true"] += int(best_cycle_beq)
        summary["best_cycle_tc_beq_true"] += int(best_cycle_tc_beq)
        if best_cycle_beq and not baseline_beq:
            summary["best_cycle_improve"] += 1
        if baseline_beq and not best_cycle_beq:
            summary["best_cycle_degrade"] += 1
        if best_cycle_tc_beq and not baseline_beq:
            summary["best_cycle_tc_improve"] += 1
        if baseline_beq and not best_cycle_tc_beq:
            summary["best_cycle_tc_degrade"] += 1

        print(
            f"[{idx}/{len(records)}] {record.get('id')} "
            f"baseline={baseline_beq} best_cycle={best_cycle_beq} best_cycle_tc={best_cycle_tc_beq}"
        )

    summary_rates: Dict[str, Any] = {
        "count": summary["count"],
        "baseline_beq_rate": summary["baseline_beq_true"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_beq_rate": summary["best_cycle_beq_true"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_tc_beq_rate": summary["best_cycle_tc_beq_true"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_improve_rate": summary["best_cycle_improve"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_degrade_rate": summary["best_cycle_degrade"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_tc_improve_rate": summary["best_cycle_tc_improve"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_tc_degrade_rate": summary["best_cycle_tc_degrade"] / summary["count"] if summary["count"] else 0.0,
    }

    out_payload = {
        "meta": {
            "results_path": str(args.results),
            "lean_version": args.lean_version,
            "timeout": args.timeout,
        },
        "summary": summary_rates,
        "records": records,
    }

    if args.output is not None:
        args.output.write_text(json.dumps(out_payload, indent=2))
        print(f"Saved BEq+ results to {args.output}")

    print("\nSummary:")
    for key, value in summary_rates.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
