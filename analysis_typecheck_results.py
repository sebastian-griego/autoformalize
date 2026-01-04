#!/usr/bin/env python3
"""Type-check Kimina Lean candidates and summarize well-typedness metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from lean_interact import AutoLeanServer, Command, LeanREPLConfig
from lean_interact.interface import CommandResponse, LeanError
from lean_interact.project import TempRequireProject


DEFAULT_TIMEOUT = 60
ALL_KEY = "__all__"


@dataclass
class MethodStats:
    count: int = 0
    top1_typecheck_sum: int = 0
    any_typecheck_sum: int = 0
    expected_typecheck_sum: float = 0.0
    top1_trivial_sum: int = 0
    top1_exact_fallback_sum: int = 0
    top1_beq_true_sum: int = 0
    top1_beq_false_sum: int = 0
    top1_beq_missing_sum: int = 0
    top1_beq_and_typecheck_sum: int = 0
    top1_beq_true_typecheck_false_sum: int = 0
    top1_beq_false_typecheck_true_sum: int = 0
    avg_candidates_sum: int = 0
    checked_candidates_sum: int = 0
    timeout_errors: int = 0
    lean_errors: int = 0
    other_errors: int = 0

    def as_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "count": 0,
                "top1_typecheck_rate": 0.0,
                "any_typecheck_rate": 0.0,
                "expected_typecheck": 0.0,
                "top1_trivial_rate": 0.0,
                "top1_exact_fallback_rate": 0.0,
                "top1_beq_true_rate": 0.0,
                "top1_beq_false_rate": 0.0,
                "top1_beq_missing_rate": 0.0,
                "top1_beq_true_rate_given_present": 0.0,
                "top1_beq_and_typecheck_rate": 0.0,
                "top1_beq_true_typecheck_false_rate": 0.0,
                "top1_beq_false_typecheck_true_rate": 0.0,
                "avg_candidates": 0.0,
                "avg_checked_candidates": 0.0,
                "timeout_errors": 0,
                "lean_errors": 0,
                "other_errors": 0,
            }
        beq_present = self.top1_beq_true_sum + self.top1_beq_false_sum
        beq_rate_present = self.top1_beq_true_sum / beq_present if beq_present else 0.0
        return {
            "count": self.count,
            "top1_typecheck_rate": self.top1_typecheck_sum / self.count,
            "any_typecheck_rate": self.any_typecheck_sum / self.count,
            "expected_typecheck": self.expected_typecheck_sum / self.count,
            "top1_trivial_rate": self.top1_trivial_sum / self.count,
            "top1_exact_fallback_rate": self.top1_exact_fallback_sum / self.count,
            "top1_beq_true_rate": self.top1_beq_true_sum / self.count,
            "top1_beq_false_rate": self.top1_beq_false_sum / self.count,
            "top1_beq_missing_rate": self.top1_beq_missing_sum / self.count,
            "top1_beq_true_rate_given_present": beq_rate_present,
            "top1_beq_and_typecheck_rate": self.top1_beq_and_typecheck_sum / self.count,
            "top1_beq_true_typecheck_false_rate": self.top1_beq_true_typecheck_false_sum / self.count,
            "top1_beq_false_typecheck_true_rate": self.top1_beq_false_typecheck_true_sum / self.count,
            "avg_candidates": self.avg_candidates_sum / self.count,
            "avg_checked_candidates": self.checked_candidates_sum / self.count,
            "timeout_errors": self.timeout_errors,
            "lean_errors": self.lean_errors,
            "other_errors": self.other_errors,
        }


def _is_method_block(value: Any) -> bool:
    return isinstance(value, dict) and "posterior" in value


def _sorted_candidates(posterior: Dict[str, Any], top_k: int | None) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for candidate, prob in posterior.items():
        try:
            prob_val = float(prob)
        except (TypeError, ValueError):
            prob_val = 0.0
        items.append((candidate, prob_val))
    items.sort(key=lambda pair: pair[1], reverse=True)
    if top_k is not None:
        items = items[:top_k]
    return items


def _build_command(lean_header: str, candidate: str) -> str:
    header = lean_header.strip()
    if header:
        return header + "\n\n" + candidate.strip() + "\n"
    return candidate.strip() + "\n"


def _is_trivial(candidate: str) -> bool:
    return ": True := by sorry" in candidate


def _is_exact_fallback(candidate: str, theorem_name: str) -> bool:
    return candidate.strip() == f"theorem {theorem_name} : True := by sorry"


def _theorem_name(record: Dict[str, Any]) -> str:
    idx = record.get("index")
    if idx is not None:
        return f"autoformalized_theorem_{idx}"
    return "autoformalized_theorem"


def _check_candidate(
    server: AutoLeanServer,
    candidate: str,
    lean_header: str,
    env_id: int | None,
    timeout: int,
) -> Tuple[bool, str | None]:
    if not candidate.strip():
        return False, "empty"
    if env_id is None:
        request = Command(cmd=_build_command(lean_header, candidate))
    else:
        request = Command(cmd=candidate.strip() + "\n", env=env_id)
    try:
        response = server.run(request, timeout=timeout)
    except TimeoutError:
        return False, "timeout"
    except Exception:
        return False, "exception"
    if isinstance(response, LeanError):
        return False, "lean_error"
    if isinstance(response, CommandResponse):
        return response.lean_code_is_valid(), None
    return False, "unexpected"


def _get_header_env(
    server: AutoLeanServer,
    header_cache: Dict[str, int | None],
    lean_header: str,
    timeout: int,
) -> int | None:
    if lean_header in header_cache:
        return header_cache[lean_header]
    header = lean_header.strip()
    if not header:
        header_cache[lean_header] = None
        return None
    response = server.run(Command(cmd=header + "\n"), timeout=timeout, add_to_session_cache=True)
    if isinstance(response, LeanError):
        header_cache[lean_header] = None
        return None
    if isinstance(response, CommandResponse) and not response.lean_code_is_valid():
        header_cache[lean_header] = None
        return None
    env_id = response.env if isinstance(response, CommandResponse) else None
    header_cache[lean_header] = env_id
    return env_id


def _record_stats(stats: MethodStats, candidate_count: int, checked_count: int) -> None:
    stats.count += 1
    stats.avg_candidates_sum += candidate_count
    stats.checked_candidates_sum += checked_count


def analyze_results(
    results_paths: Iterable[Path],
    *,
    top_k: int | None,
    timeout: int,
    lean_version: str,
    cache_headers: bool,
    limit_per_file: int | None,
) -> Dict[str, Dict[str, MethodStats]]:
    config = LeanREPLConfig(
        project=TempRequireProject(lean_version=lean_version, require="mathlib"),
        verbose=False,
    )
    server = AutoLeanServer(config=config)
    header_cache: Dict[str, int | None] = {}
    stats_by_source: Dict[str, Dict[str, MethodStats]] = {}

    for path in results_paths:
        data = json.loads(path.read_text())
        records = data.get("records", [])
        if limit_per_file is not None:
            records = records[:limit_per_file]
        for record in records:
            lean_header = record.get("lean_header", "")
            env_id = None
            if cache_headers:
                env_id = _get_header_env(server, header_cache, lean_header, timeout=timeout)
            theorem_name = _theorem_name(record)

            for method_name, block in record.items():
                if not _is_method_block(block):
                    continue
                posterior = block.get("posterior", {}) or {}
                candidates = _sorted_candidates(posterior, top_k)
                method_stats = stats_by_source.setdefault(str(path), {}).setdefault(method_name, MethodStats())
                overall_stats = stats_by_source.setdefault(ALL_KEY, {}).setdefault(method_name, MethodStats())
                _record_stats(method_stats, len(posterior), len(candidates))
                _record_stats(overall_stats, len(posterior), len(candidates))
                if not candidates:
                    continue

                top1_candidate, _ = candidates[0]
                beq_labels = block.get("beq_labels", {}) or {}
                top1_beq = beq_labels.get(top1_candidate)
                for stats in (method_stats, overall_stats):
                    if top1_beq is None:
                        stats.top1_beq_missing_sum += 1
                    elif top1_beq:
                        stats.top1_beq_true_sum += 1
                    else:
                        stats.top1_beq_false_sum += 1

                    if _is_trivial(top1_candidate):
                        stats.top1_trivial_sum += 1
                    if _is_exact_fallback(top1_candidate, theorem_name):
                        stats.top1_exact_fallback_sum += 1

                any_valid = False
                expected_valid = 0.0
                top1_valid = False

                for candidate, prob in candidates:
                    valid, error = _check_candidate(
                        server,
                        candidate,
                        lean_header,
                        env_id,
                        timeout=timeout,
                    )
                    for stats in (method_stats, overall_stats):
                        if error == "timeout":
                            stats.timeout_errors += 1
                        elif error == "lean_error":
                            stats.lean_errors += 1
                        elif error not in (None, "empty"):
                            stats.other_errors += 1

                    if valid:
                        any_valid = True
                        expected_valid += prob
                    if candidate == top1_candidate:
                        top1_valid = valid

                for stats in (method_stats, overall_stats):
                    if top1_valid:
                        stats.top1_typecheck_sum += 1
                    if any_valid:
                        stats.any_typecheck_sum += 1
                    stats.expected_typecheck_sum += expected_valid
                    if top1_beq is True and top1_valid:
                        stats.top1_beq_and_typecheck_sum += 1
                    if top1_beq is True and not top1_valid:
                        stats.top1_beq_true_typecheck_false_sum += 1
                    if top1_beq is False and top1_valid:
                        stats.top1_beq_false_typecheck_true_sum += 1

    return stats_by_source


def _format_stats(stats_by_source: Dict[str, Dict[str, MethodStats]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    formatted: Dict[str, Dict[str, Dict[str, float]]] = {}
    for source, method_map in stats_by_source.items():
        formatted[source] = {method: stats.as_dict() for method, stats in method_map.items()}
    return formatted


def _print_summary(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    for source, methods in summary.items():
        print(f"\n=== {source} ===")
        for method_name in sorted(methods):
            stats = methods[method_name]
            print(
                f"{method_name:18s} "
                f"top1_tc={stats['top1_typecheck_rate']:.3f} "
                f"any_tc={stats['any_typecheck_rate']:.3f} "
                f"exp_tc={stats['expected_typecheck']:.3f} "
                f"top1_beq={stats['top1_beq_true_rate_given_present']:.3f} "
                f"top1_beq_and_tc={stats['top1_beq_and_typecheck_rate']:.3f} "
                f"top1_trivial={stats['top1_trivial_rate']:.3f} "
                f"avg_cands={stats['avg_candidates']:.2f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Type-check Kimina results and summarize well-typedness.")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Result JSON files to analyze")
    parser.add_argument("--top-k", type=int, default=None, help="Only check the top-k candidates per method")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Lean REPL timeout per candidate")
    parser.add_argument(
        "--lean-version",
        type=str,
        default="v4.8.0",
        help="Lean version for TempRequireProject",
    )
    parser.add_argument(
        "--no-header-cache",
        action="store_true",
        help="Disable caching of Lean headers across candidates",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Limit the number of records processed per file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_paths = [path for path in args.results if path.exists()]
    if not results_paths:
        raise SystemExit("No valid results paths supplied.")
    stats_by_source = analyze_results(
        results_paths,
        top_k=args.top_k,
        timeout=args.timeout,
        lean_version=args.lean_version,
        cache_headers=not args.no_header_cache,
        limit_per_file=args.limit_per_file,
    )
    summary = _format_stats(stats_by_source)
    if args.output is not None:
        payload = {
            "meta": {
                "paths": [str(path) for path in results_paths],
                "top_k": args.top_k,
                "timeout": args.timeout,
                "lean_version": args.lean_version,
                "header_cache": not args.no_header_cache,
                "limit_per_file": args.limit_per_file,
            },
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))
    _print_summary(summary)


if __name__ == "__main__":
    main()
