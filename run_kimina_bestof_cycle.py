#!/usr/bin/env python3
"""Sample multiple Kimina candidates and rerank by cycle-consistency score."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent / "LeanInteract"
if PROJECT_ROOT.exists():
    sys.path.append(str(PROJECT_ROOT))

from lean_interact import AutoLeanServer, Command, LeanREPLConfig
from lean_interact.interface import CommandResponse, LeanError
from lean_interact.project import TempRequireProject

from run_kimina_genlm_full_pipeline import (
    AlwaysAcceptPotential,
    ModelConfig,
    build_formalization_prompt,
    build_informalization_prompt,
    build_models,
    build_prompted_llm,
    clean_candidate_output,
    decode_sequence,
    run_sampler,
    summarize_sequences,
)


DEFAULT_PRIMARY_MODEL_ID = "AI-MO/Kimina-Autoformalizer-7B"
DEFAULT_REFERENCE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "PAug/ProofNetVerif"
DEFAULT_SPLIT = "valid"
DEFAULT_TIMEOUT = 60
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|[^\s]")


@dataclass
class SummaryStats:
    count: int = 0
    baseline_typecheck_sum: int = 0
    best_cycle_typecheck_sum: int = 0
    best_cycle_tc_typecheck_sum: int = 0
    best_lenpen_typecheck_sum: int = 0
    best_lenpen_tc_typecheck_sum: int = 0
    baseline_cycle_sum: float = 0.0
    best_cycle_sum: float = 0.0
    best_cycle_tc_sum: float = 0.0
    baseline_lenpen_sum: float = 0.0
    best_lenpen_sum: float = 0.0
    best_lenpen_tc_sum: float = 0.0
    best_cycle_improve_sum: int = 0
    best_cycle_change_sum: int = 0
    best_lenpen_improve_sum: int = 0
    best_lenpen_change_sum: int = 0
    avg_candidates_sum: int = 0
    avg_typechecked_candidates_sum: int = 0

    def as_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "count": 0,
                "baseline_typecheck_rate": 0.0,
                "best_cycle_typecheck_rate": 0.0,
                "best_cycle_tc_typecheck_rate": 0.0,
                "best_lenpen_typecheck_rate": 0.0,
                "best_lenpen_tc_typecheck_rate": 0.0,
                "baseline_cycle_avg": 0.0,
                "best_cycle_avg": 0.0,
                "best_cycle_tc_avg": 0.0,
                "baseline_lenpen_avg": 0.0,
                "best_lenpen_avg": 0.0,
                "best_lenpen_tc_avg": 0.0,
                "best_cycle_improve_rate": 0.0,
                "best_cycle_change_rate": 0.0,
                "best_lenpen_improve_rate": 0.0,
                "best_lenpen_change_rate": 0.0,
                "avg_candidates": 0.0,
                "avg_typechecked_candidates": 0.0,
            }
        return {
            "count": self.count,
            "baseline_typecheck_rate": self.baseline_typecheck_sum / self.count,
            "best_cycle_typecheck_rate": self.best_cycle_typecheck_sum / self.count,
            "best_cycle_tc_typecheck_rate": self.best_cycle_tc_typecheck_sum / self.count,
            "best_lenpen_typecheck_rate": self.best_lenpen_typecheck_sum / self.count,
            "best_lenpen_tc_typecheck_rate": self.best_lenpen_tc_typecheck_sum / self.count,
            "baseline_cycle_avg": self.baseline_cycle_sum / self.count,
            "best_cycle_avg": self.best_cycle_sum / self.count,
            "best_cycle_tc_avg": self.best_cycle_tc_sum / self.count,
            "baseline_lenpen_avg": self.baseline_lenpen_sum / self.count,
            "best_lenpen_avg": self.best_lenpen_sum / self.count,
            "best_lenpen_tc_avg": self.best_lenpen_tc_sum / self.count,
            "best_cycle_improve_rate": self.best_cycle_improve_sum / self.count,
            "best_cycle_change_rate": self.best_cycle_change_sum / self.count,
            "best_lenpen_improve_rate": self.best_lenpen_improve_sum / self.count,
            "best_lenpen_change_rate": self.best_lenpen_change_sum / self.count,
            "avg_candidates": self.avg_candidates_sum / self.count,
            "avg_typechecked_candidates": self.avg_typechecked_candidates_sum / self.count,
        }


def _theorem_name(index: int) -> str:
    return f"autoformalized_theorem_{index}"


def _format_cycle_prompt(reference_tokenizer, lean_code: str, original_nl: str) -> List[int]:
    prompt_messages = build_informalization_prompt(lean_code, original_nl)
    try:
        return reference_tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        prompt_text = "\n".join(message["content"] for message in prompt_messages)
        return reference_tokenizer(prompt_text, add_special_tokens=True).input_ids


def score_cycle(
    lean_code: str,
    original_nl: str,
    reference_model,
    reference_tokenizer,
) -> float:
    if not lean_code.strip() or not original_nl.strip():
        return float("-inf")
    prompt_ids = _format_cycle_prompt(reference_tokenizer, lean_code, original_nl)
    target_ids = reference_tokenizer.encode(original_nl, add_special_tokens=False)
    if not prompt_ids or not target_ids:
        return float("-inf")
    device = reference_model.device
    full_ids = torch.tensor([prompt_ids + target_ids], device=device)
    attention_mask = torch.ones_like(full_ids)
    with torch.no_grad():
        outputs = reference_model(input_ids=full_ids, attention_mask=attention_mask)
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
    norm = max(1.0, float(len(target_ids)))
    return logp / norm


def _extract_theorem_block(text: str) -> str:
    if not text:
        return ""
    snippet = text.strip()
    lower = snippet.lower()
    idx = lower.rfind("theorem")
    if idx != -1:
        snippet = snippet[idx:]
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


def _statement_len(text: str) -> int:
    snippet = _strip_theorem_name(_extract_theorem_block(text))
    return len(TOKEN_RE.findall(snippet))


def _build_header_env(
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


def typecheck_candidate(
    server: AutoLeanServer,
    candidate: str,
    lean_header: str,
    env_id: int | None,
    timeout: int,
) -> bool:
    if not candidate.strip():
        return False
    if env_id is None:
        cmd = lean_header.strip() + "\n\n" + candidate.strip() + "\n"
        request = Command(cmd=cmd)
    else:
        request = Command(cmd=candidate.strip() + "\n", env=env_id)
    try:
        response = server.run(request, timeout=timeout)
    except TimeoutError:
        return False
    except Exception:
        return False
    if isinstance(response, LeanError):
        return False
    if isinstance(response, CommandResponse):
        return response.lean_code_is_valid()
    return False


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _update_stats_from_record(stats: SummaryStats, record: Dict[str, Any]) -> None:
    stats.count += 1

    def _as_float(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    baseline_cycle = _as_float(record.get("baseline_cycle_score"))
    best_cycle = _as_float(record.get("best_cycle_score"))
    best_cycle_tc = _as_float(record.get("best_cycle_tc_score"))
    baseline_lenpen = _as_float(record.get("baseline_length_penalized_score"))
    best_lenpen = _as_float(record.get("best_lenpen_score"))
    best_lenpen_tc = _as_float(record.get("best_lenpen_tc_score"))

    if baseline_cycle is not None and math.isfinite(baseline_cycle):
        stats.baseline_cycle_sum += baseline_cycle
    if best_cycle is not None and math.isfinite(best_cycle):
        stats.best_cycle_sum += best_cycle
    if best_cycle_tc is not None and math.isfinite(best_cycle_tc):
        stats.best_cycle_tc_sum += best_cycle_tc
    if baseline_lenpen is not None and math.isfinite(baseline_lenpen):
        stats.baseline_lenpen_sum += baseline_lenpen
    if best_lenpen is not None and math.isfinite(best_lenpen):
        stats.best_lenpen_sum += best_lenpen
    if best_lenpen_tc is not None and math.isfinite(best_lenpen_tc):
        stats.best_lenpen_tc_sum += best_lenpen_tc

    stats.baseline_typecheck_sum += int(bool(record.get("baseline_typecheck", False)))
    stats.best_cycle_typecheck_sum += int(bool(record.get("best_cycle_typecheck", False)))
    stats.best_cycle_tc_typecheck_sum += int(bool(record.get("best_cycle_tc_typecheck", False)))
    stats.best_lenpen_typecheck_sum += int(bool(record.get("best_lenpen_typecheck", False)))
    stats.best_lenpen_tc_typecheck_sum += int(bool(record.get("best_lenpen_tc_typecheck", False)))

    if best_cycle is not None and baseline_cycle is not None and best_cycle > baseline_cycle:
        stats.best_cycle_improve_sum += 1
    if best_lenpen is not None and baseline_lenpen is not None and best_lenpen > baseline_lenpen:
        stats.best_lenpen_improve_sum += 1

    baseline_candidate = record.get("baseline_candidate", "")
    if record.get("best_cycle_candidate") and record.get("best_cycle_candidate") != baseline_candidate:
        stats.best_cycle_change_sum += 1
    if record.get("best_lenpen_candidate") and record.get("best_lenpen_candidate") != baseline_candidate:
        stats.best_lenpen_change_sum += 1

    stats.avg_candidates_sum += int(record.get("num_candidates", 0) or 0)
    stats.avg_typechecked_candidates_sum += int(record.get("num_typechecked_candidates", 0) or 0)


def _compute_shard_indices(
    total: int,
    start_index: int,
    n_examples: int | None,
    num_shards: int,
    shard_id: int,
) -> List[int]:
    end_index = total if n_examples is None else min(total, start_index + n_examples)
    base = list(range(start_index, end_index))
    if num_shards <= 1:
        return base
    return [idx for pos, idx in enumerate(base) if pos % num_shards == shard_id]


async def async_main(args: argparse.Namespace) -> None:
    dataset = load_dataset(DATASET_NAME, split=args.split)
    indices = _compute_shard_indices(
        len(dataset),
        start_index=args.start_index,
        n_examples=args.n_examples,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    if not indices:
        print("[INFO] No examples to process; exiting.")
        return

    if args.results_path.exists():
        try:
            existing = json.loads(args.results_path.read_text())
            records = existing.get("records", [])
        except Exception:
            records = []
    else:
        records = []
    processed_ids = {
        rec.get("id")
        for rec in records
        if isinstance(rec, dict) and rec.get("id") is not None
    }

    model_cfg = ModelConfig(
        primary_model_id=args.primary_model_id,
        reference_model_id=args.reference_model_id,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=args.four_bit,
    )
    models = build_models(model_cfg)
    reference_model = models["reference_model"]
    reference_tokenizer = models["reference_tok"]

    repl_config = None
    server = None
    header_cache: Dict[str, int | None] = {}
    if args.typecheck:
        repl_kwargs = {}
        if args.repl_cache_dir is not None:
            repl_kwargs["cache_dir"] = str(args.repl_cache_dir)
        repl_config = LeanREPLConfig(
            project=TempRequireProject(lean_version=args.lean_version, require="mathlib"),
            verbose=False,
            **repl_kwargs,
        )
        server = AutoLeanServer(config=repl_config)

    stats = SummaryStats()
    for record in records:
        if isinstance(record, dict):
            _update_stats_from_record(stats, record)
    results_path = args.results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)

    def persist() -> None:
        payload = {
            "meta": {
                "model": model_cfg.primary_model_id,
                "reference_model": model_cfg.reference_model_id,
                "max_tokens": model_cfg.max_new_tokens,
                "temperature": model_cfg.temperature,
                "top_p": model_cfg.top_p,
                "num_candidates": args.num_candidates,
                "length_penalty": args.length_penalty,
                "dataset": DATASET_NAME,
                "split": args.split,
                "start_index": args.start_index,
                "n_examples": args.n_examples,
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
                "typecheck": args.typecheck,
                "lean_version": args.lean_version,
            },
            "summary": stats.as_dict(),
            "records": records,
        }
        results_path.write_text(json.dumps(payload, indent=2))

    for idx, dataset_idx in enumerate(indices, start=1):
        example = dataset[dataset_idx]
        example_id = example["id"]
        if example_id in processed_ids:
            print(f"[{idx}/{len(indices)}] {example_id} already processed, skipping.")
            continue

        theorem_name = _theorem_name(dataset_idx)
        nl_statement = example["nl_statement"].strip()
        lean_header = example["lean4_src_header"]

        kimina_llm = build_prompted_llm(
            models["primary_async"],
            models["primary_tok"],
            model_cfg,
            nl_statement,
            theorem_name,
        )
        accept_all = AlwaysAcceptPotential(kimina_llm.vocab)

        start = time.time()
        baseline_sequences = await run_sampler(
            unit_potential=kimina_llm,
            condition=accept_all,
            n_particles=1,
            max_tokens=model_cfg.max_new_tokens,
            ess_threshold=0.0,
        )
        baseline_posterior, baseline_raw, _ = summarize_sequences(baseline_sequences, theorem_name)
        baseline_candidate = next(iter(baseline_posterior.keys()), "")
        baseline_cycle = score_cycle(
            baseline_candidate,
            nl_statement,
            reference_model,
            reference_tokenizer,
        )
        if not math.isfinite(baseline_cycle):
            baseline_cycle = float("-inf")
        baseline_len = _statement_len(baseline_candidate)
        baseline_lenpen = baseline_cycle - (args.length_penalty * baseline_len)
        baseline_typecheck = False
        env_id = None
        if args.typecheck and server is not None:
            env_id = _build_header_env(server, header_cache, lean_header, timeout=args.timeout)
            baseline_typecheck = typecheck_candidate(
                server,
                baseline_candidate,
                lean_header,
                env_id,
                timeout=args.timeout,
            )

        candidate_sequences = await run_sampler(
            unit_potential=kimina_llm,
            condition=accept_all,
            n_particles=args.num_candidates,
            max_tokens=model_cfg.max_new_tokens,
            ess_threshold=0.0,
        )
        candidate_posterior, candidate_raw, _ = summarize_sequences(candidate_sequences, theorem_name)
        candidates = _dedupe(candidate_posterior.keys())
        candidate_set = set(candidates)

        candidate_info: List[Dict[str, Any]] = []
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
        typechecked_count = 0

        for candidate in candidates:
            candidate_len = _statement_len(candidate)
            cycle_score = score_cycle(
                candidate,
                nl_statement,
                reference_model,
                reference_tokenizer,
            )
            if not math.isfinite(cycle_score):
                cycle_score = float("-inf")
            lenpen_score = cycle_score - (args.length_penalty * candidate_len)
            candidate_typecheck = False
            if args.typecheck and server is not None:
                candidate_typecheck = typecheck_candidate(
                    server,
                    candidate,
                    lean_header,
                    env_id,
                    timeout=args.timeout,
                )
            if candidate_typecheck:
                typechecked_count += 1
            candidate_info.append(
                {
                    "candidate": candidate,
                    "cycle_score": cycle_score,
                    "length_penalized_score": lenpen_score,
                    "statement_len": candidate_len,
                    "typecheck": candidate_typecheck,
                }
            )
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

        baseline_in_candidates = baseline_candidate in candidate_set
        if baseline_candidate and not baseline_in_candidates:
            baseline_entry = {
                "candidate": baseline_candidate,
                "cycle_score": baseline_cycle,
                "length_penalized_score": baseline_lenpen,
                "statement_len": baseline_len,
                "typecheck": baseline_typecheck,
                "source": "baseline",
            }
            candidate_info.append(baseline_entry)
            if baseline_cycle > best_cycle_score:
                best_cycle_score = baseline_cycle
                best_cycle_candidate = baseline_candidate
                best_cycle_typecheck = baseline_typecheck
            if baseline_typecheck and baseline_cycle > best_cycle_tc_score:
                best_cycle_tc_score = baseline_cycle
                best_cycle_tc_candidate = baseline_candidate
            if baseline_lenpen > best_lenpen_score:
                best_lenpen_score = baseline_lenpen
                best_lenpen_candidate = baseline_candidate
                best_lenpen_typecheck = baseline_typecheck
            if baseline_typecheck and baseline_lenpen > best_lenpen_tc_score:
                best_lenpen_tc_score = baseline_lenpen
                best_lenpen_tc_candidate = baseline_candidate

        if not best_cycle_candidate:
            best_cycle_candidate = baseline_candidate
            best_cycle_score = baseline_cycle
            best_cycle_typecheck = baseline_typecheck
        if not best_cycle_tc_candidate:
            best_cycle_tc_candidate = best_cycle_candidate
            best_cycle_tc_score = best_cycle_score
            best_cycle_tc_typecheck = best_cycle_typecheck
        else:
            best_cycle_tc_typecheck = True
        if not best_lenpen_candidate:
            best_lenpen_candidate = baseline_candidate
            best_lenpen_score = baseline_lenpen
            best_lenpen_typecheck = baseline_typecheck
        if not best_lenpen_tc_candidate:
            best_lenpen_tc_candidate = best_lenpen_candidate
            best_lenpen_tc_score = best_lenpen_score
            best_lenpen_tc_typecheck = best_lenpen_typecheck
        else:
            best_lenpen_tc_typecheck = True

        record = {
            "index": dataset_idx,
            "id": example_id,
            "nl_statement": nl_statement,
            "lean_header": lean_header,
            "ground_truth": example.get("lean4_formalization", ""),
            "baseline_candidate": baseline_candidate,
            "baseline_cycle_score": baseline_cycle,
            "baseline_length_penalized_score": baseline_lenpen,
            "baseline_statement_len": baseline_len,
            "baseline_typecheck": baseline_typecheck,
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
            "num_candidates": len(candidates),
            "num_candidates_with_baseline": len(candidate_info),
            "baseline_in_candidates": baseline_in_candidates,
            "num_typechecked_candidates": typechecked_count,
            "baseline_raw": baseline_raw,
            "candidate_raw": candidate_raw,
            "candidates": candidate_info,
            "runtime_sec": time.time() - start,
        }
        records.append(record)
        processed_ids.add(example_id)

        _update_stats_from_record(stats, record)

        persist()
        print(
            f"[{idx}/{len(indices)}] {example_id} "
            f"baseline_tc={int(baseline_typecheck)} best_cycle_tc={int(best_cycle_typecheck)} "
            f"best_lenpen_tc={int(best_lenpen_tc_typecheck)} "
            f"cands={len(candidates)} typechecked={typechecked_count}"
        )

    print("\nSummary:")
    for key, value in stats.as_dict().items():
        print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-of-N Kimina sampling with cycle reranking.")
    parser.add_argument("--primary-model-id", type=str, default=DEFAULT_PRIMARY_MODEL_ID)
    parser.add_argument("--reference-model-id", type=str, default=DEFAULT_REFERENCE_MODEL_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--four-bit", action="store_true")
    parser.add_argument("--typecheck", action="store_true")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--lean-version", type=str, default="v4.8.0")
    parser.add_argument(
        "--repl-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for Lean REPL artifacts (useful when home is full).",
    )
    parser.add_argument("--results-path", type=Path, default=Path("kimina_bestof_cycle.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
