"""End-to-end Kimina autoformalization pipeline with GenLM steering and BEq+ evaluation.

This script reproduces the MATH AI 2025 style pipeline for translating natural language
math statements into Lean 4 theorems using Kimina with cycle-consistency and Lean typing
potentials, and evaluates semantic correctness through the BEq+ benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent / "LeanInteract"
if PROJECT_ROOT.exists():
    sys.path.append(str(PROJECT_ROOT))

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None
from genlm.backend.llm.hf import AsyncTransformer
from genlm.control import AWRS, PromptedLLM
from genlm.control.constant import EndOfSequence
from genlm.control.potential.base import Potential
from genlm.control.potential.product import Product
from lean_interact import AutoLeanServer, Command, LeanREPLConfig
from lean_interact.interface import LeanError
from lean_interact.project import TempRequireProject

from examples.beq_plus import DEFAULT_TIMEOUT, beq_plus

DEFAULT_PRIMARY_MODEL_ID = "AI-MO/Kimina-Autoformalizer-7B"
DEFAULT_REFERENCE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
N_PARTICLES = 5
DATASET_NAME = "PAug/ProofNetVerif"
SPLIT = "valid"
N_EXAMPLES = 100
RESULTS_PATH = Path("kimina_genlm_full_results.json")


@dataclass
class ModelConfig:
    primary_model_id: str = DEFAULT_PRIMARY_MODEL_ID
    reference_model_id: str = DEFAULT_REFERENCE_MODEL_ID
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    load_in_4bit: bool = False


@dataclass
class SMCConfig:
    num_particles: int = N_PARTICLES
    smc_mode: str = "all"
    potential_stride: int = 1


def load_hf_causal_lm(model_id: str, *, load_in_4bit: bool) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quant_config = None
    if load_in_4bit and BitsAndBytesConfig is not None:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            quantization_config=quant_config,
        )
    finally:
        torch.set_grad_enabled(grad_state)
    model.eval()
    return model, tokenizer


def build_models(cfg: ModelConfig) -> Dict[str, Any]:
    primary_model, primary_tok = load_hf_causal_lm(cfg.primary_model_id, load_in_4bit=cfg.load_in_4bit)
    reference_model, reference_tok = load_hf_causal_lm(cfg.reference_model_id, load_in_4bit=cfg.load_in_4bit)
    primary_async = AsyncTransformer(primary_model, primary_tok)
    return {
        "primary_model": primary_model,
        "primary_tok": primary_tok,
        "primary_async": primary_async,
        "reference_model": reference_model,
        "reference_tok": reference_tok,
    }


@dataclass
class MethodRun:
    posterior: Dict[str, float]
    raw_particles: List[Dict[str, Any]]
    runtime_sec: float
    avg_tokens: float


@dataclass
class MethodSummary:
    score_sum: float = 0.0
    runtime_sum: float = 0.0
    token_sum: float = 0.0

    def update(self, score: float, runtime: float, avg_tokens: float) -> None:
        self.score_sum += score
        self.runtime_sum += runtime
        self.token_sum += avg_tokens


def build_formalization_prompt(nl_statement: str, theorem_name: str) -> str:
    """Construct the Kimina user message following the BEq+ prompt format."""
    intro = (
        "Autoformalize the following natural-language mathematics statement in Lean 4. "
        "Return exactly one theorem declaration named "
        f"`{theorem_name}` with a proof stub of the form `:= by sorry`.\n"
        "Do not include `import` or `open` commands because a header is provided separately.\n"
        "Only translate the statement: give hypotheses and conclusion precisely and leave the "
        "proof as `sorry`.\n\n"
    )
    return intro + nl_statement.strip()


def build_informalization_prompt(lean_stmt: str, original_nl: str) -> List[Dict[str, str]]:
    """Build a chat-style prompt for the backtranslation LM.

    The original natural-language statement is intentionally held out so that we can score its
    log probability after feeding the Lean code to the backtranslation model.
    """
    _ = original_nl  # kept for clarity and future extensions
    lean_block = f"```lean4\n{lean_stmt.strip()}\n```"
    return [
        {
            "role": "system",
            "content": (
                "You are an expert at translating Lean 4 theorem statements into clear, concise "
                "natural language descriptions."
            ),
        },
        {
            "role": "user",
            "content": (
                "Paraphrase the following Lean theorem in a single English sentence. "
                "Focus purely on the mathematical statement and hypotheses, ignore proof terms.\n\n"
                f"{lean_block}\n\nRespond with the informal statement only."
            ),
        },
    ]


def clean_candidate_output(text: str, theorem_name: str) -> str:
    """Normalize Kimina output into a single Lean theorem declaration."""
    if not text:
        return f"theorem {theorem_name} : True := by sorry"

    snippet = text.strip()
    if "```" in snippet:
        parts = snippet.split("```")
        code_blocks = [parts[i] for i in range(1, len(parts), 2)]
        if code_blocks:
            snippet = code_blocks[-1].strip()
    lower = snippet.lower()
    idx = lower.find("theorem")
    if idx != -1:
        snippet = snippet[idx:]
    snippet = snippet.strip()
    if not snippet:
        return f"theorem {theorem_name} : True := by sorry"

    # Force the declaration to start with `theorem`.
    snippet = re.sub(r"^(lemma|example)\b", "theorem", snippet, flags=re.IGNORECASE)

    match = re.search(r"theorem\s+([^\s:({]+)", snippet)
    if match:
        current_name = match.group(1)
        if current_name != theorem_name:
            snippet = snippet[: match.start(1)] + theorem_name + snippet[match.end(1) :]
    else:
        snippet = f"theorem {theorem_name} : True := by sorry"
        return snippet

    head, _, tail = snippet.partition(":=")
    head = head.strip()
    if ":" not in head:
        head = head.rstrip()
        head = f"{head} : True"
    if not tail.strip():
        snippet = f"{head} := by sorry"
    else:
        proof_stub = tail.strip()
        if not proof_stub.startswith("by"):
            snippet = f"{head} := by sorry"
        else:
            body = proof_stub[2:].strip()
            if not body or body.startswith("?_"):
                snippet = f"{head} := by sorry"
            elif "sorry" not in body:
                snippet = f"{head} := by sorry"
            else:
                snippet = f"{head} := by sorry"

    return snippet.strip()


class AlwaysAcceptPotential(Potential):
    """Boolean potential that never rejects any prefix or completion."""

    async def prefix(self, context):
        return 0.0

    async def complete(self, context):
        return 0.0


class LeanWellTypedPotential(Potential):
    """Binary potential that rewards Lean snippets that type-check under the dataset header."""

    def __init__(
        self,
        vocabulary,
        lean_server: AutoLeanServer,
        lean_header: str,
        theorem_name: str,
        potential_stride: int = 1,
    ):
        super().__init__(vocabulary)
        self.server = lean_server
        self.lean_header = lean_header.strip()
        self.theorem_name = theorem_name
        self._lock = asyncio.Lock()
        self.potential_stride = max(1, potential_stride)
        self._score_cache: Dict[bytes, float] = {}

    async def prefix(self, context) -> float:
        if not context:
            return 0.0
        context_bytes = self._context_bytes(context)
        cached = self._maybe_reuse_score(context, context_bytes)
        if cached is not None:
            return cached
        text = context_bytes.decode("utf-8", errors="ignore")
        if not text.strip():
            return 0.0
        if self._has_noise(text):
            return float("-inf")
        if not self._looks_complete(text):
            return 0.0
        score = await self._check_well_typed(text)
        self._score_cache[context_bytes] = score
        return score

    async def complete(self, context) -> float:
        if not context:
            return 0.0
        context_bytes = self._context_bytes(context)
        cached = self._maybe_reuse_score(context, context_bytes)
        if cached is not None:
            return cached
        text = context_bytes.decode("utf-8", errors="ignore")
        if not self._looks_complete(text):
            return float("-inf")
        score = await self._check_well_typed(text)
        self._score_cache[context_bytes] = score
        return score

    def _context_bytes(self, context: Iterable[Any], limit: int | None = None) -> bytes:
        byte_stream = []
        count = 0
        for token in context:
            if isinstance(token, EndOfSequence):
                break
            if limit is not None and count >= limit:
                break
            byte_stream.append(token)
            count += 1
        return b"".join(byte_stream)

    def _context_length(self, context: Iterable[Any]) -> int:
        return sum(1 for token in context if not isinstance(token, EndOfSequence))

    def _maybe_reuse_score(self, context: Iterable[Any], context_bytes: bytes) -> float | None:
        if self.potential_stride <= 1:
            return None
        token_len = self._context_length(context)
        if token_len == 0 or token_len % self.potential_stride == 0:
            return None
        prev_len = token_len - (token_len % self.potential_stride)
        if prev_len <= 0:
            return None
        prev_bytes = self._context_bytes(context, limit=prev_len)
        cached = self._score_cache.get(prev_bytes)
        if cached is not None:
            self._score_cache[context_bytes] = cached
        return cached

    def _has_noise(self, text: str) -> bool:
        if text.count("\n\n\n") > 2:
            return True
        if text.count("/-") > 2 or text.count("--") > 4:
            return True
        return False

    def _looks_complete(self, text: str) -> bool:
        snippet = text.strip()
        return "theorem" in snippet.lower() and ":= by" in snippet

    async def _check_well_typed(self, generated_text: str) -> float:
        cleaned = clean_candidate_output(generated_text, self.theorem_name)
        lean_code = f"{self.lean_header}\n\n{cleaned}\n"
        command = Command(cmd=lean_code)
        async with self._lock:
            try:
                response = await self.server.async_run(command, timeout=DEFAULT_TIMEOUT)
            except Exception:
                return float("-inf")
        if isinstance(response, LeanError):
            return float("-inf")
        return 0.0


class CycleConsistencyPotential(Potential):
    """Continuous potential scored via the reference LM's likelihood of the original NL statement."""

    def __init__(
        self,
        vocabulary,
        original_nl: str,
        reference_model: AutoModelForCausalLM,
        reference_tokenizer,
        cfg: ModelConfig,
        potential_stride: int = 1,
    ):
        super().__init__(vocabulary)
        self.original_nl = original_nl.strip()
        self.reference_model = reference_model
        self.reference_tokenizer = reference_tokenizer
        self.cfg = cfg
        self.potential_stride = max(1, potential_stride)
        self._score_cache: Dict[bytes, float] = {}

    async def prefix(self, context) -> float:
        return 0.0

    async def complete(self, context) -> float:
        if not context:
            return float("-inf")
        context_bytes = self._context_bytes(context)
        cached = self._maybe_reuse_score(context, context_bytes)
        if cached is not None:
            return cached
        lean_code = context_bytes.decode("utf-8", errors="ignore")
        if not lean_code.strip() or not self.original_nl:
            return float("-inf")
        try:
            score = await asyncio.to_thread(self._score_cycle, lean_code)
        except Exception:
            return float("-inf")
        self._score_cache[context_bytes] = score
        return score

    def _score_cycle(self, lean_code: str) -> float:
        prompt_messages = build_informalization_prompt(lean_code, self.original_nl)
        try:
            prompt_ids = self.reference_tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = "\n".join(m["content"] for m in prompt_messages)
            prompt_ids = self.reference_tokenizer(
                prompt_text,
                add_special_tokens=True,
            ).input_ids

        target_ids = self.reference_tokenizer.encode(
            self.original_nl,
            add_special_tokens=False,
        )
        if not target_ids or not prompt_ids:
            return float("-inf")

        device = self.reference_model.device
        full_ids = torch.tensor([prompt_ids + target_ids], device=device)
        attention_mask = torch.ones_like(full_ids)
        with torch.no_grad():
            outputs = self.reference_model(
                input_ids=full_ids,
                attention_mask=attention_mask,
            )
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
        return (logp / norm) * self.cfg.top_p

    def _context_bytes(self, context: Iterable[Any], limit: int | None = None) -> bytes:
        byte_stream = []
        count = 0
        for token in context:
            if isinstance(token, EndOfSequence):
                break
            if limit is not None and count >= limit:
                break
            byte_stream.append(token)
            count += 1
        return b"".join(byte_stream)

    def _context_length(self, context: Iterable[Any]) -> int:
        return sum(1 for token in context if not isinstance(token, EndOfSequence))

    def _maybe_reuse_score(self, context: Iterable[Any], context_bytes: bytes) -> float | None:
        if self.potential_stride <= 1:
            return None
        token_len = self._context_length(context)
        if token_len == 0 or token_len % self.potential_stride == 0:
            return None
        prev_len = token_len - (token_len % self.potential_stride)
        if prev_len <= 0:
            return None
        prev_bytes = self._context_bytes(context, limit=prev_len)
        cached = self._score_cache.get(prev_bytes)
        if cached is not None:
            self._score_cache[context_bytes] = cached
        return cached


def decode_sequence(context: List[Any]) -> str:
    byte_stream = []
    for token in context:
        if isinstance(token, EndOfSequence):
            break
        if isinstance(token, bytes):
            byte_stream.append(token)
    try:
        return b"".join(byte_stream).decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        return ""


def summarize_sequences(sequences, theorem_name: str) -> Tuple[Dict[str, float], List[Dict[str, Any]], float]:
    posterior_chart = sequences.decoded_posterior
    aggregate: Dict[str, float] = {}
    for raw_text, prob in posterior_chart.items():
        cleaned = clean_candidate_output(raw_text, theorem_name)
        if not cleaned.strip():
            continue
        aggregate[cleaned] = aggregate.get(cleaned, 0.0) + float(prob)
    total = sum(aggregate.values())
    if total > 0:
        aggregate = {k: v / total for k, v in aggregate.items()}

    raw_particles = []
    for context, log_w in zip(sequences.contexts, sequences.log_weights):
        raw_particles.append({"text": decode_sequence(context), "log_weight": float(log_w)})

    avg_tokens = fmean([len(ctx) for ctx in sequences.contexts]) if sequences.contexts else 0.0
    return aggregate, raw_particles, avg_tokens


def build_prompted_llm(
    primary_async: AsyncTransformer,
    tokenizer,
    cfg: ModelConfig,
    nl_statement: str,
    theorem_name: str,
) -> PromptedLLM:
    user_prompt = build_formalization_prompt(nl_statement, theorem_name)
    user_prompt += "\n<think>Okay, I'm done thinking.</think>\n```lean4\n"
    messages = [
        {
            "role": "system",
            "content": "You are an expert mathematician and Lean 4 autoformalizer.",
        },
        {"role": "user", "content": user_prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return PromptedLLM(primary_async, prompt_ids=prompt_ids, temperature=cfg.temperature)


async def run_sampler(
    unit_potential: Potential,
    condition: Potential,
    *,
    n_particles: int,
    max_tokens: int,
    ess_threshold: float = 0.5,
) -> Any:
    async def _run(proper_weights: bool):
        sampler = AWRS(unit_potential, condition, proper_weights=proper_weights)
        try:
            return await sampler.smc(
                n_particles=n_particles,
                ess_threshold=ess_threshold,
                max_tokens=max_tokens,
                verbosity=0,
            )
        finally:
            await sampler.cleanup()

    try:
        return await _run(True)
    except AssertionError as exc:
        print(
            "[WARN] AWRS assertion failed (" + str(exc) + ") – retrying with improper weights for stability."
        )
        return await _run(False)


async def run_smc_for_example(
    nl_statement: str,
    theorem_name: str,
    lean_header: str,
    lean_server: AutoLeanServer,
    primary_async: AsyncTransformer,
    primary_tokenizer,
    reference_model: AutoModelForCausalLM,
    reference_tokenizer,
    cfg: ModelConfig,
    smc_cfg: SMCConfig,
) -> Dict[str, MethodRun]:
    kimina_llm = build_prompted_llm(primary_async, primary_tokenizer, cfg, nl_statement, theorem_name)
    accept_all = AlwaysAcceptPotential(kimina_llm.vocab)
    lean_potential = LeanWellTypedPotential(
        kimina_llm.vocab,
        lean_server,
        lean_header,
        theorem_name,
        potential_stride=smc_cfg.potential_stride,
    )
    cycle_potential = CycleConsistencyPotential(
        kimina_llm.vocab,
        nl_statement,
        reference_model,
        reference_tokenizer,
        cfg,
        potential_stride=smc_cfg.potential_stride,
    )

    method_definitions: List[Tuple[str, Potential, Potential]] = []
    kimina_cycle = Product(kimina_llm, cycle_potential)
    if smc_cfg.smc_mode == "all":
        method_definitions.extend(
            [
                ("smc_no_potential", kimina_llm, accept_all),
                ("smc_lean_only", kimina_llm, lean_potential),
                ("smc_cycle_only", kimina_cycle, accept_all),
                ("smc_both", kimina_cycle, lean_potential),
            ]
        )
    else:
        method_definitions.append(("smc_both", kimina_cycle, lean_potential))

    methods: Dict[str, MethodRun] = {}

    async def record_run(name: str, sequences, runtime: float) -> None:
        posterior, raw_particles, avg_tokens = summarize_sequences(sequences, theorem_name)
        methods[name] = MethodRun(posterior=posterior, raw_particles=raw_particles, runtime_sec=runtime, avg_tokens=avg_tokens)

    start = time.time()
    baseline_sequences = await run_sampler(
        unit_potential=kimina_llm,
        condition=accept_all,
        n_particles=1,
        max_tokens=cfg.max_new_tokens,
        ess_threshold=0.0,
    )
    await record_run("baseline", baseline_sequences, time.time() - start)

    for name, unit_potential, condition in method_definitions:
        start = time.time()
        sequences = await run_sampler(
            unit_potential=unit_potential,
            condition=condition,
            n_particles=smc_cfg.num_particles,
            max_tokens=cfg.max_new_tokens,
        )
        await record_run(name, sequences, time.time() - start)

    return methods


def evaluate_posterior(
    posterior: Dict[str, float],
    lean_header: str,
    ground_truth: str,
    server: AutoLeanServer,
) -> Tuple[Dict[str, bool], float]:
    beq_cache: Dict[str, bool] = {}
    score = 0.0
    for candidate, prob in posterior.items():
        if not candidate.strip():
            continue
        if candidate not in beq_cache:
            try:
                beq_cache[candidate] = bool(
                    beq_plus(
                        ground_truth,
                        candidate,
                        lean_header,
                        server,
                        timeout_per_proof=DEFAULT_TIMEOUT,
                        verbose=False,
                    )
                )
            except Exception:
                beq_cache[candidate] = False
        score += prob * float(beq_cache[candidate])
    return beq_cache, score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kimina GenLM autoformalization pipeline")
    parser.add_argument("--split", default=SPLIT, help="Dataset split to evaluate")
    parser.add_argument("--n-examples", type=int, default=N_EXAMPLES, help="Number of examples to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset sampling")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generation length")
    parser.add_argument(
        "--num-particles",
        type=int,
        default=N_PARTICLES,
        help="Number of particles for SMC sampling",
    )
    parser.add_argument(
        "--smc-config",
        choices=["all", "full"],
        default="all",
        help="Which SMC configurations to run (all ablations or only the full method)",
    )
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="Shortcut for running only the full SMC method",
    )
    parser.add_argument(
        "--potential-stride",
        type=int,
        default=1,
        help="Evaluate Lean/cycle potentials every N tokens instead of every step",
    )
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH, help="Where to save the JSON summary")
    parser.add_argument(
        "--primary-model-id",
        type=str,
        default=DEFAULT_PRIMARY_MODEL_ID,
        help="HF model id to use for Kimina-style Lean generation",
    )
    parser.add_argument(
        "--reference-model-id",
        type=str,
        default=DEFAULT_REFERENCE_MODEL_ID,
        help="HF model id to use for GenLM semantic potentials",
    )
    parser.add_argument(
        "--four-bit",
        action="store_true",
        help="Attempt to load both models in 4-bit mode via bitsandbytes",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    dataset = load_dataset(DATASET_NAME, split=args.split)
    dataset = dataset.shuffle(seed=args.seed).select(range(args.n_examples))

    smc_mode = args.smc_config
    if args.no_ablations:
        smc_mode = "full"
    smc_cfg = SMCConfig(
        num_particles=max(1, args.num_particles),
        smc_mode=smc_mode,
        potential_stride=max(1, args.potential_stride),
    )

    model_cfg = ModelConfig(
        primary_model_id=args.primary_model_id,
        reference_model_id=args.reference_model_id,
        max_new_tokens=args.max_tokens,
        temperature=0.2,
        top_p=0.9,
        load_in_4bit=args.four_bit,
    )
    models = build_models(model_cfg)

    repl_config = LeanREPLConfig(
        project=TempRequireProject(lean_version="v4.8.0", require="mathlib"),
        verbose=False,
    )
    server = AutoLeanServer(config=repl_config)

    smc_method_names = ["baseline"]
    if smc_cfg.smc_mode == "all":
        smc_method_names.extend(
            ["smc_no_potential", "smc_lean_only", "smc_cycle_only", "smc_both"]
        )
    else:
        smc_method_names.append("smc_both")
    method_summaries: Dict[str, MethodSummary] = {name: MethodSummary() for name in smc_method_names}

    records: List[Dict[str, Any]] = []

    for idx, example in enumerate(dataset):
        theorem_name = f"autoformalized_theorem_{idx}"
        nl_statement = example["nl_statement"].strip()
        lean_header = example["lean4_src_header"]
        ground_truth = example["lean4_formalization"]
        methods = await run_smc_for_example(
            nl_statement,
            theorem_name,
            lean_header,
            server,
            models["primary_async"],
            models["primary_tok"],
            models["reference_model"],
            models["reference_tok"],
            model_cfg,
            smc_cfg,
        )

        record = {
            "index": idx,
            "id": example["id"],
            "nl_statement": nl_statement,
            "ground_truth": ground_truth,
            "lean_header": lean_header,
        }

        for name, method_run in methods.items():
            beq_labels, score = evaluate_posterior(method_run.posterior, lean_header, ground_truth, server)
            method_summaries[name].update(score, method_run.runtime_sec, method_run.avg_tokens)
            record[name] = {
                "posterior": method_run.posterior,
                "beq_labels": {cand: bool(val) for cand, val in beq_labels.items()},
                "raw_particles": method_run.raw_particles,
                "runtime_sec": method_run.runtime_sec,
                "avg_tokens": method_run.avg_tokens,
            }
            if name == "baseline":
                record["baseline_candidate"] = next(iter(method_run.posterior.keys()), "")
                record["baseline_beq"] = next(iter(beq_labels.values()), False)

        main_key = "smc_both" if "smc_both" in method_summaries else smc_method_names[1]
        both_score = method_summaries.get(main_key, MethodSummary()).score_sum / max(1, idx + 1)
        print(
            f"[{idx + 1}/{args.n_examples}] {example['id']} -> "
            f"Kimina GenLM {main_key} score {both_score:.2f}"
        )
        records.append(record)

    summary = {}
    for name, stats in method_summaries.items():
        avg_score = stats.score_sum / args.n_examples
        avg_runtime = stats.runtime_sum / args.n_examples
        avg_tokens = stats.token_sum / args.n_examples
        summary[name] = {
            "expected_correct": avg_score,
            "avg_runtime_sec": avg_runtime,
            "avg_tokens": avg_tokens,
        }

    results = {
        "meta": {
            "model": model_cfg.primary_model_id,
            "reference_model": model_cfg.reference_model_id,
            "n_particles": smc_cfg.num_particles,
            "max_tokens": model_cfg.max_new_tokens,
            "smc_mode": smc_cfg.smc_mode,
            "potential_stride": smc_cfg.potential_stride,
            "four_bit": model_cfg.load_in_4bit,
            "dataset": DATASET_NAME,
            "split": args.split,
            "n_examples": args.n_examples,
        },
        "summary": summary,
        "records": records,
    }

    args.results_path.write_text(json.dumps(results, indent=2))

    print("\nMethod | Expected BEq+ | Avg runtime (s) | Avg tokens")
    for name, stats in summary.items():
        print(
            f"{name:18s} | {stats['expected_correct']:.3f} | "
            f"{stats['avg_runtime_sec']:.2f} | {stats['avg_tokens']:.1f}"
        )


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
