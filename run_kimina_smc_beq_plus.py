import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent / "LeanInteract"
if PROJECT_ROOT.exists():
    sys.path.append(str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from genlm.control import PromptedLLM, AWRS
from genlm.control.potential import Potential
from transformers import AutoTokenizer, AutoModelForCausalLM

from lean_interact import AutoLeanServer, LeanREPLConfig
from lean_interact.project import TempRequireProject
from lean_interact.utils import clean_last_theorem_string
from examples.beq_plus import beq_plus, DEFAULT_TIMEOUT

N_SAMPLES = 50
MODEL_NAME = "AI-MO/Kimina-Autoformalizer-7B"
BACKWARD_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEED = 42
N_PARTICLES = 3
MAX_TOKENS = 256

back_tokenizer = AutoTokenizer.from_pretrained(BACKWARD_MODEL)
back_model = AutoModelForCausalLM.from_pretrained(
    BACKWARD_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
back_model.eval()

_kimina_llm: PromptedLLM | None = None


def clean_candidate_output(text: str) -> str:
    """Strip to the final theorem block so Lean code can be embedded safely."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        code_blocks = [parts[i] for i in range(1, len(parts), 2)]
        if code_blocks:
            text = code_blocks[-1].strip()
    lower = text.lower()
    idx = lower.find("theorem")
    if idx != -1:
        text = text[idx:]
    return text


def build_prompt(nl_statement: str, theorem_name: str) -> tuple[str, list[dict]]:
    intro = (
        "Please autoformalize the following natural language problem in Lean 4.\n"
        f"Return only the final theorem declaration named `{theorem_name}` in the form\n"
        f"theorem {theorem_name} ... := by ...\n"
        "Avoid `import` or `open` commands because the header is provided separately.\n\n"
    )
    prompt = intro + nl_statement.strip()
    messages = [
        {"role": "system", "content": "You are an expert mathematician and Lean 4 formalizer."},
        {"role": "user", "content": prompt},
    ]
    return prompt, messages


def make_kimina_llm(messages: list[dict]) -> PromptedLLM:
    global _kimina_llm  # noqa: PLW0603 - cache for reuse across dataset loop
    if _kimina_llm is None:
        _kimina_llm = PromptedLLM.from_name(
            MODEL_NAME,
            temperature=0.2,
        )

    tokenizer = _kimina_llm.model.tokenizer
    _kimina_llm.prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return _kimina_llm


class LeanWellTypedPotential(Potential):
    def __init__(self, vocabulary, server: AutoLeanServer, src_header: str, timeout: int = DEFAULT_TIMEOUT):
        super().__init__(vocabulary)
        self.server = server
        self.src_header = src_header
        self.timeout = timeout

    async def prefix(self, context):
        return 0.0

    async def complete(self, context):
        text = b"".join(context).decode("utf8", errors="ignore")
        lower = text.lower()
        idx = lower.rfind("theorem")
        if idx == -1:
            return float("-inf")
        candidate = text[idx:].strip()

        base_thm_name = "base_theorem"
        try:
            formal_1 = (
                self.src_header
                + "\n\n"
                + clean_last_theorem_string(candidate, base_thm_name, add_sorry=True)
                + "\n\n"
            )
        except ValueError:
            return float("-inf")

        from examples.beq_plus import check_proof_sub

        formal_2_start_line = formal_1.count("\n") + 1
        formal_2_code = clean_last_theorem_string(candidate, base_thm_name, add_sorry=False) + " := by"
        formal_code = formal_1 + formal_2_code

        try:
            res = check_proof_sub(
                self.server,
                formal_code,
                formal_2_start_line,
                "sorry",
                self.timeout,
            )
        except Exception:
            return float("-inf")

        if res is None:
            return float("-inf")
        return 0.0


class CycleConsistencyPotential(Potential):
    def __init__(self, vocabulary, nl_statement: str):
        super().__init__(vocabulary)
        self.nl_statement = nl_statement

    async def prefix(self, context):
        return 0.0

    async def complete(self, context):
        text = b"".join(context).decode("utf8", errors="ignore")
        lower = text.lower()
        idx = lower.rfind("theorem")
        if idx == -1:
            return float("-inf")
        lean_code = text[idx:].strip()

        prompt = (
            "You are an assistant that explains Lean 4 theorem statements in natural language.\n\n"
            "Lean theorem:\n```lean\n"
            + lean_code
            + "\n```\n\n"
            "Restate this as a single English sentence, with all hypotheses and conclusions.\n"
            "Output only the sentence, with no extra text.\n"
        )

        inputs = back_tokenizer(prompt, return_tensors="pt").to(back_model.device)
        target_ids = back_tokenizer(
            self.nl_statement,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(back_model.device)

        with torch.no_grad():
            full_ids = torch.cat([inputs.input_ids, target_ids], dim=1)
            attn_mask = torch.ones_like(full_ids)
            outputs = back_model(full_ids, attention_mask=attn_mask)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

        T = target_ids.size(1)
        start = inputs.input_ids.size(1)
        idxs = torch.arange(T, device=target_ids.device)
        tok_log_probs = log_probs[0, start - 1 + idxs, target_ids[0]]
        logp = float(tok_log_probs.sum().cpu())

        return logp / float(T)


async def smc_autoformalize(
    server: AutoLeanServer,
    nl_statement: str,
    src_header: str,
    theorem_name: str,
    n_particles: int = N_PARTICLES,
) -> dict[str, float]:
    _, messages = build_prompt(nl_statement, theorem_name)
    llm = make_kimina_llm(messages)

    lean_potential = LeanWellTypedPotential(llm.vocab, server, src_header)
    cycle_potential = CycleConsistencyPotential(llm.vocab, nl_statement)

    constraint = lean_potential * cycle_potential

    token_sampler = AWRS(llm, constraint)

    sequences = await token_sampler.smc(
        n_particles=n_particles,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=0,
    )

    return sequences.decoded_posterior


def main():
    dataset = load_dataset("PAug/ProofNetVerif", split="valid")
    dataset = dataset.shuffle(seed=SEED).select(range(N_SAMPLES))

    repl_config = LeanREPLConfig(
        project=TempRequireProject(lean_version="v4.8.0", require="mathlib"),
        verbose=False,
    )
    server = AutoLeanServer(config=repl_config)

    records = []
    eq_count = 0

    for idx, example in enumerate(dataset):
        theorem_name = f"autoformalized_theorem_{idx}"
        nl_statement = example["nl_statement"].strip()
        src_header = example["lean4_src_header"]

        posterior = asyncio.run(
            smc_autoformalize(
                server=server,
                nl_statement=nl_statement,
                src_header=src_header,
                theorem_name=theorem_name,
            )
        )

        if not posterior:
            best_lean = ""
        else:
            best_lean = max(posterior.items(), key=lambda kv: kv[1])[0]

        best_lean_clean = clean_candidate_output(best_lean)

        beq_result = None
        beq_error = None

        try:
            beq_bool = beq_plus(
                example["lean4_formalization"],
                best_lean_clean,
                src_header,
                server,
                timeout_per_proof=DEFAULT_TIMEOUT,
                verbose=False,
            )
            beq_result = bool(beq_bool)
            if beq_result:
                eq_count += 1
        except Exception as exc:  # pylint: disable=broad-except
            beq_error = str(exc)

        records.append(
            {
                "index": idx,
                "id": example["id"],
                "nl_statement": nl_statement,
                "smc_posterior": posterior,
                "kimina_smc_best": best_lean,
                "kimina_smc_best_clean": best_lean_clean,
                "ground_truth": example["lean4_formalization"],
                "kimina_equivalent": beq_result,
                "beq_error": beq_error,
            }
        )
        print(f"[{idx + 1}/{N_SAMPLES}] {example['id']} -> Kimina SMC equivalent? {beq_result}")

    out_path = Path("kimina_smc_beq_plus_results.json")
    out_path.write_text(
        json.dumps(
            {"records": records, "equivalent": eq_count, "total": N_SAMPLES},
            indent=2,
        )
    )
    print(f"Saved SMC results to {out_path}")
    print(f"{eq_count}/{N_SAMPLES} equivalents.")


if __name__ == "__main__":
    main()
