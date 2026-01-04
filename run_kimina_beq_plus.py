import json
import sys
from pathlib import Path

# Ensure local project modules are importable when running outside the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent / "LeanInteract"
if PROJECT_ROOT.exists():
    sys.path.append(str(PROJECT_ROOT))

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from examples.beq_plus import beq_plus, DEFAULT_TIMEOUT
from lean_interact import AutoLeanServer, LeanREPLConfig
from lean_interact.project import TempRequireProject

N_SAMPLES = 50
MODEL_NAME = "AI-MO/Kimina-Autoformalizer-7B"
SEED = 42


def build_prompt(problem: str, theorem_name: str) -> str:
    intro = (
        "Please autoformalize the following natural-language problem in Lean 4. "
        f"Return only the final theorem declaration named `{theorem_name}` in the form\\n"
        f"theorem {theorem_name} ... := by ... and avoid `import` or `open` commands because the header is provided separately.\n\n"
    )
    return intro + problem


def clean_candidate_output(text: str) -> str:
    """Strip to the final theorem block so Lean code can be embedded safely."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        # pick the last code block if multiple appear
        code_blocks = [parts[i] for i in range(1, len(parts), 2)]
        if code_blocks:
            text = code_blocks[-1].strip()
    lower = text.lower()
    idx = lower.find("theorem")
    if idx != -1:
        text = text[idx:]
    return text


def main():
    dataset = load_dataset("PAug/ProofNetVerif", split="valid")
    dataset = dataset.shuffle(seed=SEED).select(range(N_SAMPLES))

    # Initialize Kimina model once
    llm = LLM(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=2048)

    # Lean server for BEq+
    repl_config = LeanREPLConfig(project=TempRequireProject(lean_version="v4.8.0", require="mathlib"), verbose=False)
    server = AutoLeanServer(config=repl_config)

    records = []
    eq_count = 0
    for idx, example in enumerate(dataset):
        theorem_name = f"autoformalized_theorem_{idx}"
        nl_statement = example["nl_statement"].strip()
        prompt = build_prompt(nl_statement, theorem_name)
        messages = [
            {"role": "system", "content": "You are an expert mathematician and Lean 4 formalizer."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generation = llm.generate(text, sampling_params=sampling_params)
        raw_output = generation[0].outputs[0].text.strip()
        output_text = clean_candidate_output(raw_output)

        beq_result = None
        beq_error = None
        try:
            beq_bool = beq_plus(
                example["lean4_formalization"],
                output_text,
                example["lean4_src_header"],
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
                "kimina_output_raw": raw_output,
                "kimina_output": output_text,
                "ground_truth": example["lean4_formalization"],
                "kimina_equivalent": beq_result,
                "beq_error": beq_error,
            }
        )
        print(f"[{idx+1}/{N_SAMPLES}] {example['id']} -> Kimina equivalent? {beq_result}")

    out_path = Path("kimina_beq_plus_results.json")
    out_path.write_text(json.dumps({"records": records, "equivalent": eq_count, "total": N_SAMPLES}, indent=2))
    print(f"Saved results to {out_path}. {eq_count}/{N_SAMPLES} equivalents.")


if __name__ == "__main__":
    main()
