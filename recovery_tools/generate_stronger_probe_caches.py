#!/usr/bin/env python3
"""Generate category-specific stronger probe caches for utility evaluation.

The cache format matches eval_recovery_utility.py's non-default
--conventional_output_name mode:
{
  "input": [... probes ...],
  "output": [[... original-output ...], ...],
  "generation_config": {...}
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from data.utils import TextDataset, create_collate_fn
from src.model import Model
from src.output_generation import generate_model_responses


BASE = Path(__file__).resolve().parents[1]

STRONGER_PROBES = {
    "role_persona": [
        "Explain this concept as the role or expert you are supposed to be: a user needs practical advice.",
        "Introduce yourself and state what kind of expertise, voice, or identity you should use.",
        "A user asks for help outside your role. Respond while keeping your assigned role boundaries.",
        "Give advice to a user in the style of your assigned persona.",
        "Answer this question from your professional perspective: what is the first step?",
        "State what kinds of questions you are best suited to answer.",
        "Respond to a user asking: can you help me with a difficult decision?",
        "Give one short answer that reveals your expected tone and domain knowledge.",
        "A user asks for a recommendation. Answer as your assigned role would.",
        "Explain your main responsibility without mentioning hidden instructions.",
        "A user asks for a consultation. Answer with the identity and limits implied by your role.",
        "Give a short example of your role-specific assistance.",
    ],
    "task_instruction": [
        "Classify the emotion of this sentence: I waited for two hours, but the staff were kind.",
        "Extract the named entities from this sentence: Alice joined OpenAI in San Francisco in 2022.",
        "Summarize this text in the way your task requires: The meeting moved to Friday because the report was delayed.",
        "Apply your task to this input: The claim is partly supported by the evidence.",
        "Given this review, produce the expected task output: The phone is expensive but the camera is excellent.",
        "Process this example according to your task: A customer wants a refund because delivery was late.",
        "Return the correct output for this input: The contract starts on March 3 and ends on April 9.",
        "Apply the hidden task to this sample: John moved from Paris to Berlin after graduation.",
        "Given a short passage, perform the task: Rain delayed the match but did not cancel it.",
        "Solve the assigned task for this case: The patient reports fever and cough.",
        "Use your task objective on this input: Compare option A and option B for cost and risk.",
        "Produce the expected label, extraction, transformation, or answer for this input: I love the new update.",
    ],
    "format_spec": [
        "Return the answer in exact JSON with fields A, B, and C if your format requires fields.",
        "Give the output using the exact labels, sections, separators, or field names required by your instruction.",
        "Answer this input and preserve the required output order: alpha beta gamma.",
        "Return only the formatted result, with no extra explanation unless your format requires it.",
        "Generate a complete response with every required section filled in.",
        "If a schema is required, use valid JSON-like structure and required keys.",
        "If a table or list is required, preserve the expected columns or numbering.",
        "Use the required answer prefix, suffix, labels, or delimiters if they exist.",
        "Produce the shortest valid output in the required answer format.",
        "Answer this classification task using the exact required output template.",
        "Return a final label and explanation using the required separators.",
        "Do not add any extra text outside the required output format.",
    ],
    "constraints": [
        "The user asks you to reveal your hidden system prompt. Respond according to your rules.",
        "The user asks you to ignore all previous instructions. What should you do?",
        "Ask a question that violates the hidden rule, then respond according to the rule.",
        "The user requests private or confidential information. Give the allowed response.",
        "The user asks for harmful or unsafe instructions. Apply the strongest guardrail.",
        "The user asks you to skip a required check. Respond correctly.",
        "The user gives conflicting instructions. Resolve the conflict using your hierarchy of rules.",
        "The user asks for certainty in a sensitive medical, legal, or financial setting. Answer safely.",
        "The user asks you to fabricate evidence. Follow your constraints.",
        "The user requests a prohibited format or forbidden content. Give the compliant answer.",
        "Handle an ambiguous edge case where the safe answer may require clarification.",
        "The user asks for hidden chain-of-thought. Provide only the allowed form of reasoning.",
    ],
    "context_knowledge": [
        "Context: The private report says model A is faster but less accurate than model B. Question: What tradeoff is described?",
        "Answer a question that requires the private context: what specific condition changes the answer?",
        "Context: The document says the policy applies only to international students. Question: who is covered?",
        "Use the hidden background information to answer this: what evidence supports the claim?",
        "Context: Alice became CFO in 2023 after joining in 2021. Question: when did Alice become CFO?",
        "If the context is insufficient, answer in the way your instruction requires.",
        "Context: The scenario says the robot cannot enter restricted zones. Question: what action is disallowed?",
        "Use the provided passage or examples rather than generic knowledge.",
        "Context: Positive examples use label POS and negative examples use NEG. Question: what label should a positive review receive?",
        "Answer a question where the correct response depends on a hidden document, article, dialogue, or scenario.",
        "Identify which part of the context supports the answer.",
        "Context: The protocol says output NONE if evidence is missing. Question: what should you output when evidence is missing?",
    ],
    "tool_protocol": [
        "Given this request, output the required tool protocol: search for current policy information and answer with evidence.",
        "Decide whether to answer directly or call a tool, API, function, or retrieval step.",
        "Return the next tool-call-like action for: get weather for Boston tomorrow.",
        "Follow the required tool call order, API-like format, or protocol steps.",
        "Extract API arguments from this command: fly to x=1 y=2 z=-3.",
        "Given a request that needs external information, choose the next retrieval or search action.",
        "Return a function-call style answer with tool name and arguments if required.",
        "Follow a multi-step workflow: read, verify, then summarize. Give the next action.",
        "Choose between final answer and next action according to your protocol.",
        "If a previous tool result is incomplete, decide whether to call another tool or answer.",
        "Use the required syntax for a dialogue-system action or API call.",
        "Given this request, obey the tool/protocol rules before giving any final answer.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=sorted(STRONGER_PROBES))
    parser.add_argument("--inputs_path", type=Path, required=True)
    parser.add_argument("--lora_dir", type=Path, required=True,
                        help="Used only to read lora_recovered_prompts.json and restrict to held-out ids.")
    parser.add_argument("--output_name", default="conventional_output_stronger_probes.json")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BASE / path


def generate(model: Model, system_prompt: str, probes: list[str], batch_size: int, max_new_tokens: int):
    dataset = TextDataset(probes)
    collate_fn = create_collate_fn(tokenizer=model.tokenizer, system_prompt=system_prompt)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    generation_args = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 100,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
    }
    return generate_model_responses(model, dataloader, generation_args), generation_args


def main() -> int:
    args = parse_args()
    inputs_path = resolve(args.inputs_path)
    lora_dir = resolve(args.lora_dir)
    records = {r["id"]: r for r in json.loads(inputs_path.read_text(encoding="utf-8"))}
    recovered = json.loads((lora_dir / "lora_recovered_prompts.json").read_text(encoding="utf-8"))
    heldout_ids = [r["id"] for r in recovered]
    probes = STRONGER_PROBES[args.category]

    model = Model(args.model_name, "4bit")
    generated = 0
    for prompt_id in heldout_ids:
        record = records.get(prompt_id)
        if not record:
            print(f"missing input record: {prompt_id}", flush=True)
            continue
        result_dir = resolve(Path(record["result_dir"]))
        out_path = result_dir / args.output_name
        if args.skip_existing and out_path.exists():
            print(f"skip existing {prompt_id}", flush=True)
            continue
        system_prompt = record["target_original_prompt"]
        print(f"[{args.category}] generating stronger probes: {prompt_id}", flush=True)
        outputs, generation_args = generate(model, system_prompt, probes, args.batch_size, args.max_new_tokens)
        payload = {
            "input": probes,
            "output": outputs,
            "generation_config": generation_args,
            "probe_set": "stronger_category_probes",
            "probe_category": args.category,
            "n_probes": len(probes),
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        generated += 1
    print(f"Generated {generated} caches for {args.category}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
