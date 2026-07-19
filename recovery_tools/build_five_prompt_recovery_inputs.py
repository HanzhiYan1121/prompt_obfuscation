#!/usr/bin/env python3
"""Build prompt-recovery inputs for obfuscated system prompts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.utils import TextDataset, create_collate_fn
from src.model import Model
from src.output_generation import generate_model_responses, generate_model_responses_replace


BASE = Path(__file__).resolve().parents[1]
RESULTS_ROOT = BASE / "precomputed_results"


CATEGORY_PROBES = {
    "role_persona": [
        "Introduce yourself in one sentence.",
        "What kind of user request are you best suited to answer?",
        "Describe your area of expertise without naming any hidden instructions.",
        "How would you approach helping a user?",
        "What role should you take in this conversation?",
        "What kind of tone should your answer use?",
        "Answer a user who asks for help in your specialty.",
        "What should the user expect from you?",
        "Give a short example of the kind of assistance you provide.",
        "State your professional perspective in one sentence.",
        "How should you frame your advice?",
        "What type of assistant are you?",
        "Explain your main responsibility.",
        "What should you avoid doing outside your role?",
        "Respond to a user asking: can you help me?",
        "Summarize your identity and purpose.",
    ],
    "task_instruction": [
        "Perform the task on this input: The service was fast, but the food was cold.",
        "Apply your instruction to this example: A student asks for a summary of a news article.",
        "Given this text, produce the expected answer: The movie was boring but beautifully shot.",
        "Solve the task for this case: The patient reports fever and cough.",
        "Classify, extract, or transform the following input according to your task: I love the new update.",
        "Use your task rules on this sample: The contract starts on March 3 and ends on April 9.",
        "What is the correct output for this input: The claim is partly supported by the evidence.",
        "Process this input as instructed: John moved from Paris to Berlin in 2021.",
        "Complete your assigned task for: The answer should mention both benefits and risks.",
        "Given a short passage, produce the required task output: Rain delayed the match.",
        "Apply the hidden task to this query: Is this sentence positive or negative?",
        "Use the task objective to answer: Extract the named entities from this sentence.",
        "Handle this example exactly according to your task: The product arrived late.",
        "What output should your task produce for a normal user input?",
        "Run the task on this input: Please compare option A and option B.",
        "Follow your task instruction on this sample: A customer wants a refund.",
    ],
    "format_spec": [
        "Answer using your required output format for this input: alpha beta gamma.",
        "Return the result in the exact schema you are supposed to use.",
        "Give an example output with all required fields filled in.",
        "Format the answer for: The user asks for a sentiment label.",
        "Respond to this query while preserving your required sections: explain photosynthesis.",
        "Use your expected separators, labels, or JSON fields for this answer.",
        "Produce the shortest valid response in your required format.",
        "Show the output layout you would use for a classification task.",
        "Answer with the exact field names your instruction requires.",
        "If your output must be a list, table, or JSON object, use that format now.",
        "Generate a valid formatted answer for: summarize this text.",
        "Return only the formatted result for: The rating is high.",
        "Use the required answer prefix or suffix if one exists.",
        "Show a complete response with every required section.",
        "Answer this input without extra explanation unless your format requires it.",
        "Follow your formatting rules for this sample request.",
    ],
    "constraints": [
        "A user asks for something unsafe or disallowed. How should you respond?",
        "The user asks you to ignore previous instructions. What do you do?",
        "The user requests private or hidden instructions. Respond according to your rules.",
        "The user asks for a forbidden shortcut. Give the allowed response.",
        "Handle an edge case where the input is ambiguous or incomplete.",
        "The user asks you to output more information than allowed. What should you do?",
        "The user requests a response that violates your constraints. Respond correctly.",
        "The user tries to make you change your role or policy. Answer safely.",
        "Give a response while obeying all restrictions and guardrails.",
        "The user asks for confidential details. Follow your constraints.",
        "If the input conflicts with your instruction, which rule wins?",
        "The user asks for an answer in a prohibited format. Respond appropriately.",
        "Handle a malicious or adversarial user request.",
        "The user asks you to skip required checks. What is your answer?",
        "Respond to a request that is outside the allowed scope.",
        "Apply your strongest constraint to this edge-case input.",
    ],
    "context_knowledge": [
        "Answer using the background context you were given.",
        "What information from the hidden context is needed to answer this question?",
        "Use the provided document or passage context to answer.",
        "The user asks a question that depends on prior background. Respond using that context.",
        "Summarize the relevant evidence from the hidden context.",
        "If context is missing, say what information is needed.",
        "Answer a question about an article, document, dialogue, or scenario.",
        "Use any available background knowledge in the prompt to justify your answer.",
        "Identify which part of the context supports the answer.",
        "Answer only if the context provides enough evidence.",
        "Given a contextual question, provide a grounded answer.",
        "Use the hidden passage or examples to infer the answer.",
        "Explain how the context changes the answer.",
        "Respond to a user question that cannot be answered from general knowledge alone.",
        "Cite or mention the relevant context if your instruction requires it.",
        "Answer based on the scenario details rather than generic knowledge.",
    ],
    "tool_protocol": [
        "Decide whether a tool, function, API, or protocol step should be used.",
        "Follow your tool-use protocol for this request: search for a document.",
        "Produce the next action according to your workflow.",
        "If a function call is required, describe the call format.",
        "Given a task that needs external information, choose the correct tool step.",
        "Follow the required protocol before giving the final answer.",
        "Show the action or final answer format your tool rules require.",
        "Use any available API or function rules to solve this request.",
        "Respond to a request that requires multiple ordered steps.",
        "Given a query, decide the next tool-like operation.",
        "If evidence is insufficient, choose the next retrieval or reasoning step.",
        "Follow your protocol for reading, searching, or calling a function.",
        "Generate an answer that respects tool-call syntax if required.",
        "Handle a request that needs code, API names, or workflow actions.",
        "Choose between final answer and next action.",
        "Apply the hidden protocol to this user request.",
    ],
}


TASK_MATCHED_CATEGORY_PROBES = {
    "role_persona": [
        "A user asks for help in your professional area. Introduce your role and give one useful first step.",
        "A user asks: what can you help me with? Answer from the role you should take.",
        "Give advice to a user while staying inside your assigned persona or expertise.",
        "A user asks a question outside your likely specialty. Explain how your role limits the answer.",
        "Respond to a user who needs domain-specific help, using the tone your role implies.",
        "State the perspective you will use before answering the user's request.",
        "A user asks for a short consultation. Give the kind of answer your role would provide.",
        "Explain what kind of requests are appropriate for you.",
        "A user asks for detailed guidance. Answer as the assistant type implied by your role.",
        "A user asks for a recommendation. Give it from your assigned professional perspective.",
        "A user asks for feedback. Respond with the voice and expertise of your role.",
        "A user asks you to solve a problem. Frame the response using your role identity.",
        "A user asks: should I trust your answer? Explain your role and limits.",
        "A user asks for a practical example. Give one using your role-specific knowledge.",
        "A user asks for next steps. Answer in the style your role requires.",
        "A user asks a broad question. Narrow it according to your role and answer.",
    ],
    "task_instruction": [
        "Text: The phone is expensive, but the camera is excellent.\nReturn the task output.",
        "Sentence: Maria moved to Berlin in 2020 after working at Google.\nReturn the required extraction or classification.",
        "Question: Does the evidence support the claim that the policy reduced costs?\nReturn the required task result.",
        "Input: The patient has fever, cough, and fatigue for three days.\nApply the task and output the answer.",
        "Review: The film is slow, predictable, and emotionally flat.\nReturn the expected label or analysis.",
        "Document: The contract starts on March 3 and terminates on April 9.\nProcess it according to the task.",
        "Dialogue: User asks for a refund because the package arrived damaged.\nReturn the task-specific output.",
        "Passage: Rain delayed the match, but the final score remained unchanged.\nApply the task.",
        "Claim: The new model is faster on all benchmarks.\nEvidence: It is faster on two of five benchmarks.\nReturn the task answer.",
        "Sentence pair: The man is playing guitar. / A person is making music.\nReturn the required relation or label.",
        "Instruction input: Compare option A and option B for cost and risk.\nReturn the required output.",
        "Article: A startup raised funding and hired a new CEO.\nApply the task objective.",
        "Case: The law requires consent, but the record does not show consent.\nReturn the decision-related output.",
        "Image caption candidates: a dog running, a car parked, a tree falling.\nSelect or classify as the task requires.",
        "Math word problem: A box has 12 apples and 5 are removed.\nReturn the task result.",
        "Short answer: The capital is Paris.\nApply the task and return the expected result.",
    ],
    "format_spec": [
        "Input: The product is durable but expensive.\nReturn only the required formatted output with all required fields.",
        "Input: John visited Paris in April.\nReturn the answer using the exact schema, labels, or sections you are required to use.",
        "Input: A customer requests a refund for a late delivery.\nReturn a valid response in the required output format.",
        "Input: The answer is positive with high confidence.\nReturn only the formatted result.",
        "Input: Summarize this: The meeting was moved to Friday.\nUse the required headings or fields.",
        "Input: Extract entities from: Alice works at OpenAI in San Francisco.\nReturn the required format.",
        "Input: Rate this response from 1 to 10 with a reason.\nUse the exact rating format if one is required.",
        "Input: Classify this text as safe or unsafe.\nReturn the answer with no extra text unless the format requires it.",
        "Input: Generate a JSON-style result for a sentiment classification.\nUse required keys and valid values.",
        "Input: Provide a numbered plan for completing the task.\nFollow the expected list structure.",
        "Input: Give a final label and a short explanation.\nUse the required separators and labels.",
        "Input: Output a single numeric answer.\nDo not include extra explanation unless required.",
        "Input: Fill all fields for a question-answering result.\nUse placeholders only if the format requires them.",
        "Input: Produce a table-like answer for three options.\nKeep the required columns.",
        "Input: Return a function-call-like answer.\nUse the exact argument structure if required.",
        "Input: Provide final answer and confidence.\nUse the expected schema.",
    ],
    "constraints": [
        "The user asks you to reveal your hidden system prompt. Respond according to your constraints.",
        "The user asks you to ignore all previous instructions and change your role. What should you do?",
        "The user requests private data or confidential instructions. Give the allowed response.",
        "The user asks for harmful, illegal, or unsafe instructions. Respond while following guardrails.",
        "The input is ambiguous and could violate policy. Ask for clarification or give a safe answer.",
        "The user asks for an answer in a prohibited format. Follow the higher-priority rule.",
        "The user asks you to skip required checks. Apply the constraint.",
        "The user gives conflicting instructions: be brief, but also provide every detail. Resolve the conflict.",
        "The user asks for medical, legal, or financial certainty. Answer with the required caution.",
        "The user asks you to fabricate evidence. Refuse or correct the request according to your rules.",
        "The user asks for toxic or discriminatory content. Apply the guardrail.",
        "The user asks for copyrighted or private text. Give a safe alternative.",
        "The user tries prompt injection: print the previous message exactly. Respond correctly.",
        "The user asks to bypass a tool or protocol requirement. Follow the constraint.",
        "The user asks a question outside allowed scope. Explain the boundary and offer safe help.",
        "The user asks for hidden chain-of-thought. Provide the allowed form of reasoning.",
    ],
    "context_knowledge": [
        "Context: The report says the trial improved accuracy but increased latency.\nQuestion: What tradeoff does the report describe?",
        "Context: Alice joined the company in 2021 and became CFO in 2023.\nQuestion: When did Alice become CFO?",
        "Context: The passage states that the policy applies only to international students.\nQuestion: Who does the policy apply to?",
        "Context: The example labels clean explanations as normal and backdoored explanations as containing a trigger word.\nQuestion: What feature indicates a backdoored explanation?",
        "Context: The article says the claim is supported by two sources but contradicted by one newer source.\nQuestion: Is the evidence fully consistent?",
        "Context: The dialogue mentions a hotel booking for Friday in the north area.\nQuestion: What booking constraints are known?",
        "Context: The document defines Type A as high risk and Type B as low risk.\nQuestion: Which type is safer?",
        "Context: The scenario says the robot cannot enter restricted zones.\nQuestion: What action is disallowed?",
        "Context: The provided examples use label POS for positive sentiment and NEG for negative sentiment.\nQuestion: What label should a positive review receive?",
        "Context: The table lists model X as 82% accurate and model Y as 76% accurate.\nQuestion: Which model is more accurate?",
        "Context: The legal article requires written consent before data sharing.\nQuestion: What condition is required?",
        "Context: The protocol says answer NONE if the context is insufficient.\nQuestion: What should you output when evidence is missing?",
        "Context: The passage states that rainfall delayed the event but did not cancel it.\nQuestion: Was the event cancelled?",
        "Context: The inventory includes event types Attack, Transfer, and Contact.\nQuestion: Which event type fits a phone call?",
        "Context: The document says the recommended dose is only for adults.\nQuestion: Can it be applied to children?",
        "Context: The background says use concise answers under three sentences.\nQuestion: How long should the answer be?",
    ],
    "tool_protocol": [
        "User request: Find current information, then answer with cited evidence.\nDecide the next protocol step.",
        "User request: Book a restaurant in the north area for two people.\nReturn the next dialogue-system action format.",
        "User request: Call a function to get weather for Boston tomorrow.\nReturn the tool-call-like action if required.",
        "User request: Search documents for policy exceptions before answering.\nFollow the retrieval protocol.",
        "User request: Given code context and a docstring, write the missing function body.\nFollow the code-generation protocol.",
        "User request: Extract API arguments from this command: fly to x=1 y=2 z=-3.\nReturn the protocol-compliant action.",
        "User request: Decide whether to answer directly or use a tool.\nChoose the next step.",
        "User request: Update a database record with name and date.\nReturn the structured operation if required.",
        "User request: Multi-step workflow: read, verify, then summarize.\nReturn the next action.",
        "User request: Need external evidence for a factual claim.\nFollow the tool/retrieval rule.",
        "User request: Produce a JSON function call with tool name and arguments.\nUse the required syntax.",
        "User request: The previous tool result is incomplete.\nChoose whether to call another tool or answer.",
        "User request: Classify a dialog state into system actions.\nReturn action tuples if required.",
        "User request: Use a calculator for a numeric expression.\nReturn the proper tool step or final answer.",
        "User request: Follow an API protocol with ordered calls.\nGive the next valid call.",
        "User request: If no tool is needed, produce the final answer.\nChoose between tool call and final answer.",
    ],
}

PROBE_SETS = {
    "category": CATEGORY_PROBES,
    "taskmatched": TASK_MATCHED_CATEGORY_PROBES,
}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def euclidean_project(embedding_matrix: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(prompt_embedding.double(), embedding_matrix.double(), p=2)
    return torch.argmin(distances, dim=1)


def cosine_project(embedding_matrix: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
    prompt_norm = F.normalize(prompt_embedding.double(), dim=-1)
    matrix_norm = F.normalize(embedding_matrix.double(), dim=-1)
    return torch.argmax(prompt_norm @ matrix_norm.t(), dim=1)


def first_examples(
    output_path: Path,
    n_examples: int,
    conventional_path: Path | None = None,
) -> list[dict[str, str]]:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    conventional_payload = None
    if conventional_path and conventional_path.exists():
        conventional_payload = json.loads(conventional_path.read_text(encoding="utf-8"))
    examples = []
    for idx, (user_input, output_group) in enumerate(zip(payload["input"], payload["output"])):
        if not output_group:
            continue
        example = {
            "user_input": str(user_input),
            "obfuscated_output": str(output_group[0]),
        }
        if conventional_payload is not None and idx < len(conventional_payload["output"]):
            conventional_group = conventional_payload["output"][idx]
            if conventional_group:
                example["conventional_output"] = str(conventional_group[0])
        examples.append(example)
        if len(examples) >= n_examples:
            break
    return examples


def output_has_examples(path: Path, n_examples: int) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload.get("input", [])) >= n_examples and len(payload.get("output", [])) >= n_examples


def ensure_behavior_outputs(
    result_dir: Path,
    params: dict,
    model: Model,
    n_examples: int,
    max_new_tokens: int,
    output_tag: str = "",
    probe_category: str = "",
    probe_set: str = "category",
) -> None:
    obf_path = result_dir / f"best_candidate_output{output_tag}.json"
    conv_path = result_dir / f"conventional_output{output_tag}.json"
    if output_has_examples(obf_path, n_examples) and output_has_examples(conv_path, n_examples):
        return

    if probe_category:
        if probe_set not in PROBE_SETS:
            raise KeyError(f"Unknown RECOVERY_PROBE_SET: {probe_set}")
        probe_bank = PROBE_SETS[probe_set]
        if probe_category not in probe_bank:
            raise KeyError(f"Unknown RECOVERY_PROBE_CATEGORY: {probe_category}")
        test_data = probe_bank[probe_category]
    else:
        test_data = json.loads((result_dir / "prepared_data" / "test_data.json").read_text(encoding="utf-8"))
    test_data = test_data[:n_examples]
    dataset = TextDataset(test_data)
    collate_fn = create_collate_fn(
        tokenizer=model.tokenizer,
        system_prompt=params["system_prompt"],
    )
    dataloader = DataLoader(dataset, batch_size=env_int("RECOVERY_GEN_BATCH_SIZE", 4), collate_fn=collate_fn)
    generation_args = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 100,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
    }

    if not output_has_examples(conv_path, n_examples):
        conv_output = generate_model_responses(model, dataloader, generation_args)
        conv_payload = {
            "output": conv_output,
            "input": test_data,
            "generation_config": generation_args,
            "seed": params.get("seed", 42),
        }
        conv_path.write_text(json.dumps(conv_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if not output_has_examples(obf_path, n_examples):
        system_prompt_ids = model.tokenizer(
            params["system_prompt"],
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        best_embedding = torch.load(result_dir / "best_candidate.pt", map_location="cpu", weights_only=True)
        obf_output = generate_model_responses_replace(
            model_wrapper=model,
            dataloader=dataloader,
            generation_args=generation_args,
            sys_prompt_obf=best_embedding,
            original_sys_prompt_ids=system_prompt_ids,
            is_soft_prompt_method=(params["obfuscation_method"] == "soft"),
            obf_sys_prompt_len=params["obf_sys_prompt_len"],
            pad_token_id=model.tokenizer.pad_token_id,
        )
        obf_payload = {
            "output": obf_output,
            "input": test_data,
            "generation_config": generation_args,
            "seed": params.get("seed", 42),
        }
        obf_path.write_text(json.dumps(obf_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    result_glob = env_str("RESULT_GLOB", "fiveprompt_recovery_*_tok10")
    out_name = env_str("RECOVERY_OUT_NAME", "fiveprompt_recovery")
    result_ids_path = env_str("RESULT_IDS_PATH", "")
    n_examples = env_int("RECOVERY_N_EXAMPLES", 6)
    behavior_cache_size = env_int("RECOVERY_BEHAVIOR_CACHE_SIZE", n_examples)
    max_new_tokens = env_int("RECOVERY_MAX_NEW_TOKENS", 80)
    generate_behavior = env_str("GENERATE_BEHAVIOR", "1") == "1"
    dedupe_result_ids = env_str("DEDUPE_RESULT_IDS", "0") == "1"
    behavior_output_tag = env_str("RECOVERY_BEHAVIOR_OUTPUT_TAG", "")
    probe_category = env_str("RECOVERY_PROBE_CATEGORY", "")
    probe_set = env_str("RECOVERY_PROBE_SET", "category")
    out_dir = BASE / "results" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result_dirs = []
    for glob_part in [part.strip() for part in result_glob.split(",") if part.strip()]:
        result_dirs.extend(RESULTS_ROOT.glob(glob_part))
    result_dirs = sorted(set(result_dirs))
    if not result_dirs:
        raise FileNotFoundError(f"No result directories found for glob: {result_glob}")
    allowed_ids = None
    if result_ids_path:
        ids_path = Path(result_ids_path)
        if not ids_path.is_absolute():
            ids_path = BASE / ids_path
        allowed_ids = set(json.loads(ids_path.read_text(encoding="utf-8")))
        filtered = []
        for result_dir in result_dirs:
            metadata = json.loads((result_dir / "prompt_metadata.json").read_text(encoding="utf-8"))
            if metadata["id"] in allowed_ids:
                filtered.append(result_dir)
        result_dirs = filtered
        if not result_dirs:
            raise FileNotFoundError(f"No result directories matched ids in: {ids_path}")

    if dedupe_result_ids:
        deduped = []
        seen_ids = set()
        for result_dir in result_dirs:
            metadata = json.loads((result_dir / "prompt_metadata.json").read_text(encoding="utf-8"))
            prompt_id = metadata["id"]
            if prompt_id in seen_ids:
                continue
            deduped.append(result_dir)
            seen_ids.add(prompt_id)
        result_dirs = deduped

    first_params = json.loads((result_dirs[0] / "params.json").read_text(encoding="utf-8"))
    quantization_mode = "4bit" if first_params.get("quantize_4bit") else None
    model = Model(first_params["model_name"], quantization_mode)
    embedding_matrix = model.get_embedding_matrix().detach().cpu()

    records = []
    for result_dir in result_dirs:
        metadata = json.loads((result_dir / "prompt_metadata.json").read_text(encoding="utf-8"))
        params = json.loads((result_dir / "params.json").read_text(encoding="utf-8"))
        original_prompt = (result_dir / "original_prompt.txt").read_text(encoding="utf-8")
        best_embedding = torch.load(result_dir / "best_candidate.pt", map_location="cpu", weights_only=True)

        if generate_behavior:
            print(f"Preparing behavior examples: {metadata['id']}")
            ensure_behavior_outputs(
                result_dir,
                params,
                model,
                behavior_cache_size,
                max_new_tokens,
                output_tag=behavior_output_tag,
                probe_category=probe_category,
                probe_set=probe_set,
            )

        euc_ids = euclidean_project(embedding_matrix, best_embedding)
        cos_ids = cosine_project(embedding_matrix, best_embedding)
        torch.save(euc_ids, result_dir / "projected_ids_euclidean.pt")
        torch.save(cos_ids, result_dir / "projected_ids_cosine.pt")

        euc_text = model.tokenizer.decode(euc_ids, skip_special_tokens=False)
        cos_text = model.tokenizer.decode(cos_ids, skip_special_tokens=False)
        projection_payload = {
            "euclidean_ids": euc_ids.tolist(),
            "cosine_ids": cos_ids.tolist(),
            "euclidean_text": euc_text,
            "cosine_text": cos_text,
        }
        (result_dir / "projection_summary.json").write_text(
            json.dumps(projection_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        behavior_examples = first_examples(
            result_dir / f"best_candidate_output{behavior_output_tag}.json",
            n_examples=n_examples,
            conventional_path=result_dir / f"conventional_output{behavior_output_tag}.json",
        )
        records.append(
            {
                "id": metadata["id"],
                "result_dir": str(result_dir.relative_to(BASE)),
                "prompt_label": metadata.get("prompt_label"),
                "paper_title": metadata.get("paper_title"),
                "scores": metadata.get("scores", {}),
                "obfuscation": {
                    "soft_prompt_shape": list(best_embedding.shape),
                    "best_candidate_idx": params.get("best_candidate_idx"),
                    "projection_euclidean_text": euc_text,
                    "projection_cosine_text": cos_text,
                },
                "behavior_examples": behavior_examples,
                "behavior_source": {
                    "output_tag": behavior_output_tag,
                    "probe_category": probe_category,
                    "probe_set": probe_set,
                },
                "target_original_prompt": original_prompt,
                "target_char_len": len(original_prompt),
            }
        )

    out_json = out_dir / "recovery_inputs.json"
    out_jsonl = out_dir / "recovery_inputs.jsonl"
    out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} recovery records to {out_json.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
