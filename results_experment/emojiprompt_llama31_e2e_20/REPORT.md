# EmojiPrompt Llama-3.1 Strict Black-Box Report

## Setting

- Defense: EmojiPrompt-style open-source adaptation for hidden system prompts.
- Obfuscator model: meta-llama/Meta-Llama-3.1-8B-Instruct
- Victim model: meta-llama/Meta-Llama-3.1-8B-Instruct
- Recovery model: Qwen/Qwen3-32B with six frozen LoRA adapters from sixcat200.
- Recovery input: user input + protected model output only.
- Original prompts, labels, categories, and obfuscated prompts are used only for evaluation/reporting.
- Prompts evaluated: 20 / 20
- Evidence / selection / eval queries: 4 / 2 / 10
- Semantic metric: bertscore; rho=0.15; semantic epsilon=10.0
- Post-sampling: True with epsilon=10.0

## Isolation Audit

- Forbidden term hit in recovery prompts: False
- Evidence-selection overlap: 0
- Evidence-eval overlap: 0
- Selection-eval overlap: 0

## Main Results

| Metric | Value |
|---|---:|
| Text F1: recovered prompt vs original prompt | 0.097 |
| Functional F1: recovered output vs original output | 0.340 |
| Functional F1: recovered output vs protected output | 0.415 |
| Defense utility: protected output vs original output | 0.369 |
| Null baseline: null output vs original output | 0.377 |
| Mismatch baseline: wrong prompt output vs original output | 0.339 |
| Normalized recovery gain vs mismatch | 0.017 |
| JSON parse rate | 0.750 |

## By Category

| Category | N | Functional F1 vs original | Functional F1 vs protected | Text F1 |
|---|---:|---:|---:|---:|
| role_persona | 4 | 0.430 | 0.499 | 0.086 |
| task_instruction | 4 | 0.367 | 0.402 | 0.096 |
| format_spec | 3 | 0.332 | 0.441 | 0.032 |
| constraints | 3 | 0.253 | 0.490 | 0.210 |
| context_knowledge | 3 | 0.274 | 0.284 | 0.122 |
| tool_protocol | 3 | 0.343 | 0.351 | 0.042 |
