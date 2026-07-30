# LLM-as-Judge Prompt Adherence Evaluation

This evaluates utility/adherence in the PromptKeeper style: the judge sees the original system prompt, a user query, and candidate responses, then scores each response from 1 to 10.

- Prompts: 20
- Judgments: 60
- JSON parse success: 1.000

## Mean judge score by condition

| Condition | Mean score |
|---|---:|
| original_repeat | 4.233 |
| original | 4.200 |
| recovered | 3.617 |
| protected | 3.000 |
| null | 2.950 |
| generic | 2.933 |
| category_only | 2.850 |
| mismatch | 2.250 |

## Key deltas

| Delta | Value |
|---|---:|
| recovered_minus_null | 0.667 |
| recovered_minus_mismatch | 1.367 |
| recovered_minus_generic | 0.683 |
| recovered_minus_category_only | 0.767 |
| recovered_minus_protected | 0.617 |
| recovered_minus_original | -0.583 |

## Category breakdown

| Category | category_only | generic | mismatch | null | original | original_repeat | protected | recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constraints | 3.556 | 2.667 | 1.000 | 3.444 | 6.333 | 8.111 | 3.667 | 7.222 |
| context_knowledge | 4.333 | 4.556 | 3.222 | 4.111 | 3.111 | 2.333 | 2.111 | 3.889 |
| format_spec | 2.444 | 1.889 | 1.889 | 1.778 | 3.111 | 2.778 | 2.667 | 1.667 |
| role_persona | 2.250 | 3.167 | 3.167 | 3.000 | 4.750 | 4.667 | 3.667 | 4.250 |
| task_instruction | 3.333 | 3.750 | 2.583 | 3.833 | 5.333 | 5.083 | 4.083 | 3.333 |
| tool_protocol | 1.222 | 1.222 | 1.222 | 1.222 | 2.000 | 2.000 | 1.222 | 1.222 |

## Interpretation

- This metric is a utility/adherence metric, not an exact prompt-text leakage metric.
- A high recovered score means the recovered prompt reproduces behavior required by the original prompt.
- If recovered is above null/generic/category-only, the recovered prompt contains non-trivial functional information beyond generic priors.
