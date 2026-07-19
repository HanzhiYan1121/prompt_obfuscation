# Qwen3-32B Zero-Shot Recovery Baseline

| Metric | Value |
|---|---:|
| Prompts | 20 |
| Utility F1 vs original | 0.380 |
| Utility F1 vs protected | 0.486 |
| Text F1 vs original prompt | 0.073 |
| JSON parse rate | 1.000 |

| # | ID | Category | Utility F1 | Protected F1 | Text F1 | JSON |
|---:|---|---|---:|---:|---:|---:|
| 1 | AllPrev-003 (main_prompt_template_speaker) | role_persona | 0.502 | 0.540 | 0.047 | 1 |
| 2 | AllPrev-008 (appendix_context_prompt) | role_persona | 0.482 | 0.461 | 0.049 | 1 |
| 3 | AllPrev-051 (system_prompt_partial_extrac) | role_persona | 0.383 | 0.575 | 0.043 | 1 |
| 4 | AllPrev-082 (multi_turn_clarification_tem) | role_persona | 0.367 | 0.462 | 0.083 | 1 |
| 5 | AllPrev-021 (section_p_ethics_category_pr) | task_instruction | 0.528 | 0.520 | 0.103 | 1 |
| 6 | Prompt335-014 | task_instruction | 0.298 | 0.460 | 0.053 | 1 |
| 7 | Prompt335-024 | task_instruction | 0.344 | 0.472 | 0.042 | 1 |
| 8 | Prompt335-042 | task_instruction | 0.439 | 0.382 | 0.148 | 1 |
| 9 | AllPrev-084 (tipa_prompt_example) | format_spec | 0.182 | 0.458 | 0.071 | 1 |
| 10 | AllPrev-094 (csc_prompt_3) | format_spec | 0.493 | 0.489 | 0.075 | 1 |
| 11 | Prompt335-020 | format_spec | 0.392 | 0.501 | 0.000 | 1 |
| 12 | Prompt335-219 | constraints | 0.495 | 0.538 | 0.107 | 1 |
| 13 | Prompt417-034 | constraints | 0.210 | 0.433 | 0.175 | 1 |
| 14 | Prompt417-115 | constraints | 0.060 | 0.493 | 0.057 | 1 |
| 15 | AllPrev-010 (user_prompt_for_synthetic_co) | context_knowledge | 0.300 | 0.500 | 0.111 | 1 |
| 16 | AllPrev-075 (g_1_w_o_saliency_prompt) | context_knowledge | 0.361 | 0.547 | 0.067 | 1 |
| 17 | Prompt335-057 | context_knowledge | 0.281 | 0.419 | 0.056 | 1 |
| 18 | Prompt335-017 | tool_protocol | 0.499 | 0.477 | 0.080 | 1 |
| 19 | Prompt335-106 | tool_protocol | 0.493 | 0.508 | 0.051 | 1 |
| 20 | Prompt335-148 | tool_protocol | 0.482 | 0.482 | 0.040 | 1 |
