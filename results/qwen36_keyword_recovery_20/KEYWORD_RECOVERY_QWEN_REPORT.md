# Qwen-assisted Keyword Recovery Evaluation

This evaluates keyword-level recoverability between the original hidden prompt and the selected recovered prompt.

- Prompts: 20
- JSON parse success: 1.000
- Keyword Precision: 0.018
- Keyword Recall: 0.015
- Keyword F1: 0.017

## Category breakdown

| Category | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| constraints | 3 | 0.048 | 0.048 | 0.048 |
| context_knowledge | 3 | 0.074 | 0.056 | 0.063 |
| format_spec | 3 | 0.000 | 0.000 | 0.000 |
| role_persona | 4 | 0.000 | 0.000 | 0.000 |
| task_instruction | 4 | 0.000 | 0.000 | 0.000 |
| tool_protocol | 3 | 0.000 | 0.000 | 0.000 |

## Per-sample scores

| ID | Category | Precision | Recall | F1 | Original K | Recovered K | Matched |
|---|---|---:|---:|---:|---:|---:|---:|
| AllPrev-003 (main_prompt_template_speaker) | role_persona | 0.000 | 0.000 | 0.000 | 2 | 7 | 0 |
| AllPrev-008 (appendix_context_prompt) | role_persona | 0.000 | 0.000 | 0.000 | 1 | 6 | 0 |
| AllPrev-051 (system_prompt_partial_extrac) | role_persona | 0.000 | 0.000 | 0.000 | 10 | 5 | 0 |
| AllPrev-082 (multi_turn_clarification_tem) | role_persona | 0.000 | 0.000 | 0.000 | 7 | 8 | 0 |
| AllPrev-021 (section_p_ethics_category_pr) | task_instruction | 0.000 | 0.000 | 0.000 | 8 | 12 | 0 |
| Prompt335-014 | task_instruction | 0.000 | 0.000 | 0.000 | 14 | 12 | 0 |
| Prompt335-024 | task_instruction | 0.000 | 0.000 | 0.000 | 13 | 13 | 0 |
| Prompt335-042 | task_instruction | 0.000 | 0.000 | 0.000 | 14 | 8 | 0 |
| AllPrev-084 (tipa_prompt_example) | format_spec | 0.000 | 0.000 | 0.000 | 8 | 9 | 0 |
| AllPrev-094 (csc_prompt_3) | format_spec | 0.000 | 0.000 | 0.000 | 9 | 5 | 0 |
| Prompt335-020 | format_spec | 0.000 | 0.000 | 0.000 | 2 | 13 | 0 |
| Prompt335-219 | constraints | 0.143 | 0.143 | 0.143 | 7 | 7 | 1 |
| Prompt417-034 | constraints | 0.000 | 0.000 | 0.000 | 13 | 10 | 0 |
| Prompt417-115 | constraints | 0.000 | 0.000 | 0.000 | 18 | 9 | 0 |
| AllPrev-010 (user_prompt_for_synthetic_co) | context_knowledge | 0.222 | 0.167 | 0.190 | 12 | 9 | 2 |
| AllPrev-075 (g_1_w_o_saliency_prompt) | context_knowledge | 0.000 | 0.000 | 0.000 | 19 | 10 | 0 |
| Prompt335-057 | context_knowledge | 0.000 | 0.000 | 0.000 | 13 | 8 | 0 |
| Prompt335-017 | tool_protocol | 0.000 | 0.000 | 0.000 | 15 | 8 | 0 |
| Prompt335-106 | tool_protocol | 0.000 | 0.000 | 0.000 | 14 | 7 | 0 |
| Prompt335-148 | tool_protocol | 0.000 | 0.000 | 0.000 | 6 | 5 | 0 |
