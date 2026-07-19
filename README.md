# Prompt Obfuscation and Recovery

This repository studies the privacy--utility trade-off of protected system prompts.
It reproduces soft-prompt obfuscation, evaluates the boundary between broad natural-language behavior and exact symbolic behavior, and develops structured prompt-recovery evaluations.

## Current research claim

Protection can make exact prompt wording difficult to extract while still preserving some behaviorally useful structure. Recovery must therefore distinguish exact text, structured prompt attributes, and functional imitation.

The eight-defense study is a **mechanism-inspired common-interface benchmark**. It is not a claim that the eight original defense systems have been reproduced or broken end to end.

## Clone with third-party dependencies

The external InferDPT and EmojiCrypt implementations are pinned as Git submodules. Clone them together with the main repository:

```bash
git clone --recurse-submodules git@github.com:HanzhiYan1121/prompt_obfuscation.git
cd prompt_obfuscation
```

## Start here

- `docs/PROJECT_SUMMARY.md`: complete experiment history, established findings, limitations, and recommended next steps.
- `prompt_protection_attack_paper/AnonymousSubmission2027.tex`: current AAAI manuscript.
- `prompt_protection_attack_paper/`: manuscript references and final figures.
- `results/`: experiment outputs, model adapters, splits, evaluations, and tables.
- `precomputed_results/`: cached foundation-reproduction artifacts.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/` | Prompt collections and prepared inputs. |
| `system prompt/`, `robot system prompt/` | Source prompt collections used by the foundation experiments. |
| `src/` | Reusable implementation modules. |
| `dataset_tools/` | Prompt classification, selection, and benchmark-input preparation. |
| `recovery_tools/` | Structured recovery training, generation, active probing, and utility evaluation. |
| `analysis_tools/` | Result aggregation, ablation summaries, calibration, and error analysis. |
| `bash_scripts/` | Shell helpers for reproducible runs. |
| `results/` | Retained experiment results; `historical_figures/` stores derived figures from earlier stages. |
| `references/emoji_prism/` | Retained reading material for related protection methods. |
| `prompt_protection_attack_paper/` | Current paper submission materials. |

## Active experiment entry points

- Foundation obfuscation: `obfuscate.py`, `evaluate_obfuscation.py`, `evaluate_prompt_extraction.py`.
- Structured recovery: `recovery_tools/prepare_structured_6d_recovery_targets_v2.py`, `recovery_tools/train_acl100_recovery_lora.py`, `recovery_tools/eval_recovery_utility.py`.
- Larger-scale recovery: `run_6category200_qwen3_full_chain.sh` and `run_6category150_full_chain.sh`.
- Active recovery pilot: `recovery_tools/run_role_adaptive_probe_pilot.py` and `analysis_tools/aggregate_role_adaptive_probe_pilot.py`.
- Eight-defense benchmark preparation: `dataset_tools/prepare_privacy_defense_recovery_inputs.py` and `run_privacybench100_5defense_recovery.sh`.

## Notes

- `data/`, `results/`, and `precomputed_results/` are local research assets and are intentionally excluded from Git; distribute required artifacts separately.
- The current paper reports continuous Text F1, ROUGE-L F1, and Utility F1.
