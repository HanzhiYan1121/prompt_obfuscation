# Research Context — Soft Prompt Obfuscation

## Paper
**"Soft Prompt Obfuscation"** — obfuscates a readable system prompt into a continuous 10-token embedding that preserves model behavior but is unreadable.

Working directory: `/scratch/Hanzhi/paper reproduce/prompt_obfuscation/prompt_obfuscation/`

---

## What We've Done

### 1. Reproduced the paper's 8 styles (soft obfuscation)
All 8 styles were obfuscated and evaluated. Results in `precomputed_results/soft_*_truthfulqa_full/`.

| Style | Status |
|-------|--------|
| pirate | ✅ |
| robot | ✅ |
| manga_miko | ✅ (case_study dir) |
| friendly | ✅ |
| poem | ✅ |
| southern_usa | ✅ |
| stutter | ✅ |
| david_attenborough | ✅ |

### 2. Ran Attack 3: Soft Fluency Deobfuscation on all 8 styles
Results: **ALL 8 FAILED** — no keyword recovered, all scores worse than random baseline.

Summary table saved at: `results/soft_deobf_summary.csv` / `.json`

Key finding: soft deobfuscation cosine_similarity (0.16–0.28) is consistently BELOW random baseline (0.25–0.32).

### 3. Token Projection Analysis
For each style's best_candidate.pt, projected each embedding position to nearest token.
- **pirate** → "pirate" appears in projection ✅
- **poem** → "poem" appears in projection ✅
- **manga_miko** → "anime", "-chan" appear in hard deobf ✅
- robot, friendly, southern_usa, stutter, david_attenborough → ❌

Figure saved at: `figures/keyword_recovery_summary.png`

---

## Current Attack Angle: Breaking the Utility Claim

The paper claims obfuscated prompts preserve output utility.
We hypothesize: **complex prompts cannot be compressed into 10 tokens → utility breaks down**.

### Experiment Design
3 complexity levels vs paper's simple prompts:

| Name | Prompt tokens | Compression ratio (10/N) |
|------|--------------|--------------------------|
| paper baseline (pirate) | ~33 | 0.30 |
| complex_L1 | 79 | 0.127 |
| complex_L2 | 109 | 0.092 |
| complex_L3 | 166 | 0.060 |

**Prompts used:**

- **L1**: "Always structure your response in exactly three parts: 1) A one-sentence direct answer. 2) A brief explanation in no more than two sentences. 3) One concrete example. Never use bullet points. Never start a sentence with 'I'."

- **L2**: "Adapt your response style based on the topic: If the question is about science or technology, respond like a university professor using precise terminology. If the question is about history or culture, respond like a storyteller with vivid language. If the question is about food or daily life, respond like a friendly neighbor. For all other topics, be concise and factual. Always end your response with a one-sentence reflection."

- **L3**: "You are a highly structured assistant with the following strict rules: Rule 1: Begin every response with 'Analysis:' followed by a one-sentence summary. Rule 2: List exactly three key points, each starting with a dash and a number (e.g., -1., -2., -3.). Rule 3: After the key points, add a section labeled 'Confidence:' rated Low, Medium, or High, with a one-sentence justification. Rule 4: Never use passive voice. Rule 5: Never mention your own limitations or say you cannot answer. Rule 6: Keep total response under 120 words. Violation of any rule is not acceptable."

Results saved at: `results/complexity_vs_utility.csv` / `.json` (COMPLETE — all 8 baselines + 3 complex)

Full results:
| Name               | Type     | Tokens | Ratio  | Cosine | BERTScore | ROUGE-1 | SacreBLEU |
|--------------------|----------|--------|--------|--------|-----------|---------|-----------|
| pirate             | baseline | 33     | 0.303  | 0.8476 | 0.9122    | 0.5846  | 49.20     |
| robot              | baseline | 35     | 0.286  | 0.8371 | 0.9302    | 0.6181  | 49.23     |
| poem               | baseline | 32     | 0.313  | 0.8259 | 0.8992    | 0.5367  | 35.77     |
| stutter            | baseline | 32     | 0.313  | 0.7764 | 0.9005    | 0.5647  | 25.82     |
| friendly           | baseline | 45     | 0.222  | 0.8550 | 0.9175    | 0.5923  | 48.90     |
| southern_usa       | baseline | 34     | 0.294  | 0.8402 | 0.9047    | 0.5494  | 43.47     |
| david_attenborough | baseline | 41     | 0.244  | 0.8072 | 0.9120    | 0.5830  | 47.33     |
| manga_miko         | baseline | 87     | 0.115  | 0.8234 | 0.9050    | 0.5548  | 47.62     |
| complex_L1         | complex  | 79     | 0.127  | 0.8537 | 0.9111    | 0.5646  | 42.51     |
| complex_L2         | complex  | 109    | 0.092  | 0.8151 | 0.9037    | 0.5301  | 39.43     |
| complex_L3         | complex  | 166    | 0.060  | 0.8602 | 0.9002    | 0.5351  | 37.26     |

Key findings from complexity experiment:
- **SacreBLEU**: clearest monotonic decline for complex (42.5 → 39.4 → 37.3)
- **ROUGE-1**: r=-0.53, p=0.079 across all points (borderline significant)
- **Cosine Similarity**: L3 rebounds (0.86) — does NOT support the compression hypothesis
- **Conclusion**: Partial support. Lower-level lexical metrics (SacreBLEU, ROUGE-1) degrade with complexity; semantic similarity (cosine) does not.

Figure saved at: `figures/complexity_vs_utility.png`

---

## Experiment: Long Context Degradation ✅ COMPLETE

**Script**: `eval_long_context.py`
**Results**: `results/long_context_degradation.json`
**Setup**: `soft_pirate_truthfulqa_full`, n_samples=40, GPU 5

**Hypothesis**: As user messages grow longer, the 10-token soft prompt's relative influence shrinks → obfuscated outputs diverge from conventional.

| ctx_tokens | Cosine | BERTScore | ROUGE-1 | SacreBLEU |
|------------|--------|-----------|---------|-----------|
| 0          | 0.8550 | 0.9142    | 0.5932  | 50.58     |
| 128        | 0.8685 | 0.9160    | 0.5978  | 51.82     |
| 256        | 0.8675 | 0.9146    | 0.5955  | 50.92     |
| 512        | 0.8743 | 0.9182    | 0.6048  | 52.89     |
| 1024       | 0.8777 | 0.9164    | 0.5981  | 52.75     |

**Conclusion**: Hypothesis **NOT supported**. All metrics stable or slightly increasing with context length. The soft prompt is robust to long user contexts — the 10-token embedding at position 0 is unaffected by neutral filler appended to the user turn.

---

## Experiment: Multi-Turn Degradation ✅ COMPLETE

**Script**: `eval_multi_turn.py`
**Results**: `results/multi_turn_degradation.json`
**Setup**: `soft_pirate_truthfulqa_full`, n_triplets=30, GPU 3

**Hypothesis**: Soft prompt optimized for single-turn Q&A; its influence weakens as conversation history grows.

| Turn | Cosine | BERTScore | ROUGE-1 | SacreBLEU |
|------|--------|-----------|---------|-----------|
| 1    | 0.8681 | 0.9115    | 0.5773  | 49.37     |
| 2    | 0.8627 | 0.9184    | 0.6220  | 53.88     |
| 3    | 0.8367 | 0.9179    | 0.6141  | 51.99     |

**Conclusion**: Hypothesis **weakly supported at best**. Cosine shows modest decline (0.868→0.837, -3.6%), but BERTScore is flat and SacreBLEU/ROUGE-1 actually peak at Turn 2 before dropping. No clear monotonic degradation across metrics.

---

## Pending / Next Steps

1. Possible additional attacks:
   - Test with `obf_sys_prompt_len=20` (longer embedding) to see if utility recovers at high complexity
   - Output-behavior-based keyword recovery (use model outputs to infer style)
   - Focus on SacreBLEU/ROUGE as the more sensitive utility metrics

---

## Key Files
```
precomputed_results/
  soft_pirate_truthfulqa_full/      # paper baseline
  soft_robot_truthfulqa_full/
  ... (all 8 styles)
  complex_L1/                        # our complexity attack
  complex_L2/
  complex_L3/

results/
  soft_deobf_summary.csv            # attack3 results for 8 styles
  complexity_vs_utility.csv         # complexity vs utility experiment
  long_context_degradation.json     # long context experiment (pending)
  multi_turn_degradation.json       # multi-turn experiment (pending)

figures/
  keyword_recovery_summary.png      # which keywords are recoverable
  complexity_vs_utility.png         # complexity attack figure

eval_long_context.py               # long context degradation script
eval_multi_turn.py                 # multi-turn degradation script
bash_scripts/run_long_context_eval.sh
bash_scripts/run_multi_turn_eval.sh
```

## Environment
- Python: `/scratch/Hanzhi/paper reproduce/venv/bin/python3`
- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct` (4-bit quantized)
- GPUs: A100-SXM4-80GB × 8 (GPU 0,1,2,3,4,5,6,7)
