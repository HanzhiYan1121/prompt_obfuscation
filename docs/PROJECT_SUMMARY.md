# Prompt Obfuscation and Recovery Project: Complete Experiment Summary

Last updated: 2026-07-18

## 中文总览

这个项目最初复现了 soft-prompt obfuscation：把可读 system prompt 压缩成十个连续向量，在隐藏原始文字的同时尽量保持模型行为。基础复现证明，Pirate 和 Robot 这类简单风格能够很好保留；105 次直接提取对 readable prompt 得到 33 次 exact 和 41 次 near-exact recovery，但对 tok10 soft prompt 都是 0 次。最近邻 token projection 和 fluency deobfuscation 也只能得到很低的文字重合，因此 exact prompt text 确实更难直接取回。

随后，项目把重点转向 utility boundary。Drone code 实验表明，输出即使格式看起来正确，API adherence 和 trajectory 仍可全部为 0；换成同一任务的自然语言步骤后，结构和语义又能部分恢复。JSON sentiment 位于两者之间：sentiment 大体能保留，但严格 JSON schema 和 `confidence` 字段不能保留。因此形成了项目的第一个核心结论：**soft prompt 更容易保留宽松自然语言行为，而难以保留精确 API、代码、schema、标签和执行顺序。**

之后，项目发展出 prompt recovery 主线。恢复证据包括 soft-vector projection、prompt metadata、behavior examples 和 matched probes；恢复目标从完整自由文本改成 role、task、format、constraints、context、tool protocol 六字段 JSON。Role/Persona 的早期 ACL100 QLoRA 得到很高结果，但由于模板重复和 metadata 泄漏，只能视为 pipeline validation。扩大到异构六分类后，恢复明显更难，也证明不同 prompt category 必须分开评估。

后续实验逐步加入 Text F1、ROUGE-L F1 和 Utility F1。Text F1 比较 recovered prompt 与 original prompt 的 token 重合；Utility F1 则把两种 prompt 分别放回模型，在相同 probes 上比较输出。Constraints n=12 的 Text F1 为 0.422，但 Utility F1 只有 0.178，说明“文字看起来接近”不等于“行为真正相同”。多 seed 实验进一步证明 Constraints 的下降不是随机噪声，而 Task 的 n=12 文本峰值并不是稳定的行为提升。

六分类实验从 100/category 扩展到 150/category，再扩展到 200/category。matched original-test probes 在所有类别上都优于宽泛的手写 category probes；但更强却不匹配的 diagnostic probes 又使六类 Utility 全部下降。这说明 probe 既要针对类别，也必须匹配具体任务分布。200/category 的 Qwen3-32B 最终实验已经完整产生六类 160/40 split 的恢复和 Utility 结果，其中 Task Utility F1 最高，为 0.421；Role 为 0.329，Format 为 0.311，Constraints 为 0.307，Context 为 0.348，Tool 为 0.346。为了排除“证据 probe 与评估 probe 重叠”带来的分布内红利，进一步做了 query-disjoint 的完整第二步：证据用查询 [0:6]，评估用不相交的 [6:16]，并对全部 240 个 held-out prompt 重新生成原始输出。结果六类 Utility 全部下降但仍非零（平均从 0.344 降到 0.306，平均降 0.037），说明 soft-prompt obfuscation 的功能泄漏在 query-disjoint 黑盒评估下依然成立，而不只是分布内重建；不过 probe 仍来自同一任务分布，属于 query-disjoint 而非 cross-distribution。

模型与 target 消融表明，单纯把 Qwen2.5-14B 换成 32B 只对部分类别有帮助；重新构造的 silver v2 targets 虽然让六类 Text F1 全部提高，却使大多数类别的 Utility 降低；增加完整 `executable_prompt` 后，Format、Context 和 Tool 得到提升，但 Role、Task 和 Constraints 下降。因此恢复目标需要按类别设计，不能只追求更干净或更长的统一 target。

主动攻击 pilot 在 10 个 Role prompts 上生成四个 recovery candidates，并用有限查询重新排序。Adaptive reranking 把 Utility 从 greedy 的 0.291 提高到 0.356，但 random-probe reranking 已达到 0.350。当前证据说明 candidate diversity 有用，但还不能证明 adaptive probe selection 优于 random selection。

最后，项目把攻击扩展到八种保护机制。当前结果覆盖 8 个 mechanism-inspired `-style` transformations、6 个类别、每类 100 个实例、48 个 Qwen3-32B QLoRA adapters 和 960 个 held-out evaluations。八种方法的平均 Utility F1 位于 0.361--0.433，说明统一接口下都存在部分 functional reconstruction。不过这些是对核心机制的 benchmark abstraction，不是八篇论文的完整端到端复现，也不能直接否定其原始隐私保证。

当前 AAAI 稿件已经加入 system model、symbolic threat model、attacker observation mapping、完整实验协议和 threat-model figure，并删除了 Success@0.50。删除原因是不同类别的 random floor 和 self-similarity reference 不同，固定 0.50 不能稳定代表“成功”；人工检查也发现历史上超过 0.50 的 26 个样本只有 17 个属于可信的部分恢复。当前论文因此只保留连续的 Text F1、ROUGE-L F1 和 Utility F1。

## 1. Project Overview

This project started by reproducing **Prompt Obfuscation for Large Language Models** and then developed into a broader study of the privacy--utility tradeoff in prompt protection.

The project now addresses three connected questions:

1. **Utility preservation:** after a readable system prompt is converted into a learned soft prompt, which behaviors remain and which behaviors fail?
2. **Prompt recovery:** even when exact wording is hidden, can an attacker infer the prompt's role, task, format, constraints, context, or tool protocol from indirect evidence?
3. **Cross-defense recovery:** do similar residual signals also appear under other prompt-protection ideas such as token perturbation, prompt confusion, emoji encoding, sanitization, and representation splitting?

The main conclusion is not that every protected prompt can be copied exactly. The supported conclusion is:

> Prompt protection can hide exact wording while still exposing behaviorally useful structure, because a useful protected prompt must preserve enough task information to control model behavior.

The project therefore separates three levels of recovery:

- **Exact recovery:** reproduce the original prompt word for word.
- **Structural recovery:** infer fields such as role, task, format, constraints, context, and tool protocol.
- **Functional recovery:** construct a different prompt that produces partly similar behavior on matched inputs.

## 2. Common Experimental Pipeline

### 2.1 Soft-prompt obfuscation

The original readable system prompt is converted into a short continuous prompt, usually ten learned embedding vectors (`tok10`). The vectors are optimized so that the protected model output resembles the output under the original prompt.

The main protected-model evaluator is Llama-3.1-8B-Instruct in 4-bit precision. The soft prompt is saved as `best_candidate.pt`, with all candidates stored in `obfuscated_system_prompt_list.pt`.

### 2.2 Recovery evidence

The recovery pipeline has used several forms of indirect evidence:

- nearest-token projections of the soft vectors;
- prompt metadata and six-dimensional category scores;
- behavior examples, meaning input--output traces produced under the hidden or protected prompt;
- matched probes drawn from the original task's input style;
- in later work, multiple recovery candidates and actively selected probes.

### 2.3 Six-dimensional prompt representation

Prompts are analyzed through six dimensions:

1. Role/Persona
2. Task Instruction
3. Format Specification
4. Constraints/Guardrails
5. Context Knowledge
6. Tool/Protocol

The dominant dimension determines the category, but a prompt may contain information in several fields. Structured recovery predicts all six fields rather than assuming that only the dominant field exists.

### 2.4 Metric terminology

- **Token F1** is the underlying bag-of-token overlap function. It ignores token position and is not a semantic metric.
- **Text F1** applies Token F1 to recovered and reference prompt representations.
- **ROUGE-L F1** measures longest-common-subsequence overlap and therefore reflects sequence order more than Text F1.
- **Utility F1** applies Token F1 to model outputs generated under the recovered and original prompts on the same probes.
- **Keyword recall** measures whether selected prompt-specific keywords are recovered, but it is unreliable when equivalent concepts use different words.
- **Success@0.50** was used in historical experiments as a thresholded Utility F1 statistic. It is no longer a primary metric in the current paper.

The current paper reports continuous **Text F1, ROUGE-L F1, and Utility F1**. Token F1 is the common scoring operation, not an additional independent experiment-level metric.

## 3. Phase I: Foundation Reproduction

### 3.1 Eight style prompts

The original soft-obfuscation pipeline was run on eight styles: pirate, robot, manga/miko, friendly, poem, southern USA, stutter, and David Attenborough. The main detailed reproduction uses Pirate and Robot prompts on 800 TruthfulQA examples.

For `tok10`, Pirate obfuscated-versus-original outputs reached BERTScore 0.9122, SacreBLEU 49.20, and ROUGE-L 0.4606. Robot reached BERTScore 0.9302, SacreBLEU 49.23, and ROUGE-L 0.5266. These values were close to rerunning the readable prompt with another generation seed.

**Finding:** simple style behavior can be preserved very well even when the exact text is replaced by continuous vectors.

### 3.2 Direct extraction attack

The model was queried with 105 prompt-extraction instructions.

| Prompt form | Exact extraction | Near extraction | Mean ROUGE-L recall |
|---|---:|---:|---:|
| Readable prompt | 33/105 | 41/105 | 0.4985 |
| Obfuscated tok10 prompt | 0/105 | 0/105 | 0.1293 |

**Finding:** direct prompt injection failed to reconstruct the soft prompt text, although related style words could still appear.

### 3.3 Nearest-token projection

Each soft vector was mapped to its closest vocabulary token using Euclidean and cosine distance.

| Metric | Projection | Random baseline |
|---|---:|---:|
| Levenshtein similarity | 0.162 | 0.115 |
| Jaccard | 0.064 | 0.000 |
| LCS ratio | 0.196 | 0.122 |
| Cosine similarity | 0.387 | 0.269 |

**Finding:** projection retained a few weak clues but did not reconstruct readable prompt text.

### 3.4 Fluency and optimization-based recovery

Soft and hard deobfuscation were tested for Pirate and Robot styles. For Pirate, hard recovery reached Jaccard 0.077, close to the paper reference of 0.084. Soft recovery could preserve behavior at BERTScore 0.9122 while having Jaccard 0.000.

This produced the first important project distinction:

> A behaviorally equivalent prompt can exist even when its text has almost no overlap with the original prompt.

Across all eight styles, soft fluency recovery did not reliably recover style keywords and often scored below the random semantic baseline. Hard recovery gave slightly better word overlap but was unstable and sometimes collapsed to blank-prompt behavior.

## 4. Phase II: Utility-Preservation Boundary

### 4.1 Prompt complexity

Three increasingly complex prompts were compressed into ten vectors. Their lengths were 79, 109, and 166 tokens. SacreBLEU decreased from 42.51 to 39.43 to 37.26, while semantic similarity did not decline monotonically.

**Finding:** prompt complexity weakens lexical preservation, but complexity alone does not fully explain semantic utility loss.

### 4.2 Long-context robustness

Neutral user context was increased from 0 to 1,024 tokens. BERTScore remained around 0.914--0.918 and SacreBLEU around 50.6--52.9.

**Finding:** the hypothesis that long user context would wash out the ten-vector prompt was not supported.

### 4.3 Multi-turn robustness

Across three turns, cosine similarity decreased from 0.868 to 0.837, but BERTScore remained flat and ROUGE/SacreBLEU were not monotonic.

**Finding:** multi-turn degradation was weak and metric-dependent, not a robust failure mode.

### 4.4 Experiments 1--5: out-of-domain drone code

Soft prompts trained on CNN/DailyMail were tested on 24 AirSim code tasks with lengths 10, 20, 40, 80, and 160. Format scores stayed around 0.92--0.97, but API adherence and trajectory scores were both 0.00 for every length.

**Finding:** outputs can look syntactically plausible while being functionally useless.

### 4.5 Experiments 6--10: in-domain drone training

The soft prompts were retrained directly on drone instructions. Format rose to about 0.97--0.99, but API adherence and trajectory remained 0.00.

**Finding:** the failure was not merely caused by CNN-to-drone distribution shift.

### 4.6 Experiments 11--12: longer training and L4 reranking

Longer output training and evaluator-based candidate reranking were applied to tok10 and tok20 drone prompts. The selected result had Format 0.85, API 0.00, and Trajectory 0.00.

**Finding:** stronger optimization and task-specific candidate selection did not recover exact code behavior.

### 4.7 Experiment 13: JSON sentiment

| Metric | Original | Obfuscated |
|---|---:|---:|
| Valid JSON | 1.000 | 0.855 |
| Entity field | 1.000 | 0.570 |
| Sentiment field | 1.000 | 0.855 |
| Confidence field | 1.000 | 0.000 |
| Loose schema | 1.000 | 0.470 |
| Strict schema | 1.000 | 0.000 |
| Sentiment agreement | n/a | 0.647 |

**Finding:** coarse sentiment survived, but strict schema details did not.

### 4.8 Experiment 14: drone natural language

The same drone task was changed from Python/API output to numbered English instructions.

| Dimension | Drone code | Drone natural language |
|---|---:|---:|
| Surface | 0.85 | 0.88 |
| Structure | 0.00 | 0.93 |
| Semantics | 0.00 | 0.54 |

**Finding:** changing only the output form restored partial utility. This is the strongest evidence for the **symbolic precision boundary**: loose natural language is easier to preserve than exact code, API names, schemas, and execution order.

## 5. Phase III: Prompt-Recovery Development

### 5.1 Recovery A: five representative system prompts

DetGPT, SPADE, BlindSpots, SMARTMiner, and FeedbackLearn were converted to tok10 prompts and attacked using projections and behavior examples.

| Metric | Mean |
|---|---:|
| ROUGE-L F1 | 0.147 |
| Token F1 | 0.246 |
| Keyword recall | 0.228 |

SMARTMiner had the strongest partial keyword recovery at about 0.625.

**Finding:** exact recovery failed, but task semantics and keywords leaked partially.

### 5.2 Recovery B: ACL100 Role/Persona validation

One hundred short role/persona prompts were split into 80 training and 20 held-out examples.

| Setting | ROUGE-L F1 | Token F1 | Exact recovery | Role recall |
|---|---:|---:|---:|---:|
| Direct Llama-3.1-8B | 0.133 | 0.160 | 0.000 | 0.530 |
| QLoRA | 0.971 | 0.971 | 0.900 | 1.000 |

**Interpretation:** this validated the supervised pipeline, but it was not a general security result. The prompts were short, repetitive, and template-like, and metadata could reveal role/template information.

### 5.3 Recovery C: 101 prompts and six-dimensional taxonomy

The first heterogeneous set contained 40 Task, 29 Role, 17 Mixed, 6 Context, 5 Format, 2 Constraints, and 2 Tool prompts.

- Mean obfuscation utility: about 0.539.
- Mixed/Complex utility: about 0.465.
- Role and Task utility: about 0.558 and 0.553.
- Direct recovery Token F1: about 0.159.
- Heterogeneous QLoRA Token F1: about 0.181.

**Finding:** category heterogeneity makes recovery much harder; results should be reported by category.

### 5.4 Recovery D: structured Role/Persona recovery

| Setting | ROUGE-L F1 | Token F1 |
|---|---:|---:|
| Full-prompt QLoRA | 0.190 | 0.253 |
| Six-field JSON QLoRA | 0.337 | 0.387 |
| Six-field JSON + Qwen2.5-14B | 0.387 | 0.419 |

**Finding:** structured targets and a stronger recovery model improved results step by step, but this was structural rather than exact wording recovery.

### 5.5 Recovery E: remaining five categories

The Role method was extended to Task, Format, Constraints, Context, and Tool. Categories with fewer than 100 prompts used all available prompts; Task was capped at 100. Qwen2.5-14B category-specific QLoRA models compared six and twelve behavior examples.

| Category | Token F1 n=6 | Token F1 n=12 | Change |
|---|---:|---:|---:|
| Task | 0.384 | 0.458 | +0.074 |
| Format | 0.325 | 0.351 | +0.026 |
| Constraints | 0.313 | 0.422 | +0.109 |
| Context | 0.387 | 0.386 | -0.001 |
| Tool | 0.343 | 0.334 | -0.009 |

**Initial finding:** more examples appeared helpful for Task and Constraints but not Context or Tool. Later behavior-level evaluation revised this conclusion.

## 6. Phase IV: Evidence Quantity and Behavioral Evaluation

### 6.1 Recovery F: n-sweep to 30 examples

| Category | n=6 | n=12 | n=18 | n=24 | n=30 |
|---|---:|---:|---:|---:|---:|
| Task Text F1 | 0.384 | 0.458 | 0.393 | 0.415 | 0.402 |
| Constraints Text F1 | 0.313 | 0.422 | 0.386 | 0.415 | 0.394 |

**Finding:** both text-overlap curves peaked at n=12 and then degraded. More generic evidence was not automatically better.

### 6.2 Recovery G: Utility F1 and historical threshold calibration

Task n=12 was calibrated using a random-prompt floor, recovered prompts, and an original-prompt self-similarity reference.

| Condition | Mean Utility F1 |
|---|---:|
| Random prompt | 0.336 |
| Recovered prompt | 0.396 |
| Self-similarity reference | 0.489 |

The old normalized lift was `(0.396-0.336)/(0.489-0.336)=0.39`.

The critical result was disagreement between text and behavior. For Constraints at n=12, Text F1 was 0.422 but Utility F1 was only 0.178. At n=18, Utility F1 fell to 0.123.

**Finding:** wording overlap can rank a recovery highly even when the reconstructed prompt produces the wrong behavior.

### 6.3 Recovery H: multi-seed stability

Three generation seeds, 42, 123, and 2025, were used without retraining.

- Constraints cross-n Utility range: 0.223.
- Constraints average seed standard deviation: 0.013.
- Structural-to-noise ratio: 16.96.
- Task n=6 versus n=12 range: 0.012.
- Task seed standard deviation: 0.020.
- Task ratio: 0.62.

**Finding:** the Constraints collapse was much larger than seed noise, while the apparent Task n=12 gain was within seed noise.

### 6.4 Recovery I: Best-of-5 reranking

Five recovered candidates were generated and ranked on one half of cached queries, then evaluated on the other half.

- Task n=12: Greedy 0.369, Best-of-5 0.375, gain +0.006.
- Constraints n=6: Greedy 0.348, Best-of-5 0.375, gain +0.027.

**Finding:** candidate reranking helped slightly, but it did not create a structural breakthrough.

## 7. Phase V: Balanced Six-Category Experiments

### 7.1 Recovery J: 100 prompts per category

Each category used 80 training and 20 held-out prompts with Qwen2.5-14B QLoRA.

| Category | Text F1 | Utility F1 |
|---|---:|---:|
| Role/Persona | 0.426 | 0.453 |
| Task Instruction | 0.397 | 0.342 |
| Format Spec | 0.314 | 0.304 |
| Constraints | 0.274 | 0.269 |
| Context Knowledge | 0.324 | 0.311 |
| Tool/Protocol | 0.285 | 0.324 |

**Finding:** Role/Persona was strongest. Tool/Protocol demonstrated that low wording overlap can coexist with moderate functional overlap.

### 7.2 Recovery K: probe comparison on 100/category

The recovery model was fixed; only evaluation inputs changed.

| Category | Generic category probes | Matched original-test probes |
|---|---:|---:|
| Role | 0.327 | 0.469 |
| Task | 0.248 | 0.355 |
| Format | 0.192 | 0.351 |
| Constraints | 0.222 | 0.269 |
| Context | 0.305 | 0.366 |
| Tool | 0.236 | 0.332 |

**Finding:** matched probes improved Utility F1 in all six categories. Probe distribution is part of the evaluation definition, not a minor implementation detail.

### 7.3 Scale-up to 150 prompts per category

Each category used 120 training and 30 held-out prompts. The evaluation used matched probes.

| Category | Text F1 | Utility F1 |
|---|---:|---:|
| Role/Persona | 0.389 | 0.372 |
| Task Instruction | 0.370 | 0.412 |
| Format Spec | 0.343 | 0.304 |
| Constraints | 0.282 | 0.289 |
| Context Knowledge | 0.326 | 0.341 |
| Tool/Protocol | 0.313 | 0.328 |

**Finding:** Task became strongest under matched task-distribution probes; Constraints remained hardest.

### 7.4 Error analysis on the 150/category run

Manual and field-level analysis identified these main failure modes:

- missing dominant fields;
- low overlap in the dominant field;
- unsupported or spurious fields;
- duplicated content across fields;
- collapse into a generic assistant prompt;
- valid JSON with the wrong operational details;
- high Text F1 but low Utility F1.

Among 26 historical cases above Utility F1 0.50, only 17 were judged plausible partial task recoveries, giving about 65.4% manual precision. This demonstrated that a fixed threshold was not equivalent to 100% successful recovery.

### 7.5 Per-category threshold study

The random floor and self-similarity reference differed by category. No acceptable common threshold was found for Task or Format under the chosen calibration criteria, while recommended values differed for other categories.

**Decision:** remove Success@0.50 from the current paper and retain continuous Text F1, ROUGE-L F1, and Utility F1.

## 8. Phase VI: Model and Target Ablations

All experiments below use 150 prompts per category, a 120/30 split, n=6 behavior examples, and matched probes unless stated otherwise.

### 8.1 Experiment 3: Qwen2.5-14B versus Qwen2.5-32B

| Category | Utility 14B | Utility 32B |
|---|---:|---:|
| Role | 0.372 | 0.380 |
| Task | 0.412 | 0.400 |
| Format | 0.304 | 0.308 |
| Constraints | 0.289 | 0.287 |
| Context | 0.341 | 0.317 |
| Tool | 0.328 | 0.367 |

**Finding:** the larger model helped Tool most and slightly helped Role/Format, but did not improve every category. Capacity was not the only bottleneck.

### 8.2 Experiment 4: rebuilt six-field targets

Regex v1 targets were replaced by Qwen2.5-32B-generated silver v2 targets followed by conservative validation.

| Audit statistic | Regex v1 | Silver v2 |
|---|---:|---:|
| Exact cross-field duplication | 219 | 0 |
| Near cross-field duplication | 321 | 0 |
| Mean target/original length ratio | 1.044 | 0.932 |
| Dominant-field coverage | 0.881 | 0.808 |

**Finding:** v2 labels were cleaner but more conservative. They were automatically generated silver labels, not human gold labels.

### 8.3 Experiment 5: QLoRA with silver v2 targets

Text F1 improved in all six categories. Utility improved only for Role, from 0.372 to 0.380; Task, Format, Constraints, Context, and Tool decreased.

**Finding:** cleaner semantic summaries can improve textual supervision while deleting exact labels, keys, placeholders, exceptions, or procedure steps needed for behavior.

### 8.4 Experiment 6: dual target with executable prompt

The model predicted both six structured fields and a complete `executable_prompt` that could be inserted directly as the system prompt during Utility evaluation.

| Category | Utility v1 | Utility v2 | Utility dual |
|---|---:|---:|---:|
| Role | 0.372 | 0.380 | 0.364 |
| Task | 0.412 | 0.326 | 0.393 |
| Format | 0.304 | 0.297 | 0.369 |
| Constraints | 0.289 | 0.251 | 0.248 |
| Context | 0.341 | 0.295 | 0.395 |
| Tool | 0.328 | 0.297 | 0.346 |

**Finding:** executable targets helped Format, Context, and Tool because they preserved schemas, placeholders, background payloads, and procedures. They hurt Role, Task, and Constraints through role drift, changed task labels, omitted negations, and divided training capacity.

## 9. Phase VII: Active and Adaptive Recovery

### 9.1 Role/Persona adaptive-probe pilot

Ten held-out Role prompts were evaluated with query-level separation. Four candidates were generated; a pool of twelve candidate-conditioned probes was created; four real queries were allowed for candidate reranking; final evaluation used separate queries.

| Selection method | Mean Utility F1 |
|---|---:|
| Greedy single candidate | 0.291 |
| Random-probe reranking | 0.350 |
| Adaptive-probe reranking | 0.356 |
| Oracle candidate selection | 0.399 |

Adaptive minus Greedy was +0.065 with bootstrap 95% CI `[+0.011,+0.130]`. Adaptive minus Random was only +0.006 with CI `[-0.012,+0.030]`.

**Finding:** candidate diversity clearly helped; the current adaptive disagreement rule did not establish an advantage over random reranking.

## 10. Phase VIII: 200/category and Qwen3-32B

The six category sets were expanded to 200 prompts each. Every category used 160 training and 40 held-out prompts, Qwen3-32B with QLoRA, six behavior examples, and matched probes.

An initial run failed because the Qwen3 chat-template call passed the wrong keyword. The pipeline was corrected/resumed, and all six final directories contain training configs, splits, recovered prompts, Text/ROUGE evaluation, and matched Utility evaluation.

| Category | ROUGE-L F1 | Text F1 | Utility F1 |
|---|---:|---:|---:|
| Role/Persona | 0.301 | 0.383 | 0.329 |
| Task Instruction | 0.402 | 0.438 | 0.421 |
| Format Spec | 0.250 | 0.329 | 0.311 |
| Constraints | 0.218 | 0.310 | 0.307 |
| Context Knowledge | 0.309 | 0.380 | 0.348 |
| Tool/Protocol | 0.314 | 0.392 | 0.346 |

Compared with the 150/category Qwen2.5-14B baseline, Utility improved slightly for Task, Format, Constraints, Context, and Tool but decreased for Role. This comparison changes both model family and dataset size, so it is a scale-up result rather than a clean model-only ablation.

### 10.1 Stronger diagnostic probes

The same Qwen3 recovery models were evaluated with stronger category-specific probes without retraining.

| Category | Matched Utility | Stronger-probe Utility |
|---|---:|---:|
| Role | 0.329 | 0.292 |
| Task | 0.421 | 0.317 |
| Format | 0.311 | 0.173 |
| Constraints | 0.307 | 0.235 |
| Context | 0.348 | 0.268 |
| Tool | 0.346 | 0.249 |

**Finding:** stronger but mismatched generic diagnostics reduced scores in every category. The best evaluation probes must be both category-aware and task-matched.

### 10.2 Query-disjoint black-box attack (Step 2 completion)

The earlier matched Utility F1 conditioned the recovery model on test queries `[0:6]` and then evaluated on the same queries, so it measured in-distribution reconstruction. This run separates the two: evidence stays on queries `[0:6]` (the probes whose real protected-model outputs, `obfuscated_output`, produced the recovered prompts), while evaluation uses the evidence-disjoint queries `[6:16]`. To cover every held-out prompt, the original prompt was re-run on `test_data[6:16]` for all 240 held-out prompts to build disjoint ground-truth outputs; the recovered prompt was then run on the same disjoint probes and compared with Token F1. The victim model stays Llama-3.1-8B-Instruct, the architecture the tok10 soft prompt is bound to.

| Category | Overlap Utility F1 (n=6, all 40) | Disjoint Utility F1 (full 40) | Change |
|---|---:|---:|---:|
| Role/Persona | 0.329 | 0.305 | -0.024 |
| Task | 0.421 | 0.354 | -0.066 |
| Format | 0.311 | 0.266 | -0.045 |
| Constraints | 0.307 | 0.281 | -0.027 |
| Context | 0.348 | 0.308 | -0.040 |
| Tool | 0.346 | 0.324 | -0.022 |
| Average | 0.344 | 0.306 | -0.037 |

Each category covers all 40 held-out prompts with 10 disjoint pairs each. A biased earlier pass over only the 15--32 prompts that had leftover cached queries showed a larger -0.066 average drop, because that subset over-represented harder prompts.

**Finding:** removing the evidence/evaluation probe overlap lowers Utility F1 in every category, by 0.037 on average (about 11% relative), but disjoint Utility F1 stays clearly nonzero at 0.27--0.35. The functional-leakage claim for soft-prompt obfuscation therefore holds under a clean query-disjoint (out-of-distribution) black-box evaluation, not only under in-distribution reconstruction. The probes remain drawn from the same task distribution, so this is query-disjoint but not cross-distribution.

## 11. Phase IX: EmojiPrompt-Style Pilot

A lightweight EmojiPrompt-style pilot used 30 prompts per category, with 24 training and 6 held-out prompts, Qwen2.5-14B QLoRA, and matched probes.

| Category | Text F1 | ROUGE-L F1 | Utility F1 |
|---|---:|---:|---:|
| Role | 0.418 | 0.390 | 0.482 |
| Task | 0.491 | 0.433 | 0.456 |
| Format | 0.394 | 0.321 | 0.296 |
| Constraints | 0.422 | 0.332 | 0.199 |
| Context | 0.329 | 0.259 | 0.339 |
| Tool | 0.308 | 0.267 | 0.284 |

**Scope:** this was a mechanism-style pilot, not a full EmojiPrompt reproduction with its complete LDP post-sampling or learned optimization procedure.

## 12. Phase X: Eight-Defense Common-Interface Benchmark

### 12.1 Methods represented

The benchmark contains eight protection transformations:

1. ConfusionPrompt-style: genuine and pseudo prompt fragments are mixed.
2. InferDPT-style: prompt tokens are perturbed.
3. Split-and-Denoise-style: noisy representation identifiers approximate split representation protection.
4. Confidential Prompting-style: virtual prompt sets approximate hidden demonstrations.
5. NOIR-style: randomized token identifiers represent protected token mappings.
6. EmojiCrypt/EmojiPrompt-style: content is represented through emoji-coded tokens.
7. DP-OPT-style: noisy keyword votes approximate private prompt optimization.
8. PrEEmpt-style: sensitive spans are replaced by typed placeholders.

### 12.2 Dataset and model

- Six categories with 100 category instances each.
- 600 category instances but 522 unique prompt IDs because score-based filling can reuse a mixed prompt in more than one category.
- Eight transformations, creating 4,800 category-instance--transformation pairs.
- 80 training and 20 held-out prompts for every method/category pair.
- 48 Qwen3-32B QLoRA adapters.
- 960 held-out method/category evaluations.
- Six matched behavior traces per recovery input.

### 12.3 Average results

| Protection transformation | Text F1 | ROUGE-L F1 | Utility F1 |
|---|---:|---:|---:|
| ConfusionPrompt-style | 0.667 | 0.596 | 0.406 |
| InferDPT-style | 0.671 | 0.588 | 0.433 |
| Split-and-Denoise-style | 0.380 | 0.312 | 0.366 |
| Confidential Prompting-style | 0.481 | 0.391 | 0.388 |
| NOIR-style | 0.389 | 0.314 | 0.387 |
| EmojiPrompt-style | 0.546 | 0.477 | 0.391 |
| DP-OPT-style | 0.444 | 0.353 | 0.361 |
| PrEEmpt-style | 0.679 | 0.606 | 0.425 |

PrEEmpt-style had the highest text overlap; InferDPT-style had the highest behavior-level Utility F1. Every condition had nonzero Utility F1 in the common-interface benchmark.

### 12.4 Category pattern

Average Utility F1 across transformations was approximately:

| Category | Utility F1 |
|---|---:|
| Role/Persona | 0.461 |
| Task Instruction | 0.411 |
| Context Knowledge | 0.409 |
| Format Specification | 0.388 |
| Tool/Protocol | 0.359 |
| Constraints | 0.339 |

**Finding:** category differences were often larger than method differences. Role and Task exposed repeated, observable behavior more readily; exact constraints and tool procedures were harder to elicit with six generic matched traces.

### 12.5 Critical scope statement

These are **mechanism-inspired benchmark approximations**, not complete end-to-end reproductions of the eight original papers. The benchmark does not reproduce every cryptographic protocol, differential-privacy guarantee, trusted execution environment, local denoiser, cloud--edge communication layer, or learned transformation.

The behavior traces currently use cached original-prompt outputs as an ideal high-utility proxy. Therefore, the benchmark asks whether a utility-preserving interface can leave recoverable structure; it does not prove that all eight deployed defenses are broken under their original threat models.

## 13. Current Paper Development

The AAAI-format manuscript now contains:

- an abstract and literature review with verified references;
- a symbolic system model separating private prompt `P`, internal protected state `H`, server-visible artifact `Z`, model output `Y`, and structural target `S`;
- a threat model defining the bounded cloud-side service operator and its observation view;
- a threat-model mapping figure showing client state, transmitted data, cloud execution, attacker observations, and active probes;
- the foundation reproduction and eight-transformation benchmark protocol;
- Qwen3-32B QLoRA settings and complete 80/20 held-out protocol;
- continuous Text F1, ROUGE-L F1, and Utility F1 results;
- a corrected information-theoretic statement: once complete `Z`, `X`, and model parameters are conditioned on, deterministic output `Y` adds no conditional Shannon information, although query outputs can still be operationally useful to a bounded recovery algorithm.

Success@0.50 has been removed from the current manuscript. The paper also explicitly labels the eight methods as `-style` transformations and does not claim full reproduction of their original security guarantees.

## 14. What Has Been Established Reliably

1. Soft prompts can preserve simple style behavior while making exact prompt extraction much harder.
2. Utility preservation has a symbolic precision boundary: exact APIs, code, schemas, labels, and rule details fail before broad natural-language intent.
3. Exact text recovery and functional recovery are different objectives.
4. Nearest-token projections are weak alone, but behavior traces and supervised examples expose more structure.
5. Recovery difficulty depends strongly on prompt category.
6. More behavior examples are not monotonically helpful; n=12 text gains did not consistently survive Utility evaluation.
7. Text F1 can disagree sharply with Utility F1, especially for Constraints and exact formats.
8. Matched probes are more informative than broad category probes. Using the same probes for evidence and evaluation measures in-distribution reconstruction; for the 200/category soft-prompt experiment a query-disjoint evaluation (evidence on `[0:6]`, evaluation on `[6:16]`, all 40 held-out prompts) still leaves nonzero functional leakage (average Utility F1 0.306, down 0.037 from the overlapping matched score).
9. Larger recovery models help selected categories but do not solve target or evidence problems.
10. Cleaner six-field labels improve textual recovery but can remove executable details.
11. Adding an executable target helps format-, context-, and procedure-heavy prompts, but can hurt roles and brittle constraints.
12. Candidate diversity helps active recovery; the current adaptive probe selector is not yet better than random reranking.
13. Across eight mechanism-style transformations, nonzero functional recovery remains under the common-interface benchmark.

## 15. What Has Not Yet Been Established

1. The project has not fully reproduced all eight original defense systems end to end.
2. The current eight-defense benchmark does not validate or invalidate the original papers' formal privacy guarantees.
3. The eight-defense common-interface Utility F1 still uses overlapping evidence and evaluation probes, so it is not yet a query-level out-of-distribution attack result. (The 200/category soft-prompt experiment now has a query-disjoint version; see 10.2. The same disjoint protocol has not yet been applied to the eight-defense benchmark.) Both remain query-disjoint at most, not cross-distribution.
4. The six-field targets are automatically generated and are not fully human-verified gold annotations.
5. Utility F1 is lexical overlap between outputs; it is not a direct correctness, schema-validity, policy-compliance, or tool-execution metric.
6. The active attack has only been piloted on ten Role/Persona prompts.
7. The 200/category Qwen3 result changes both model and data scale, so it cannot isolate either factor.

## 16. Recommended Next Experimental Sequence

1. Fully reproduce one defense at a time using official code and original parameters.
2. Define a defense-specific client/cloud/attacker mapping before running recovery.
3. Separate evidence probes, candidate-selection probes, and final evaluation probes.
4. Replace proxy traces with outputs from the actual protected execution path.
5. Add category-specific functional metrics: schema validity, constraint compliance, hidden-context accuracy, and tool-call correctness.
6. Human-audit the six-field targets and a stratified sample of recovered prompts.
7. Expand the adaptive attack only after its selector consistently beats random reranking.
8. Report exact text leakage, structured leakage, and functional imitation as separate outcomes.

## 17. Main Artifacts

- Project handoff: `CODEX_EXPERIMENT_README.md`
- Foundation reproduction: `REPRODUCTION_REPORT_BILINGUAL.md`
- Utility experiment log: `EXPERIMENTS.md`
- 150/category error analysis: `results/sixcat150_recovery_error_analysis.md`
- 200/category stronger-probe comparison: `results/sixcat200_stronger_probe_comparison.md`
- 200/category query-disjoint Step-2 attack: `results/sixcat200_step2_disjoint_probe_attack.md`
- Eight-defense tables: `results/privacybench100_8defense_combined_tables.md`
- Active Role pilot: `results/role_adaptive_probe_pilot/summary.md`
- Current AAAI manuscript: `prompt_protection_attack_paper/AnonymousSubmission2027.tex`
- Threat-model figure: `prompt_protection_attack_paper/figures/threat_model_mapping.pdf`

## 18. One-Paragraph Project Summary

The project began by reproducing soft-prompt obfuscation and confirmed that a ten-vector prompt can preserve simple style behavior while resisting direct textual extraction. It then identified a symbolic precision boundary: broad natural-language behavior survives better than exact code, API, schema, and constraint details. The work subsequently developed a six-category supervised recovery pipeline using projection clues, behavior traces, structured JSON targets, QLoRA, matched probes, and Utility F1. Results showed that recovery is partial, category-dependent, sensitive to probe distribution, and often different under text and behavior metrics. Model scaling, cleaner targets, executable prompts, Best-of-N selection, and adaptive probing produced selective rather than universal gains. The latest common-interface benchmark applies Qwen3-32B recovery to eight mechanism-inspired protection transformations and finds nonzero functional reconstruction across all conditions, while explicitly stopping short of claiming full end-to-end breaks of the original defense papers.
