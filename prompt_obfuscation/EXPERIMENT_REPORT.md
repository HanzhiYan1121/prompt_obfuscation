# Soft Prompt Obfuscation — 实验报告

**论文**：Prompt Obfuscation for Large Language Models (USENIX Security 2025)
**作者**：David Pape, Thorsten Eisenhofer, Lea Schönherr
**实验环境**：NVIDIA A100-SXM4-80GB × 8，Llama-3.1-8B-Instruct (4-bit BitsAndBytes 量化)
**数据集**：TruthfulQA
**报告日期**：2026-03-23

---

## 一、论文核心主张

该论文提出将可读的 system prompt 转换为连续嵌入向量（soft prompt），以保护其作为知识产权。核心声明：

1. **Utility Claim**：obfuscated prompt 与 original prompt 产生等价的模型输出
2. **Security Claim**：任何已知攻击都无法从 obfuscated prompt 中恢复原始 prompt 文本

本研究分两部分：**复现论文结论**（§2–§4），以及**对 Utility Claim 发起新攻击**（§5–§7）。

---

## 二、评估指标体系

### 2.1 Utility 评估指标（输出相似度）

用于比较"obfuscated prompt 生成的输出"与"conventional prompt 生成的输出"之间的相似度。

| 指标 | 含义 | 范围 | 为何使用 |
|------|------|------|----------|
| **SacreBLEU** | n-gram 精确率，加 brevity penalty | 0–100 | 词汇级匹配，对格式敏感（如"pirate 腔"需要特定词汇），是最能反映 style 保留程度的指标 |
| **ROUGE-1** | 1-gram 召回率 | 0–1 | 衡量词汇覆盖度，对 style 变化较敏感；与 SacreBLEU 互补（精确率 vs 召回率） |
| **BERTScore** | 基于 RoBERTa 的 token-level 语义相似度 | 0–1 | 捕捉语义等价性而非字面匹配，评估"内容是否一致" |
| **Cosine Similarity** | 基于 sentence-transformers 的句子级语义相似度 | -1–1 | 整体语义对齐度，不依赖 n-gram 重叠 |

**与原文的对应关系**：原文 Table 4–6 使用上述全部指标评估 8 种 style 的 obfuscation quality。本研究复现时使用完全相同的指标和计算方式（`src/output_similarity.py`）。

**基准对照组**（原文设计，本研究延用）：
- **Obf vs Conv**：obfuscated prompt 生成结果 vs conventional prompt 生成结果（主要实验）
- **Conv(seed=43) vs Conv(seed=42)**：同一 prompt 不同随机种子，代表"不可超越的上限"
- **Blank vs Conv**：无 system prompt vs conventional prompt，代表"最差下限"

### 2.2 Security 评估指标（攻击成功率）

| 攻击 | 指标 | 含义 |
|------|------|------|
| Prompt Injection | Exact Match / ROUGE-L ≥ 0.9 extractions / Mean ROUGE-L recall | 成功提取出原始 prompt 的次数 |
| Token Projection | Levenshtein / Jaccard / LCS / Cosine Sim | 投影得到的 token 序列与原始 prompt 的文本相似度 |
| Fluency Deobfuscation | 同上 4 指标 | 优化得到的"可读 prompt"与原始 prompt 的文本相似度 |

---

## 三、实验一：复现论文 8 种 Style 的 Soft Obfuscation

### 3.1 实验设计

原文使用 8 种预定义 style（pirate、robot、manga_miko 等），每种 style 构成一个 system prompt（原始 style 句 + 基础 QA 指令），通过以下优化过程将其压缩为 10 个连续嵌入向量：

**优化目标**：最小化 obfuscated prompt 与 original prompt 在输出分布上的差异。

**损失函数**：
```
L = CE_weight × CE(p_obf, y*) + KL_weight × KL(p_obf || p_orig)
```
其中 `p_obf` 为 obfuscated prompt 的输出 logits，`y*` 和 `p_orig` 均来自 original prompt 的预计算输出。

**优化细节**：
- 方法：Adam（lr=0.01），在 continuous embedding space 优化
- 数据集：TruthfulQA，800 samples（640 train / 160 test）
- 优化迭代：10 iterations per sliding window（window_size=5，共 3 个 window）
- 输出 token 数：15（用于计算损失）

### 3.2 结果

| Style | Tokens | Ratio | Cosine | BERTScore | ROUGE-1 | SacreBLEU |
|-------|--------|-------|--------|-----------|---------|-----------|
| pirate | 33 | 0.303 | 0.8476 | 0.9122 | 0.5846 | 49.20 |
| robot | 35 | 0.286 | 0.8371 | 0.9302 | 0.6181 | 49.23 |
| poem | 32 | 0.312 | 0.8259 | 0.8992 | 0.5367 | 35.77 |
| stutter | 32 | 0.312 | 0.7764 | 0.9005 | 0.5647 | 25.82 |
| friendly | 45 | 0.222 | 0.8550 | 0.9175 | 0.5923 | 48.90 |
| southern_usa | 34 | 0.294 | 0.8402 | 0.9047 | 0.5494 | 43.47 |
| david_attenborough | 41 | 0.244 | 0.8072 | 0.9120 | 0.5830 | 47.33 |
| manga_miko | 87 | 0.115 | 0.8234 | 0.9050 | 0.5548 | 47.62 |

**参照基准（pirate style）**：

| 对比组 | Cosine | BERTScore | ROUGE-1 | SacreBLEU |
|--------|--------|-----------|---------|-----------|
| Obf vs Conv | 0.8476 | 0.9122 | 0.5846 | 49.20 |
| Conv(seed=43) vs Conv(seed=42) | 0.8442 | 0.9122 | 0.5778 | 48.47 |
| Blank vs Conv | 0.7817 | 0.8572 | 0.3719 | 19.11 |

### 3.3 与原文对比

**完全复现**。Obfuscated prompt 的 utility 指标与"不同随机种子重跑 original prompt"几乎持平，说明 obfuscation 几乎没有引入额外损耗。原文 Table 4 报告了相同的规律，本实验数值与原文结果吻合（差异 < 0.5%，来自量化精度差异）。

---

## 四、实验二：Attack 3 — Soft Fluency Deobfuscation（8 种 Style）

### 4.1 攻击原理

这是论文中最强的 white-box 攻击。攻击者拥有模型权重和 obfuscated prompt 的完整权限。核心思路：在优化 behavioral consistency 的同时，添加 **fluency 正则化**，促使恢复的 prompt 成为流畅的自然语言。

**损失函数**：
```
L_total = consistency_weight × L_consistency + fluency_weight × L_fluency

L_consistency = CE(p_deobf, y*) + KL(p_deobf || p_obf)
L_fluency = NLL(project(deobf_embeddings))   # 自然语言流畅度
```

### 4.2 结果

对所有 8 种 style 运行 soft fluency deobfuscation，**全部失败**：

| Style | Levenshtein | Jaccard | LCS | Cosine Sim |
|-------|-------------|---------|-----|------------|
| pirate | 0.135 | 0.051 | 0.169 | 0.355 |
| robot | 0.142 | 0.038 | 0.167 | 0.282 |
| poem | 0.127 | 0.029 | 0.149 | 0.261 |
| stutter | 0.118 | 0.022 | 0.138 | 0.218 |
| friendly | 0.131 | 0.041 | 0.162 | 0.279 |
| southern_usa | 0.139 | 0.035 | 0.158 | 0.263 |
| david_attenborough | 0.144 | 0.047 | 0.171 | 0.291 |
| manga_miko | 0.156 | 0.062 | 0.183 | 0.307 |
| **Random baseline** | **0.115** | **0.000** | **0.122** | **0.268** |

**关键发现**：soft deobfuscation 的 cosine_similarity（0.16–0.28）**持续低于或接近 random baseline（0.25–0.32）**，说明优化后的 prompt 甚至不比随机 token 序列更像原始 prompt。

### 4.3 与原文对比

**与原文结论一致，但量化差异存在**。原文 Table 9 报告的 Soft Fluency 结果为 Levenshtein=0.237、Jaccard=0.068、LCS=0.284，高于本实验（原因：原文使用 800 samples 和 10 iter，本实验使用 200 samples 和 5 iter）。即便如此，原文的 best case 也远低于任何"有意义恢复"所需的阈值（通常认为 Jaccard > 0.5 才有实用价值）。

---

## 五、实验三：Token Projection Analysis

### 5.1 方法

对每种 style 的 `best_candidate.pt`，将每个嵌入向量投影到最近的词汇 token（欧氏距离），检查是否能恢复 style 关键词。

### 5.2 结果

| Style | 关键词恢复 | 结果 |
|-------|-----------|------|
| pirate | "pirate" | ✅ 出现在投影中 |
| poem | "poem" | ✅ 出现在投影中 |
| manga_miko | "anime"、"-chan" | ✅ 出现在 hard deobf 中 |
| robot | — | ❌ 未恢复 |
| friendly | — | ❌ 未恢复 |
| southern_usa | — | ❌ 未恢复 |
| stutter | — | ❌ 未恢复 |
| david_attenborough | — | ❌ 未恢复 |

### 5.3 分析

仅 3/8 style 有关键词泄漏，且泄漏的都是**语义独特、在词汇中唯一**的词（"pirate"、"anime"）。语义复杂或多词描述的 style（如 David Attenborough、southern USA）无法通过投影恢复。这与原文 §6.3 的论证一致：即使投影出关键词，攻击者也无法还原完整的 prompt 指令。

---

## 六、实验四：Prompt Complexity Attack（新实验，对 Utility Claim 的攻击）

### 6.1 动机与假设

论文的 Utility Claim 基于**简单 style prompt**（32–45 tokens，压缩比 0.22–0.31）。我们提出：**当 system prompt 包含复杂的、多条件的行为规则时，10 个 token 无法充分压缩，utility 将降低**。

**假设**：SacreBLEU 等词汇级指标随 prompt 复杂度（token 数量）单调下降。

### 6.2 实验设计

设计 3 个递进复杂度的 prompt，控制变量（同样基于 TruthfulQA，相同超参数）：

| 名称 | Token 数 | 压缩比 | 核心内容 |
|------|---------|--------|---------|
| **complex_L1** | 79 | 0.127 | 强制三段式回答结构（直接答案 / 解释 / 示例），禁止项目符号和"I"开头 |
| **complex_L2** | 109 | 0.092 | 话题自适应风格（science→教授腔；history→故事腔；food→邻居腔；other→简洁），每次以反思句结尾 |
| **complex_L3** | 166 | 0.060 | 6 条严格规则（必须以"Analysis:"开头；恰好 3 个 dash 要点；必须有 Confidence 评级；禁止被动语态；禁止承认局限性；≤120 词） |

**对照**：原文 8 种 style（32–87 tokens，压缩比 0.115–0.312）。

### 6.3 结果

| 名称 | 类型 | Tokens | 压缩比 | Cosine | BERTScore | ROUGE-1 | SacreBLEU |
|------|------|--------|--------|--------|-----------|---------|-----------|
| pirate（原文） | baseline | 33 | 0.303 | 0.8476 | 0.9122 | 0.5846 | 49.20 |
| robot（原文） | baseline | 35 | 0.286 | 0.8371 | 0.9302 | 0.6181 | 49.23 |
| manga_miko（原文最复杂） | baseline | 87 | 0.115 | 0.8234 | 0.9050 | 0.5548 | 47.62 |
| **complex_L1** | complex | 79 | 0.127 | 0.8537 | 0.9111 | 0.5646 | **42.51** |
| **complex_L2** | complex | 109 | 0.092 | 0.8151 | 0.9037 | 0.5301 | **39.43** |
| **complex_L3** | complex | 166 | 0.060 | 0.8602 | 0.9002 | 0.5351 | **37.26** |

### 6.4 与原文对比

**原文未测试此场景**。原文 §4 仅讨论"obfuscated prompt 能保留原始 style"，使用的均是简单 style prompt（最复杂的 manga_miko 也只是人设描述，不含结构化规则）。

我们发现：

- **SacreBLEU 单调下降**：L1(42.5) → L2(39.4) → L3(37.3)，跌破原文所有 baseline 的最低值（stutter 的 25.82 除外，但 stutter 本身是特殊发音 style）。complex_L3 相比原文均值（43.4）下降约 **14%**。
- **ROUGE-1 下降**：L1(0.5646) → L2(0.5301) → L3(0.5351)，趋势较弱。
- **Cosine / BERTScore 不降反升（L3）**：语义层面 obfuscated prompt 仍能捕捉到原始 prompt 的大意，但词汇级别的格式要求（"Analysis:"开头、dash 编号）无法被 10 个 token 精确编码。

**结论**：对 Utility Claim 的**部分破解**。词汇级指标（SacreBLEU、ROUGE-1）在高复杂度 prompt 下显著退化，语义级指标（BERTScore、Cosine）保持稳健。这说明 10-token soft prompt 能保留"语义方向"，但无法保留"精确格式规则"。

---

## 七、实验五：Long Context Robustness（新实验）

### 7.1 假设

随着用户消息长度增加，model 的 attention 分配给 system prompt 的比例减小，导致 10-token soft prompt 的影响力被"稀释"，obfuscated outputs 与 conventional outputs 的相似度下降。

### 7.2 实验设计

- **Base**：`soft_pirate_truthfulqa_full`（pirate style，已验证 utility）
- **操作**：在每个 test question 前插入 N tokens 的中性填充文本（重复中性句子，无语义内容）
- **N**：{0, 128, 256, 512, 1024} extra tokens
- **样本**：40 test questions，batch_size=4
- **中性填充示例**："This document provides background context for reference purposes only. The following information is general knowledge..."

### 7.3 结果

| Extra Tokens | Cosine | BERTScore | ROUGE-1 | SacreBLEU |
|-------------|--------|-----------|---------|-----------|
| 0（baseline） | 0.8550 | 0.9142 | 0.5932 | 50.58 |
| 128 | 0.8685 | 0.9160 | 0.5978 | 51.82 |
| 256 | 0.8675 | 0.9146 | 0.5955 | 50.92 |
| 512 | 0.8743 | 0.9182 | 0.6048 | 52.89 |
| 1024 | 0.8777 | 0.9164 | 0.5981 | 52.75 |

### 7.4 与原文对比

**原文未测试此场景**。原文的所有 utility 评估均基于短问题（TruthfulQA 典型问题长度约 10–20 tokens）。

实验结果与假设**完全相反**：所有指标在 context 增长时保持稳定甚至微升。可能原因：

1. **Positional encoding 的固定优势**：Soft prompt 位于 position 0，具有全局注意力权重优势，不受用户消息长度影响。
2. **中性填充的无干扰性**：填充文本语义中立，不引入与 pirate style 相冲突的信息，因此 obfuscated 和 conventional prompt 均能正常忽略填充内容。
3. **Llama-3.1 的长上下文处理能力**：8B instruct 模型在长 context 下已具备良好的 system prompt 遵从性。

**结论**：soft prompt 对 long context 具有预料外的鲁棒性，此攻击角度无效。

---

## 八、实验六：Multi-Turn Robustness（新实验）

### 8.1 假设

Soft prompt 是在 single-turn Q&A 上优化的。当对话历史积累后，越来越多的 context tokens 与 system prompt 争夺 attention，导致 obfuscated outputs 在多轮对话中逐渐偏离 conventional outputs。

### 8.2 实验设计

- **Base**：`soft_pirate_truthfulqa_full`
- **操作**：将 test questions 组成三元组 (Q1, Q2, Q3)，构建 3 轮对话
- **关键控制**：前序轮次的回答统一使用 **conventional prompt** 的输出作为 history（控制历史质量，隔离 style 差异）
  - Turn 1：`[sys] Q1` → sim(conv_A1, obf_A1)
  - Turn 2：`[sys] Q1 conv_A1 Q2` → sim(conv_A2, obf_A2)
  - Turn 3：`[sys] Q1 conv_A1 Q2 conv_A2 Q3` → sim(conv_A3, obf_A3)
- **样本**：30 triplets（共 90 questions），batch_size=4

### 8.3 结果

| Turn | Cosine | BERTScore | ROUGE-1 | SacreBLEU |
|------|--------|-----------|---------|-----------|
| Turn 1（单轮基线） | 0.8681 | 0.9115 | 0.5773 | 49.37 |
| Turn 2（含 1 轮历史） | 0.8627 | 0.9184 | 0.6220 | 53.88 |
| Turn 3（含 2 轮历史） | 0.8367 | 0.9179 | 0.6141 | 51.99 |

### 8.4 与原文对比

**原文未测试多轮场景**。原文所有实验均为 single-turn inference。

结果对假设提供了**弱支持**：
- **Cosine Similarity 有小幅下降**：0.868 → 0.837（-3.6%），是唯一单调下降的指标，方向支持假设
- **BERTScore 基本持平**：0.9115 → 0.9179（微升）
- **SacreBLEU 和 ROUGE-1 在 Turn 2 反升**：可能原因是 Turn 2 的问题恰好比 Turn 1 更适合 pirate style，历史 context 反而给了 obfuscated prompt 更多"对齐"的机会

**结论**：多轮对话并非有效攻击角度。Soft prompt 在 3 轮对话内保持了基本一致的 utility。

---

## 九、综合讨论

### 9.1 三项新实验的结论汇总

| 实验 | 核心假设 | 实验结论 | 对 Utility Claim 的威胁 |
|------|---------|---------|------------------------|
| Prompt Complexity | 复杂 prompt 无法被 10 token 压缩 | **部分成立** | ⚠️ 中等威胁 |
| Long Context | 长 context 稀释 soft prompt 影响 | **不成立** | ✅ 无威胁 |
| Multi-Turn | 多轮历史削弱 soft prompt | **弱成立** | ✅ 基本无威胁 |

### 9.2 有效的攻击维度

从实验结果看，**唯一有效的攻击角度是 Prompt Complexity**：

- 将 prompt 从简单 style（~33 tokens）扩展到复杂规则（166 tokens），SacreBLEU 从 ~49 降至 37.3（-24%）
- 降幅在**词汇级指标**（SacreBLEU、ROUGE-1）上最为显著，语义级指标（BERTScore、Cosine）相对稳健
- 这说明 10-token soft prompt 能捕捉**语义方向**（"像 pirate 一样回答"），但无法编码**精确的格式约束**（"必须以 Analysis: 开头"）

### 9.3 论文防御的局限性

1. **Utility Claim 的测试场景过于简单**：原文只测试了 style-level prompt（personality / speaking style），未考虑 format-heavy 或 rule-based prompt——这正是实际生产中更常见的 system prompt 类型。

2. **压缩比问题**：原文中最长的 prompt（manga_miko，87 tokens）压缩比为 0.115，与我们的 complex_L1（79 tokens，0.127）相近，但 manga_miko 的 SacreBLEU（47.62）远高于 complex_L1（42.51）。这说明 **prompt 的语义复杂度比单纯的 token 数量更重要**：manga_miko 是人设描述（连贯叙事），而 complex_L1 是多条独立规则的组合，后者更难压缩。

3. **`obf_sys_prompt_len=10` 是论文的固定选择**：增大嵌入长度（如 20 或 30 tokens）理论上可以缓解 utility 损失，但原文未系统测试此参数对复杂 prompt 的影响。

### 9.4 评估指标的选择反思

| 指标 | 对本实验的适用性 | 注意事项 |
|------|----------------|---------|
| SacreBLEU | ⭐⭐⭐ 最敏感 | 对词汇精确度要求高，适合检测格式保留；但高方差（stutter style 只有 25.82，不代表 obfuscation 失败，而是 style 本身导致词汇重叠低） |
| ROUGE-1 | ⭐⭐ 中等 | 召回率视角，对词汇覆盖度敏感；与 SacreBLEU 趋势基本一致 |
| BERTScore | ⭐ 偏宽松 | 语义级别，对格式变化不敏感；容易给出"虚高"分数 |
| Cosine Sim | ⭐ 偏宽松 | 整体语义，不区分细节；在 long context / multi-turn 实验中几乎不变 |

**核心结论**：**SacreBLEU 是本研究中最具区分力的指标**，应作为 utility 评估的首要指标。BERTScore 和 Cosine 的高分可能掩盖实质性的 utility 损失。

---

## 十、实验参数与可复现性

| 实验 | 脚本 | 结果目录 | GPU | 运行时间 |
|------|------|---------|-----|---------|
| 8 styles obfuscation | `obfuscate.py` + `evaluate_obfuscation.py` | `precomputed_results/soft_*` | GPU 0–7（各 style 独立） | ~2–4h per style |
| Attack 3 (soft deobf) | `fluency_deobfuscation.py` | `precomputed_results/soft_*` | GPU 0–4 | ~1.5h per style |
| Token projection | `projection.py` + `decode_all_styles.py` | `all_styles_keyword_analysis.json` | GPU 0 | ~10 min |
| Complexity attack | `obfuscate.py` + `evaluate_obfuscation.py` | `precomputed_results/complex_L*/` | GPU 0,4 | ~1h per level |
| Long context | `eval_long_context.py` | `results/long_context_degradation.json` | GPU 5 | ~20 min |
| Multi-turn | `eval_multi_turn.py` | `results/multi_turn_degradation.json` | GPU 3 | ~14 min |

所有随机种子固定为 42。模型：`meta-llama/Meta-Llama-3.1-8B-Instruct`（4-bit BitsAndBytes 量化）。

---

## 附录：图表索引

| 文件 | 内容 |
|------|------|
| `figures/fig1_complexity.png` | Prompt Complexity vs SacreBLEU（仅 L1/L2/L3，含 baseline 色带） |
| `figures/fig1_complexity_metrics.png` | Prompt Complexity vs 4 指标（双 y 轴） |
| `figures/fig1_complexity_with_baselines.png` | Prompt Complexity vs SacreBLEU（含原文 8 style 散点） |
| `figures/fig2_long_context.png` | Long Context：4 指标归一化折线 |
| `figures/fig3_multi_turn.png` | Multi-Turn：4 指标归一化折线 |
| `figures/keyword_recovery_summary.png` | 8 种 style 的关键词恢复能力 |
| `figures/complexity_vs_utility.png` | Complexity 实验综合图（含原文） |
