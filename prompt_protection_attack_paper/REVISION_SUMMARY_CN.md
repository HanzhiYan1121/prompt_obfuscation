# AnonymousSubmission2027 批注合并与修订总结

## 修订基线

- 以 `AnonymousSubmission2027.tex` 为内容基线，因为该版本已经完成文献压缩、威胁模型澄清、实验参数补充，并且已经放弃 `Success@0.50`。
- 参考 `paper (1).tex` 与 `prompt_attack (1).pdf` 中的彩色行内批注，处理仍未充分落实的建议。
- 按作者要求，不再围绕旧批注重写 System Model 和 Threat Model；修订稿沿用 AnonymousSubmission2027 中已经定稿的符号、攻击者角色、可见信息和 mutual-information 说明。

## 逐项处理结果

1. **Introduction 不完整**
   - 保留 AnonymousSubmission2027 已补充的问题背景、现有评估缺口、攻击思路、贡献和主要数值结果。
   - 将结论严格限定为 common-interface benchmark 下的 partial reconstruction，避免把结果表述成对原始防御系统的端到端攻破。

2. **Literature Review 太长、像论文罗列**
   - 使用 AnonymousSubmission2027 的精简 Related Work。
   - 保留按 protection family、protected surface 和 retained signal 组织的总结表，替代逐篇展开介绍。

3. **Method 与 Experiment 混在一起**
   - 拆分为独立的 `Recovery Method` 和 `Experimental Setup`。
   - Method 只解释攻击设计动机、证据通道、matched probes、六字段结构恢复和 executable prompt 重建。
   - Experimental Setup 集中放置 foundation reproduction、八种 benchmark transformations、数据划分、QLoRA 参数、end-to-end track 和评估指标。

4. **缺少方法设计动机和总览图**
   - 新增 `Design Rationale and Attack Overview`，明确三项设计选择：结构恢复代替逐字恢复、matched probes 用于显化行为、executable reconstruction 用于避免只看文字重合。
   - 新增论文内生的五阶段流程图：client protection -> server view -> structured recovery -> executable prompt -> continuous evaluation。

5. **Recovery model 描述不清楚**
   - 明确训练数据、测试输入、四类证据、六字段 JSON 输出、字段拼接顺序、80/20 split、48 个 QLoRA adapters、Qwen3-32B 配置和 greedy decoding 设置。
   - 明确恢复模型在测试时不接收原始 prompt；当前缓存的原始输出仅作为 protected-system output 的 high-utility proxy。

6. **如何解释约 0.4 的 Utility F1**
   - 明确 0.433 表示六个 matched probes 上输出 Token F1 的平均值，不表示“43.3% 的 prompt 被成功恢复”。
   - 不再宣称某个固定数值天然代表“sufficient leakage”；结果被解释为当前 observable view 下的连续 partial reconstruction 程度。

7. **放弃 Success@0.50**
   - 摘要、算法、指标、结果、图注、局限性和结论均只保留 Text F1、ROUGE-L F1 和 Utility F1。
   - 新增删除阈值的科学理由：不同类别的 generic-output floor 和 self-similarity 不同，固定 cutoff 未经统一校准。

8. **热力图 0.495 不可读**
   - 重绘热力图，根据单元格亮度自动使用黑色或白色数字；0.495 等深色格使用白字。
   - 将热力图改为双栏宽图，提高方法名、类别名和三位小数标注的可读性。
   - 同时重绘平均结果图，完全移除 Success@0.50 系列，只展示三个连续指标。

9. **Discussion 小标题不合适**
   - 将 `Design Lessons` 改为 `Toward Behavior-Aware Prompt Protection`，并把内容改写为更明确的防御评估与设计方向。

10. **缺少集中说明限制条件**
    - 新增 `Limitations`，集中说明：八种方法不是完整端到端复现、行为 trace 使用 proxy、metadata 可能泄漏、证据与评估 probes 重叠、targets 自动生成、Utility F1 仅为词汇重合、当前只有一个 split seed。

## 修订稿的主结论边界

修订稿支持的结论是：在统一的、包含 protected artifact、metadata 和 matched behavior traces 的 observable interface 下，八种 mechanism-inspired transformations 均存在不同程度的文本和功能结构恢复。

修订稿不支持以下更强结论：

- 八篇原始防御论文的正式隐私保证已经被推翻；
- 当前结果是 query-disjoint 的泛化攻击；
- 行为 traces 单独就足以完成恢复；
- Utility F1 超过任意固定阈值即可定义为成功；
- 0.4 左右的 Utility F1 可以直接换算为任务成功率或安全失效率。

## 产出文件

- `AnonymousSubmission2027_revised.tex`：合并批注后的完整修订稿，不覆盖原文件。
- `figures/privacybench8_method_average_revised.png`：仅包含 Text F1、ROUGE-L F1、Utility F1 的平均结果图。
- `figures/privacybench8_utility_heatmap_revised.png`：修正数字对比度的双栏热力图。
- `make_revised_result_figures_pillow.py`：两张修订图的可复现生成脚本。
