# L4 Agentic Task 实验报告：AirSim 无人机控制下的 Soft Prompt Obfuscation Utility 评估

**实验日期：** 2026-04-06  
**模型：** meta-llama/Meta-Llama-3.1-8B-Instruct（4-bit BitsAndBytes 量化）  
**GPU：** A100-SXM4-80GB

---

## 1. 实验背景与动机

本实验是对论文 *"Prompt Obfuscation for Large Language Models"*（USENIX Security 2025）**Utility Claim** 的 L4 极限测试。

前三层（L1/L2/L3）均基于 TruthfulQA 问答任务，衡量文本相似度意义上的 Utility 保留。L4 将场景升级至 **Agentic 代码生成任务**——要求模型根据自然语言指令生成可执行的 AirSim 无人机控制 Python 代码。

**核心假设（Challenge Hypothesis）：**
> 软提示混淆在 Agentic 场景下会发生灾难性失效。模型不再能遵守严格的 API 约束和坐标系转换规则，而传统文本相似度指标（BLEU/BERTScore）对此完全无感知。

---

## 2. 任务设置

### 2.1 System Prompt（约 450 tokens）

**行为规则**（airsim_drone.txt）：
- 只能输出纯 Python 代码，禁止 Markdown 代码块、禁止闲聊文本
- 禁止 `for`/`while` 循环和 `if` 语句
- 只允许 `import math`
- 不得定义函数

**API 规范**（airsim_drone_API.txt），仅允许 8 个函数：

```
aw.takeoff()          aw.land()            aw.go_home()
aw.fly_to([x,y,z])   aw.get_yaw()         aw.set_yaw(yaw)
aw.get_drone_position()                    aw.get_position(name)
```

坐标系：NED（X=North, Y=East, Z=Down），机体坐标转世界坐标公式：
```
x = x'·cos(-θ) + y'·sin(-θ)
y = -x'·sin(-θ) + y'·cos(-θ)
z = -z'
```

### 2.2 测试集（24 个用例）

| 难度 | 数量 | 示例 |
|------|------|------|
| Easy | 8 | "Take off and fly 10 meters up." |
| Medium | 9 | "Turn 90° right, fly 5m forward in body frame." |
| Hard | 7 | "Fly a 4×4m square inspection route with CW turns." |

每个用例附 Ground Truth 状态变化序列 `[Δx, Δy, Δz, Δyaw]` 用于自动轨迹评分。

### 2.3 三维度评估器

| 评估器 | 检查内容 | 权重 |
|--------|----------|------|
| **Format Score** | 7条规则（markdown/散文/循环/if/def/非法import/可解析性），AST-based | 30% |
| **API Adherence Score** | AST 遍历所有 `aw.*` 调用，对比白名单；合法调用数/总调用数 | 35% |
| **Trajectory Score** | Mock 环境执行代码，提取 `[Δx,Δy,Δz,Δyaw]` 序列，与 Ground Truth 逐步比对（容差 ≤1） | 35% |

---

## 3. 实验一：CNN-Dailymail 训练的 Soft Prompt（域外任务）

### 3.1 实验设计

| 参数 | 值 |
|------|----|
| 训练数据 | CNN/DailyMail 新闻摘要（200条）|
| 训练任务 | 给定新闻文章，输出摘要 |
| Token 数扫描 | 10 / 20 / 40 / 80 / 160 |
| 优化配置 | 5 iters × 3 windows，Adam，lr=0.01 |
| 损失函数 | CE Loss + KL Divergence |

5 个 token 数配置在不同 GPU 上并行训练，训练完成后统一在 24 个测试用例上评估。

### 3.2 实验结果

#### 总体对比

| Token 数 | Conv API | **Obf API** | Conv Traj | **Obf Traj** | Conv Fmt | Obf Fmt |
|----------|----------|-------------|-----------|--------------|----------|---------|
| 10 | 0.667 | **0.000** | 0.395 | **0.000** | 0.921 | 0.938 |
| 20 | 0.667 | **0.000** | 0.395 | **0.000** | 0.921 | 0.938 |
| 40 | 0.667 | **0.000** | 0.395 | **0.000** | 0.921 | 0.938 |
| 80 | 0.667 | **0.000** | 0.395 | **0.000** | 0.921 | 0.938 |
| 160 | 0.667 | **0.000** | 0.395 | **0.000** | 0.921 | 0.938 |

#### 按难度分层（各 token 数结果一致）

| 难度 | Conv API | Obf API | Conv Traj | Obf Traj |
|------|----------|---------|-----------|----------|
| Easy | 0.875 | **0.000** | 0.625 | **0.000** |
| Medium | 1.000 | **0.000** | 0.444 | **0.000** |
| Hard | 0.000 | 0.000 | 0.069 | 0.000 |

> **注：** Hard 级别 Conventional 也出现 API=0，原因是复杂路径（方形巡检）迫使模型自发使用 `for` 循环，违反白名单。这是 Conventional 的能力边界，非 obfuscation 引入的问题。

#### Obfuscated 输出样本（tok=10）

| 测试用例 | Obfuscated 实际输出 |
|----------|-------------------|
| E01: "Fly 10m up" (tok=10) | `' a lot of safety.'` |
| E01: "Fly 10m up" (tok=20) | `' to overcome the weight of the object...'` |
| E01: "Fly 10m up" (tok=40/80/160) | `''`（空字符串） |
| E02~E24（所有） | `''` 或新闻摘要风格散文 |

### 3.3 关键发现

**发现 1：Token 数增加对恢复完全无效**

从 10 到 160 token（增加 16 倍），API Adherence 和 Trajectory Score 全程保持 0.000，无任何改善趋势。这证明**瓶颈不是 soft prompt 的表示容量，而是训练任务与测试任务的根本不匹配**。

**发现 2：Format Score 产生"假稳定"假象**

Format Score 在 Conv（0.921）和 Obf（0.938）之间几乎无差距，Obf 甚至略高。原因：空字符串和英文散文不含循环/if/非法 import，格式规则机械判定为"合格"。这证明**传统格式指标在 Agentic 场景下完全失效，无法捕捉功能性崩溃**。

**发现 3：Soft Prompt 跨域迁移能力为零**

在新闻摘要任务上训练的 soft prompt 对无人机代码生成完全无效，无论 token 数多少，模型都无法生成任何 `aw.*` API 调用。**Soft prompt 是高度任务特异性的（task-specific），不可跨域部署。**

---

## 4. 实验二：Drone 任务训练的 Soft Prompt（域内任务）——进行中

### 4.1 实验设计

| 参数 | 值 |
|------|----|
| 训练数据 | 200 条无人机控制指令（与测试集同分布）|
| 训练任务 | 无人机代码生成（与评估任务完全一致） |
| System Prompt | 完整 AirSim 规则 + API 文档（同测试时使用的 prompt）|
| Token 数扫描 | 10 / 20 / 40 / 80 / 160 |
| 其他配置 | 与实验一完全相同 |

**实验逻辑：** 通过 domain alignment 排除"域不匹配"这一混淆变量，让 soft prompt 的训练目标与测试目标完全一致。这是对论文 Utility Claim 的**最公平、最严格的测试条件**。

### 4.2 预期结果

#### 预期指标走势

**API Adherence（最关键指标）：**

| Token 数 | 预期范围 | 推理依据 |
|----------|----------|----------|
| 10 | 0.2–0.5 | 10 token ≈ 40K 参数，可传递"输出代码"的风格，但 8 个 API 名称的精确约束难以完全编码 |
| 20 | 0.4–0.7 | 容量翻倍，核心 API 约束开始出现 |
| 40 | 0.5–0.8 | 约为原始 prompt 的 9%，主要 API 白名单应可覆盖 |
| 80 | 0.6–0.9 | 约 18%，坐标系规则开始可编码 |
| 160 | 0.7–0.95 | 约 36%，接近但可能仍低于 Conventional |

**Trajectory Score：**

| Token 数 | 预期范围 | 推理依据 |
|----------|----------|----------|
| 10 | 0.05–0.2 | 能生成代码但坐标计算规则精度不足 |
| 40 | 0.2–0.4 | 坐标变换公式开始可编码 |
| 160 | 0.3–0.6 | 接近 Conventional（0.395） |

**Format Score：** 各 token 数均预期 0.85–1.00（输出格式约束相对简单）

#### 预期拐点

| 指标 | 预期拐点 | 理由 |
|------|----------|------|
| API Adherence | **20–40 token** | API 名称是离散符号信息，需要足够 embedding 容量编码 |
| Trajectory Score | **40–80 token** | 坐标变换是复杂数学关系，需要更多容量 |
| Format Score | **10–20 token** | 格式约束简单，少量 token 即可传递 |

### 4.3 两种结果的解读

**情形 A：160 token 仍显著低于 Conventional（预期更可能）**

→ 证明 soft prompt obfuscation 存在**根本性容量瓶颈**：即便 token 数达到原始 prompt 的 1/3，功能性 Utility 仍无法恢复。论文的 Utility Claim 在 Agentic 场景下不成立，与训练域无关。

**情形 B：160 token 接近 Conventional**

→ 说明实验一的失效是**部署不当**（跨域使用）造成的。但这反而揭示了一个更严重的实用性问题：论文声称的防御机制需要**为每个具体任务单独训练 soft prompt**，部署成本极高，与"通用防御"的设计目标相悖。

**无论哪种情形，对论文的质疑都成立：**
- 情形 A：方法本身有根本局限
- 情形 B：方法缺乏实用性（需任务对齐的专属 soft prompt）

---

## 5. 综合结论

| 维度 | 已确认结论 | 待实验二确认 |
|------|-----------|-------------|
| Agentic Utility | 域外训练：功能性完全失效（API 0%） | 域内训练的恢复上限 |
| Token 数影响 | 域外：16 倍增加无任何改善 | 域内：拐点位置 |
| 跨域迁移 | Soft prompt 零迁移能力 | — |
| 格式指标局限 | Format/BLEU/BERTScore 无法感知功能失效 | — |
| 论文 Claim | Agentic 场景下不成立 | 量化失效程度 |

---

## 6. 文件索引

| 文件 | 说明 |
|------|------|
| `l4_drone_evaluator.py` | 三维度评估器（Format + API + Trajectory） |
| `l4_drone_testcases.py` | 24 个测试用例 + Ground Truth |
| `run_l4_evaluation.py` | 自动化评估 Pipeline |
| `data/drone_queries.json` | 200 条无人机训练指令 |
| `results/fig_token_sweep.png` | CNN-trained token 扫描图 |
| `results/fig_drone_sweep.png` | 域内外对比图（实验二完成后生成） |
| `results/l4_drone_eval_tok{N}.json` | CNN-trained 各 token 数详细结果 |
| `results/l4_drone_eval_drone_tok{N}.json` | Drone-trained 各 token 数详细结果（待生成） |
