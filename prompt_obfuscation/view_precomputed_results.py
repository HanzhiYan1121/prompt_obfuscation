"""
查看预计算结果的脚本 - 不需要GPU，不需要下载模型
用于快速验证论文中的实验结果
"""
import json
import torch
from pathlib import Path

RESULTS_DIR = Path("precomputed_results/soft_pirate_truthfulqa_full")

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ========== 1. 基本信息 ==========
print_separator("实验配置 (params.json)")
with open(RESULTS_DIR / "params.json") as f:
    params = json.load(f)
for k, v in params.items():
    print(f"  {k}: {v}")

# ========== 2. 混淆效果 (Table 4 对应) ==========
print_separator("混淆效果评估 - Output Similarity (对应论文 Table 4)")

print("\n  [obfuscated prompt vs original prompt 的输出相似度]")
with open(RESULTS_DIR / "best_candidate_scores.json") as f:
    obf_scores = json.load(f)
for k, v in obf_scores.items():
    print(f"    {k}: {v:.4f}")

print("\n  [blank prompt vs original prompt 的输出相似度 (基线)]")
with open(RESULTS_DIR / "blank_output_scores.json") as f:
    blank_scores = json.load(f)
for k, v in blank_scores.items():
    print(f"    {k}: {v:.4f}")

print("\n  [original prompt (seed 43) vs original prompt 的输出相似度 (参考)]")
with open(RESULTS_DIR / "original_output_scores.json") as f:
    orig_scores = json.load(f)
for k, v in orig_scores.items():
    print(f"    {k}: {v:.4f}")

# ========== 3. 攻击1: Prompt Injection (Table 7) ==========
print_separator("攻击1: Prompt Injection 结果 (对应论文 Table 7)")

try:
    with open(RESULTS_DIR / "successful_outputs_conventional.json") as f:
        conv_extractions = json.load(f)
    print(f"  原始prompt被成功提取次数: {len(conv_extractions)} / 105")
    if conv_extractions:
        print(f"  示例提取结果 (前100字):")
        first = conv_extractions[0]
        if isinstance(first, dict):
            text = first.get('output', first.get('text', str(first)))
        else:
            text = str(first)
        print(f"    {text[:100]}...")
except FileNotFoundError:
    print("  [未找到conventional提取结果文件]")

try:
    with open(RESULTS_DIR / "successful_outputs_obfuscated.json") as f:
        obf_extractions = json.load(f)
    print(f"\n  混淆prompt被成功提取次数: {len(obf_extractions)} / 105")
    if obf_extractions:
        print(f"  提取内容: {obf_extractions}")
    else:
        print(f"  >> 混淆成功! 无法提取任何系统prompt <<")
except FileNotFoundError:
    print("  [未找到obfuscated提取结果文件]")

# ========== 4. 攻击2: Token Space Projection (Table 8) ==========
print_separator("攻击2: Token Space Projection 结果 (对应论文 Table 8)")

for metric_file, label in [
    ("euclidean_sys_prompt_scores.json", "欧氏距离投影"),
    ("cosine_sys_prompt_scores.json", "余弦相似度投影"),
    ("random_sys_prompt_scores.json", "随机基线"),
]:
    try:
        with open(RESULTS_DIR / metric_file) as f:
            scores = json.load(f)
        print(f"\n  [{label}] vs 原始prompt:")
        for k, v in scores.items():
            print(f"    {k}: {v:.4f}")
    except FileNotFoundError:
        print(f"\n  [{label}]: 文件未找到")

# ========== 5. 攻击3: Fluency Optimization (Table 9 & 10) ==========
print_separator("攻击3: Fluency Optimization 结果 (对应论文 Table 9 & 10)")

for score_file, label in [
    ("best_deobf_sys_prompt_soft_scores.json", "Soft去混淆 (Table 9)"),
    ("best_deobf_sys_prompt_hard_scores.json", "Hard去混淆 (Table 10)"),
]:
    try:
        with open(RESULTS_DIR / score_file) as f:
            scores = json.load(f)
        print(f"\n  [{label}] vs 原始prompt:")
        for k, v in scores.items():
            print(f"    {k}: {v:.4f}")
    except FileNotFoundError:
        print(f"\n  [{label}]: 文件未找到")

# 尝试解码去混淆的prompt
print_separator("去混淆的Prompt示例")
print("  (需要tokenizer才能解码，这里显示token IDs)")

for pt_file, label in [
    ("best_deobf_sys_prompt_soft.pt", "Soft去混淆结果"),
    ("best_deobf_sys_prompt_hard.pt", "Hard去混淆结果"),
]:
    try:
        tensor = torch.load(RESULTS_DIR / pt_file, weights_only=True, map_location='cpu')
        print(f"\n  [{label}]")
        print(f"    Token IDs: {tensor.tolist()}")
        print(f"    Shape: {tensor.shape}")
    except FileNotFoundError:
        print(f"\n  [{label}]: 文件未找到")

print(f"\n{'='*60}")
print("  原始系统prompt:")
print(f"  {params['system_prompt']}")
print(f"{'='*60}")
