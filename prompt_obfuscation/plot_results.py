"""
Plot reproduction results for the Prompt Obfuscation paper (USENIX Security 2025).
Generates 4 figures summarizing all 3 attacks and obfuscation quality.
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.dpi'] = 150

RESULTS_DIR = Path("precomputed_results/soft_pirate_truthfulqa_full")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load results
with open("reproduction_results.json") as f:
    results = json.load(f)

# Color palette
COLORS = {
    'obfuscated': '#2196F3',    # blue
    'conventional': '#F44336',   # red
    'blank': '#9E9E9E',         # gray
    'original': '#4CAF50',      # green
    'euclidean': '#FF9800',     # orange
    'cosine': '#9C27B0',        # purple
    'random': '#607D8B',        # blue-gray
    'soft': '#00BCD4',          # cyan
    'hard': '#E91E63',          # pink
}

# ==================== Figure 1: Attack 1 - Prompt Injection ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1a: Extraction success count
ax = axes[0]
categories = ['Exact Match', 'Approximate\n(ROUGE-L ≥ 0.9)']
obf_vals = [0, 0]
conv_vals = [33, 41]
x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, conv_vals, width, label='Conventional Prompt', color=COLORS['conventional'], edgecolor='white')
bars2 = ax.bar(x + width/2, obf_vals, width, label='Obfuscated Prompt', color=COLORS['obfuscated'], edgecolor='white')
ax.set_ylabel('Successful Extractions (out of 105)')
ax.set_title('(a) Prompt Extraction Success Rate')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylim(0, 55)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

# 1b: Average recall
ax = axes[1]
categories = ['Conventional\nPrompt', 'Obfuscated\nPrompt']
values = [0.4985, 0.1293]
colors = [COLORS['conventional'], COLORS['obfuscated']]
bars = ax.bar(categories, values, color=colors, edgecolor='white', width=0.5)
ax.set_ylabel('Average ROUGE-L Recall')
ax.set_title('(b) Average Extraction Recall')
ax.set_ylim(0, 0.7)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

fig.suptitle('Attack 1: Prompt Injection (Black-Box) — Table 7', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'attack1_prompt_injection.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: figures/attack1_prompt_injection.png")

# ==================== Figure 2: Attack 2 - Token Space Projection ====================
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Levenshtein', 'Jaccard', 'LCS', 'Cosine Sim.']
euclidean_vals = [0.1622, 0.0641, 0.1959, 0.3867]
cosine_vals = [0.1622, 0.0641, 0.1959, 0.3867]
random_vals = [0.1149, 0.0000, 0.1216, 0.2685]

x = np.arange(len(metrics))
width = 0.25
bars1 = ax.bar(x - width, euclidean_vals, width, label='Euclidean Projection', color=COLORS['euclidean'], edgecolor='white')
bars2 = ax.bar(x, cosine_vals, width, label='Cosine Projection', color=COLORS['cosine'], edgecolor='white')
bars3 = ax.bar(x + width, random_vals, width, label='Random Baseline', color=COLORS['random'], edgecolor='white')

ax.set_ylabel('Similarity Score')
ax.set_title('Attack 2: Token Space Projection (White-Box) — Table 8', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 0.55)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.008, f'{h:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'attack2_token_projection.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: figures/attack2_token_projection.png")

# ==================== Figure 3: Attack 3 - Fluency Optimization ====================
fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['Levenshtein', 'Jaccard', 'LCS', 'Cosine Sim.']
# Paper's precomputed results
soft_paper = [0.2365, 0.0682, 0.2838, 0.2317]
hard_paper = [0.2297, 0.0839, 0.2568, 0.4560]
# Our reproduced results (Soft: dataset=200, iter=3)
soft_repro = [0.1351, 0.0511, 0.1689, 0.3553]
random_vals = [0.1149, 0.0000, 0.1216, 0.2685]

x = np.arange(len(metrics))
width = 0.18
bars1 = ax.bar(x - 1.5*width, soft_paper, width, label='Soft Deobf. (Paper, Table 9)', color=COLORS['soft'], edgecolor='white')
bars2 = ax.bar(x - 0.5*width, soft_repro, width, label='Soft Deobf. (Reproduced)', color='#26C6DA', edgecolor='white', hatch='//')
bars3 = ax.bar(x + 0.5*width, hard_paper, width, label='Hard Deobf. (Paper, Table 10)', color=COLORS['hard'], edgecolor='white')
bars4 = ax.bar(x + 1.5*width, random_vals, width, label='Random Baseline', color=COLORS['random'], edgecolor='white')

ax.set_ylabel('Similarity to Original Prompt')
ax.set_title('Attack 3: Fluency Optimization (White-Box) — Tables 9 & 10\n(Reproduced Soft: dataset=200, iter=3)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0, 0.6)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.008, f'{h:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'attack3_fluency_optimization.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: figures/attack3_fluency_optimization.png")

# ==================== Figure 4: Obfuscation Quality (Output Similarity) ====================
fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'BERTScore', 'Cosine Sim.']
obf_vals = [49.20/100, 0.5846, 0.3551, 0.4606, 0.5170, 0.9122, 0.8476]
blank_vals = [19.11/100, 0.3719, 0.1554, 0.2615, 0.3059, 0.8572, 0.7817]
orig_vals = [48.47/100, 0.5778, 0.3496, 0.4544, 0.5111, 0.9122, 0.8442]

x = np.arange(len(metrics))
width = 0.25
bars1 = ax.bar(x - width, obf_vals, width, label='Obfuscated vs Original', color=COLORS['obfuscated'], edgecolor='white')
bars2 = ax.bar(x, orig_vals, width, label='Original (seed 43) vs Original', color=COLORS['original'], edgecolor='white')
bars3 = ax.bar(x + width, blank_vals, width, label='Blank vs Original', color=COLORS['blank'], edgecolor='white')

ax.set_ylabel('Similarity Score')
ax.set_title('Obfuscation Quality: Output Similarity — Table 4', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.legend(loc='upper left')
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'obfuscation_quality.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: figures/obfuscation_quality.png")

# ==================== Figure 5: Summary Overview (all attacks) ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a: Attack 1 summary
ax = axes[0, 0]
labels = ['Conventional', 'Obfuscated']
exact = [33, 0]
approx = [41, 0]
x = np.arange(len(labels))
width = 0.3
ax.bar(x - width/2, exact, width, label='Exact Match', color='#EF5350')
ax.bar(x + width/2, approx, width, label='Approx. (ROUGE ≥ 0.9)', color='#EF9A9A')
ax.set_ylabel('Successful Extractions / 105')
ax.set_title('Attack 1: Prompt Injection\n(Black-Box)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(fontsize=9)
ax.set_ylim(0, 55)
# Add value labels
for i, (e, a) in enumerate(zip(exact, approx)):
    ax.text(i - width/2, e + 1, str(e), ha='center', fontweight='bold', fontsize=11)
    ax.text(i + width/2, a + 1, str(a), ha='center', fontweight='bold', fontsize=11)

# 5b: Attack 2 summary
ax = axes[0, 1]
metrics_short = ['Lev.', 'Jac.', 'LCS', 'Cos.']
eucl = [0.1622, 0.0641, 0.1959, 0.3867]
rand = [0.1149, 0.0000, 0.1216, 0.2685]
x = np.arange(len(metrics_short))
width = 0.3
ax.bar(x - width/2, eucl, width, label='Projection', color=COLORS['euclidean'])
ax.bar(x + width/2, rand, width, label='Random', color=COLORS['random'])
ax.set_ylabel('Similarity to Original')
ax.set_title('Attack 2: Token Projection\n(White-Box)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_short)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.55)

# 5c: Attack 3 summary
ax = axes[1, 0]
soft = [0.2365, 0.0682, 0.2838, 0.2317]
soft_r = [0.1351, 0.0511, 0.1689, 0.3553]
hard = [0.2297, 0.0839, 0.2568, 0.4560]
rand = [0.1149, 0.0000, 0.1216, 0.2685]
x = np.arange(len(metrics_short))
width = 0.18
ax.bar(x - 1.5*width, soft, width, label='Soft (Paper)', color=COLORS['soft'])
ax.bar(x - 0.5*width, soft_r, width, label='Soft (Repro.)', color='#26C6DA', hatch='//')
ax.bar(x + 0.5*width, hard, width, label='Hard (Paper)', color=COLORS['hard'])
ax.bar(x + 1.5*width, rand, width, label='Random', color=COLORS['random'])
ax.set_ylabel('Similarity to Original')
ax.set_title('Attack 3: Fluency Optimization\n(White-Box)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_short)
ax.legend(fontsize=8)
ax.set_ylim(0, 0.6)

# 5d: Obfuscation quality
ax = axes[1, 1]
metrics_q = ['BLEU', 'ROUGE-L', 'METEOR', 'BERTSc.', 'Cosine']
obf_q = [49.20/100, 0.4606, 0.5170, 0.9122, 0.8476]
blank_q = [19.11/100, 0.2615, 0.3059, 0.8572, 0.7817]
orig_q = [48.47/100, 0.4544, 0.5111, 0.9122, 0.8442]
x = np.arange(len(metrics_q))
width = 0.22
ax.bar(x - width, obf_q, width, label='Obfuscated', color=COLORS['obfuscated'])
ax.bar(x, orig_q, width, label='Orig. (seed 43)', color=COLORS['original'])
ax.bar(x + width, blank_q, width, label='Blank', color=COLORS['blank'])
ax.set_ylabel('Output Similarity')
ax.set_title('Obfuscation Quality\n(Output Preservation)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_q, fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)

fig.suptitle('Reproduction Results: Prompt Obfuscation for LLMs (USENIX Security 2025)\n'
             'Model: Llama-3.1-8B-Instruct (4-bit) | Dataset: TruthfulQA | Style: Pirate',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'summary_all_attacks.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: figures/summary_all_attacks.png")

print("\nAll figures saved to figures/ directory.")
