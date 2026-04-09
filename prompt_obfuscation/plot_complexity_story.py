"""
Two-panel figure:
  A: Baseline band + complex L1/L2/L3 bars for SacreBLEU
  B: Multi-metric line plot for complex prompts, normalized to baseline mean
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

with open("results/complexity_vs_utility.json") as f:
    rows = json.load(f)

baselines = [r for r in rows if r["type"] == "baseline"]
complex_  = [r for r in rows if r["type"] == "complex"]   # L1, L2, L3

BLUE  = "#2471a3"
LBLUE = "#aed6f1"
RED   = "#c0392b"
GREY  = "#7f8c8d"

METRICS = [
    ("cosine_similarity", "Cosine Sim."),
    ("bertscore",         "BERTScore"),
    ("rouge1",            "ROUGE-1"),
    ("sacrebleu",         "SacreBLEU"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle(
    "Soft Prompt Obfuscation: Does It Fail for Complex Prompts?\n"
    "Llama-3.1-8B-Instruct · 10-token obfuscated embedding · TruthfulQA",
    fontsize=12, fontweight="bold", y=1.03
)

# ── Panel A: SacreBLEU — baseline band + L1/L2/L3 bars ─────────────────
b_sbleu  = np.array([r["sacrebleu"] for r in baselines])
b_mean, b_std = b_sbleu.mean(), b_sbleu.std()

cx_labels = ["Complex L1\n(79 tokens)", "Complex L2\n(109 tokens)", "Complex L3\n(166 tokens)"]
cy_sbleu  = [r["sacrebleu"] for r in complex_]

x = np.arange(3)
bars = ax1.bar(x, cy_sbleu, color=RED, width=0.45,
               edgecolor="white", linewidth=0.8, zorder=3, label="Complex prompts (ours)")

# baseline band
ax1.axhspan(b_mean - b_std, b_mean + b_std, color=LBLUE, alpha=0.35, zorder=1,
            label=f"Baseline mean ± 1σ  ({b_mean:.1f} ± {b_std:.1f})")
ax1.axhline(b_mean, color=BLUE, linestyle="--", linewidth=1.5, alpha=0.9, zorder=2)

# value labels
for bar, val in zip(bars, cy_sbleu):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=RED)

# arrow annotation for trend
ax1.annotate("", xy=(2, cy_sbleu[2] + 0.5), xytext=(0, cy_sbleu[0] - 0.5),
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.8, connectionstyle="arc3,rad=-0.1"))
ax1.text(1.5, min(cy_sbleu) - 3.5,
         f"−{cy_sbleu[0]-cy_sbleu[2]:.1f} pts\n({(cy_sbleu[0]-cy_sbleu[2])/cy_sbleu[0]*100:.0f}% drop)",
         ha="center", fontsize=9, color=RED, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(cx_labels, fontsize=10)
ax1.set_ylabel("SacreBLEU", fontsize=11)
ax1.set_ylim(0, b_mean + b_std + 12)
ax1.set_title("(A)  SacreBLEU Degrades with Prompt Complexity", fontsize=11, fontweight="bold", pad=8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
ax1.set_axisbelow(True)
ax1.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#cccccc")

# ── Panel B: All metrics normalized to baseline mean ────────────────────
cx_tokens = [r["prompt_tokens"] for r in complex_]

for metric, label in METRICS:
    b_mean_m = np.mean([r[metric] for r in baselines])
    cy_norm  = [r[metric] / b_mean_m for r in complex_]
    ax2.plot(cx_tokens, cy_norm, marker="o", markersize=7, linewidth=2, label=label)

ax2.axhline(1.0, color=GREY, linestyle="--", linewidth=1.3, alpha=0.8, label="Baseline mean (norm.)")
ax2.fill_between([70, 175], 0.94, 1.06, color=GREY, alpha=0.08)

ax2.set_xlabel("Prompt Token Count (complex prompts)", fontsize=11)
ax2.set_ylabel("Utility relative to baseline mean", fontsize=11)
ax2.set_title("(B)  Relative Utility across Metrics\n(1.0 = baseline mean)", fontsize=11, fontweight="bold", pad=8)
ax2.set_xticks(cx_tokens)
ax2.set_xticklabels(["L1\n(79t)", "L2\n(109t)", "L3\n(166t)"], fontsize=10)
ax2.set_ylim(0.7, 1.08)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
ax2.set_axisbelow(True)
ax2.legend(fontsize=9, loc="lower left", framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/complexity_story.png", dpi=160, bbox_inches="tight")
print("Saved: figures/complexity_story.png")
