"""
Plot: Prompt Complexity vs Obfuscation Utility
X = prompt token count, Y = utility metrics
Baseline styles (blue) vs Complex prompts (red)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

with open("results/complexity_vs_utility.json") as f:
    rows = json.load(f)

baselines = [r for r in rows if r["type"] == "baseline"]
complex_  = [r for r in rows if r["type"] == "complex"]

METRICS = [
    ("cosine_similarity", "Cosine Similarity"),
    ("bertscore",         "BERTScore"),
    ("rouge1",            "ROUGE-1"),
    ("sacrebleu",         "SacreBLEU"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(
    "Soft Prompt Obfuscation: Prompt Complexity vs Utility\n"
    "(10-token obfuscated embedding, Llama-3.1-8B-Instruct, TruthfulQA)",
    fontsize=13, fontweight="bold", y=1.01
)

BLUE   = "#2980b9"
RED    = "#c0392b"
GREY   = "#7f8c8d"
LBLUE  = "#aed6f1"
LRED   = "#f1948a"

for ax, (metric, label) in zip(axes.flat, METRICS):
    bx = [r["prompt_tokens"]       for r in baselines]
    by = [r[metric]                for r in baselines]
    bn = [r["name"]                for r in baselines]

    cx = [r["prompt_tokens"]       for r in complex_]
    cy = [r[metric]                for r in complex_]
    cn = [r["name"]                for r in complex_]

    # scatter
    ax.scatter(bx, by, color=BLUE,  s=80, zorder=4, label="Baseline styles (paper)")
    ax.scatter(cx, cy, color=RED,   s=100, marker="^", zorder=5, label="Complex prompts (ours)")

    # labels for baselines
    for x, y, n in zip(bx, by, bn):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=7.5, color=GREY)

    # labels for complex
    for x, y, n in zip(cx, cy, cn):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, -10),
                    fontsize=8, color=RED, fontweight="bold")

    # trend line over ALL points
    all_x = bx + cx
    all_y = by + cy
    slope, intercept, r, p, _ = stats.linregress(all_x, all_y)
    xfit = np.linspace(min(all_x) - 5, max(all_x) + 5, 200)
    yfit = slope * xfit + intercept
    ax.plot(xfit, yfit, "--", color=GREY, linewidth=1.4, alpha=0.7,
            label=f"Trend (r={r:.2f}, p={p:.3f})")

    ax.set_xlabel("Prompt Token Count", fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, framealpha=0.85)

plt.tight_layout()
out = "figures/complexity_vs_utility.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
