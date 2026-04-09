import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = "precomputed_results"
STYLES = ["pirate", "robot"]
METRICS = ["levenshtein", "jaccard", "lcs", "cosine_similarity"]
METRIC_LABELS = ["Levenshtein", "Jaccard", "LCS", "Cosine\nSimilarity"]

def load(style, filename):
    path = f"{BASE}/soft_{style}_truthfulqa_full/{filename}"
    with open(path) as f:
        return json.load(f)

# Paper reference data (Table 9 & 10), pirate style only
PAPER_PIRATE = {
    "Soft\n(Paper)": {
        "levenshtein": 0.2365, "jaccard": 0.0682,
        "lcs": 0.2838, "cosine_similarity": 0.2317
    },
    "Hard\n(Paper)": {
        "levenshtein": 0.2297, "jaccard": 0.0839,
        "lcs": 0.2568, "cosine_similarity": 0.4560
    },
}

# Load our experimental data
data = {}
for style in STYLES:
    data[style] = {
        "Random":       load(style, "random_sys_prompt_scores.json"),
        "Soft\n(Ours)": load(style, "best_deobf_sys_prompt_soft_scores.json"),
        "Hard\n(Ours)": load(style, "best_deobf_sys_prompt_hard_scores.json"),
    }

# Merge paper reference into pirate data
for key, vals in PAPER_PIRATE.items():
    data["pirate"][key] = vals

# Conditions for each style
PIRATE_CONDS = ["Random", "Soft\n(Ours)", "Soft\n(Paper)", "Hard\n(Ours)", "Hard\n(Paper)"]
ROBOT_CONDS  = ["Random", "Soft\n(Ours)", "Hard\n(Ours)"]
CONDS_BY_STYLE = {"pirate": PIRATE_CONDS, "robot": ROBOT_CONDS}

# Colors: grey=random, blue tones=soft, red tones=hard
# Ours = solid/darker, Paper = lighter + hatched
COLORS = {
    "Random":        "#95a5a6",
    "Soft\n(Ours)":  "#2980b9",
    "Soft\n(Paper)": "#85c1e9",
    "Hard\n(Ours)":  "#c0392b",
    "Hard\n(Paper)": "#f1948a",
}

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(
    "Attack 3: Fluency Deobfuscation (Table 9 & 10)\n"
    "Ours: iter=5, dataset=800  |  Paper: iter=10, dataset=800  |  "
    "Higher score = attacker recovered more of the original prompt",
    fontsize=12, fontweight="bold", y=1.03
)

for row, style in enumerate(STYLES):
    conds = CONDS_BY_STYLE[style]
    x = np.arange(len(conds))

    for col, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = axes[row][col]
        values = [data[style][c][metric] for c in conds]

        for i, (cond, val) in enumerate(zip(conds, values)):
            color = COLORS[cond]
            hatch = "//" if "Paper" in cond else None
            ax.bar(i, val, width=0.6, color=color, edgecolor="white",
                   linewidth=0.8, hatch=hatch, alpha=0.92)
            ax.text(i, val + 0.004,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(conds, fontsize=8.5)
        ax.set_ylim(0, max(values) * 1.38 + 0.025)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        if col == 0:
            note = " (w/ Paper ref.)" if style == "pirate" else " (no paper ref.)"
            row_label = ("Pirate" if style == "pirate" else "Robot") + note
            ax.set_ylabel(f"{row_label}\n\nScore", fontsize=9, fontweight="bold")
        else:
            ax.set_ylabel("")

# Legend
legend_entries = [
    mpatches.Patch(color=COLORS["Random"],        label="Random Baseline"),
    mpatches.Patch(color=COLORS["Soft\n(Ours)"],  label="Soft Deobf.  (Ours,  iter=5)"),
    mpatches.Patch(color=COLORS["Soft\n(Paper)"], label="Soft Deobf.  (Paper, iter=10)", hatch="//"),
    mpatches.Patch(color=COLORS["Hard\n(Ours)"],  label="Hard / GCG  (Ours,  iter=5)"),
    mpatches.Patch(color=COLORS["Hard\n(Paper)"], label="Hard / GCG  (Paper, iter=10)", hatch="//"),
]
fig.legend(handles=legend_entries, loc="lower center", ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.06), frameon=True,
           edgecolor="#cccccc")

plt.tight_layout()
out = "figures/attack3_fluency_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
