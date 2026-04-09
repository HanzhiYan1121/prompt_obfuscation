#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

with open("results/multi_turn_n_degradation.json") as f:
    data = json.load(f)

points = data["data"]
turns  = [p["turn"] for p in points]

C_COSINE = "#2166ac"
C_BERT   = "#4dac26"
C_ROUGE  = "#d7191c"
C_BLEU   = "#984ea3"

METRICS = {
    "cosine_similarity": ("Cosine Sim",  C_COSINE),
    "bertscore":         ("BERTScore",   C_BERT),
    "rouge1":            ("ROUGE-1",     C_ROUGE),
    "sacrebleu":         ("SacreBLEU",  C_BLEU),
}

ref = {k: points[0]["scores"][k] for k in METRICS}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# ── Left: normalised scores ───────────────────────────────────────────────────
ax = axes[0]
for metric, (label, color) in METRICS.items():
    vals = [p["scores"][metric] / ref[metric] for p in points]
    ax.plot(turns, vals, marker="o", markersize=5, color=color, lw=2, label=label)

ax.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6, label="Turn 1 baseline")


ax.set_xlabel("Conversation turn", fontsize=10)
ax.set_ylabel("Normalised score (relative to Turn 1)", fontsize=10)
ax.set_title(f"Multi-Turn Utility ({data['n_turns']} turns, {data['n_groups']} groups)\nNormalised", fontsize=11)
ax.set_xticks(turns)
ax.set_ylim(0.88, 1.12)
ax.legend(fontsize=8.5, loc="lower left")
ax.grid(True, alpha=0.3)

# ── Right: raw scores (dual axis) ─────────────────────────────────────────────
ax2_l = axes[1]
ax2_r = ax2_l.twinx()

for metric, (label, color) in METRICS.items():
    vals = [p["scores"][metric] for p in points]
    if metric == "sacrebleu":
        ax2_r.plot(turns, vals, marker="o", markersize=5, color=color,
                   lw=2, label=label, ls="--")
        ax2_r.set_ylabel("SacreBLEU", fontsize=10, color=C_BLEU)
        ax2_r.tick_params(axis="y", labelcolor=C_BLEU)
    else:
        ax2_l.plot(turns, vals, marker="o", markersize=5, color=color, lw=2, label=label)

ax2_l.set_xlabel("Conversation turn", fontsize=10)
ax2_l.set_ylabel("Cosine Sim / BERTScore / ROUGE-1", fontsize=10)
ax2_l.set_title(f"Multi-Turn Utility ({data['n_turns']} turns, {data['n_groups']} groups)\nRaw scores", fontsize=11)
ax2_l.set_xticks(turns)
ax2_l.set_ylim(0.55, 0.97)
ax2_r.set_ylim(44, 66)
ax2_l.grid(True, alpha=0.3)

lines1, labs1 = ax2_l.get_legend_handles_labels()
lines2, labs2 = ax2_r.get_legend_handles_labels()
ax2_l.legend(lines1 + lines2, labs1 + labs2, fontsize=8.5, loc="upper left")

# Trend line for cosine
cosine_vals = [p["scores"]["cosine_similarity"] for p in points]
z = np.polyfit(turns, cosine_vals, 1)
trend = np.poly1d(z)
ax2_l.plot(turns, trend(turns), color=C_COSINE, lw=1, ls=":", alpha=0.6)

fig.suptitle("Soft Prompt Obfuscation — Multi-Turn Robustness\n(pirate style, TruthfulQA, Llama-3.1-8B-Instruct 4-bit)",
             fontsize=11, y=1.01)

Path("figures").mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig("figures/fig3_multi_turn_n.png", dpi=150, bbox_inches="tight")
print("Saved → figures/fig3_multi_turn_n.png")

# Print table
print(f"\n{'Turn':>5} | {'Cosine':>8} | {'BERTScore':>10} | {'ROUGE-1':>8} | {'SacreBLEU':>10}")
print("-" * 52)
for p in points:
    s = p["scores"]
    print(f"{p['turn']:>5} | {s['cosine_similarity']:>8.4f} | {s['bertscore']:>10.4f} | "
          f"{s['rouge1']:>8.4f} | {s['sacrebleu']:>10.2f}")

# Cosine trend
slope = z[0]
print(f"\nCosine trend: {slope:+.4f} per turn "
      f"({'declining' if slope < 0 else 'rising'})")
