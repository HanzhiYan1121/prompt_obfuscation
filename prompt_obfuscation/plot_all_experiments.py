#!/usr/bin/env python3
"""
Three separate figures:
  Fig 1 — Prompt Complexity vs Utility (complex L1/L2/L3 only)
  Fig 2 — Long Context Degradation
  Fig 3 — Multi-Turn Degradation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── data ──────────────────────────────────────────────────────────────────────
with open("results/complexity_vs_utility.json") as f:
    complexity_data = json.load(f)

with open("results/long_context_degradation.json") as f:
    lc_raw = json.load(f)

with open("results/multi_turn_degradation.json") as f:
    mt_raw = json.load(f)

# ── palette ───────────────────────────────────────────────────────────────────
C_COMPLEX = "#e6550d"
C_COSINE  = "#2166ac"
C_BERT    = "#4dac26"
C_ROUGE   = "#d7191c"
C_BLEU    = "#984ea3"

METRICS = {
    "cosine_similarity": ("Cosine Sim", C_COSINE),
    "bertscore":         ("BERTScore",  C_BERT),
    "rouge1":            ("ROUGE-1",    C_ROUGE),
    "sacrebleu":         ("SacreBLEU", C_BLEU),
}

Path("figures").mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Prompt Complexity (complex only, SacreBLEU vs compression ratio)
# ══════════════════════════════════════════════════════════════════════════════
C_BASELINE = "#9ecae1"

baselines = [d for d in complexity_data if d["type"] == "baseline"]
complexes = [d for d in complexity_data if d["type"] == "complex"]

bleu_base  = [d["sacrebleu"] for d in baselines]
bleu_cplx  = [d["sacrebleu"] for d in complexes]
ratio_cplx = [d["compression_ratio"] for d in complexes]

fig1, ax = plt.subplots(figsize=(6, 4.5))

# Baseline band + mean (no scatter dots)
ax.axhspan(min(bleu_base), max(bleu_base),
           alpha=0.12, color=C_BASELINE, label="Baseline range")
ax.axhline(np.mean(bleu_base), color=C_BASELINE, lw=1.2, ls="--", alpha=0.7,
           label=f"Baseline mean ({np.mean(bleu_base):.1f})")

# Complex points + connecting line
xs = sorted(complexes, key=lambda d: d["compression_ratio"])
ax.plot([d["compression_ratio"] for d in xs],
        [d["sacrebleu"] for d in xs],
        color=C_COMPLEX, lw=1.5, zorder=3)
for d in complexes:
    ax.scatter(d["compression_ratio"], d["sacrebleu"],
               color=C_COMPLEX, s=90, zorder=4, edgecolors="white", linewidths=0.5)
    lbl = d["name"].replace("complex_", "")
    ax.annotate(lbl, (d["compression_ratio"], d["sacrebleu"]),
                textcoords="offset points", xytext=(4, 5),
                fontsize=8.5, color=C_COMPLEX, fontweight="bold")

ax.set_xlabel("Compression ratio (10 / prompt tokens)", fontsize=10)
ax.set_ylabel("SacreBLEU", fontsize=10)
ax.set_title("Prompt Complexity vs Utility", fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_xlim(-0.02, 0.36)
ax.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig("figures/fig1_complexity.png", dpi=150, bbox_inches="tight")
print("Saved → figures/fig1_complexity.png")
plt.close(fig1)

# ── Figure 1b — all 4 metrics, dual axis, prompt tokens x-axis ────────────────
complexes_sorted = sorted(complexes, key=lambda d: d["prompt_tokens"])
tokens = [d["prompt_tokens"] for d in complexes_sorted]
xlabels = [f"{t} tok" for t in tokens]

fig1b, ax = plt.subplots(figsize=(6, 4.5))
ax2 = ax.twinx()

for metric, (label, color) in METRICS.items():
    vals = [d[metric] for d in complexes_sorted]
    base_mean = np.mean([d[metric] for d in baselines])
    if metric == "sacrebleu":
        ax2.plot(tokens, vals, marker="o", markersize=7, color=color,
                 lw=2, label=label, zorder=4)
        ax2.axhline(base_mean, color=color, lw=1, ls="--", alpha=0.5,
                    label=f"Baseline mean ({base_mean:.1f})")
        ax2.set_ylabel("SacreBLEU", fontsize=10, color=C_BLEU)
        ax2.tick_params(axis="y", labelcolor=C_BLEU)
    else:
        ax.plot(tokens, vals, marker="o", markersize=7, color=color,
                lw=2, label=label, zorder=4)
        ax.axhline(base_mean, color=color, lw=1, ls="--", alpha=0.5)

for i, d in enumerate(complexes_sorted):
    lbl = d["name"].replace("complex_", "")
    ax.annotate(lbl, (tokens[i], d["cosine_similarity"]),
                textcoords="offset points", xytext=(5, -12),
                fontsize=9, color="black", fontweight="bold")

ax.set_xlabel("Prompt length (tokens)", fontsize=10)
ax.set_ylabel("Cosine Sim / BERTScore / ROUGE-1", fontsize=10)
ax.set_title("Prompt Complexity vs Utility\n(soft obfuscation, 10-token embedding)", fontsize=11)
ax.set_xticks(tokens)
ax.set_xticklabels(xlabels)
ax.set_ylim(0.48, 0.97)
ax2.set_ylim(30, 55)
ax.grid(True, alpha=0.3)

lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8.5, loc="upper right")

fig1b.tight_layout()
fig1b.savefig("figures/fig1_complexity_metrics.png", dpi=150, bbox_inches="tight")
print("Saved → figures/fig1_complexity_metrics.png")
plt.close(fig1b)

# ── Figure 1c — same as above but WITH paper baseline scatter dots ─────────────
fig1b, ax = plt.subplots(figsize=(6, 4.5))

ax.axhspan(min(bleu_base), max(bleu_base),
           alpha=0.12, color=C_BASELINE, label="Baseline range")
ax.axhline(np.mean(bleu_base), color=C_BASELINE, lw=1.2, ls="--", alpha=0.7,
           label=f"Baseline mean ({np.mean(bleu_base):.1f})")

ratio_base = [d["compression_ratio"] for d in baselines]
ax.scatter(ratio_base, bleu_base, color=C_BASELINE, s=70,
           label="Paper baselines (8 styles)", zorder=3, edgecolors="white", linewidths=0.5)

xs = sorted(complexes, key=lambda d: d["compression_ratio"])
ax.plot([d["compression_ratio"] for d in xs],
        [d["sacrebleu"] for d in xs],
        color=C_COMPLEX, lw=1.5, zorder=3)
for d in complexes:
    ax.scatter(d["compression_ratio"], d["sacrebleu"],
               color=C_COMPLEX, s=90, zorder=4, edgecolors="white", linewidths=0.5)
    lbl = d["name"].replace("complex_", "")
    ax.annotate(lbl, (d["compression_ratio"], d["sacrebleu"]),
                textcoords="offset points", xytext=(4, 5),
                fontsize=8.5, color=C_COMPLEX, fontweight="bold")

ax.set_xlabel("Compression ratio (10 / prompt tokens)", fontsize=10)
ax.set_ylabel("SacreBLEU", fontsize=10)
ax.set_title("Prompt Complexity vs Utility", fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_xlim(-0.02, 0.36)
ax.grid(True, alpha=0.3)

fig1b.tight_layout()
fig1b.savefig("figures/fig1_complexity_with_baselines.png", dpi=150, bbox_inches="tight")
print("Saved → figures/fig1_complexity_with_baselines.png")
plt.close(fig1b)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Long Context
# ══════════════════════════════════════════════════════════════════════════════
lc_points = lc_raw["data"]
ctx_lens  = [p["extra_context_tokens"] for p in lc_points]
ref_lc    = {k: lc_points[0]["scores"][k] for k in METRICS}

fig2, ax = plt.subplots(figsize=(6, 4.5))

for metric, (label, color) in METRICS.items():
    vals = [p["scores"][metric] / ref_lc[metric] for p in lc_points]
    ax.plot(ctx_lens, vals, marker="o", markersize=6, color=color, lw=2, label=label)

# Annotate SacreBLEU raw values
for p in lc_points:
    x   = p["extra_context_tokens"]
    raw = p["scores"]["sacrebleu"]
    ny  = raw / ref_lc["sacrebleu"]
    ax.annotate(f"{raw:.1f}", (x, ny),
                textcoords="offset points", xytext=(0, 7),
                fontsize=7.5, ha="center", color=C_BLEU)

ax.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6, label="ctx=0 baseline")
ax.set_xlabel("Extra context tokens prepended to user message", fontsize=10)
ax.set_ylabel("Normalised score (relative to ctx=0)", fontsize=10)
ax.set_title("Long-Context Robustness\n(pirate style, TruthfulQA)", fontsize=11)
ax.set_xticks(ctx_lens)
ax.set_ylim(0.93, 1.10)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig("figures/fig2_long_context.png", dpi=150, bbox_inches="tight")
print("Saved → figures/fig2_long_context.png")
plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Multi-Turn
# ══════════════════════════════════════════════════════════════════════════════
mt_points = mt_raw["data"]
turns     = [p["turn"] for p in mt_points]
ref_mt    = {k: mt_points[0]["scores"][k] for k in METRICS}

fig3, ax = plt.subplots(figsize=(6, 4.5))

for metric, (label, color) in METRICS.items():
    vals = [p["scores"][metric] / ref_mt[metric] for p in mt_points]
    ax.plot(turns, vals, marker="o", markersize=6, color=color, lw=2, label=label)

# Annotate SacreBLEU raw values
for p in mt_points:
    raw = p["scores"]["sacrebleu"]
    ny  = raw / ref_mt["sacrebleu"]
    ax.annotate(f"{raw:.1f}", (p["turn"], ny),
                textcoords="offset points", xytext=(0, 7),
                fontsize=7.5, ha="center", color=C_BLEU)

ax.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6, label="Turn 1 baseline")
ax.set_xlabel("Conversation turn", fontsize=10)
ax.set_ylabel("Normalised score (relative to Turn 1)", fontsize=10)
ax.set_title("Multi-Turn Robustness\n(pirate style, TruthfulQA)", fontsize=11)
ax.set_xticks(turns)
ax.set_xticklabels(["Turn 1\n(single)", "Turn 2\n(+1 history)", "Turn 3\n(+2 history)"])
ax.set_ylim(0.93, 1.13)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig("figures/fig3_multi_turn.png", dpi=150, bbox_inches="tight")
print("Saved → figures/fig3_multi_turn.png")
plt.close(fig3)
