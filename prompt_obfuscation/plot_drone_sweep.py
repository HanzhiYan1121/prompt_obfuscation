"""
Compare drone-trained soft prompt vs CNN-trained soft prompt vs conventional,
across token counts 10/20/40/80/160.
"""
import json, statistics, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results")
TOKEN_COUNTS = [10, 20, 40, 80, 160]

def load_avg(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        records = json.load(f)
    def avg(vals):
        v = [x for x in vals if x is not None]
        return statistics.mean(v) if v else 0.0
    return {
        "conv_api":  avg([r["conv_api"]       for r in records]),
        "obf_api":   avg([r["obf_api"]        for r in records]),
        "conv_traj": avg([r["conv_trajectory"] for r in records]),
        "obf_traj":  avg([r["obf_trajectory"]  for r in records]),
        "conv_fmt":  avg([r["conv_format"]     for r in records]),
        "obf_fmt":   avg([r["obf_format"]      for r in records]),
    }

cnn_data   = {}
drone_data = {}
for t in TOKEN_COUNTS:
    d = load_avg(OUT / f"l4_drone_eval_tok{t}.json")
    if d: cnn_data[t] = d
    d = load_avg(OUT / f"l4_drone_eval_drone_tok{t}.json")
    if d: drone_data[t] = d

toks = sorted(set(list(cnn_data.keys()) + list(drone_data.keys())))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Soft Prompt Token Count vs Utility — Drone-Trained vs CNN-Trained",
             fontsize=13, fontweight="bold")

METRICS = [
    ("API Adherence Rate", "api"),
    ("Trajectory Score",   "traj"),
    ("Format Score",       "fmt"),
]

colors = {"conv": "#4C72B0", "cnn": "#DD8452", "drone": "#55A868"}

for ax, (title, key) in zip(axes, METRICS):
    # Conventional baseline (same for both, use CNN results)
    conv_vals = [cnn_data[t][f"conv_{key}"] for t in toks if t in cnn_data]
    toks_cnn  = [t for t in toks if t in cnn_data]
    toks_drn  = [t for t in toks if t in drone_data]
    cnn_vals  = [cnn_data[t][f"obf_{key}"]   for t in toks_cnn]
    drn_vals  = [drone_data[t][f"obf_{key}"] for t in toks_drn]

    ax.plot(toks_cnn, conv_vals, "o--", color=colors["conv"],
            linewidth=2, markersize=7, label="Conventional")
    ax.plot(toks_cnn, cnn_vals,  "s-",  color=colors["cnn"],
            linewidth=2, markersize=7, label="Obf (CNN-trained)")
    ax.plot(toks_drn, drn_vals,  "^-",  color=colors["drone"],
            linewidth=2, markersize=7, label="Obf (Drone-trained)")

    for t, v in zip(toks_drn, drn_vals):
        ax.annotate(f"{v:.2f}", (t, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=colors["drone"])
    for t, v in zip(toks_cnn, cnn_vals):
        ax.annotate(f"{v:.2f}", (t, v), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=colors["cnn"])

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Soft Prompt Tokens", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(TOKEN_COUNTS)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9)

plt.tight_layout()
out = OUT / "fig_drone_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# Table
print(f"\n{'Tokens':>8} {'Conv':>8} {'CNN-obf':>10} {'Drone-obf':>12}  (API Adherence)")
print("-" * 46)
for t in TOKEN_COUNTS:
    c = f"{cnn_data[t]['conv_api']:.3f}"   if t in cnn_data   else "N/A"
    n = f"{cnn_data[t]['obf_api']:.3f}"    if t in cnn_data   else "N/A"
    d = f"{drone_data[t]['obf_api']:.3f}"  if t in drone_data else "N/A"
    print(f"{t:>8} {c:>8} {n:>10} {d:>12}")
