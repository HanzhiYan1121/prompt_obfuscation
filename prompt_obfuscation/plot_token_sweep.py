"""
Plot API adherence / trajectory / format score vs soft-prompt token count.
Reads results/l4_drone_eval_tokN.json for N in {10, 20, 40, 80, 160}.
"""
import json, statistics, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results")
TOKEN_COUNTS = [10, 20, 40, 80, 160]

def load_avg(tok):
    path = OUT / f"l4_drone_eval_tok{tok}.json"
    if not path.exists():
        # fallback: original 10-token result
        path = OUT / "l4_drone_eval.json"
        if tok != 10 or not path.exists():
            return None
    with open(path) as f:
        records = json.load(f)

    def avg(vals):
        v = [x for x in vals if x is not None]
        return statistics.mean(v) if v else 0.0

    return {
        "conv_api":   avg([r["conv_api"]        for r in records]),
        "obf_api":    avg([r["obf_api"]         for r in records]),
        "conv_traj":  avg([r["conv_trajectory"]  for r in records]),
        "obf_traj":   avg([r["obf_trajectory"]   for r in records]),
        "conv_fmt":   avg([r["conv_format"]      for r in records]),
        "obf_fmt":    avg([r["obf_format"]       for r in records]),
    }

data = {}
for t in TOKEN_COUNTS:
    d = load_avg(t)
    if d:
        data[t] = d

toks   = sorted(data.keys())
labels = [str(t) for t in toks]

conv_api  = [data[t]["conv_api"]  for t in toks]
obf_api   = [data[t]["obf_api"]   for t in toks]
conv_traj = [data[t]["conv_traj"] for t in toks]
obf_traj  = [data[t]["obf_traj"]  for t in toks]
conv_fmt  = [data[t]["conv_fmt"]  for t in toks]
obf_fmt   = [data[t]["obf_fmt"]   for t in toks]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Soft Prompt Token Count vs Utility (L4 AirSim Drone)",
             fontsize=13, fontweight="bold")

METRICS = [
    ("API Adherence",    conv_api,  obf_api),
    ("Trajectory Score", conv_traj, obf_traj),
    ("Format Score",     conv_fmt,  obf_fmt),
]

for ax, (title, conv, obf) in zip(axes, METRICS):
    ax.plot(toks, conv, "o--", color="#4C72B0", linewidth=2,
            markersize=7, label="Conventional")
    ax.plot(toks, obf,  "s-",  color="#DD8452", linewidth=2,
            markersize=7, label="Obfuscated")
    # Annotate obf values
    for t, v in zip(toks, obf):
        ax.annotate(f"{v:.2f}", (t, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#DD8452")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Soft Prompt Tokens", fontsize=10)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks(toks)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9)

plt.tight_layout()
out = OUT / "fig_token_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# Print table
print(f"\n{'Tokens':>8} {'Conv API':>10} {'Obf API':>10} {'Conv Traj':>10} {'Obf Traj':>10}")
print("-" * 52)
for t in toks:
    d = data[t]
    print(f"{t:>8} {d['conv_api']:>10.3f} {d['obf_api']:>10.3f} "
          f"{d['conv_traj']:>10.3f} {d['obf_traj']:>10.3f}")
