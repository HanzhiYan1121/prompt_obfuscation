"""Generate the client/cloud/adversary mapping used in the threat model."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"

COLORS = {
    "ink": "#17202A",
    "muted": "#4F5B66",
    "client_edge": "#3F6F9F",
    "client_fill": "#EDF4FA",
    "cloud_edge": "#3C7A57",
    "cloud_fill": "#EEF7F1",
    "attack_edge": "#A84942",
    "attack_fill": "#FBF0EF",
    "payload": "#F6E7B0",
    "node": "#FFFFFF",
    "arrow": "#283747",
}


def rounded_box(ax, x, y, w, h, *, edge, face, text, size=10.5, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.07",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=COLORS["ink"],
        fontweight=weight,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, *, color=None, style="-|>", lw=1.6, rad=0.0, dashed=False):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color or COLORS["arrow"],
        connectionstyle=f"arc3,rad={rad}",
        linestyle="--" if dashed else "-",
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def boundary(ax, x, y, w, h, *, edge, face, title, subtitle):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=face,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.34, title, fontsize=12.5, fontweight="bold", color=edge)
    ax.text(x + 0.18, y + h - 0.63, subtitle, fontsize=8.7, color=COLORS["muted"])
    return patch


def build_figure():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(14.2, 5.35))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 5.35)
    ax.axis("off")

    boundary(
        ax,
        0.20,
        1.20,
        4.20,
        3.85,
        edge=COLORS["client_edge"],
        face=COLORS["client_fill"],
        title="Trusted client boundary",
        subtitle="Protection runs locally; private state is retained here",
    )
    boundary(
        ax,
        4.90,
        1.20,
        4.55,
        3.85,
        edge=COLORS["cloud_edge"],
        face=COLORS["cloud_fill"],
        title="Cloud inference service",
        subtitle="Receives the protected payload and executes the model",
    )
    boundary(
        ax,
        9.95,
        1.20,
        4.05,
        3.85,
        edge=COLORS["attack_edge"],
        face=COLORS["attack_fill"],
        title="Service-operator adversary $\\mathcal{A}$",
        subtitle="Cloud provider or compromised logging/control plane",
    )

    rounded_box(
        ax,
        0.52,
        3.20,
        1.02,
        0.68,
        edge=COLORS["client_edge"],
        face=COLORS["node"],
        text="Private prompt\n$P$",
        size=10.2,
    )
    rounded_box(
        ax,
        1.82,
        3.20,
        1.00,
        0.68,
        edge=COLORS["client_edge"],
        face=COLORS["node"],
        text="Protect\n$T(\\cdot;\\xi_T)$",
        size=10.2,
    )
    rounded_box(
        ax,
        3.10,
        3.20,
        0.98,
        0.68,
        edge=COLORS["client_edge"],
        face=COLORS["node"],
        text="Local state\n$H$",
        size=10.2,
    )
    arrow(ax, (1.54, 3.54), (1.82, 3.54))
    arrow(ax, (2.82, 3.54), (3.10, 3.54))

    ax.text(
        2.30,
        2.20,
        "(1) Remains on client",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["client_edge"],
    )
    ax.text(
        2.30,
        1.83,
        "$P$, $H$, and transformation randomness $\\xi_T$",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["ink"],
    )

    rounded_box(
        ax,
        5.18,
        3.16,
        1.18,
        0.76,
        edge="#A17C16",
        face=COLORS["payload"],
        text="Payload\n$Z=\\Pi(H)$",
        size=10.5,
        weight="bold",
    )
    arrow(ax, (4.08, 3.54), (5.18, 3.54), color="#8A6A12", lw=2.0)
    ax.text(
        4.63,
        3.88,
        "(2) only $Z$ crosses",
        ha="center",
        fontsize=9.2,
        color="#765A0C",
        fontweight="bold",
    )

    rounded_box(
        ax,
        6.87,
        3.16,
        1.48,
        0.76,
        edge=COLORS["cloud_edge"],
        face=COLORS["node"],
        text="Cloud model\n$M_{\\theta_M}(Z,X_i)$",
        size=10.4,
    )
    rounded_box(
        ax,
        8.66,
        3.16,
        0.55,
        0.76,
        edge=COLORS["cloud_edge"],
        face=COLORS["node"],
        text="$Y_i$",
        size=12.0,
        weight="bold",
    )
    arrow(ax, (6.36, 3.54), (6.87, 3.54))
    arrow(ax, (8.35, 3.54), (8.66, 3.54))

    ax.text(
        7.18,
        2.22,
        "(3) Cloud processes",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["cloud_edge"],
    )
    ax.text(
        7.18,
        1.84,
        "$Z$ and probe $X_i$ with model parameters $\\theta_M$",
        ha="center",
        fontsize=9.5,
        color=COLORS["ink"],
    )

    rounded_box(
        ax,
        10.35,
        3.00,
        3.25,
        1.05,
        edge=COLORS["attack_edge"],
        face=COLORS["node"],
        text="(4) Observes $\\theta_M$, $Z$, $X_{1:q}$, $Y_{1:q}$\n"
        "and auxiliary information $A,D_{\\mathrm{aux}}$",
        size=10.2,
    )
    arrow(ax, (9.21, 3.54), (10.35, 3.54), color=COLORS["attack_edge"], lw=1.9)
    ax.text(
        9.78,
        3.87,
        "logs / responses",
        ha="center",
        fontsize=8.8,
        color=COLORS["attack_edge"],
        fontweight="bold",
    )

    rounded_box(
        ax,
        10.63,
        1.65,
        2.70,
        0.62,
        edge=COLORS["attack_edge"],
        face=COLORS["node"],
        text="Recovery: $(\\hat{S},\\hat{P})=\\mathcal{A}(V_q)$",
        size=10.2,
        weight="bold",
    )
    arrow(ax, (11.98, 3.00), (11.98, 2.27), color=COLORS["attack_edge"])

    arrow(
        ax,
        (10.63, 1.82),
        (7.78, 3.12),
        color=COLORS["attack_edge"],
        lw=1.7,
        rad=-0.20,
        dashed=True,
    )
    ax.text(
        9.18,
        1.56,
        "(5) active replay / probe $X_i$,  $i\\leq q$",
        ha="center",
        fontsize=9.3,
        color=COLORS["attack_edge"],
        fontweight="bold",
    )

    ax.text(
        7.10,
        0.55,
        "Trust boundary: the adversary sees the complete cloud-side interface, but never the client-local $P$, $H$, or $\\xi_T$.",
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["ink"],
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf_path = FIG_DIR / "threat_model_mapping.pdf"
    png_path = FIG_DIR / "threat_model_mapping.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
