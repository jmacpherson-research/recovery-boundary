#!/usr/bin/env python3
"""Generate Paper 3 figures from DB-verified data (52-corpus version).

Produces:
  fig_recovery.pdf — Horizontal bar chart of recovery rates
  fig_pca.pdf      — PCA scatter plot (PC1 vs PC2)

All data hard-coded from corpus.db queries (2026-02-27).
PCA sign convention: naturals at HIGH PC1, AI at LOW PC1.

Usage:
  python generate_figures.py            # generates both figures
  python generate_figures.py --show     # also displays interactively
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys

# ============================================================
# VERIFIED DATA (from gen_tables.py / corpus.db)
# ============================================================

# Recovery rates (only corpora where recovery is defined)
RECOVERY = {
    # Natural
    "annomi": 0.957, "fomc": 0.949, "law_of_one": 0.912,
    "scotus": 0.898, "cwg": 0.868, "news_reuters": 0.791,
    "seth": 0.780, "carla_prose": 0.697, "bashar": 0.697,
    "ll_research": 0.578, "acim": 0.576, "dailydialog": 0.455,
    "academic_arxiv": 0.429, "gutenberg_fiction": 0.333, "vatican": 0.274,
    # Multi-turn AI (standard) — updated S113 from DB: 28/82=0.341, 30/38=0.789
    "multiturn_claude": 0.341, "multiturn_gpt4o": 0.789,
    # Cross-domain AI
    "xdom_claude_fomc": 0.000, "xdom_claude_scotus": 0.000,
    "xdom_claude_therapy": 0.000, "xdom_gpt4o_fomc": 0.000,
    "xdom_gpt4o_scotus": 0.000, "xdom_gpt4o_therapy": 0.000,
    # Edited AI
    "edit_e1": 0.000, "edit_e2": 0.000, "edit_e3": 0.000,
    # Two-pass
    "synth_twopass_claude": 0.000,
    # Adversarial
    "adv_constraint": 1.000, "adv_style": 0.833,
    # Reasoning MT (entity-only)
    "reasoning_deepseek_r1_mt": 0.630, "reasoning_o3mini_mt": 0.895,
}

# PCA coordinates for all 52 corpora
PCA = {
    # Natural
    "law_of_one": (0.22, -0.34), "seth": (1.31, -0.40),
    "bashar": (3.71, -1.49), "acim": (4.41, 0.56),
    "cwg": (2.79, -0.42), "scotus": (0.19, -0.57),
    "fomc": (-1.09, -0.42), "vatican": (3.28, -2.04),
    "carla_prose": (2.38, -1.24), "ll_research": (1.65, -0.86),
    "annomi": (1.73, -2.27), "dailydialog": (1.69, 0.83),
    "gutenberg_fiction": (0.89, -0.05), "news_reuters": (-0.85, 0.52),
    "academic_arxiv": (-0.88, -0.01),
    # Calibration
    "ecc_hamming74": (-1.59, -3.40), "sat_3sat": (-1.09, -1.66),
    "gencode_splice": (-1.03, -1.46),
    # AI Single-Shot
    "synth_g0": (-0.67, 1.03), "synth_g1": (-0.23, 0.35),
    "synth_g2": (-0.41, 0.26), "synth_g3": (0.14, 0.28),
    "synth_g4": (-0.69, 0.25),
    "synth_gpt4o_g0": (-1.47, 0.24), "synth_gemini_g0": (-1.19, 0.29),
    "synth_llama70b_g0": (-1.24, -0.16),
    "synth_fiction_g0": (-0.51, 0.83), "synth_news_g0": (0.17, 1.16),
    "synth_academic_g0": (-0.95, -0.02), "synth_dialogue_g0": (-1.00, 0.55),
    # AI Multi-Turn
    "multiturn_claude": (0.07, -0.38), "multiturn_gpt4o": (-1.52, -0.27),
    # AI Reasoning
    "reasoning_deepseek_r1": (-0.68, 0.50), "reasoning_o3mini": (-1.56, 0.23),
    "reasoning_deepseek_r1_mt": (-0.76, 0.30), "reasoning_o3mini_mt": (-1.39, -0.14),
    # AI Temperature Sweep
    "synth_claude_t02": (-0.34, 0.87), "synth_claude_t15": (-0.27, 1.13),
    "synth_gpt4o_t02": (-1.59, -0.22), "synth_gpt4o_t15": (-0.01, 1.78),
    # AI Cross-Domain
    "xdom_claude_fomc": (-1.37, 0.06), "xdom_claude_scotus": (-0.61, 0.64),
    "xdom_claude_therapy": (-1.74, 0.01),
    "xdom_gpt4o_fomc": (-1.62, -0.08), "xdom_gpt4o_scotus": (-0.72, 0.11),
    "xdom_gpt4o_therapy": (-1.61, -0.17),
    # AI Edited / Adversarial / Special
    "edit_e1": (1.36, 0.76), "edit_e2": (0.96, 1.02),
    "edit_e3": (1.83, -0.75),
    "adv_style": (-0.49, -0.65), "adv_constraint": (0.09, -0.30),
    "synth_twopass_claude": (2.29, 5.20),
}

# Classification for coloring
NATURAL = {
    "law_of_one", "seth", "bashar", "acim", "cwg", "scotus", "fomc",
    "vatican", "carla_prose", "ll_research", "annomi", "dailydialog",
    "gutenberg_fiction", "news_reuters", "academic_arxiv",
}
CALIBRATION = {"ecc_hamming74", "sat_3sat", "gencode_splice"}
REASONING_MT = {"reasoning_deepseek_r1_mt", "reasoning_o3mini_mt"}
ADVERSARIAL = {"adv_style", "adv_constraint"}
EDITED = {"edit_e1", "edit_e2", "edit_e3"}
MULTI_TURN = {"multiturn_claude", "multiturn_gpt4o"}
TWOPASS = {"synth_twopass_claude"}

# Display names for labels
DISPLAY = {
    "law_of_one": "Law of One", "seth": "Seth", "bashar": "Bashar",
    "acim": "ACIM", "cwg": "CWG", "scotus": "SCOTUS", "fomc": "FOMC",
    "vatican": "Vatican", "carla_prose": "Carla Prose",
    "ll_research": "L/L Research", "annomi": "AnnoMI",
    "dailydialog": "DailyDialog", "gutenberg_fiction": "Gutenberg",
    "news_reuters": "Reuters", "academic_arxiv": "arXiv",
    "ecc_hamming74": "Hamming", "sat_3sat": "3-SAT",
    "gencode_splice": "GenCode",
    "synth_g0": "G0", "synth_g1": "G1", "synth_g2": "G2",
    "synth_g3": "G3", "synth_g4": "G4",
    "synth_gpt4o_g0": "GPT-4o G0", "synth_gemini_g0": "Gemini G0",
    "synth_llama70b_g0": "Llama G0",
    "synth_fiction_g0": "Fiction G0", "synth_news_g0": "News G0",
    "synth_academic_g0": "Academic G0", "synth_dialogue_g0": "Dialogue G0",
    "multiturn_claude": "MT Claude", "multiturn_gpt4o": "MT GPT-4o",
    "reasoning_deepseek_r1": "DS-R1 SS", "reasoning_o3mini": "o3-mini SS",
    "reasoning_deepseek_r1_mt": "DS-R1 MT", "reasoning_o3mini_mt": "o3-mini MT",
    "synth_claude_t02": "Claude T=0.2", "synth_claude_t15": "Claude T=1.5",
    "synth_gpt4o_t02": "GPT-4o T=0.2", "synth_gpt4o_t15": "GPT-4o T=1.5",
    "xdom_claude_fomc": "xCl FOMC", "xdom_claude_scotus": "xCl SCOTUS",
    "xdom_claude_therapy": "xCl Therapy",
    "xdom_gpt4o_fomc": "xGPT FOMC", "xdom_gpt4o_scotus": "xGPT SCOTUS",
    "xdom_gpt4o_therapy": "xGPT Therapy",
    "edit_e1": "Edit E1", "edit_e2": "Edit E2", "edit_e3": "Edit E3",
    "adv_style": "Adv Style", "adv_constraint": "Adv Constraint",
    "synth_twopass_claude": "Two-Pass",
}

# Manual label offsets for PCA plot (to avoid overlaps)
LABEL_OFFSETS = {
    "acim": (0.15, 0.15),
    "bashar": (0.15, 0.15),
    "vatican": (0.15, -0.2),
    "carla_prose": (0.15, -0.15),
    "scotus": (0.15, -0.25),
    "fomc": (0.1, -0.25),
    "annomi": (0.15, -0.2),
    "cwg": (0.15, -0.2),
    "seth": (0.15, -0.15),
    "ll_research": (0.15, -0.15),
    "law_of_one": (0.15, 0.15),
    "dailydialog": (0.15, 0.15),
    "news_reuters": (0.1, 0.15),
    "academic_arxiv": (0.1, 0.15),
    "gutenberg_fiction": (0.15, 0.15),
    "synth_twopass_claude": (0.15, 0.2),
    "reasoning_deepseek_r1_mt": (0.1, 0.15),
    "reasoning_o3mini_mt": (0.1, 0.15),
    "multiturn_claude": (0.15, -0.2),
    "multiturn_gpt4o": (-0.15, -0.25),
    "ecc_hamming74": (0.15, -0.2),
    "sat_3sat": (0.1, -0.15),
    "gencode_splice": (0.1, -0.2),
    "synth_g0": (0.1, 0.15),
    "adv_constraint": (0.1, -0.2),
    "adv_style": (0.1, -0.2),
    "edit_e3": (0.15, -0.2),
}


def get_color(name):
    """Return color based on corpus classification."""
    if name in NATURAL:
        return "#2166AC"   # blue
    elif name in CALIBRATION:
        return "#999999"   # grey
    elif name in REASONING_MT:
        return "#E08214"   # orange
    elif name in ADVERSARIAL:
        return "#B2182B"   # dark red
    elif name in EDITED:
        return "#762A83"   # purple
    elif name in MULTI_TURN:
        return "#D6604D"   # salmon
    elif name in TWOPASS:
        return "#92C5DE"   # light blue
    else:
        return "#4DAF4A"   # green (standard AI)


def get_marker(name):
    """Return marker based on corpus classification."""
    if name in NATURAL:
        return "o"
    elif name in CALIBRATION:
        return "D"
    elif name in REASONING_MT:
        return "^"
    elif name in ADVERSARIAL:
        return "s"
    elif name in EDITED:
        return "P"
    elif name in MULTI_TURN:
        return "v"
    elif name in TWOPASS:
        return "X"
    else:
        return "o"  # standard AI


# ============================================================
# FIGURE 1: RECOVERY BAR CHART
# ============================================================

def gen_recovery_figure(outpath="fig_recovery.pdf"):
    """Generate horizontal bar chart of recovery rates."""
    # Sort by recovery rate (descending)
    items = sorted(RECOVERY.items(), key=lambda x: x[1], reverse=True)
    names = [DISPLAY.get(k, k) for k, _ in items]
    rates = [v * 100 for _, v in items]
    colors = [get_color(k) for k, _ in items]

    fig, ax = plt.subplots(figsize=(8, 12))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Recovery Rate (%)", fontsize=10)
    ax.set_xlim(0, 105)
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Add value labels
    for i, (rate, name) in enumerate(zip(rates, names)):
        if rate > 0:
            ax.text(rate + 1, i, f"{rate:.1f}%", va="center", fontsize=6)
        else:
            ax.text(1, i, "0.0%", va="center", fontsize=6)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2166AC", label="Natural"),
        mpatches.Patch(facecolor="#E08214", label="Reasoning MT"),
        mpatches.Patch(facecolor="#B2182B", label="Adversarial"),
        mpatches.Patch(facecolor="#D6604D", label="Multi-Turn AI"),
        mpatches.Patch(facecolor="#4DAF4A", label="Standard AI"),
        mpatches.Patch(facecolor="#762A83", label="Edited AI"),
        mpatches.Patch(facecolor="#92C5DE", label="Two-Pass"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
              framealpha=0.9)

    ax.set_title("Recovery Rates Across 30 Corpora\n(where recovery is measurable)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved {outpath}")
    return fig


# ============================================================
# FIGURE 2: PCA SCATTER PLOT
# ============================================================

def gen_pca_figure(outpath="fig_pca.pdf"):
    """Generate PCA scatter plot for all 52 corpora."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each corpus
    for name, (pc1, pc2) in PCA.items():
        color = get_color(name)
        marker = get_marker(name)
        size = 80 if name in NATURAL else (60 if name in CALIBRATION else 50)
        zorder = 10 if name in NATURAL else 5

        ax.scatter(pc1, pc2, c=color, marker=marker, s=size,
                   edgecolors="black", linewidths=0.5, zorder=zorder)

    # Add labels for key corpora (not all — too crowded)
    label_set = (
        NATURAL | CALIBRATION | REASONING_MT | ADVERSARIAL |
        MULTI_TURN | TWOPASS | {"synth_g0", "synth_gpt4o_g0"}
    )
    for name in label_set:
        if name not in PCA:
            continue
        pc1, pc2 = PCA[name]
        dx, dy = LABEL_OFFSETS.get(name, (0.1, 0.1))
        disp = DISPLAY.get(name, name)
        fontsize = 6 if name not in NATURAL else 7
        ax.annotate(disp, (pc1, pc2), xytext=(pc1 + dx, pc2 + dy),
                    fontsize=fontsize, color="black", alpha=0.8,
                    arrowprops=dict(arrowstyle="-", color="grey",
                                    lw=0.3, alpha=0.5) if abs(dx) > 0.15 or abs(dy) > 0.15 else None)

    # Axis labels with variance explained
    ax.set_xlabel("PC1 (32.3% variance explained)", fontsize=10)
    ax.set_ylabel("PC2 (19.9% variance explained)", fontsize=10)

    # Reference lines
    ax.axhline(y=0, color="grey", linewidth=0.3, linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="grey", linewidth=0.3, linestyle="--", alpha=0.5)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166AC",
                   markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                   label="Natural (15)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#999999",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                   label="Calibration (3)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4DAF4A",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                   label="AI Single-Shot (12)"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#D6604D",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                   label="AI Multi-Turn (2)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#E08214",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                   label="Reasoning MT (2)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4DAF4A",
                   markersize=6, markeredgecolor="black", markeredgewidth=0.5,
                   label="AI Temp/XDom/etc (13)"),
        plt.Line2D([0], [0], marker="P", color="w", markerfacecolor="#762A83",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                   label="Edited (3)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#B2182B",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                   label="Adversarial (2)"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#92C5DE",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                   label="Two-Pass (1)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              framealpha=0.9, ncol=1)

    ax.set_title("52-Corpus PCA: 8-Dimensional Structural Vector",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved {outpath}")
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    show = "--show" in sys.argv

    fig1 = gen_recovery_figure()
    fig2 = gen_pca_figure()

    if show:
        plt.show()
    else:
        plt.close("all")

    print("Done. Figures saved to fig_recovery.pdf and fig_pca.pdf")
