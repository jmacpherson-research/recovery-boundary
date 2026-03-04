import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Paths relative to project root
PROJECT = r"C:\NDE_Research_Project"
JSON_PATH = os.path.join(PROJECT, "topic_coherence_results.json")
OUT_FIG = os.path.join(PROJECT, "fig_coherence.pdf")

# Load coherence data
with open(JSON_PATH) as f:
    data = json.load(f)

# Recovery rates from corpus.db (updated for 90-session MT expansion)
recovery = {
    "multiturn_claude": 34.1, "multiturn_gpt4o": 78.9,
    "mt_xdom_claude_econ": 75.0, "mt_xdom_claude_legal": 51.1,
    "mt_xdom_claude_therapy": 30.0, "mt_xdom_gpt4o_econ": 43.8,
    "mt_xdom_gpt4o_legal": 57.1, "mt_xdom_gpt4o_therapy": 50.0,
    "law_of_one": 91.2, "ll_research": 57.8, "bashar": 69.7,
    "seth": 78.0, "acim": 57.6, "cwg": 86.8, "fomc": 94.9,
    "scotus": 89.8, "annomi": 95.7, "vatican": 27.4,
    "carla_prose": 69.7, "gutenberg_fiction": 33.3,
    "news_reuters": 79.1, "academic_arxiv": 42.9, "dailydialog": 45.5,
}

# Merge
for cid in data:
    data[cid]["recovery_rate"] = recovery.get(cid)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))

colors = {"natural": "#2196F3", "ai_diverse": "#F44336"}
markers = {"natural": "o", "ai_diverse": "^"}
sizes = {"natural": 70, "ai_diverse": 100}
label_map = {"natural": "Natural", "ai_diverse": "AI (multi-turn)"}
labels_used = set()

ai_x, ai_y = [], []
nat_x, nat_y = [], []

for cid, r in data.items():
    if r.get("recovery_rate") is None:
        continue
    t = r["type"]
    if t not in colors:
        continue
    label = label_map[t] if t not in labels_used else None
    labels_used.add(t)

    ax.scatter(r["coherence_mean"], r["recovery_rate"],
              c=colors[t], marker=markers[t], s=sizes[t], label=label,
              edgecolors="black", linewidths=0.5, zorder=3)

    # Smart label offset
    offset = (5, 5)
    ha = "left"
    if r["label"] in ("ACIM", "Vatican"):
        offset = (5, -10)
    elif r["label"] in ("MT Claude (spiritual)",):
        offset = (-5, -12)
        ha = "right"
    elif r["label"] in ("MT GPT-4o (spiritual)",):
        offset = (-5, 5)
        ha = "right"

    ax.annotate(r["label"], (r["coherence_mean"], r["recovery_rate"]),
               fontsize=6.5, ha=ha, va="bottom",
               xytext=offset, textcoords="offset points", color="#333333")

    if t.startswith("ai"):
        ai_x.append(r["coherence_mean"])
        ai_y.append(r["recovery_rate"])
    else:
        nat_x.append(r["coherence_mean"])
        nat_y.append(r["recovery_rate"])

# Trend lines
r_ai, p_ai = None, None
if len(ai_x) >= 3:
    r_ai, p_ai = stats.pearsonr(ai_x, ai_y)
    z = np.polyfit(ai_x, ai_y, 1)
    xline = np.linspace(min(ai_x) - 0.03, max(ai_x) + 0.03, 50)
    ax.plot(xline, np.polyval(z, xline), '--', color='#D32F2F', alpha=0.6, linewidth=1.5,
           label=f"AI trend (r={r_ai:.2f}, p={p_ai:.3f})")

r_nat, p_nat = None, None
if len(nat_x) >= 3:
    r_nat, p_nat = stats.pearsonr(nat_x, nat_y)
    z = np.polyfit(nat_x, nat_y, 1)
    xline = np.linspace(min(nat_x) - 0.03, max(nat_x) + 0.03, 50)
    ax.plot(xline, np.polyval(z, xline), '--', color='#1565C0', alpha=0.6, linewidth=1.5,
           label=f"Natural trend (r={r_nat:.2f}, p={p_nat:.3f})")

ax.set_xlabel("Topic Coherence Index\n(mean cosine similarity between consecutive segments)", fontsize=11)
ax.set_ylabel("Corpus-Level Recovery Rate (%)", fontsize=11)
ax.set_title("Recovery Rate vs. Topic Coherence", fontsize=13, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_xlim(0.08, 0.55)
ax.set_ylim(-5, 100)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
print(f"Saved {OUT_FIG}")

# Key stats
if r_ai is not None:
    print(f"\nAI trend: r={r_ai:.3f}, p={p_ai:.4f}")
if r_nat is not None:
    print(f"Natural trend: r={r_nat:.3f}, p={p_nat:.4f}")
print(f"\nSpiritual MT mean coherence: {np.mean([data[c]['coherence_mean'] for c in ['multiturn_claude','multiturn_gpt4o'] if c in data]):.4f}")
print(f"All AI mean coherence: {np.mean([data[c]['coherence_mean'] for c in data if data[c].get('type')=='ai_diverse']):.4f}")
print(f"Natural coherence range: {min(nat_x):.4f} - {max(nat_x):.4f}")
