"""
Generate fig_trace.pdf: Illustrative chunk-by-chunk compressibility traces
showing natural recovery event vs AI persistent excursion.

Schematic illustration using realistic parameters from corpus.db.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(2026)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

# === Panel A: Natural text with recovery ===
n = 30
baseline = 0.42
sigma = 0.035

# Build a realistic-looking trace
nat = np.random.normal(baseline, sigma * 0.5, n)
# Gentle autocorrelation
for i in range(1, n):
    nat[i] = 0.6 * nat[i] + 0.4 * nat[i-1]

# Insert excursion (chunks 10-14) with gradual recovery (chunks 15-19)
nat[10] = baseline + 0.9 * sigma
nat[11] = baseline + 1.8 * sigma
nat[12] = baseline + 2.1 * sigma   # peak
nat[13] = baseline + 1.6 * sigma
nat[14] = baseline + 1.0 * sigma   # declining
nat[15] = baseline + 0.5 * sigma   # near threshold
nat[16] = baseline + 0.1 * sigma   # recovered
nat[17] = baseline - 0.2 * sigma
nat[18] = baseline - 0.1 * sigma

chunks = np.arange(1, n + 1)
threshold = baseline + 0.5 * sigma

ax1.plot(chunks, nat, 'o-', color='#2166ac', markersize=3.5, linewidth=1.3, zorder=3)
ax1.axhline(baseline, color='#666666', linestyle='-', linewidth=0.7, alpha=0.6)
ax1.axhline(threshold, color='#d6604d', linestyle='--', linewidth=0.8, alpha=0.7)

# Shade excursion (above threshold)
ax1.axvspan(10.5, 15.5, alpha=0.12, color='#d6604d')
# Shade recovery zone
ax1.axvspan(15.5, 17.5, alpha=0.12, color='#2166ac')

# Annotations
ax1.annotate('Excursion', xy=(12, nat[12]), xytext=(5, baseline + 2.5 * sigma),
            fontsize=8, ha='center', color='#d6604d', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d6604d', lw=1.0))
ax1.annotate('Recovery', xy=(16.5, nat[16]), xytext=(22, baseline - 1.5 * sigma),
            fontsize=8, ha='center', color='#2166ac', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.0))

# Labels for reference lines
ax1.text(n + 0.3, baseline, r'$\mu$', fontsize=8, va='center', color='#666666')
ax1.text(n + 0.3, threshold, r'$\mu{+}0.5\sigma$', fontsize=8, va='center', color='#d6604d')

ax1.set_xlabel('Chunk index', fontsize=10)
ax1.set_ylabel('Compressibility ($c_i$)', fontsize=10)
ax1.set_title('(a) Natural text: excursion with recovery', fontsize=10, fontweight='bold')
ax1.set_xlim(0.5, n + 2.5)
ax1.set_ylim(baseline - 3 * sigma, baseline + 3 * sigma)

# === Panel B: Standard AI with persistent excursion ===
n_ai = 30
baseline_ai = 0.48
sigma_ai = 0.010

ai = np.random.normal(baseline_ai, sigma_ai * 0.4, n_ai)
for i in range(1, n_ai):
    ai[i] = 0.6 * ai[i] + 0.4 * ai[i-1]

# Excursion at chunk 10 that never recovers
ai[10] = baseline_ai + 0.8 * sigma_ai
ai[11] = baseline_ai + 1.5 * sigma_ai
ai[12] = baseline_ai + 1.8 * sigma_ai
for i in range(13, n_ai):
    ai[i] = baseline_ai + (1.2 + 0.4 * np.random.randn()) * sigma_ai
    ai[i] = max(ai[i], baseline_ai + 0.4 * sigma_ai)  # stays elevated

chunks_ai = np.arange(1, n_ai + 1)
threshold_ai = baseline_ai + 0.5 * sigma_ai

ax2.plot(chunks_ai, ai, 's-', color='#b2182b', markersize=3.5, linewidth=1.3, zorder=3)
ax2.axhline(baseline_ai, color='#666666', linestyle='-', linewidth=0.7, alpha=0.6)
ax2.axhline(threshold_ai, color='#d6604d', linestyle='--', linewidth=0.8, alpha=0.7)

# Shade persistent excursion
ax2.axvspan(10.5, n_ai + 0.5, alpha=0.12, color='#d6604d')

ax2.annotate('Excursion onset', xy=(11, ai[11]), xytext=(5, baseline_ai + 2.5 * sigma_ai),
            fontsize=8, ha='center', color='#d6604d', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d6604d', lw=1.0))
ax2.annotate('No recovery\n(persists to session end)', xy=(25, ai[25]),
            xytext=(22, baseline_ai - 1.5 * sigma_ai),
            fontsize=8, ha='center', color='#b2182b', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.0))

ax2.text(n_ai + 0.3, baseline_ai, r'$\mu$', fontsize=8, va='center', color='#666666')
ax2.text(n_ai + 0.3, threshold_ai, r'$\mu{+}0.5\sigma$', fontsize=8, va='center', color='#d6604d')

ax2.set_xlabel('Chunk index', fontsize=10)
ax2.set_title('(b) Standard AI: persistent excursion', fontsize=10, fontweight='bold')
ax2.set_xlim(0.5, n_ai + 2.5)
ax2.set_ylim(baseline_ai - 3 * sigma_ai, baseline_ai + 3 * sigma_ai)

plt.tight_layout()
plt.savefig('fig_trace.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_trace.png', bbox_inches='tight', dpi=150)
print("Saved fig_trace.pdf and fig_trace.png")
