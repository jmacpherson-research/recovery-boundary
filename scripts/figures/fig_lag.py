import matplotlib.pyplot as plt
import numpy as np

# Natural lag distribution (recovered excursions only)
nat_lags = {1: 17989, 2: 5177, 3: 4943, 4: 1638, 5: 1338}
nat_total_recovered = sum(nat_lags.values())  # 31,085
nat_pcts = {k: 100*v/nat_total_recovered for k,v in nat_lags.items()}

# Reasoning MT lag distribution
reas_lags = {1: 60, 2: 5, 3: 3, 4: 3, 5: 1}
reas_total = sum(reas_lags.values())  # 72
reas_pcts = {k: 100*v/reas_total for k,v in reas_lags.items()}

fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

x = np.array([1, 2, 3, 4, 5])
width = 0.35

bars1 = ax.bar(x - width/2, [nat_pcts[i] for i in x], width, 
               label=f'Natural (N={nat_total_recovered:,})', color='#2171b5', alpha=0.9)
bars2 = ax.bar(x + width/2, [reas_pcts[i] for i in x], width,
               label=f'Reasoning MT (N={reas_total})', color='#cb181d', alpha=0.9)

ax.set_xlabel('Recovery Lag (chunks after excursion)', fontsize=10)
ax.set_ylabel('Percentage of Recovered Excursions', fontsize=10)
ax.set_title('Recovery Lag Distribution', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['1', '2', '3', '4', '5'])
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 100)

# Add percentage labels on natural bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=7)

# Note: standard AI has NO recovered excursions (all lag=-1), so not shown
ax.text(0.98, 0.55, 'Standard AI: 0 recovered\nexcursions (not shown)',
        transform=ax.transAxes, fontsize=7, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))

plt.tight_layout()
plt.savefig('/sessions/ecstatic-sweet-archimedes/mnt/COWORK/fig_lag.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/sessions/ecstatic-sweet-archimedes/mnt/COWORK/fig_lag.png', dpi=150, bbox_inches='tight')
print("fig_lag.pdf and fig_lag.png saved")
print(f"Natural: lag=1: {nat_pcts[1]:.1f}%, lag=2: {nat_pcts[2]:.1f}%, lag=3: {nat_pcts[3]:.1f}%, lag=4: {nat_pcts[4]:.1f}%, lag=5: {nat_pcts[5]:.1f}%")
print(f"Natural cumulative at lag=1: {nat_pcts[1]:.1f}%, at lag=2: {nat_pcts[1]+nat_pcts[2]:.1f}%, at lag=3: {nat_pcts[1]+nat_pcts[2]+nat_pcts[3]:.1f}%")
print(f"Reasoning MT: lag=1: {reas_pcts[1]:.1f}%")
