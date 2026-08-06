import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("experiments/5M_arch_tournament_results.json") as f:
    data = json.load(f)

# Organize by arch -> iter -> [seed1_wr, seed2_wr]
arch_data = {}
for net in data["networks"]:
    arch = net["arch"]
    it = net["iter"]
    wr = net["avg_winrate"]
    if arch not in arch_data:
        arch_data[arch] = {}
    if it not in arch_data[arch]:
        arch_data[arch][it] = []
    arch_data[arch][it].append(wr)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
ax.set_facecolor('white')

colors = {"deep14": "#E74C3C", "mid8": "#3498DB", "wide4": "#2ECC71"}
markers = {"deep14": "o", "mid8": "s", "wide4": "D"}
labels = {"deep14": "Deep14 (d=14, e=160, 5.17M)", "mid8": "Mid8 (d=8, e=256, 5.24M)", "wide4": "Wide4 (d=4, e=352, 5.19M)"}

for arch in ["deep14", "mid8", "wide4"]:
    iters_sorted = sorted(arch_data[arch].keys())
    means = []
    lows = []
    highs = []
    for it in iters_sorted:
        vals = arch_data[arch][it]
        mean = np.mean(vals)
        means.append(mean)
        lows.append(min(vals))
        highs.append(max(vals))

    means = np.array(means)
    lows = np.array(lows)
    highs = np.array(highs)
    iters_arr = np.array(iters_sorted)

    ax.plot(iters_arr, means, color=colors[arch], marker=markers[arch],
            linewidth=2, markersize=8, label=labels[arch], zorder=3)
    ax.fill_between(iters_arr, lows, highs, color=colors[arch], alpha=0.2, zorder=2)

    # Plot individual seed points
    for it in iters_sorted:
        vals = arch_data[arch][it]
        for v in vals:
            ax.scatter(it, v, color=colors[arch], s=30, alpha=0.5, zorder=4)

ax.set_xlabel("Training Iteration", fontsize=13)
ax.set_ylabel("Average Tournament Winrate (%)", fontsize=13)
ax.set_title("5M Architecture Comparison — Small Maps (S)\nTruncation=512, 500 games/matchup, 2 seeds per arch", fontsize=14)
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_ylim(20, 80)

plt.tight_layout()
plot_path = "experiments/5M_arch_winrate_progression.png"
plt.savefig(plot_path, dpi=150, facecolor='white', bbox_inches='tight')
print(f"Plot saved to {plot_path}")
