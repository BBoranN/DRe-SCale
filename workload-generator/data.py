import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

data = [41, 1, 15, 1, 1, 1, 2, 14, 1, 1, 15, 2, 1, 43, 1, 3, 12, 1, 1, 1, 16, 1, 1, 15, 2, 1, 43, 1, 4, 11, 1, 1, 1, 16, 1, 1, 1, 14, 2, 1, 5, 38, 1, 4, 11, 1, 1, 1, 2, 14, 1, 1, 1, 14, 1, 1, 1, 43, 1, 15, 2, 1, 4, 12, 1, 1, 4, 11, 1, 1, 1, 2, 41, 1, 1, 14, 2, 1, 4, 12, 1, 1, 2, 13, 2, 1, 43, 1, 1, 14, 1, 1, 1, 3, 13, 1, 1, 15, 2, 1, 4, 39, 1, 2, 13, 2, 1, 16, 1, 1, 1, 14, 1, 1, 1, 43, 1, 1, 14, 1, 1, 1, 1, 15, 1, 1, 15, 2, 1, 4, 39, 1, 15, 1, 1, 1, 6, 10, 1, 1, 15, 1, 1, 1, 6, 37, 1, 2, 13, 1, 1, 1, 1, 15, 1, 1, 15, 2, 1, 43, 1, 15, 2, 1, 16, 1, 1, 15, 2, 1, 1, 42, 1, 15, 1, 1, 1, 2, 14, 1, 1, 15, 2, 1, 1, 42, 1, 15, 2, 1, 3, 13, 1, 1, 3, 12, 2, 1, 1, 42, 1, 2, 13, 1, 1, 1, 16, 1, 1, 15, 1, 1, 1, 3, 40, 1, 2, 13, 1, 1, 1, 5, 11, 1, 1, 4, 11, 2, 1, 1, 42, 1, 15, 1, 1, 1, 5, 11, 1, 1, 15, 2, 1, 43, 1, 2, 13, 1, 1, 1, 6, 10, 1, 1, 15, 2, 1, 4, 39, 1, 15, 1, 1, 1, 2, 14, 1, 1, 15, 2, 1, 5, 38, 1, 5, 10, 1, 1, 1, 6, 10, 1, 1, 5, 10, 2, 1, 4, 39, 1, 2, 13, 1, 1, 1, 5, 11, 1, 1, 1, 14, 2, 1, 3, 40, 1, 2, 13, 2, 1, 16, 1, 1, 15, 1, 1, 1, 1, 42, 1, 4, 11, 2, 1, 4, 12, 1, 1, 3, 12, 2, 1, 1, 42, 1, 2, 13, 2, 1, 2, 14, 1, 1, 15, 2, 1]

fig = plt.figure(figsize=(14, 8))
fig.suptitle("Invocation Data", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Time series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(data, linewidth=0.8, color="#3a7dc9", alpha=0.85)
ax1.set_title("Time Series")
ax1.set_xlabel("Index")
ax1.set_ylabel("Value")
ax1.grid(True, linestyle="--", alpha=0.4)

# Histogram
ax2 = fig.add_subplot(gs[1, 0])
unique_vals, counts = np.unique(data, return_counts=True)
ax2.bar(unique_vals, counts, color="#3a7dc9", edgecolor="white", linewidth=0.5)
ax2.set_title("Value Distribution")
ax2.set_xlabel("Value")
ax2.set_ylabel("Frequency")
ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

# CDF
ax3 = fig.add_subplot(gs[1, 1])
sorted_data = np.sort(data)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax3.step(sorted_data, cdf, color="#e07b39", linewidth=1.5, where="post")
ax3.set_title("Cumulative Distribution (CDF)")
ax3.set_xlabel("Value")
ax3.set_ylabel("CDF")
ax3.grid(True, linestyle="--", alpha=0.4)

plt.savefig("invocation_data_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to invocation_data_plot.png")
print(f"n={len(data)}, min={min(data)}, max={max(data)}, mean={np.mean(data):.2f}, median={np.median(data):.1f}")
