"""Figure 3: Per-workload metric breakdown across all three variants."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

variants = ["Single-agent", "Multi-agent Baseline", "Multi-agent Guarded"]
short     = ["Single", "Baseline", "Guarded"]

# ── Support Triage metrics ────────────────────────────────────────────────────
st_labels = ["Task\nSuccess", "Approval\nRouting", "Tool Order\nAccuracy",
             "Cost\nEfficiency", "Latency\nScore", "ARS"]
st_data = {
    "Single-agent":        [0.0000, 0.3333, 0.0000, 0.7000, 0.9000, 0.3750],
    "Multi-agent Baseline":[0.3333, 0.3333, 1.0000, 0.5500, 0.8000, 0.4467],
    "Multi-agent Guarded": [1.0000, 1.0000, 1.0000, 0.4600, 0.7500, 0.8545],
}

# ── Research & Retrieval metrics ──────────────────────────────────────────────
rr_labels = ["Task\nSuccess", "Keyword\nCoverage", "Citation\nRecall",
             "Retrieval\nBudget", "Cost\nEfficiency", "Latency\nScore", "ARS"]
rr_data = {
    "Single-agent":        [0.0000, 0.4444, 0.5000, 1.0000, 0.7200, 0.9000, 0.5040],
    "Multi-agent Baseline":[0.6667, 1.0000, 1.0000, 0.6667, 0.5000, 0.8200, 0.7063],
    "Multi-agent Guarded": [1.0000, 1.0000, 1.0000, 1.0000, 0.4600, 0.7500, 0.8545],
}

COLORS = ["#64748B", "#7C3AED", "#059669"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

def grouped_bars(ax, labels, data_dict, title):
    x = np.arange(len(labels))
    n = len(data_dict)
    w = 0.22
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)
    for i, (variant, vals) in enumerate(data_dict.items()):
        bars = ax.bar(x + offsets[i], vals, w, label=short[i],
                      color=COLORS[i], edgecolor="white", linewidth=0.7)
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=7.2, color="#1E293B", rotation=90)
    # Highlight ARS bar group with subtle background
    ax.axvspan(x[-1] - 0.45, x[-1] + 0.45, color="#FEFCE8", alpha=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7, zorder=0)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, loc="upper left")
    ax.axhline(1.0, color="#94A3B8", linewidth=0.8, linestyle="--", zorder=0)

grouped_bars(ax1, st_labels, st_data,
             "Support Triage  –  Per-Metric Results")
grouped_bars(ax2, rr_labels, rr_data,
             "Research & Retrieval  –  Per-Metric Results")

fig.suptitle("Figure 3  –  Per-Workload Metric Breakdown by Runtime Variant",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figure3-workload-breakdown.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("Saved figure3-workload-breakdown.png")
