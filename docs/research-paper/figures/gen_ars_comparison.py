"""Figure 2: Overall Pair B benchmark comparison – Task Success and ARS."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

variants = ["Single-agent", "Multi-agent\nBaseline", "Multi-agent\nGuarded"]
task_success = [0.0000, 0.5000, 1.0000]
ars          = [0.4395, 0.5765, 0.8545]

x = np.arange(len(variants))
width = 0.32

C_TASK = "#2563EB"
C_ARS  = "#059669"
C_EDGE = "white"

fig, ax = plt.subplots(figsize=(8, 5.2))

bars1 = ax.bar(x - width/2, task_success, width,
               label="Mean Task Success", color=C_TASK,
               edgecolor=C_EDGE, linewidth=0.8)
bars2 = ax.bar(x + width/2, ars, width,
               label="Mean ARS", color=C_ARS,
               edgecolor=C_EDGE, linewidth=0.8)

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
            f"{h:.4f}", ha="center", va="bottom", fontsize=9.5, color="#1E293B")

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
            f"{h:.4f}", ha="center", va="bottom", fontsize=9.5, color="#1E293B")

# Reference lines
ax.axhline(1.0, color="#94A3B8", linewidth=0.9, linestyle="--", zorder=0)
ax.axhline(0.5, color="#CBD5E1", linewidth=0.6, linestyle=":", zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=11)
ax.set_xlabel("Runtime Variant", fontsize=11)
ax.set_title("Figure 2  –  Pair B Benchmark: Task Success and ARS by Variant",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=10, frameon=True, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
ax.grid(axis="y", color="#E2E8F0", linewidth=0.7, zorder=0)

plt.tight_layout()
plt.savefig("figure2-ars-comparison.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("Saved figure2-ars-comparison.png")
