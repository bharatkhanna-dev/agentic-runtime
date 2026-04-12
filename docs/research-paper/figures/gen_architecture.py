"""Figure 1: agentic-runtime system architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

# ── colour palette ───────────────────────────────────────────────────────────
C_ORCH   = "#2563EB"   # blue   – orchestrator
C_GUARD  = "#DC2626"   # red    – guardrail layer
C_TOOL   = "#059669"   # green  – tool registry
C_WORK   = "#7C3AED"   # purple – workload executors
C_EVAL   = "#D97706"   # amber  – evaluation harness
C_LG     = "#64748B"   # slate  – LangGraph substrate
C_TEXT   = "#F8FAFC"   # near-white for labels inside boxes
C_DARK   = "#1E293B"   # dark for external label

def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=11):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cy = y + h / 2 + (0.18 if sublabel else 0)
    ax.text(x + w / 2, cy, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=C_TEXT, zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.28, sublabel,
                ha="center", va="center", fontsize=8.5,
                color=C_TEXT, alpha=0.88, zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#94A3B8", label=None):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
        zorder=2
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.05, my + 0.1, label,
                ha="left", va="bottom", fontsize=8, color=C_DARK)

# ── LangGraph substrate (background band) ────────────────────────────────────
bg = FancyBboxPatch(
    (0.4, 0.35), 11.2, 7.2,
    boxstyle="round,pad=0.1",
    facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=2, zorder=1
)
ax.add_patch(bg)
ax.text(6.0, 7.28, "LangGraph Execution Substrate",
        ha="center", va="center", fontsize=10,
        color=C_LG, fontweight="bold", zorder=4)

# ── Orchestrator (centre-top) ─────────────────────────────────────────────────
box(ax, 3.8, 5.4, 4.4, 1.4, C_ORCH,
    "Orchestrator",
    "RunState · NodeLifecycle · Retries · Termination")

# ── Tool Registry (left) ──────────────────────────────────────────────────────
box(ax, 0.7, 5.4, 2.6, 1.4, C_TOOL,
    "Tool Registry",
    "ToolSpec · Permissions\n(allow / approval / deny)")

# ── Guardrail Layer (right) ───────────────────────────────────────────────────
box(ax, 8.7, 5.4, 2.8, 1.4, C_GUARD,
    "Guardrail Layer",
    "before_tool_call\naction · output checkpoints")

# ── Workload Executors (bottom-left) ─────────────────────────────────────────
box(ax, 0.7, 2.8, 2.6, 1.9, C_WORK,
    "Support Triage",
    "approve / escalate\ntool-order rules")

box(ax, 3.8, 2.8, 4.4, 1.9, C_WORK,
    "Research & Retrieval",
    "search_corpus · draft_answer\nretrieval budget · citation recall")

# ── Evaluation Harness (bottom-right) ────────────────────────────────────────
box(ax, 8.7, 2.8, 2.8, 1.9, C_EVAL,
    "Evaluation Harness",
    "ARS · Task Success\nGuardrail Compliance")

# ── RunState detail box (bottom-centre) ──────────────────────────────────────
rs_rect = FancyBboxPatch(
    (4.6, 0.6), 2.8, 1.5,
    boxstyle="round,pad=0.08",
    facecolor="#DBEAFE", edgecolor=C_ORCH, linewidth=1.5, zorder=3
)
ax.add_patch(rs_rect)
ax.text(6.0, 1.35, "RunState",
        ha="center", va="center", fontsize=10,
        fontweight="bold", color=C_ORCH, zorder=4)
ax.text(6.0, 0.93,
        "run_id · step_count · token_count\nnode_results · guardrail_events",
        ha="center", va="center", fontsize=8.2,
        color="#1E3A5F", zorder=4)

# ── Arrows ────────────────────────────────────────────────────────────────────
# Tool Registry → Orchestrator
arrow(ax, 3.3, 6.1, 3.8, 6.1, label="tool lookup")
# Orchestrator → Guardrail Layer
arrow(ax, 8.2, 6.1, 8.7, 6.1, label="policy check")
# Guardrail → Orchestrator (allow/deny)
arrow(ax, 8.7, 6.5, 8.2, 6.6, color=C_GUARD, label="allow / deny")
# Orchestrator → Support Triage
arrow(ax, 4.8, 5.4, 3.0, 4.7, label="dispatch")
# Orchestrator → Research
arrow(ax, 6.0, 5.4, 6.0, 4.7, label="dispatch")
# Orchestrator → Eval
arrow(ax, 7.2, 5.4, 9.2, 4.7, label="results")
# Support Triage → RunState
arrow(ax, 2.0, 2.8, 5.2, 2.1, color=C_WORK)
# Research → RunState
arrow(ax, 6.0, 2.8, 6.0, 2.1, color=C_WORK)
# Eval → RunState
arrow(ax, 9.8, 2.8, 7.4, 2.1, color=C_EVAL)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_ORCH,  label="Orchestrator"),
    mpatches.Patch(facecolor=C_GUARD, label="Guardrail Layer"),
    mpatches.Patch(facecolor=C_TOOL,  label="Tool Registry"),
    mpatches.Patch(facecolor=C_WORK,  label="Workload Executors"),
    mpatches.Patch(facecolor=C_EVAL,  label="Evaluation Harness"),
]
ax.legend(handles=legend_items, loc="lower left",
          bbox_to_anchor=(0.0, 0.0), fontsize=9,
          frameon=True, framealpha=0.9, ncol=5)

plt.title("Figure 1  –  agentic‑runtime System Architecture",
          fontsize=13, fontweight="bold", pad=12, color=C_DARK)
plt.tight_layout()
plt.savefig("figure1-architecture.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("Saved figure1-architecture.png")
