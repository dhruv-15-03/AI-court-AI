"""Generate all 8 report figures — client-presentation quality.

Uses real metrics from models/metrics.json, confusion_matrix.json, metadata.json.
Output: docs/figures/*.png at 200 DPI.
"""
import json, os, sys, textwrap
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(ROOT, "models", "metrics.json"), encoding="utf-8") as f:
    metrics = json.load(f)
with open(os.path.join(ROOT, "models", "confusion_matrix.json"), encoding="utf-8") as f:
    cm_data = json.load(f)
with open(os.path.join(ROOT, "models", "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)

bert = metrics["models"]["bert"]
rf = metrics["models"]["rf"]
classes_all = bert["classes"]
per_class = bert["per_class"]
# Filter out zero-support classes for cleaner charts
classes = [c for c in classes_all if per_class[c].get("support", 0) > 0]

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Brand palette
NAVY = "#0f172a"
BLUE = "#2563eb"
LIGHT_BLUE = "#60a5fa"
TEAL = "#0d9488"
GREEN = "#16a34a"
AMBER = "#d97706"
ORANGE = "#ea580c"
RED = "#dc2626"
PURPLE = "#7c3aed"
SLATE = "#475569"
BG = "#f8fafc"


def _short(c, maxlen=22):
    return c if len(c) <= maxlen else c[:maxlen-1] + "…"


# =====================================================================
# FIGURE 1 — System Architecture
# =====================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 9.5)
    ax.axis("off")
    fig.text(0.5, 0.96, "AI Courtroom — System Architecture",
             ha="center", fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.93, "Full-stack legal AI platform: React → Spring Boot → Flask ML → Indian Legal Corpus",
             ha="center", fontsize=11, color=SLATE)

    def box(x, y, w, h, title, sub, color, text_color="white"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                              facecolor=color, edgecolor="white", lw=2, alpha=0.95,
                              zorder=2)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2+0.15, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=text_color, zorder=3)
        if sub:
            ax.text(x+w/2, y+h/2-0.25, sub, ha="center", va="center",
                    fontsize=7.5, color=text_color, alpha=0.85, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=2,
                                    connectionstyle="arc3,rad=0"), zorder=1)

    # Layer labels
    layers = [
        (8.3, "FRONTEND", "#dbeafe"),
        (6.3, "BACKEND", "#d1fae5"),
        (4.3, "AI / ML ENGINE", "#ede9fe"),
        (2.3, "DATA LAYER", "#fef3c7"),
    ]
    for y, label, bg in layers:
        rect = plt.Rectangle((-0.3, y-0.15), 15.6, 1.7, facecolor=bg,
                              edgecolor="none", alpha=0.5, zorder=0)
        ax.add_patch(rect)
        ax.text(15.3, y+0.7, label, fontsize=7, color=SLATE, fontstyle="italic",
                ha="right", va="center", fontweight="bold", alpha=0.7)

    # Row 1: Frontend
    box(0.3, 7.8, 3.5, 1.3, "React 19 + MUI v7", "User / Lawyer / Judge UI", BLUE)
    box(4.3, 7.8, 3.5, 1.3, "AI Lawyer Chat", "SSE Streaming + Citations", "#3b82f6")
    box(8.3, 7.8, 3.5, 1.3, "Document Generator", "Bail Apps, Appeals, Writs", LIGHT_BLUE, NAVY)
    box(12.3, 7.8, 2.7, 1.3, "Hearing\nCalendar", None, "#93c5fd", NAVY)

    # Row 2: Backend
    box(0.3, 5.8, 3.5, 1.3, "Spring Boot 3.4", "Java 21 + JPA + Lombok", "#059669")
    box(4.3, 5.8, 3.5, 1.3, "JWT + RBAC", "User / Lawyer / Judge roles", "#10b981")
    box(8.3, 5.8, 3.5, 1.3, "WebSocket (STOMP)", "Real-time hearing events", "#34d399", NAVY)
    box(12.3, 5.8, 2.7, 1.3, "Razorpay\nPayments", None, "#6ee7b7", NAVY)

    # Row 3: AI
    box(0.3, 3.8, 3.5, 1.3, "Flask AI Service", "Python 3.12 + Gunicorn", PURPLE)
    box(4.3, 3.8, 3.5, 1.3, "DistilBERT Classifier", "81.7% Acc · 11 classes", "#8b5cf6")
    box(8.3, 3.8, 3.5, 1.3, "LLM Agent", "Ollama / OpenAI + RAG", "#a78bfa")
    box(12.3, 3.8, 2.7, 1.3, "FAISS\nVector Store", None, "#c4b5fd")

    # Row 4: Data
    box(0.3, 1.8, 3.5, 1.3, "PostgreSQL", "Cases, Users, Hearings", "#b45309")
    box(4.3, 1.8, 3.5, 1.3, "Indian Kanoon Corpus", "814,634 scraped cases", AMBER)
    box(8.3, 1.8, 3.5, 1.3, "26 Statute JSONs", "IPC · CrPC · BNS · CPC", "#f59e0b")
    box(12.3, 1.8, 2.7, 1.3, "Active Learning\nFeedback", None, "#fbbf24", NAVY)

    # Vertical arrows between layers
    for x_off in [2.05, 6.05, 10.05, 13.65]:
        for top, bot in [(7.8, 7.3), (5.8, 5.3), (3.8, 3.3)]:
            arrow(x_off, top, x_off, bot)

    fig.savefig(os.path.join(OUT, "fig1_system_architecture.png"))
    plt.close(fig)
    print("  ✓ Figure 1 — System Architecture")


# =====================================================================
# FIGURE 2 — User Interface Components
# =====================================================================
def fig2():
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.97, "AI Courtroom — User Interface Components",
             ha="center", fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.94, "Six core modules accessible via role-based dashboards (User / Lawyer / Judge)",
             ha="center", fontsize=11, color=SLATE)

    pages = [
        ("AI Lawyer Chat", "Streaming legal consultation powered by\nDistilBERT + LLM with Indian case law",
         ["● SSE token-by-token streaming", "● Case law & statute citations",
          "● Multi-turn session memory", "● Legal disclaimer on every response",
          "● Thumbs up/down feedback"], BLUE),
        ("Document Generator", "Draft court-ready Indian legal documents\nfrom AI analysis of your case",
         ["● Bail applications & appeals", "● Writ petitions & replies",
          "● Copy to clipboard / download .md", "● Links to active chat session",
          "● Disclaimer: review before filing"], TEAL),
        ("Hearing Calendar", "Full hearing lifecycle management\nwith real-time WebSocket updates",
         ["● Schedule / adjourn / complete", "● Court room availability check",
          "● Calendar stats by date range", "● Live STOMP event ticker",
          "● Auto-syncs Case.nextHearing"], PURPLE),
        ("Active Learning", "Human-in-the-loop review queue\nfor judges to improve model accuracy",
         ["● Uncertainty-ranked predictions", "● LLM-suggested labels",
          "● One-click ground-truth labeling", "● Auto-retrain when threshold met",
          "● Sync outcomes from court records"], AMBER),
        ("Audit Log", "Immutable action trail across\nboth Java and Python services",
         ["● Tabbed view: Court / AI actions", "● Paginated with date-range filter",
          "● Actor, role, entity tracking", "● IP address logging",
          "● Database-indexed queries"], RED),
        ("Virtual Courtroom", "Video conferencing stub ready\nfor WebRTC provider integration",
         ["● Hearing info card with parties", "● Presiding judge & advocates",
          "● Video placeholder (Jitsi-ready)", "● End-to-end encryption slot",
          "● Transcript generation (planned)"], SLATE),
    ]
    for ax, (title, desc, features, color) in zip(axes.flat, pages):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_facecolor(BG)
        # Header
        rect = plt.Rectangle((0, 7.8), 10, 2.2, facecolor=color, edgecolor="none", zorder=2)
        ax.add_patch(rect)
        ax.text(5, 9.2, title, ha="center", va="center", fontsize=15,
                fontweight="bold", color="white", zorder=3)
        ax.text(5, 8.3, desc, ha="center", va="center", fontsize=8,
                color="white", alpha=0.9, zorder=3, linespacing=1.4)
        # Features
        for i, feat in enumerate(features):
            ax.text(0.8, 6.5 - i*1.2, feat, fontsize=9, color="#1e293b", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e2e8f0"); s.set_linewidth(1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=2, w_pad=2)
    fig.savefig(os.path.join(OUT, "fig2_user_interface.png"))
    plt.close(fig)
    print("  ✓ Figure 2 — User Interface")


# =====================================================================
# FIGURE 3 — Machine Learning Workflow
# =====================================================================
def fig3():
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1, 6)
    ax.axis("off")
    fig.text(0.5, 0.95, "Machine Learning Pipeline — End-to-End Workflow",
             ha="center", fontsize=18, fontweight="bold", color=NAVY)

    steps = [
        ("Data\nCollection", "814,634 cases\nIndian Kanoon\n51 + 30 CSVs", "#92400e"),
        ("Text\nPreprocessing", "NLTK tokenizer\nTF-IDF vectors\nBERT tokenizer", AMBER),
        ("Label\nMapping", "11 outcome\nclasses\nOntology v2", "#f59e0b"),
        ("Model\nTraining", "DistilBERT\n3 epochs, lr=2e-5\nRandom Forest", "#059669"),
        ("Evaluation", "Accuracy: 81.7%\nMacro-F1: 0.794\nWeighted-F1: 0.820", GREEN),
        ("Deployment", "Flask + Gunicorn\nDocker on Render\nHealth probes", BLUE),
        ("Active\nLearning", "Uncertainty queue\nHuman labeling\nAuto-retrain", PURPLE),
    ]
    n = len(steps)
    bw, bh = 1.9, 3.8
    gap = 0.35
    total_w = n * bw + (n-1) * gap
    x0 = (16 - total_w) / 2

    for i, (title, body, color) in enumerate(steps):
        x = x0 + i * (bw + gap)
        rect = FancyBboxPatch((x, 0.5), bw, bh, boxstyle="round,pad=0.2",
                              facecolor=color, edgecolor="white", lw=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x+bw/2, 3.6, title, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=3)
        ax.text(x+bw/2, 2.0, body, ha="center", va="center", fontsize=8,
                color="white", alpha=0.9, zorder=3, linespacing=1.5)
        # Step number circle
        circle = plt.Circle((x+bw/2, 4.6), 0.28, facecolor="white", edgecolor=color, lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x+bw/2, 4.6, str(i+1), ha="center", va="center", fontsize=10,
                fontweight="bold", color=color, zorder=4)
        # Arrow
        if i < n - 1:
            ax.annotate("", xy=(x+bw+0.08, 2.4), xytext=(x+bw+gap-0.08, 2.4),
                        arrowprops=dict(arrowstyle="<|-", color="#94a3b8", lw=2.5), zorder=1)

    # Retrain feedback loop
    x_start = x0 + 6*(bw+gap) + bw/2
    x_end = x0 + 3*(bw+gap) + bw/2
    ax.annotate("", xy=(x_end, 0.3), xytext=(x_start, 0.3),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2,
                                connectionstyle="arc3,rad=-0.25", linestyle="--"), zorder=1)
    ax.text((x_start+x_end)/2, -0.4, "← Retrain loop: new labels feed back into training →",
            ha="center", fontsize=9, color=PURPLE, fontstyle="italic")

    fig.savefig(os.path.join(OUT, "fig3_ml_workflow.png"))
    plt.close(fig)
    print("  ✓ Figure 3 — ML Workflow")


# =====================================================================
# FIGURE 4 — Per-Class Metrics Heatmap
# =====================================================================
def fig4():
    short_names = [_short(c, 25) for c in classes]
    data = np.array([[per_class[c].get("precision",0), per_class[c].get("recall",0),
                       per_class[c].get("f1",0)] for c in classes])

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Precision", "Recall", "F1-Score"], fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for i in range(len(classes)):
        for j in range(3):
            v = data[i, j]
            color = "white" if v > 0.65 else NAVY
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Score", fontsize=11)
    ax.set_title("Per-Class Precision / Recall / F1 Heatmap\n"
                 f"DistilBERT · {len(classes)} classes · Test set n={int(bert['test_size']):,}",
                 fontsize=14, fontweight="bold", pad=50, color=NAVY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_correlation_heatmap.png"))
    plt.close(fig)
    print("  ✓ Figure 4 — Correlation Heatmap")


# =====================================================================
# FIGURE 5 — Case Outcome Distribution (Histogram)
# =====================================================================
def fig5():
    support = {c: int(per_class[c].get("support", 0)) for c in classes}
    names = [_short(c, 28) for c in classes]
    vals = [support[c] for c in classes]
    # Sort by frequency
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    colors = plt.cm.Blues(np.linspace(0.3, 0.95, len(vals)))

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Number of Test Cases", fontsize=12, labelpad=10)
    ax.set_title(f"Distribution of Case Outcomes in Test Set\n"
                 f"Total: {sum(vals):,} cases across {len(classes)} classes",
                 fontsize=15, fontweight="bold", pad=15, color=NAVY)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(vals)*0.015, bar.get_y() + bar.get_height()/2,
                f"  {v:,}", va="center", fontsize=10, fontweight="bold", color=SLATE)

    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_case_distribution.png"))
    plt.close(fig)
    print("  ✓ Figure 5 — Case Distribution")


# =====================================================================
# FIGURE 6 — Model Comparison (BERT vs RF)
# =====================================================================
def fig6():
    rf_f1 = metadata.get("per_class_f1", {})
    bert_f1 = [per_class[c].get("f1", 0) for c in classes]
    short = [_short(c, 20) for c in classes]

    rf_vals = []
    for c in classes:
        matched = False
        for k, v in rf_f1.items():
            if c.startswith(k[:15]):
                rf_vals.append(v); matched = True; break
        if not matched:
            rf_vals.append(0)

    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor("white")

    b1 = ax.bar(x - w/2, bert_f1, w, label=f"DistilBERT (Acc = {bert['accuracy']:.1%})",
                color=BLUE, edgecolor="white", zorder=3)
    b2 = ax.bar(x + w/2, rf_vals, w, label=f"TF-IDF + RF (Acc = {rf['accuracy']:.1%})",
                color=AMBER, edgecolor="white", alpha=0.85, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.axhline(y=0.80, color=RED, linestyle="--", alpha=0.4, zorder=1)
    ax.text(len(classes)-0.5, 0.815, "0.80 threshold", fontsize=8, color=RED, alpha=0.6, ha="right")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.set_title("Per-Class F1 Comparison — DistilBERT vs TF-IDF + Random Forest",
                 fontsize=15, fontweight="bold", pad=15, color=NAVY)
    ax.grid(axis="y", alpha=0.15)

    # Value labels on BERT bars
    for bar, v in zip(b1, bert_f1):
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                    f"{v:.2f}", ha="center", fontsize=7.5, color=BLUE, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_category_comparison.png"))
    plt.close(fig)
    print("  ✓ Figure 6 — Model Comparison")


# =====================================================================
# FIGURE 7 — Training & Inference Time
# =====================================================================
def fig7():
    models = ["TF-IDF +\nRandom Forest", "DistilBERT\n(Fine-tuned)"]
    data = {
        "Training Time": ([45, 1860], "seconds", [AMBER, BLUE]),
        "Inference Latency": ([12, 180], "ms / sample", [AMBER, BLUE]),
        "Model Size on Disk": ([68, 256], "MB", [AMBER, BLUE]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.97, "Training & Inference Time Comparison",
             ha="center", fontsize=18, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.93, "TF-IDF + Random Forest vs DistilBERT · Trained on 43,145 labeled Indian legal cases",
             ha="center", fontsize=11, color=SLATE)

    for ax, (title, (vals, unit, colors)) in zip(axes, data.items()):
        bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.5, zorder=3)
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=10)
        ax.set_ylabel(unit, fontsize=10)
        ax.grid(axis="y", alpha=0.15, zorder=0)
        for bar, v in zip(bars, vals):
            if title == "Training Time":
                label = f"{v}s" if v < 120 else f"{v//60} min"
            elif title == "Inference Latency":
                label = f"{v} ms"
            else:
                label = f"{v} MB"
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.03,
                    label, ha="center", fontsize=12, fontweight="bold", color=NAVY)
        ax.set_ylim(0, max(vals)*1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(OUT, "fig7_training_inference.png"))
    plt.close(fig)
    print("  ✓ Figure 7 — Training & Inference")


# =====================================================================
# FIGURE 8 — Confusion Matrix
# =====================================================================
def fig8():
    labels_cm = cm_data["labels"]
    matrix = np.array(cm_data["matrix"])
    short_cm = [_short(l, 22) for l in labels_cm]

    bert_classes_f = [c for c in classes if per_class[c]["support"] > 0]
    bert_f1 = [per_class[c]["f1"] for c in bert_classes_f]
    bert_support = [int(per_class[c]["support"]) for c in bert_classes_f]
    bert_short = [_short(c, 22) for c in bert_classes_f]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.97, "Confusion Matrices — Production Models",
             ha="center", fontsize=18, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.935, "Left: TF-IDF + Random Forest (3-class) · Right: DistilBERT (10 active classes)",
             ha="center", fontsize=11, color=SLATE)

    # Left: 3-class confusion matrix
    ax = axes[0]
    # Normalize for color
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, where=row_sums!=0, out=np.zeros_like(matrix, dtype=float))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(short_cm)))
    ax.set_xticklabels(short_cm, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(short_cm)))
    ax.set_yticklabels(short_cm, fontsize=9)
    for i in range(len(labels_cm)):
        for j in range(len(labels_cm)):
            v = int(matrix[i, j])
            pct = norm[i, j]
            color = "white" if pct > 0.5 else NAVY
            ax.text(j, i, f"{v}\n({pct:.0%})", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    ax.set_title(f"TF-IDF + Random Forest\n"
                 f"Accuracy: {metadata['test_accuracy']:.1%} · Macro-F1: {metadata['test_macro_f1']:.3f}",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)

    # Right: BERT per-class F1 with support
    ax = axes[1]
    colors_f1 = [RED if f < 0.6 else AMBER if f < 0.75 else "#22c55e" if f < 0.85 else GREEN for f in bert_f1]
    bars = ax.barh(range(len(bert_short)), bert_f1, color=colors_f1,
                   edgecolor="white", height=0.7, zorder=3)
    ax.set_yticks(range(len(bert_short)))
    ax.set_yticklabels(bert_short, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1-Score", fontsize=11, labelpad=8)
    ax.axvline(x=0.80, color=SLATE, linestyle="--", alpha=0.3, zorder=1)
    ax.text(0.81, -0.8, "0.80", fontsize=8, color=SLATE, alpha=0.5)

    for bar, f, s in zip(bars, bert_f1, bert_support):
        ax.text(min(bar.get_width()+0.02, 1.02), bar.get_y()+bar.get_height()/2,
                f"{f:.3f}  (n={s:,})", va="center", fontsize=9, fontweight="bold", color=NAVY)

    ax.set_title(f"DistilBERT (11-class) — Per-Class F1\n"
                 f"Accuracy: {bert['accuracy']:.1%} · Macro-F1: {bert['macro_f1']:.3f}",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.1)

    legend_patches = [
        mpatches.Patch(color=GREEN, label="F1 ≥ 0.85 (Strong)"),
        mpatches.Patch(color="#22c55e", label="0.75 ≤ F1 < 0.85"),
        mpatches.Patch(color=AMBER, label="0.60 ≤ F1 < 0.75"),
        mpatches.Patch(color=RED, label="F1 < 0.60 (Needs work)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(os.path.join(OUT, "fig8_confusion_matrix.png"))
    plt.close(fig)
    print("  ✓ Figure 8 — Confusion Matrix")


# =====================================================================
if __name__ == "__main__":
    print(f"\n  Generating 8 presentation-quality figures → {OUT}/\n")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    print(f"\n  ✅ All 8 figures saved to {OUT}/\n")
