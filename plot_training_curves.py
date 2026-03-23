import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

with open("results_summary.json", "r") as f:
    data = json.load(f)

train_loss = data["train_loss_history"]
eval_hist = data["eval_history"]
before = data["before_finetuning"]
after = data["after_finetuning"]

steps = [d["step"] for d in train_loss]
losses = [d["loss"] for d in train_loss]
smoothed = uniform_filter1d(losses, size=7)

eval_steps = [d["step"] for d in eval_hist]
eval_losses = [d["loss"] for d in eval_hist]
eval_acc = [d["accuracy"] for d in eval_hist]
eval_f1 = [d["f1_macro"] for d in eval_hist]

# ---- Style ----
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.color": "#ced4da",
    "font.family": "sans-serif",
    "font.size": 11,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("OFT Generative Finetuning — Qwen3-0.6B on AG News", fontsize=16, fontweight="bold", y=0.97)
fig.subplots_adjust(hspace=0.35, wspace=0.30, top=0.91, bottom=0.07, left=0.08, right=0.96)

# ---- Panel 1: Training Loss ----
ax = axes[0, 0]
ax.plot(steps, losses, color="#74c0fc", linewidth=0.9, alpha=0.7, label="Training Loss")
ax.plot(steps, smoothed, color="#e03131", linewidth=2.2, label="Smoothed")
ax.set_title("Training Loss Curve", fontweight="bold")
ax.set_xlabel("Training Steps")
ax.set_ylabel("Loss")
ax.legend(framealpha=0.9)

# ---- Panel 2: Evaluation Loss ----
ax = axes[0, 1]
ax.plot(eval_steps, eval_losses, "o-", color="#40c057", linewidth=2.2, markersize=8)
ax.set_title("Evaluation Loss", fontweight="bold")
ax.set_xlabel("Training Steps")
ax.set_ylabel("Loss")

# ---- Panel 3: Accuracy & F1 ----
ax = axes[1, 0]
ax.plot(eval_steps, eval_acc, "s-", color="#845ef7", linewidth=2.2, markersize=8, label="Accuracy")
ax.plot(eval_steps, eval_f1, "D-", color="#f59f00", linewidth=2.2, markersize=8, label="F1 (Macro)")
ax.set_title("Accuracy & F1 Score", fontweight="bold")
ax.set_xlabel("Training Steps")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(framealpha=0.9)

# ---- Panel 4: Before vs After bar chart ----
ax = axes[1, 1]
metrics = ["Accuracy", "F1\n(Macro)", "Precision\n(Macro)", "Recall\n(Macro)"]
before_vals = [before["accuracy"], before["f1_macro"], before["precision_macro"], before["recall_macro"]]
after_vals = [after["accuracy"], after["f1_macro"], after["precision_macro"], after["recall_macro"]]

x = np.arange(len(metrics))
w = 0.32
bars_b = ax.bar(x - w/2, before_vals, w, label="Before (Zero-shot)", color="#adb5bd", edgecolor="white", linewidth=0.8)
bars_a = ax.bar(x + w/2, after_vals, w, label="After (OFT)", color="#339af0", edgecolor="white", linewidth=0.8)

# Value labels
for bar in bars_b:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color="#495057")
for bar in bars_a:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color="#1864ab", fontweight="bold")

ax.set_title("Before vs After Finetuning", fontweight="bold")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.08)
ax.legend(framealpha=0.9, loc="upper left")

plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("Saved training_curves.png")
