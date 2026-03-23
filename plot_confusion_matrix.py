import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

labels = ["World", "Sports", "Business", "Sci/Tech"]
n = len(labels)

# Confusion matrix from full 100-sample evaluation (After OFT)
cm = np.array([
    [15,  2,  3,  2],
    [ 0, 27,  0,  0],
    [ 0,  1, 26,  5],
    [ 1,  0,  5, 13],
], dtype=int)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(6, 5))

# Custom colormap: white→blue
cmap = mcolors.LinearSegmentedColormap.from_list("", ["#f8f9fa", "#339af0", "#1864ab"])
im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="equal")

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
cbar.ax.tick_params(labelsize=11)

# Tick labels
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold", labelpad=10)
ax.set_ylabel("True Label", fontsize=13, fontweight="bold", labelpad=10)
ax.set_title("Confusion Matrix — After OFT Finetuning", fontsize=14, fontweight="bold", pad=12)

# Annotate cells
thresh = cm.max() / 2.0
for i in range(n):
    for j in range(n):
        color = "white" if cm[i, j] > thresh else "#212529"
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=18, fontweight="bold", color=color)

# Grid lines
for edge in range(n + 1):
    ax.axhline(edge - 0.5, color="white", linewidth=2)
    ax.axvline(edge - 0.5, color="white", linewidth=2)

ax.tick_params(length=0)
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved confusion_matrix.png")
