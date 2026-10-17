"""Blog charts, styled to match adarshsolanki.com (cream background, burnt-orange accent)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#FBF7F2"; INK = "#2E2622"; MUTED = "#6E625A"; GRID = "#E7DFD5"
ORANGE = "#B8501E"; TEAL = "#00899B"; GOLD = "#C28A1E"; PURPLE = "#6B5CA5"
GRAY = "#BDB2A7"; LIGHT = "#D9CFC4"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def clean(ax, grid="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if grid == "both":
        ax.grid(True, color=GRID, lw=0.8)
    elif grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def title(fig, t, sub=None):
    fig.text(0.03, 0.965, t, fontsize=15, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.03, 0.905, sub, fontsize=11, color=MUTED, va="top")


def save(fig, name):
    fig.savefig(name, dpi=200)
    plt.close(fig)


# 1. Score ladder
rows = [
    ("Qwen3-4B, untrained", 0.17, LIGHT, "4b"),
    ("+ fine-tuning (3 epochs)", 0.39, GRAY, "4b"),
    ("+ RL", 0.40, GRAY, "4b"),
    ("Fine-tuning, 8 epochs", 0.53, GRAY, "4b"),
    ("Fine-tuning + search tool", 0.74, ORANGE, "4b"),
    ("GPT-6 Luna", 0.69, LIGHT, "big"),
    ("Kimi K2.5 (~1T params)", 0.73, LIGHT, "big"),
    ("GPT-6 Luna + two sentences", 0.81, TEAL, "big"),
]
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=100)
fig.subplots_adjust(left=0.31, right=0.95, top=0.82, bottom=0.1)
y = np.arange(len(rows))[::-1].astype(float)
y[5:] -= 0.6
for yi, (lab, v, c, _) in zip(y, rows):
    ax.barh(yi, v, height=0.62, color=c)
    bold = c in (ORANGE, TEAL)
    ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=12, color=INK, fontweight="bold" if bold else "normal")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=12)
for t, r in zip(ax.get_yticklabels(), rows):
    if r[2] in (ORANGE, TEAL):
        t.set_color(INK); t.set_fontweight("bold")
ax.axvline(0.82, color=MUTED, lw=1.2, ls=(0, (4, 3)))
ax.text(0.82, y[0] + 0.55, "copy the chart + code perfectly: 0.82", ha="right", va="bottom", fontsize=10, color=MUTED)
ax.text(-0.01, y[0] + 0.55, "MY 4B MODEL", transform=ax.get_yaxis_transform(), ha="right", fontsize=9.5, color=MUTED, fontweight="bold")
ax.text(-0.01, y[5] + 0.55, "FRONTIER MODELS", transform=ax.get_yaxis_transform(), ha="right", fontsize=9.5, color=MUTED, fontweight="bold")
ax.set_xlim(0, 0.9); ax.set_ylim(y[-1] - 0.6, y[0] + 1.0)
ax.set_xlabel("F1 on the 200 test patients")
clean(ax)
title(fig, "RL added 0.01. A search tool added 0.21.", "And two sentences in the prompt took GPT-6 Luna to the top.")
save(fig, "score-ladder.png")

# 2. RL signal
W = ["Updates 1–30", "31–55", "56–85"]
never = [33.1, 33.8, 34.5]; some = [37.5, 33.8, 25.0]; always = [29.4, 32.4, 40.4]
fig, ax = plt.subplots(figsize=(10, 4.0), dpi=100)
fig.subplots_adjust(left=0.15, right=0.97, top=0.74, bottom=0.04)
yy = np.arange(3)[::-1]
segs = [(never, [0] * 3, LIGHT, INK, "Never right"), (some, never, GOLD, INK, "Sometimes right"),
        (always, [a + b for a, b in zip(never, some)], ORANGE, "white", "Always right")]
for vals, left, c, tc, lab in segs:
    ax.barh(yy, vals, left=left, height=0.62, color=c, edgecolor=BG, linewidth=2)
    for y_, v, l in zip(yy, vals, left):
        ax.text(l + v / 2, y_, f"{v:.0f}%", ha="center", va="center", fontsize=12, color=tc)
for (vals, left, c, tc, lab) in segs:
    ax.text(left[0] + vals[0] / 2, 2.52, lab, ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
ax.set_yticks(yy); ax.set_yticklabels(W)
ax.set_xlim(0, 100); ax.set_xticks([])
clean(ax, grid=None); ax.spines["bottom"].set_visible(False)
title(fig, "RL turned \"sometimes\" into \"always.\" \"Never\" never moved.",
      "How many of 4 attempts got each diagnosis right, across the RL run")
save(fig, "rl-signal.png")

# 3. Training exposure
x = ["never seen", "1", "2–4", "5 or more"]
S = [("Untrained", [2, 8, 8, 35], LIGHT, "o"), ("Fine-tuning, 3 epochs", [5, 28, 37, 73], GOLD, "s"),
     ("Fine-tuning, 8 epochs", [1, 35, 68, 84], PURPLE, "^"), ("Fine-tuning + search tool", [57, 83, 87, 96], ORANGE, "D")]
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=100)
fig.subplots_adjust(left=0.08, right=0.72, top=0.8, bottom=0.14)
for name, v, c, m in S:
    lw = 3 if c == ORANGE else 2
    ax.plot(range(4), v, color=c, lw=lw, marker=m, ms=8, markeredgecolor=BG, markeredgewidth=1.5, zorder=3)
    ax.text(3.12, v[-1], name, va="center", fontsize=11, color=INK, fontweight="bold" if c == ORANGE else "normal")
ax.annotate("1% → 57%", xy=(0, 55), xytext=(0.15, 45), fontsize=12, fontweight="bold", color=ORANGE)
ax.annotate("", xy=(0, 54), xytext=(0, 4), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
ax.set_xticks(range(4)); ax.set_xticklabels(x)
ax.set_xlim(-0.25, 3.05); ax.set_ylim(0, 102)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]])
ax.set_xlabel("How many training patients had the code")
clean(ax, grid="y")
title(fig, "Training helps codes it has seen. The tool helps the ones it hasn't.",
      "Share of chart-documented diagnoses coded right, test patients")
save(fig, "training-exposure.png")

# 4. NEW: epoch sweep (replaces the W&B loss screenshot)
ep = list(range(1, 11))
val = [0.247, 0.326, 0.445, 0.484, 0.549, 0.546, 0.570, 0.576, 0.567, 0.565]
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=100)
fig.subplots_adjust(left=0.08, right=0.82, top=0.8, bottom=0.13)
ax.plot(ep, val, color=ORANGE, lw=3, marker="o", ms=7, markeredgecolor=BG, zorder=3)
ax.plot([3, 6, 10], [0.609, 0.887, 0.986], color=MUTED, lw=2, ls=(0, (4, 3)), marker="s", ms=7, markeredgecolor=BG, zorder=3)
ax.text(10.2, 0.565, "Validation\n(new patients)", va="center", fontsize=11, color=INK, fontweight="bold")
ax.text(10.2, 0.986, "Its own training\npatients", va="center", fontsize=11, color=MUTED)
ax.annotate("picked epoch 8", xy=(8, 0.576), xytext=(6.4, 0.69), fontsize=11, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
ax.axvline(3, color=GRID, lw=1); ax.text(3.1, 0.22, "where I\nstopped for RL", fontsize=9.5, color=MUTED, va="bottom")
ax.set_xticks(ep); ax.set_xlim(0.6, 10.1); ax.set_ylim(0.2, 1.02)
ax.set_xlabel("Epochs of fine-tuning"); ax.set_ylabel("Score")
clean(ax, grid="y")
title(fig, "I stopped fine-tuning too early.",
      "Validation kept climbing until epoch 8. After that, it was just memorizing.")
save(fig, "epoch-sweep.png")

# 5. NEW: what the tool is worth, by model
M = [("Qwen3-4B (untrained)", 0.172, 0.486), ("Qwen3-30B (untrained)", 0.457, 0.644), ("GPT-6 Luna", 0.689, 0.679)]
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=100)
fig.subplots_adjust(left=0.24, right=0.95, top=0.76, bottom=0.16)
yy = np.arange(3)[::-1]
for y_, (n, a, b) in zip(yy, M):
    ax.plot([a, b], [y_, y_], color=GRID, lw=6, solid_capstyle="round", zorder=1)
    ax.scatter([a], [y_], s=110, color=LIGHT, edgecolor=MUTED, zorder=3)
    ax.scatter([b], [y_], s=110, color=ORANGE, zorder=3)
    d = b - a
    ax.text(max(a, b) + 0.025, y_, f"{d:+.2f}", va="center", fontsize=12, fontweight="bold", color=ORANGE if d > 0.05 else MUTED)
ax.set_yticks(yy); ax.set_yticklabels([m[0] for m in M], fontsize=12)
ax.set_xlim(0.1, 0.85); ax.set_ylim(-0.5, 2.5); ax.set_xlabel("F1 on test patients")
ax.scatter([], [], s=80, color=LIGHT, edgecolor=MUTED, label="without the tool")
ax.scatter([], [], s=80, color=ORANGE, label="with the tool")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=2, frameon=False, fontsize=10.5)
clean(ax)
title(fig, "The less a model knows, the more the tool is worth.", "Same search tool, no fine-tuning")
save(fig, "tool-value.png")

# 6. NEW: the picky answer key
L = [("Answer key", 3.5, MUTED), ("My 4B (tool-SFT)", 3.5, ORANGE), ("GPT-6 Luna", 5.8, LIGHT), ("Luna + two sentences", 5.0, TEAL)]
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=100)
fig.subplots_adjust(left=0.24, right=0.95, top=0.76, bottom=0.16)
yy = np.arange(4)[::-1]
for y_, (n, v, c) in zip(yy, L):
    ax.barh(y_, v, height=0.6, color=c)
    ax.text(v + 0.08, y_, f"{v:.1f}", va="center", fontsize=12, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([l[0] for l in L], fontsize=12)
ax.set_xlim(0, 7); ax.set_xlabel("Diagnoses submitted per patient")
clean(ax)
title(fig, "Luna found the right diagnoses. It just listed too many.",
      "Both models found about 70% of the key's diagnoses. Luna's extras cost it precision.")
save(fig, "picky-key.png")

# 7. NEW: cost vs score
P = [("Qwen3-4B + tool", 0.26, 0.74, ORANGE), ("Qwen3-4B, 8 epochs", 0.16, 0.53, GRAY),
     ("GPT-6 Luna", 2, 0.69, LIGHT), ("Luna + two sentences", 2, 0.81, TEAL),
     ("Kimi K2.5", 51, 0.73, LIGHT), ("Gemma 3 27B", 11, 0.26, LIGHT)]
fig, ax = plt.subplots(figsize=(10, 5.4), dpi=100)
fig.subplots_adjust(left=0.09, right=0.95, top=0.8, bottom=0.14)
for n, c_, f, col in P:
    ax.scatter(c_, f, s=160, color=col, edgecolor=MUTED if col == LIGHT else col, zorder=3)
    dx = 1.18
    ax.text(c_ * dx, f, n, va="center", fontsize=11, color=INK, fontweight="bold" if col in (ORANGE, TEAL) else "normal")
ax.set_xscale("log"); ax.set_xlim(0.1, 300); ax.set_ylim(0.15, 0.9)
ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(["$0.10", "$1", "$10", "$100"])
ax.set_xlabel("Cost per 1,000 patients (log scale)"); ax.set_ylabel("F1")
clean(ax, grid="both")
title(fig, "Frontier-level score for pennies.", "My 4B ran on one rented GPU. The others are API prices.")
save(fig, "cost-vs-score.png")
print("done")
