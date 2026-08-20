import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = ["correct_per_rollout","doc_f1","p_bar","precision","pred_per_gt","recall","scored_per_rollout"]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
S1, S2 = "#2a78d6", "#eb6834"          # validated categorical slots 1 & 2

def load_txt(p):
    v = [float(x) for x in open(p)]; n = len(v)//7
    a = np.array(v[:n*7]).reshape(n, 7)
    return {k: a[:, i] for i, k in enumerate(NAMES)}

def roll(y, w=11):
    if len(y) < w: return y
    k = np.ones(w)/w
    return np.convolve(y, k, mode="valid")

def panel(ax, v1, v2, key, title, ylabel):
    for y, c, lab in ((v1, S1, "lr 1e-6 (original)"), (v2, S2, "lr 1e-5 (re-run)")):
        d = y[key]; x = np.arange(len(d))
        ax.plot(x, d, color=c, lw=1.0, alpha=0.18)          # raw, recessive
        sm = roll(d); xs = np.arange(len(sm)) + (len(d)-len(sm))//2
        ax.plot(xs, sm, color=c, lw=2.0, label=lab)          # 2px trend
    ax.set_title(title, fontsize=12, color=INK, pad=10, loc="left")
    ax.set_xlabel("training step", fontsize=10, color=INK2)
    ax.set_ylabel(ylabel, fontsize=10, color=INK2)
    ax.grid(True, color="#e6e5e1", lw=0.8)                   # recessive grid
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#d5d4cf")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_facecolor(SURFACE)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts(): t.set_color(INK)

D1, D2 = load_txt("armD_metrics_v1.txt") if False else (np.load("armD.npy", allow_pickle=True).item(), load_txt("armD2_full.txt"))
C1, C2 = np.load("armC.npy", allow_pickle=True).item(), load_txt("armC2_full.txt")

for tag, v1, v2, title in (("armD", D1, D2, "Arm D — document-level F1 reward"),
                           ("armC", C1, C2, "Arm C — consensus-marginal reward")):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    panel(ax, v1, v2, "doc_f1", title, "rollout F1 (per-step mean)")
    n2 = len(v2["doc_f1"])
    fig.text(0.5, -0.02, f"125 steps at lr 1e-6; {n2} steps so far at lr 1e-5. "
             f"Faint = per-step, bold = 11-step rolling mean.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(f"exp208_{tag}_lr.png", facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote exp208_{tag}_lr.png   v1 n={len(v1['doc_f1'])}  v2 n={n2}")
