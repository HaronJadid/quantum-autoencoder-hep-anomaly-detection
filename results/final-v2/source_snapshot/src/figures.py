"""Figures for the README, generated from results/metrics.json."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "qae_ry": "Quantum AE, RY encoding",
    "qae_zz": "Quantum AE, ZZFeatureMap",
    "ae_matched": "Classical AE, parameter-matched (tied)",
    "ae_untied32": "Classical AE, untied (32 par.)",
    "ae_untied45": "Classical AE, untied (45 par.)",
    "ae_untied58": "Classical AE, untied (58 par.)",
    "ae_dense": "Classical AE, 6-16-4-16-6",
    "pca": "PCA (4 comp.)",
    "mj1_only": r"$m_{J_1}$ alone (no training)",
}
COLORS = {"qae_ry": "#c1272d", "qae_zz": "#e8926b", "ae_matched": "#2b6cb0",
          "ae_untied32": "#4a9fd8", "ae_untied45": "#1b4a7a",
          "ae_untied58": "#7fb3d5",
          "ae_dense": "#1a7f37", "pca": "#7a5195", "mj1_only": "#6b7280"}
ORDER = ["qae_ry", "qae_zz", "ae_matched", "ae_untied32", "ae_untied45",
         "ae_untied58", "ae_dense", "pca", "mj1_only"]


def _order(keys):
    return [k for k in ORDER if k in keys] + [k for k in keys if k not in ORDER]


def roc_figure(res: dict, path: str):
    """Background rejection vs signal efficiency -- the HEP-standard view."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for name in _order(res["roc_first_seed"]):
        tpr, fpr = (np.asarray(a) for a in res["roc_first_seed"][name])
        ok = (fpr > 0) & (tpr > 0)
        ax.plot(tpr[ok], 1.0 / fpr[ok], label=LABELS.get(name, name),
                color=COLORS.get(name), lw=1.9,
                ls="--" if name in ("mj1_only", "qae_zz") else "-")
    ax.axhline(1.0, color="k", lw=0.8, ls=":", label="random")
    ax.set_yscale("log")
    ax.set_xlabel(r"signal efficiency  $\varepsilon_S$")
    ax.set_ylabel(r"background rejection  $1/\varepsilon_B$")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    # Curves are from ONE seed. A figure title must not assert a verdict about
    # significance or stability of ordering: such a verdict depends on the run
    # and then goes stale silently inside a .png, where nobody re-reads it.
    # State the scope of the figure and point at the README.
    seed = res["config"]["seeds"][0]
    n = len(res["config"]["seeds"])
    ax.set_title(f"LHCO R&D: anomaly detection ROC (seed {seed} of {n};\n"
                 "see README for across-seed spread and significance)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def auc_bar_figure(res: dict, path: str):
    """AUC with across-seed error bars."""
    names = _order(res["summary"])
    means = [res["summary"][n]["auc"]["mean"] for n in names]
    errs = [res["summary"][n]["auc"]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=errs, color=[COLORS.get(n, "#888") for n in names],
            height=0.6, capsize=4, alpha=0.9)
    ax.axvline(0.5, color="k", ls=":", lw=1, label="random (0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(n, n) for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC")
    lo = min(0.45, min(m - e for m, e in zip(means, errs)) - 0.02)
    ax.set_xlim(lo, 1.0)
    ax.legend(fontsize=8)
    n = res["summary"][names[0]]["auc"]["n"]
    ax.set_title(f"ROC-AUC, mean $\\pm$ s.d. over {n} seeds\n"
                 "(spread is training-run variability; see README)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def loss_figure(res: dict, path: str):
    """Training and validation curves for the first seed."""
    seed = sorted(res["histories"])[0]
    hist = res["histories"][seed]
    fig, axes = plt.subplots(1, len(hist), figsize=(3.1 * len(hist), 3.0),
                             squeeze=False)
    for ax, name in zip(axes[0], _order(hist)):
        h = hist[name]
        ax.plot(h["train_loss"], label="train", color=COLORS.get(name), lw=1.6)
        ax.plot(h["val_loss"], label="val", color=COLORS.get(name), lw=1.6,
                ls="--", alpha=0.7)
        ax.axvline(h["best_epoch"] - 1, color="k", ls=":", lw=0.9)
        ax.set_title(LABELS.get(name, name), fontsize=8)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("loss")
    fig.suptitle(f"Training curves (seed {seed}); dotted line = restored epoch",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def score_figure(scores: dict, labels: np.ndarray, path: str):
    """Anomaly-score distributions, background vs signal."""
    names = _order(scores)
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.0),
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        s = np.asarray(scores[name], dtype=float)
        lo, hi = np.percentile(s, [0.1, 99.9])
        bins = np.linspace(lo, hi, 60)
        ax.hist(s[labels == 0], bins=bins, density=True, histtype="step",
                lw=1.7, color="#2b6cb0", label="background")
        ax.hist(s[labels == 1], bins=bins, density=True, histtype="step",
                lw=1.7, color="#c1272d", label="signal")
        ax.set_title(LABELS.get(name, name), fontsize=8)
        ax.set_xlabel("anomaly score")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("normalised events")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sculpting_figure(scores: dict, labels: np.ndarray, mjj: np.ndarray,
                     path: str, top_frac: float = 0.01):
    """Background mjj spectrum, inclusive vs the most anomalous 1%.

    If a curve departs from the inclusive one, that model sculpts the dijet
    mass spectrum and would distort a bump hunt.
    """
    bkg = labels == 0
    m = mjj[bkg]
    bins = np.linspace(np.percentile(m, 0.5), np.percentile(m, 99.5), 50)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(m, bins=bins, density=True, histtype="step", lw=2.2, color="k",
            label="all background")
    for name in _order(scores):
        s = np.asarray(scores[name], float)[bkg]
        sel = m[s >= np.quantile(s, 1 - top_frac)]
        ax.hist(sel, bins=bins, density=True, histtype="step", lw=1.6,
                color=COLORS.get(name), label=f"{LABELS.get(name, name)}, top 1%")
    ax.set_xlabel(r"$m_{JJ}$  [GeV]")
    ax.set_ylabel("normalised background events")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("Sculpting check: does the anomaly score reshape $m_{JJ}$?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _spread(values: dict, gap: float, lo: float, hi: float) -> dict:
    """Nudge label positions apart so none overlaps, staying inside [lo, hi].

    Forward pass pushes each label above the one below it; if that runs off
    the top, a backward pass from the ceiling pushes them back down. Both
    passes are needed -- the forward one alone silently walks the whole stack
    off the top of the axes when the points are tightly clustered, which is
    exactly what this data does.
    """
    order = sorted(values, key=values.get)
    y = {}
    prev = None
    for k in order:
        v = values[k] if prev is None else max(values[k], prev + gap)
        y[k], prev = v, v
    if prev > hi:
        prev = hi
        for k in reversed(order):
            v = min(y[k], prev)
            y[k], prev = v, v - gap
    if y[order[0]] < lo:
        prev = lo
        for k in order:
            v = max(y[k], prev)
            y[k], prev = v, v + gap
    return y


def generalisation_figure(res: dict, path: str):
    """2-prong vs 3-prong AUC, one line per model, on a shared scale.

    A slope chart rather than a grouped bar chart: the quantity of interest is
    the CHANGE each model undergoes when the signal topology changes, and a
    slope shows a change directly instead of asking the reader to subtract two
    bar heights. Both axes carry the same scale so a slope's steepness is
    comparable across models, and 0.5 is drawn because a line crossing it has
    gone from better than guessing to worse.
    """
    qqq = (res.get("generalisation_qqq") or {}).get("summary") or {}
    if not qqq:
        return False
    names = [n for n in _order(res["summary"]) if n in qqq]
    left = {n: res["summary"][n]["auc"]["mean"] for n in names}
    right = {n: qqq[n]["auc"]["mean"] for n in names}

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    lo = min(min(left.values()), min(right.values()), 0.5) - 0.055
    hi = max(max(left.values()), max(right.values()), 0.5) + 0.045
    ax.axhline(0.5, color="k", lw=1.0, ls=":", zorder=1)
    ax.text(1.5, 0.5, "  random (0.5)", va="center", ha="center", fontsize=8,
            color="k", bbox=dict(fc="white", ec="none", pad=1), zorder=4)

    for n in names:
        c = COLORS.get(n, "#888")
        ax.plot([1, 2], [left[n], right[n]], "-o", color=c, lw=2.0, ms=5.5,
                zorder=3, alpha=0.95)

    # Names all go on the right, pushed apart vertically where they would
    # collide and joined to their point by a leader line. Labelling each line
    # at whichever end "has room" sounds reasonable but does not survive
    # contact with this data: five models sit inside 0.05 AUC on the left and
    # their labels simply overprint each other.
    gap = (hi - lo) * 0.042
    placed = _spread({n: right[n] for n in names}, gap, lo, hi)
    left_placed = _spread({n: left[n] for n in names}, gap * 0.8, lo, hi)
    for n in names:
        # Both coordinates in DATA units. Mixing "offset points" for x with
        # "data" for y makes matplotlib read the y value as a relative offset,
        # so an absolute AUC of 0.73 shifts the label 0.73 up the axis and off
        # the figure.
        ax.annotate(f"{left[n]:.3f}", (1, left[n]),
                    xytext=(0.955, left_placed[n]), textcoords="data",
                    va="center", ha="right", fontsize=8,
                    color=COLORS.get(n, "#888"),
                    arrowprops=dict(arrowstyle="-", color=COLORS.get(n, "#888"),
                                    lw=0.6, alpha=0.45, shrinkA=0, shrinkB=3))
    for n in names:
        c = COLORS.get(n, "#888")
        ax.annotate(f"{LABELS.get(n, n)}  {right[n]:.3f}",
                    (2, right[n]), xytext=(2.06, placed[n]),
                    textcoords="data",
                    va="center", ha="left", fontsize=8, color=c,
                    # Labels can land on the gridlines or the 0.5 rule; a flat
                    # white pad keeps them readable without drawing a box.
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.7,
                                    alpha=0.55,
                                    shrinkA=0, shrinkB=2))
    ax.set_xlim(0.72, 3.05)
    ax.set_ylim(lo, hi)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["2-prong  (X,Y" + r"$\to$" + "qq)\ntrained-on topology",
                        "3-prong  (X,Y" + r"$\to$" + "qqq)\nnever-seen topology"],
                       fontsize=9)
    ax.set_ylabel("ROC-AUC (mean over seeds)")
    ax.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    n_seeds = res["summary"][names[0]]["auc"]["n"]
    ax.set_title("ROC-AUC on the trained-on signal and on a second signal "
                 "topology\n"
                 f"(mean over {n_seeds} seeds; same trained models, "
                 "same background test split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def circuit_figure(path: str, n_qubits: int = 6, reps: int = 3):
    """Qiskit rendering of the circuit actually used."""
    from .qae import build_qiskit_circuit
    qc, _, _ = build_qiskit_circuit(n_qubits, reps)
    fig = qc.decompose().draw(output="mpl", style="clifford", fold=40)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_all(results_path: str = "results/metrics.json",
             outdir: str = "results/figures"):
    os.makedirs(outdir, exist_ok=True)
    with open(results_path) as f:
        res = json.load(f)
    roc_figure(res, os.path.join(outdir, "roc.png"))
    auc_bar_figure(res, os.path.join(outdir, "auc.png"))
    generalisation_figure(res, os.path.join(outdir, "generalisation.png"))
    loss_figure(res, os.path.join(outdir, "training_curves.png"))
    reps = res["config"]["ansatz_reps"]
    circuit_figure(os.path.join(outdir, "circuit.png"),
                   res["config"]["n_qubits"],
                   reps["ry"] if isinstance(reps, dict) else reps)
    print(f"figures written to {outdir}")


if __name__ == "__main__":
    # Explicit paths, because the default ones are the live results directory:
    # a test run of some wrapper that forgets to pass them should fail, not
    # quietly overwrite the figures of the real study.
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--results", default="results/metrics.json")
    _ap.add_argument("--outdir", default=None,
                     help="default: a 'figures' directory beside --results")
    _a = _ap.parse_args()
    make_all(_a.results,
             _a.outdir or os.path.join(os.path.dirname(_a.results), "figures"))
