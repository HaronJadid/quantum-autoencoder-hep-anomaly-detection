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
    "ae_matched": "Classical AE, parameter-matched",
    "ae_dense": "Classical AE, 6-16-4-16-6",
    "pca": "PCA (4 comp.)",
    "mj1_only": r"$m_{J_1}$ alone (no training)",
}
COLORS = {"qae_ry": "#c1272d", "qae_zz": "#e8926b", "ae_matched": "#2b6cb0",
          "ae_dense": "#1a7f37", "pca": "#7a5195", "mj1_only": "#6b7280"}
ORDER = ["qae_ry", "qae_zz", "ae_matched", "ae_dense", "pca", "mj1_only"]


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
    # Curves are from ONE seed. Across seeds these orderings are not stable and
    # the differences are not significant -- saying so on the figure keeps it
    # from being read as a ranking.
    seed = res["config"]["seeds"][0]
    ax.set_title(f"LHCO R&D: anomaly detection ROC (seed {seed} only;\n"
                 "across-seed differences are not significant — see README)",
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
    ax.set_title(f"ROC-AUC, mean $\\pm$ s.d. over {n} seeds", fontsize=11)
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
    loss_figure(res, os.path.join(outdir, "training_curves.png"))
    reps = res["config"]["ansatz_reps"]
    circuit_figure(os.path.join(outdir, "circuit.png"),
                   res["config"]["n_qubits"],
                   reps["ry"] if isinstance(reps, dict) else reps)
    print(f"figures written to {outdir}")


if __name__ == "__main__":
    make_all()
