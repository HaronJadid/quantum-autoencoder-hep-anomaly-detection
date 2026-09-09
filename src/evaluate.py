"""Anomaly-detection metrics.

Headline numbers are ROC-AUC and background rejection 1/eps_B at fixed signal
efficiency eps_S. Both are independent of the signal fraction in the test set,
so they do not depend on how many signal events we happened to mix in.

Also computed: the mjj sculpting diagnostic. An anomaly score that correlates
with the dijet invariant mass distorts the very spectrum a resonance search
bump-hunts in, which can manufacture a fake excess. mjj is excluded from the
model inputs for this reason; this measures how much correlation survives
anyway through the features that are used.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve


def rejection_at_efficiency(scores: np.ndarray, labels: np.ndarray,
                            eps_s: float) -> dict:
    """1/eps_B at signal efficiency eps_s. Higher score = more anomalous.

    Returns the rejection, the underlying background efficiency, its binomial
    uncertainty, and the number of surviving background events -- the last of
    these tells you whether the number is statistically meaningful at all.
    """
    sig = scores[labels == 1]
    bkg = scores[labels == 0]
    if len(sig) == 0 or len(bkg) == 0:
        raise ValueError("need both signal and background events")

    # threshold admitting a fraction eps_s of signal
    thr = float(np.quantile(sig, 1.0 - eps_s))
    n_pass = int(np.sum(bkg >= thr))
    eps_b = n_pass / len(bkg)
    err = float(np.sqrt(max(eps_b * (1 - eps_b), 0.0) / len(bkg)))

    return {
        "eps_s": eps_s,
        "threshold": thr,
        "eps_b": eps_b,
        "eps_b_err": err,
        "n_bkg_pass": n_pass,
        "n_bkg_total": int(len(bkg)),
        # inf when no background survives: a lower bound set by statistics,
        # not a measured value. Reported as such.
        "rejection": (1.0 / eps_b) if n_pass > 0 else float("inf"),
        "rejection_is_lower_bound": n_pass == 0,
    }


def sculpting(scores: np.ndarray, labels: np.ndarray, mjj: np.ndarray,
              top_frac: float = 0.01) -> dict:
    """How much does the anomaly score sculpt the background mjj spectrum?

    Measured on BACKGROUND ONLY -- sculpting is a background-shaping problem.
    A |rho| near 0 and a median shift near 0 mean the selection is roughly
    mass-agnostic, which is what a bump hunt needs.
    """
    s = scores[labels == 0]
    m = mjj[labels == 0]
    rho = float(spearmanr(s, m).statistic)

    thr = float(np.quantile(s, 1.0 - top_frac))
    sel = m[s >= thr]
    med_all, med_sel = float(np.median(m)), float(np.median(sel)) if len(sel) else float("nan")
    return {
        "spearman_score_vs_mjj": rho,
        "median_mjj_inclusive": med_all,
        f"median_mjj_top{top_frac:.0%}": med_sel,
        "median_mjj_shift_frac": (med_sel - med_all) / med_all if med_all else float("nan"),
        "n_selected": int(len(sel)),
    }


def evaluate(scores: np.ndarray, labels: np.ndarray, mjj: np.ndarray | None = None,
             eps_s_points=(0.1, 0.3, 0.5)) -> dict:
    """Full metric set for one model on one test set."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite anomaly scores")

    out = {
        "auc": float(roc_auc_score(labels, scores)),
        "n_signal": int((labels == 1).sum()),
        "n_background": int((labels == 0).sum()),
        "rejection": {f"eps_s={e}": rejection_at_efficiency(scores, labels, e)
                      for e in eps_s_points},
    }
    if mjj is not None:
        out["sculpting"] = sculpting(scores, labels, mjj)
    return out


def roc_points(scores: np.ndarray, labels: np.ndarray, max_points: int = 2000):
    """Thinned ROC curve for plotting: (eps_s, eps_b)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return tpr, fpr


def paired_comparison(runs_a: list, runs_b: list, metric: str = "auc") -> dict:
    """Paired comparison of two models across seeds.

    Seeds are paired: within a seed every model sees exactly the same events,
    so the per-seed difference removes the split-to-split variation and is a
    more sensitive test than comparing two independent mean +/- s.d. summaries.

    Returns the mean difference (a minus b), its standard error, a two-sided
    paired t-statistic and p-value. With 5 seeds this has little power, so a
    non-significant result means "not resolved by this study", not "identical".
    """
    from scipy import stats

    def get(run):
        """`metric` is either a top-level key ("auc") or "rejection@<eps_s>"."""
        if metric.startswith("rejection@"):
            return run["rejection"][f"eps_s={metric.split('@')[1]}"]["rejection"]
        return run[metric]

    a = np.array([get(r) for r in runs_a], dtype=np.float64)
    b = np.array([get(r) for r in runs_b], dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError(
            f"non-finite {metric} in at least one seed; an infinite rejection is "
            "a lower bound, not a measurement, and cannot enter a t-test")
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same seeds for both models")
    d = a - b
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    if n > 1 and sem > 0:
        t = mean / sem
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    else:
        t, p = float("nan"), float("nan")
    return {
        "mean_difference": mean, "sem": float(sem), "t": float(t), "p_value": p,
        "n_seeds": n, "per_seed_difference": d.tolist(),
        "significant_at_0.05": bool(p == p and p < 0.05),
    }


def aggregate(runs: list) -> dict:
    """Mean +/- sample std across seeds for the scalar metrics."""
    def ms(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
        if len(v) == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "n": int(len(v))}

    agg = {"auc": ms([r["auc"] for r in runs])}
    for key in runs[0]["rejection"]:
        agg[f"rejection[{key}]"] = ms([r["rejection"][key]["rejection"] for r in runs])
        agg[f"eps_b[{key}]"] = ms([r["rejection"][key]["eps_b"] for r in runs])
    if "sculpting" in runs[0]:
        agg["spearman_score_vs_mjj"] = ms(
            [r["sculpting"]["spearman_score_vs_mjj"] for r in runs])
    agg["n_seeds"] = len(runs)
    return agg
