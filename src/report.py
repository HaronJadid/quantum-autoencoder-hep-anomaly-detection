"""Generate the README results table from results/metrics.json.

The table is generated, never typed. Numbers in the README therefore cannot
drift from the numbers actually produced by a run.

    python -m src.report              # print markdown to stdout
"""

from __future__ import annotations

import json
import sys

import numpy as np

ROWS = [
    ("qae_ry", "**Quantum AE** (RY encoding)"),
    ("qae_zz", "Quantum AE (ZZFeatureMap)"),
    ("ae_matched", "Classical AE, parameter-matched (tied)"),
    ("ae_untied32", "Classical AE, untied, 32 par."),
    ("ae_untied45", "Classical AE, untied, 45 par."),
    ("ae_untied58", "Classical AE, untied, 58 par."),
    ("ae_dense", "Classical AE, 6-16-4-16-6"),
    ("pca", "PCA, 4 components"),
    ("mj1_only", "Cut on $m_{J_1}$ alone (no training)"),
]


def matched_note(res: dict) -> str:
    """Suffix exposing the matched model's latent dim when it does not compress.

    The exact parameter match can only be met at a latent dimension the author
    did not choose. When that equals the input dimension the model has no
    bottleneck at all, which is the single most misleading thing about the row
    labelled as the equal-parameter head-to-head.
    """
    spec = res.get("matched_ae_spec") or {}
    latent = spec.get("latent")
    d = len(res.get("config", {}).get("features", []) or [])
    if latent is None:
        return ""
    if d and latent >= d:
        return f" *(latent {latent} — **NO bottleneck**)*"
    return f" *(latent {latent})*"


def label_for(key: str, label: str, res: dict) -> str:
    return label + (matched_note(res) if key == "ae_matched" else "")


def _pm(d, fmt="{:.4f}", n_expected=None):
    """mean ± s.d., flagging any metric averaged over fewer seeds than expected.

    A seed contributes no finite rejection when zero background events survive
    the cut, in which case the value is an unbounded lower limit rather than a
    measurement. Those seeds are excluded, and saying so matters.
    """
    if d["n"] == 0:
        return "no finite value"
    body = f"{fmt.format(d['mean'])} ± {fmt.format(d['std'])}"
    if n_expected is not None and d["n"] < n_expected:
        body += f" *({d['n']}/{n_expected} seeds)*"
    return body


def results_table(res: dict) -> str:
    s = res["summary"]
    p = res["parameter_counts"]
    n = res["config"]["seeds"]
    out = [
        f"| model | trainable par. | ROC-AUC | 1/ε_B at ε_S=0.3 | ρ(score, m_JJ) |",
        "|---|---:|---|---:|---:|",
    ]
    for key, label in ROWS:
        if key not in s:
            continue
        out.append(
            f"| {label_for(key, label, res)} | {p[key]} | "
            f"{_pm(s[key]['auc'], n_expected=len(n))} | "
            f"{_pm(s[key]['rejection[eps_s=0.3]'], '{:.1f}', n_expected=len(n))} | "
            f"{s[key]['spearman_score_vs_mjj']['mean']:+.3f} |")
    out.append("")
    out.append(f"Mean ± sample s.d. over {len(n)} seed{'s' if len(n) != 1 else ''} "
               f"({', '.join(map(str, n))}). Each seed re-initialises every "
               "model and re-draws the split, so this spread is training-run "
               "variability; it is not a test-set uncertainty and the seeds are "
               "not independent experiments (their test sets overlap heavily). "
               "See the caption under the comparison tables.")
    return "\n".join(out)


def encoding_table(res: dict) -> str:
    e = res.get("encoding_compressibility")
    if not e:
        return ""
    out = [
        "| encoding | max P(trash=\\|00⟩) over **all** ansätze | effective dim of ρ | verdict |",
        "|---|---:|---:|---|",
    ]
    names = {"zz": "ZZFeatureMap", "ry": "RY angle encoding"}
    for k, v in e.items():
        verdict = "compressible" if v["compressible"] else "**not usefully compressible**"
        out.append(
            f"| {names.get(k, k)} | {v['max_p_trash_zero']:.3f} "
            f"({v['headroom_fraction']:.0%} of available headroom) | "
            f"{v['effective_dimension']:.1f} of {v['full_dimension']} | {verdict} |")
    out.append("")
    out.append(f"Random-guess reference: {list(e.values())[0]['random_reference']:.3f}.")
    return "\n".join(out)


BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"


COMPARISONS = [
    ("qae_ry", "mj1_only", "QAE vs. a plain cut on $m_{J_1}$"),
    ("qae_ry", "qae_zz", "QAE vs. the provably-incompressible encoding"),
    ("qae_ry", "ae_matched", "QAE vs. classical AE at equal parameters"),
    ("qae_ry", "ae_untied32", "QAE vs. untied classical AE, 32 par."),
    ("qae_ry", "ae_untied45", "QAE vs. untied classical AE, 45 par."),
    ("qae_ry", "ae_untied58", "QAE vs. untied classical AE, 58 par."),
    ("qae_ry", "ae_dense", "QAE vs. the larger classical AE"),
    ("ae_dense", "mj1_only", "larger classical AE vs. $m_{J_1}$ cut"),
]


REJECTION_COMPARISONS = [
    ("qae_ry", "mj1_only", "QAE vs. a plain cut on $m_{J_1}$"),
    ("qae_zz", "mj1_only", "**incompressible** encoding vs. $m_{J_1}$ cut"),
    ("ae_matched", "mj1_only", "classical AE (matched) vs. $m_{J_1}$ cut"),
    ("pca", "mj1_only", "PCA vs. $m_{J_1}$ cut"),
    ("qae_ry", "ae_matched", "QAE vs. classical AE at equal parameters"),
    ("qae_ry", "ae_untied32", "QAE vs. untied classical AE, 32 par."),
    ("qae_ry", "ae_untied45", "QAE vs. untied classical AE, 45 par."),
    ("qae_ry", "ae_untied58", "QAE vs. untied classical AE, 58 par."),
    ("qae_ry", "qae_zz", "QAE vs. the incompressible encoding"),
]


# Top-level metrics.json keys that legitimately do not appear as a rendered
# section. Anything holding per-seed data and NOT listed here must render, or
# check_coverage raises. This exists because three separate sections have now
# reached metrics.json and silently never appeared in the README.
RENDER_SKIP_LIST = {
    "config",                  # surfaced piecemeal via config_line
    "summary",                 # the results table
    "per_seed",                # the comparison tables
    "histories",               # the objective-vs-AUC table
    "parameter_counts",        # a column of the results table
    "roc_first_seed",          # figures only, not a table
    "hit_epoch_cap",           # surfaced as a config caveat
    "any_hit_epoch_cap",       # surfaced as a config caveat
    "matched_ae_spec",         # surfaced as a label annotation + footnote
    "encoding_compressibility",  # the encoding table
    "wall_seconds",            # config line
    "wall_seconds_summed",     # timing bookkeeping from a sharded run; the
                               # config line already quotes elapsed wall time,
                               # and total CPU across workers is not a result
}


def check_coverage(res: dict, bundle: dict) -> None:
    """Fail loudly if a results section would be silently omitted.

    Two failure modes, both of which have actually happened in this project:
      1. a comparison list is defined but never rendered;
      2. a block of per-seed results is written to metrics.json and no table
         ever displays it.
    Silence is the dangerous outcome, so this raises rather than warns.
    """
    empty = [name for name in ("auc", "rej", "qqq_auc", "qqq_rej")
             if not bundle.get(name)]
    # qqq lists are legitimately empty when the generalisation run was skipped.
    has_qqq = bool((res.get("generalisation_qqq") or {}).get("per_seed"))
    expected_empty = set() if has_qqq else {"qqq_auc", "qqq_rej"}
    unexpected = [e for e in empty if e not in expected_empty]
    if unexpected:
        raise AssertionError(
            f"comparison list(s) {unexpected} produced no rows and would be "
            "silently omitted from the report. Either the model keys are "
            "missing from metrics.json or the list is misconfigured.")

    unrendered = []
    for key, value in res.items():
        if key in RENDER_SKIP_LIST:
            continue
        rendered = {
            "generalisation_qqq": has_qqq,
            "diagnostic_ae_dense_ablation":
                bool((value or {}).get("results")) if isinstance(value, dict) else False,
        }.get(key)
        if rendered is None:
            unrendered.append(key)
        elif rendered is False and isinstance(value, dict) and value:
            unrendered.append(key)
    if unrendered:
        raise AssertionError(
            f"metrics.json key(s) {unrendered} are not rendered by any table "
            "and are not in RENDER_SKIP_LIST. Add a renderer or add them to "
            "the skip list deliberately -- do not let a section disappear.")


def _collect(res: dict, pairs, metric: str, source: str = "primary"):
    """Run every paired test for one metric, returning rows + failures.

    `source` selects which per-seed block to test: the primary 2-prong results
    or the 3-prong generalisation results. Both go through identical
    machinery -- same pairs, same log-space handling, same correction -- so a
    comparison cannot be reported on one signal and silently skipped on the
    other.
    """
    from .evaluate import paired_comparison

    if source == "qqq":
        ps = ((res.get("generalisation_qqq") or {}).get("per_seed")) or {}
    else:
        ps = res.get("per_seed", {})
    note = matched_note(res)
    out = []
    for a, b, label in pairs:
        if a not in ps or b not in ps:
            continue
        # Make the matched model's missing bottleneck visible in the row a
        # reviewer reads as the fair head-to-head.
        if note and ("ae_matched" in (a, b)):
            label = label + note
        try:
            out.append((label, paired_comparison(ps[a], ps[b], metric)))
        except ValueError:
            out.append((label, None))
    return out


def _rows(collected, alpha, fmt, ratio=False):
    rows = ["| comparison | mean difference | t | p (uncorrected) | "
            "survives correction? |",
            "|---|---:|---:|---:|---|"]
    for label, r in collected:
        if r is None:
            rows.append(f"| {label} | not testable (infinite rejection) | | | |")
            continue
        survives = r["p_value"] == r["p_value"] and r["p_value"] < alpha
        mark = "**yes**" if survives else "no"
        if ratio:
            diff = (f"x{np.exp(r['mean_difference']):.2f} "
                    f"[{np.exp(r['mean_difference'] - r['sem']):.2f}, "
                    f"{np.exp(r['mean_difference'] + r['sem']):.2f}]")
        else:
            diff = (f"{fmt.format(r['mean_difference'])} ± "
                    f"{fmt.format(r['sem']).lstrip('+')}")
        rows.append(f"| {label} | {diff} | {r['t']:.2f} | "
                    f"{r['p_value']:.3f} | {mark} |")
    return "\n".join(rows)


def collect_all(res: dict) -> dict:
    """Every paired test the report will render, plus the corrected threshold.

    Computed in ONE place so the stated test count and alpha cannot drift from
    the rows actually printed. Adding a comparison list here automatically
    tightens the correction for every table.
    """
    bundle = {
        "auc": _collect(res, COMPARISONS, "auc"),
        "rej": _collect(res, REJECTION_COMPARISONS, "log_rejection@0.3"),
        "qqq_auc": _collect(res, COMPARISONS, "auc", source="qqq"),
        "qqq_rej": _collect(res, REJECTION_COMPARISONS, "log_rejection@0.3",
                            source="qqq"),
    }
    n = sum(len(v) for v in bundle.values())
    bundle["n_tests"] = n
    bundle["alpha"] = 0.05 / n if n else float("nan")
    return bundle


def comparison_table(res: dict, bundle: dict) -> str:
    """Paired per-seed tests on both headline metrics, Bonferroni-corrected.

    They disagree, and that disagreement is the main result, so both are shown.
    """
    auc_c, rej_c = bundle["auc"], bundle["rej"]
    n_tests, alpha = bundle["n_tests"], bundle["alpha"]
    n_qqq = len(bundle["qqq_auc"]) + len(bundle["qqq_rej"])

    return "\n".join([
        f"**{n_tests} paired tests are performed** across this section and the "
        f"generalisation section (2-prong AUC: {len(auc_c)}, 2-prong "
        f"rejection: {len(rej_c)}, 3-prong: {n_qqq}). Reporting them at α=0.05 "
        f"each would give a {1 - 0.95 ** n_tests:.0%} chance of at least one "
        "false positive under the null. The Bonferroni-corrected threshold is "
        f"therefore **α = 0.05/{n_tests} = {alpha:.5f}**, applied identically "
        "to every table below including the 3-prong one. Uncorrected p is "
        "shown so the raw evidence is visible.",
        "",
        "*ROC-AUC*", "",
        _rows(auc_c, alpha, "{:+.4f}"),
        "",
        "*Background rejection at ε_S = 0.3, tested in log space*", "",
        _rows(rej_c, alpha, "{:+.2f}", ratio=True),
        "",
        "The rejection tests are run on **log(1/ε_B)**, not 1/ε_B. The "
        "rejection is a ratio and is strongly right-skewed across seeds, so a "
        "t-test on the raw value violates its normality assumption and lets one "
        "large seed dominate. Differences of logs are near-symmetric; the "
        "column shows the exponentiated mean difference, i.e. a multiplicative "
        "factor (x1.00 = no difference), with ±1 s.e. exponentiated as the "
        "bracket.",
        "",
        _caption(res),
    ])


def _caption(res: dict) -> str:
    """Honest description of what the quoted spread does and does not measure."""
    from .evaluate import auc_test_se

    s = res["summary"]
    cfg = res["config"]
    n_seeds = len(cfg["seeds"])
    any_run = next(iter(res["per_seed"].values()))[0]
    n_sig, n_bkg = any_run["n_signal"], any_run["n_background"]
    # Test-set statistical uncertainty on a single AUC, computed (not assumed)
    # at the AUC values actually observed.
    ses = [auc_test_se(v["auc"]["mean"], n_sig, n_bkg) for v in s.values()]
    se_lo, se_hi = min(ses), max(ses)
    # These agree to well under a significant figure across models, so a range
    # renders as "0.0010-0.0010"; quote one value unless they actually differ.
    se_txt = (f"~{se_hi:.4f}" if round(se_hi, 4) == round(se_lo, 4)
              else f"~{se_lo:.4f}–{se_hi:.4f}")
    total_bkg = cfg["n_train"] + cfg["n_val"] + n_bkg
    overlap = n_bkg / total_bkg

    return (
        f"Two-sided paired t-tests across {n_seeds} seeds. Within a seed every "
        "model sees exactly the same events, so pairing removes the "
        "split-to-split component and is more sensitive than comparing "
        "independent mean ± s.d. summaries.\n\n"
        "**What the quoted ± measures.** It is the spread of the whole training "
        "run across seeds — weight initialisation plus which background events "
        "landed in the train/validation/test split. It is *not* a test-set "
        "statistical uncertainty, and the seeds are **not independent "
        f"experiments**: each seed's test set contains {n_bkg:,} of the same "
        f"{total_bkg:,} background events ({overlap:.0%} of the pool) and all "
        f"{n_sig:,} signal events, so the test sets overlap heavily and the "
        "quoted spread understates what independent replication would show.\n\n"
        "**Test-set uncertainty, separately.** At these sample sizes "
        f"({n_sig:,} signal, {n_bkg:,} background) the Hanley–McNeil standard "
        f"error on a single AUC is {se_txt}, far smaller than "
        "the observed across-seed spread. Almost all of the variation reported "
        "here is therefore training variability, not finite-test-set noise.\n\n"
        f"With {n_seeds} seeds the power is low: *not significant* means **not "
        "resolved by this study**, not *identical*.")


def objective_vs_auc_table(res: dict) -> str:
    """Per seed, the quantity that is optimised against the quantity that matters.

    Training and every hyperparameter selection minimise background validation
    loss. Detection performance is measured by test AUC. Nothing guarantees the
    two move together, and this pairs them directly so the reader can see the
    relationship rather than assume it.
    """
    from scipy.stats import spearmanr

    hist = res.get("histories") or {}
    ps = res.get("per_seed") or {}
    seeds = res["config"]["seeds"]
    if not hist:
        return ""

    trained = [k for k, _ in ROWS if k in ps and any(
        k in hist.get(str(s), {}) for s in seeds)]
    if not trained:
        return ""

    rows = ["| model | seed | best background val. loss | test ROC-AUC |",
            "|---|---:|---:|---:|"]
    pooled_pairs, per_model_rho = [], {}
    for key in trained:
        vals, aucs = [], []
        for i, s in enumerate(seeds):
            h = hist.get(str(s), {}).get(key)
            if not h or i >= len(ps[key]):
                continue
            v = min(h["val_loss"])
            a = ps[key][i]["auc"]
            vals.append(v)
            aucs.append(a)
            rows.append(f"| {label_for(key, dict(ROWS)[key], res)} | {s} | "
                        f"{v:.4f} | {a:.4f} |")
        if len(vals) > 2:
            rho = spearmanr(vals, aucs).statistic
            per_model_rho[key] = rho
            # Pool as within-model ranks so different loss scales (a fidelity
            # infidelity and a reconstruction MSE are not comparable numbers)
            # cannot dominate the pooled correlation.
            vr = np.argsort(np.argsort(vals))
            ar = np.argsort(np.argsort(aucs))
            pooled_pairs += list(zip(vr, ar))

    rows.append("")
    rows.append("| model | Spearman ρ(val. loss, AUC) over seeds |")
    rows.append("|---|---:|")
    for key, rho in per_model_rho.items():
        rows.append(f"| {label_for(key, dict(ROWS)[key], res)} | "
                    f"{'n/a' if rho != rho else f'{rho:+.3f}'} |")
    pooled = float("nan")
    if len(pooled_pairs) > 2:
        a, b = zip(*pooled_pairs)
        pooled = spearmanr(a, b).statistic
    rows.append(f"| **pooled (within-model ranks)** | "
                f"{'n/a' if pooled != pooled else f'{pooled:+.3f}'} |")
    rows.append("")
    rows.append(_objective_sentence(per_model_rho, pooled, len(seeds)))
    return "\n".join(rows)


def _objective_sentence(per_model_rho: dict, pooled: float, n_seeds: int) -> str:
    """One generated sentence describing the table, and nothing more."""
    finite = [r for r in per_model_rho.values() if r == r]
    if not finite:
        return ("Too few seeds to estimate a rank correlation between the "
                "selection objective and test AUC.")
    n_pos = sum(1 for r in finite if r > 0)
    n_neg = sum(1 for r in finite if r < 0)
    rng = f"{min(finite):+.2f} to {max(finite):+.2f}"
    pooled_txt = ("not estimable" if pooled != pooled else f"{pooled:+.3f}")
    if pooled == pooled and abs(pooled) < 0.3:
        strength = ("no consistent monotone relationship between the two in "
                    "this study")
    elif pooled == pooled and abs(pooled) < 0.6:
        strength = ("at most a weak monotone relationship between the two in "
                    "this study")
    else:
        strength = ("a monotone relationship between the two in this study, in "
                    f"the {'expected' if pooled < 0 else 'opposite-to-expected'} "
                    "direction")
    return (
        f"Across {n_seeds} seeds the per-model rank correlation between best "
        f"background validation loss and test ROC-AUC ranges {rng} "
        f"({n_pos} positive, {n_neg} negative), and pooled over within-model "
        f"ranks it is {pooled_txt} — {strength}. Lower validation loss is what "
        "training and every hyperparameter selection minimise; these numbers "
        "describe how that quantity tracked detection performance here, in "
        "this configuration, at this number of seeds. No wider claim is made, "
        "and this says nothing about any comparison between models.")


def generalisation_table(res: dict, bundle: dict) -> str:
    """Same trained models, scored on a second signal they never saw."""
    g = res.get("generalisation_qqq") or {}
    gs = g.get("summary") or {}
    if not gs:
        return ""
    s = res["summary"]
    rows = ["| model | AUC, 2-prong (X,Y→qq) | AUC, 3-prong (X,Y→qqq) | change |",
            "|---|---|---|---:|"]
    for key, label in ROWS:
        if key not in gs:
            continue
        a2, a3 = s[key]["auc"], gs[key]["auc"]
        rows.append(f"| {label_for(key, label, res)} | "
                    f"{a2['mean']:.4f} ± {a2['std']:.4f} | "
                    f"{a3['mean']:.4f} ± {a3['std']:.4f} | "
                    f"{a3['mean'] - a2['mean']:+.4f} |")
    rows.append("")
    alpha = bundle["alpha"]
    qa, qr = bundle["qqq_auc"], bundle["qqq_rej"]
    if qa or qr:
        rows.append("*Paired tests on the 3-prong signal — same machinery, same "
                    "corrected threshold as the 2-prong tables*")
        rows.append("")
        if qa:
            rows.append("*ROC-AUC (3-prong)*")
            rows.append("")
            rows.append(_rows(qa, alpha, "{:+.4f}"))
            rows.append("")
        if qr:
            rows.append("*Background rejection at ε_S = 0.3, log space "
                        "(3-prong)*")
            rows.append("")
            rows.append(_rows(qr, alpha, "{:+.2f}", ratio=True))
            rows.append("")
    rows.append(
        "The 3-prong sample is the LHCO W' → XY, X,Y → qqq signal at the same "
        "masses. **No model was trained, selected or tuned on either signal** — "
        "training is background-only and both depth and learning rate were "
        "chosen on background validation loss — so this is a like-for-like "
        "test of the same trained models against a different signal topology, "
        "not a transfer-learning result. It probes signal-model dependence; "
        "with two topologies at one mass point it does not establish "
        "signal-model independence.")
    return "\n".join(rows)


def ablation_table(res: dict) -> str:
    """The ae_dense 2x2 diagnostic, labelled as a diagnostic.

    Folded behind a <details> block because it is an attribution check on one
    model, not a result about anomaly detection, and should not read as one.
    """
    d = (res.get("diagnostic_ae_dense_ablation") or {}).get("results")
    if not d:
        return ""
    rows = ["<details>",
            "<summary><b>Diagnostic (not a result): ae_dense epoch budget × "
            "bottleneck ReLU</b></summary>", "",
            "| epochs | bottleneck ReLU | ae_dense ROC-AUC |",
            "|---:|---|---|"]
    for key in sorted(d):
        ep = key.split(",")[0].split("=")[1]
        relu = key.split("bottleneck_relu=")[1]
        v = d[key]
        rows.append(f"| {ep} | {'on' if relu == 'True' else 'off'} | "
                    f"{v['auc_mean']:.4f} ± {v['auc_std']:.4f} |")
    rows += ["",
             "Two edits landed together between earlier runs — the epoch budget "
             "and the bottleneck activation — so this varies them one at a time "
             "to see which moved `ae_dense`. It trains a single model with no "
             "baselines and no significance testing, and exists only to "
             "attribute a change. The initialisation RNG stream also changed "
             "between those runs, so these cells are comparable to each other "
             "but not to numbers produced before that fix.",
             "</details>"]
    return "\n".join(rows)


def render(res: dict) -> str:
    """The whole generated block, as it appears in the README."""
    bundle = collect_all(res)
    check_coverage(res, bundle)
    parts = ["<!-- generated by `python -m src.report --write-readme`; "
             "do not edit by hand -->", "", results_table(res), "",
             "**Are any of these differences real?**", "",
             comparison_table(res, bundle), ""]
    gen = generalisation_table(res, bundle)
    if gen:
        parts += ["**Generalisation to a second, never-trained-on signal**", "",
                  gen, ""]
    obj = objective_vs_auc_table(res)
    if obj:
        parts += ["**What the selection objective actually tracked**", "",
                  obj, ""]
    abl = ablation_table(res)
    if abl:
        parts += [abl, ""]
    parts += [encoding_table(res), "", config_line(res)]
    return "\n".join(parts)


def write_readme(res: dict, readme: str = "README.md") -> None:
    """Splice the generated block between the markers, leaving prose untouched."""
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    if BEGIN not in text or END not in text:
        raise ValueError(f"{readme} is missing the {BEGIN} / {END} markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{BEGIN}\n{render(res)}\n{END}{tail}"

    # The placeholder warning is only true before a real run; drop it once
    # generated numbers are in place.
    start = new.find("> **⚠️ NOT YET RUN ON REAL DATA.**")
    if start != -1:
        end = new.find("\n\n", new.find("this block is replaced by generated output.**"))
        if end != -1:
            new = new[:start] + new[end + 2:]
    with open(readme, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"updated {readme}")


def config_line(res: dict) -> str:
    cfg = res["config"]
    lines = [
        f"Configuration: {cfg['n_qubits']} qubits "
        f"({cfg['n_latent']} latent + {cfg['n_trash']} trash), "
        f"ansatz reps {cfg['ansatz_reps']}, "
        f"{cfg['n_train']:,} background training events, "
        f"{cfg['epochs']} max epochs (patience {cfg.get('patience', 'n/a')}). "
        f"{cfg['simulation']}. Wall time {res['wall_seconds']:.0f} s."
    ]

    lr = cfg.get("lr_selected")
    if lr:
        lines.append(
            "Learning rate selected per model family on background validation "
            f"loss from {cfg.get('lr_candidates')}: "
            + ", ".join(f"`{k}` {v:g}" for k, v in lr.items()) + ". "
            "Ansatz depth was selected the same way. Both use only the "
            "background training/validation splits — no labels, no signal, no "
            "test data. Selection was one coordinate pass (depth, then learning "
            "rate), not a joint search.")

    caveats = []
    if cfg.get("reps_selection_hit_boundary"):
        hit = [k for k, v in cfg.get("reps_selection", {}).items()
               if v.get("hit_boundary")]
        caveats.append(
            f"**The unconstrained depth argmin sat at the top of the candidate "
            f"grid {cfg['reps_candidates']} for "
            f"{', '.join('`'+h+'`' for h in hit)}.** The grid ran out before the "
            "optimum did, so a deeper ansatz might score lower on the "
            "objective. (Where a tie-break also fired, the depth actually used "
            "is the shallowest indistinguishable one, not this argmin.)")
    # Selections that came from the tie-break rather than a resolved minimum.
    lr_tb = {k: v for k, v in (cfg.get("lr_selection") or {}).items()
             if v.get("tie_break_fired")}
    reps_tb = {k: v for k, v in (cfg.get("reps_selection") or {}).items()
               if v.get("tie_break_fired")}
    if lr_tb or reps_tb:
        floor = cfg.get("selection_noise_floor")
        bits = []
        for k, v in lr_tb.items():
            bits.append(f"`{k}` learning rate {v['selected']} (argmin was "
                        f"{v['raw_argmin']}, gap {v['best_vs_second_gap']:.5f}, "
                        f"tied {v['tied_set']})")
        for k, v in reps_tb.items():
            bits.append(f"`{k}` ansatz depth {v['selected']} (argmin was "
                        f"{v['raw_argmin']}, gap {v['best_vs_second_gap']:.5f}, "
                        f"tied {v['tied_set']})")
        caveats.append(
            f"**{len(bits)} hyperparameter(s) came from a tie-break, not from a "
            f"resolved minimum.** Candidates within {floor} validation loss of "
            "the best are treated as indistinguishable; among those the "
            "smallest learning rate / shallowest ansatz is taken, a fixed rule "
            "applied identically to every model and using only background "
            "validation loss. Affected: " + "; ".join(bits) + ". For these the "
            "objective did not choose the value — the rule did.")

    lr_edge = [k for k, v in (cfg.get("lr_selection") or {}).items()
               if v.get("hit_boundary")]
    if lr_edge:
        grid = cfg.get("lr_candidates")
        chosen = cfg.get("lr_selected") or {}
        named = ", ".join(f"`{k}` ({chosen.get(k)})" for k in lr_edge)
        caveats.append(
            f"**The learning-rate selection sits at an end of the candidate "
            f"grid {grid}** for {named}. For those models the rate is the edge "
            "of the search rather than a resolved optimum, and a value outside "
            "the grid might train them better — so the comparison is not "
            "between fully-tuned models.")

    noise = [k for k, v in (cfg.get("reps_selection_below_noise_floor") or {}).items() if v]
    if noise:
        floor = cfg.get("selection_noise_floor")
        spreads = ", ".join(
            f"`{k}` spread {cfg['reps_selection'][k]['spread']:.5f}" for k in noise)
        caveats.append(
            f"**Depth selection was not meaningful for {', '.join('`'+n+'`' for n in noise)}** "
            f"({spreads}, below the {floor} noise floor): the candidates are "
            "indistinguishable, so the reported depth for these is arbitrary "
            "rather than chosen.")
    if res.get("any_hit_epoch_cap"):
        caveats.append(
            "**At least one model hit the epoch cap** rather than early "
            "stopping, so it was under-trained relative to models that "
            "converged. See `hit_epoch_cap` in metrics.json.")
    if caveats:
        lines.append("")
        lines.extend("- " + c for c in caveats)
    return "\n".join(lines)


def main(*argv):
    """CLI: `python -m src.report [metrics.json] [--write-readme]`."""
    flags = [a for a in argv if a.startswith("--")]
    positional = [a for a in argv if not a.startswith("--")]
    path = positional[0] if positional else "results/metrics.json"

    with open(path) as f:
        res = json.load(f)
    if "--write-readme" in flags:
        write_readme(res)
        return
    print(render(res))


if __name__ == "__main__":
    main(*sys.argv[1:])
