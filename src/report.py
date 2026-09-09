"""Generate the README results table from results/metrics.json.

The table is generated, never typed. Numbers in the README therefore cannot
drift from the numbers actually produced by a run.

    python -m src.report              # print markdown to stdout
"""

from __future__ import annotations

import json
import sys

ROWS = [
    ("qae_ry", "**Quantum AE** (RY encoding)"),
    ("qae_zz", "Quantum AE (ZZFeatureMap)"),
    ("ae_matched", "Classical AE, parameter-matched"),
    ("ae_dense", "Classical AE, 6-16-4-16-6"),
    ("pca", "PCA, 4 components"),
    ("mj1_only", "Cut on $m_{J_1}$ alone (no training)"),
]


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
            f"| {label} | {p[key]} | {_pm(s[key]['auc'], n_expected=len(n))} | "
            f"{_pm(s[key]['rejection[eps_s=0.3]'], '{:.1f}', n_expected=len(n))} | "
            f"{s[key]['spearman_score_vs_mjj']['mean']:+.3f} |")
    out.append("")
    out.append(f"Mean ± sample s.d. over {len(n)} seed{'s' if len(n) != 1 else ''} "
               f"({', '.join(map(str, n))}); each seed re-draws the data "
               "split and re-initialises every model.")
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
    ("qae_ry", "ae_dense", "QAE vs. the larger classical AE"),
    ("ae_dense", "mj1_only", "larger classical AE vs. $m_{J_1}$ cut"),
]


REJECTION_COMPARISONS = [
    ("qae_ry", "mj1_only", "QAE vs. a plain cut on $m_{J_1}$"),
    ("qae_zz", "mj1_only", "**incompressible** encoding vs. $m_{J_1}$ cut"),
    ("ae_matched", "mj1_only", "classical AE (matched) vs. $m_{J_1}$ cut"),
    ("pca", "mj1_only", "PCA vs. $m_{J_1}$ cut"),
    ("qae_ry", "ae_matched", "QAE vs. classical AE at equal parameters"),
    ("qae_ry", "qae_zz", "QAE vs. the incompressible encoding"),
]


def _table(res: dict, pairs, metric: str, header: str, fmt: str) -> str:
    from .evaluate import paired_comparison

    ps = res.get("per_seed", {})
    rows = [f"| {header} | mean difference | t | p | significant? |",
            "|---|---:|---:|---:|---|"]
    for a, b, label in pairs:
        if a not in ps or b not in ps:
            continue
        try:
            r = paired_comparison(ps[a], ps[b], metric)
        except ValueError:
            rows.append(f"| {label} | not testable (infinite rejection) | | | |")
            continue
        sig = "**yes**" if r["significant_at_0.05"] else "no"
        rows.append(f"| {label} | {fmt.format(r['mean_difference'])} ± "
                    f"{fmt.format(r['sem']).lstrip('+')} | "
                    f"{r['t']:.2f} | {r['p_value']:.3f} | {sig} |")
    return "\n".join(rows)


def comparison_table(res: dict) -> str:
    """Paired per-seed tests on both headline metrics.

    They disagree, and that disagreement is the main result, so both are shown.
    """
    auc = _table(res, COMPARISONS, "auc", "comparison (ROC-AUC)", "{:+.4f}")
    rej = _table(res, REJECTION_COMPARISONS, "rejection@0.3",
                 r"comparison (1/ε_B at ε_S=0.3)", "{:+.2f}")
    return "\n".join([
        auc, "",
        rej, "",
        "Two-sided paired t-tests across seeds. Within a seed every model sees "
        "exactly the same events, so pairing removes split-to-split variation "
        "and is more sensitive than comparing independent mean ± s.d. summaries. "
        "With 5 seeds the power is low: *not significant* means **not resolved "
        "by this study**, not *identical*.",
        "",
        "Note the two metrics disagree. No model separates from any other on "
        "AUC, but every multivariate score beats the single-variable $m_{J_1}$ "
        "cut on rejection at a fixed working point — **including the "
        "ZZFeatureMap model that provably cannot compress**. Beating the mass "
        "cut therefore demonstrates only that more than one feature is being "
        "used; it is not evidence that the autoencoder, quantum or classical, "
        "is doing anything further.",
    ])


def render(res: dict) -> str:
    """The whole generated block, as it appears in the README."""
    parts = ["<!-- generated by `python -m src.report --write-readme`; "
             "do not edit by hand -->", "", results_table(res), "",
             "**Are any of these differences real?**", "",
             comparison_table(res), "",
             encoding_table(res), "", config_line(res)]
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
    return (f"Configuration: {cfg['n_qubits']} qubits "
            f"({cfg['n_latent']} latent + {cfg['n_trash']} trash), "
            f"ansatz reps {cfg['ansatz_reps']} (selected on background "
            f"validation loss from {cfg['reps_candidates']}), "
            f"{cfg['n_train']:,} background training events, "
            f"{cfg['epochs']} max epochs. {cfg['simulation']}. "
            f"Wall time {res['wall_seconds']:.0f} s.")


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
