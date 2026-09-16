"""Generate the compact README table and full final-v2 report from JSON."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.report import render

BEGIN = '<!-- BEGIN RELEASE SUMMARY -->'
END = '<!-- END RELEASE SUMMARY -->'


def mass_diagnostics(runs):
    """Across-seed moments of background-only diagnostics, not pooled events."""
    fields = ('spearman_score_vs_mjj', 'median_mjj_shift_frac')
    return {key: {'mean': float(np.mean([r['sculpting'][key] for r in runs])),
                  'std': float(np.std([r['sculpting'][key] for r in runs], ddof=1))
                         if len(runs) > 1 else 0.0}
            for key in fields}


def summary_table(res, diagnostic, diagnostic_runs):
    def pm(value, digits=4):
        return f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"
    rows = [
        '| Model / epoch cap | Parameters | Primary AUC | Primary rejection at ε_S=0.3 | Three-prong AUC | Background ρ(score, m_JJ) | Median m_JJ shift, top 1% |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    def mass_columns(runs):
        values = mass_diagnostics(runs)
        rho = pm(values['spearman_score_vs_mjj'], 3)
        shift = {k: 100 * v for k, v in values['median_mjj_shift_frac'].items()}
        return f'{rho} | ({pm(shift, 2)})%'
    for key, label in [('qae_ry', 'RY quantum AE / 600'),
                       ('qae_zz', 'ZZ quantum AE / 600'),
                       ('ae_matched', 'Matched classical AE / 600'),
                       ('mj1_only', 'Jet-mass control / no training')]:
        a, b = res['summary'][key], res['generalisation_qqq']['summary'][key]
        rows.append(f"| {label} | {res['parameter_counts'][key]} | {pm(a['auc'])} | "
                    f"{pm(a['rejection[eps_s=0.3]'], 2)} | {pm(b['auc'])} | "
                    f"{mass_columns(res['per_seed'][key])} |")
    a, b = diagnostic['primary'], diagnostic['three_prong']
    rows.append(f"| Matched classical AE / 3,000 (diagnostic) | "
                f"{res['parameter_counts']['ae_matched']} | {pm(a['auc'])} | "
                f"{pm(a['rejection[eps_s=0.3]'], 2)} | {pm(b['auc'])} | "
                f"{mass_columns([r['primary'] for r in diagnostic_runs])} |")
    return '\n'.join(rows)


def outputs():
    base = ROOT / 'results/final-v2'
    res = json.loads((base / 'metrics.json').read_text())
    diagnostic = json.loads((base / 'budget-3000/summary.json').read_text())
    diagnostic_runs = [json.loads((base / 'budget-3000' / f'seed-{seed:02d}.json').read_text())
                       for seed in res['config']['seeds']]
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    head, rest = readme.split(BEGIN)
    _, tail = rest.split(END)
    readme = head + BEGIN + '\n' + summary_table(res, diagnostic, diagnostic_runs) + '\n' + END + tail
    return {ROOT / 'README.md': readme, base / 'report.md': render(res) + '\n'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    for path, expected in outputs().items():
        if args.check:
            assert path.read_text(encoding='utf-8') == expected, f'Stale generated content: {path}'
        else:
            path.write_text(expected, encoding='utf-8')
        print(('Checked ' if args.check else 'Generated ') + str(path.relative_to(ROOT)))


if __name__ == '__main__':
    main()
