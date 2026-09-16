"""Regression checks for release reporting and provenance."""
import json
from pathlib import Path
import sys
import unittest
import tempfile
import re
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from src.evaluate import paired_comparison
from src.report import render
from audit_release import main as audit
from build_release_report import outputs, mass_diagnostics


class ReleaseTests(unittest.TestCase):
    def test_public_navigation_and_private_preparation(self):
        self.assertFalse((ROOT / 'CV_PROJECT.md').exists())
        self.assertFalse((ROOT / 'results/metrics.json').exists())
        for relative in ('README.md', 'docs/README.md', 'results/README.md',
                         'results/legacy-v1/README.md'):
            path = ROOT / relative
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                if '://' not in target and not target.startswith('#'):
                    self.assertTrue((path.parent / target.split('#')[0]).exists(),
                                    f'{relative}: broken link {target}')

    def test_mass_diagnostics_are_across_seed_background_values(self):
        runs = [{'sculpting': {'spearman_score_vs_mjj': rho,
                              'median_mjj_shift_frac': shift}}
                for rho, shift in [(0.1, 0.02), (0.3, 0.06)]]
        values = mass_diagnostics(runs)
        self.assertAlmostEqual(values['median_mjj_shift_frac']['mean'], 0.04)
        self.assertAlmostEqual(values['median_mjj_shift_frac']['std'], np.sqrt(0.0008))
        self.assertAlmostEqual(values['spearman_score_vs_mjj']['mean'], 0.2)
        study = json.loads((ROOT / 'results/final-v2/metrics.json').read_text())
        matched = mass_diagnostics(study['per_seed']['ae_matched'])
        self.assertAlmostEqual(matched['median_mjj_shift_frac']['mean'], 0.0615180695878988)

    def test_generalisation_figure_uses_all_model_standard_deviations(self):
        from src.figures import generalisation_figure
        study = json.loads((ROOT / 'results/final-v2/metrics.json').read_text())
        with tempfile.TemporaryDirectory() as folder, \
             patch('matplotlib.axes.Axes.errorbar') as errorbar:
            self.assertTrue(generalisation_figure(study, str(Path(folder) / 'figure.png')))
            self.assertEqual(errorbar.call_count, 18)
            expected = sorted(v['auc']['std'] for block in
                              (study['summary'], study['generalisation_qqq']['summary'])
                              for v in block.values())
            actual = sorted(call.kwargs['xerr'] for call in errorbar.call_args_list)
            self.assertEqual(actual, expected)

    def test_budget_runner_small_fixture_and_resume(self):
        # Synthetic fixture exercises orchestration, not scientific results.
        from src.data import FEATURES
        from src.train import train_model
        import run_budget_diagnostic
        original = json.loads((ROOT / 'results/final-v2/metrics.json').read_text())
        original['config']['n_train'] = 8
        original['config']['n_val'] = 4
        rng = np.random.default_rng(123)
        frame = pd.DataFrame(rng.uniform(size=(24, len(FEATURES))), columns=FEATURES)
        frame['label'] = [0] * 20 + [1] * 4
        frame['mjj'] = np.arange(24) + 1000
        extra = frame.iloc[-4:].drop(columns=['label'])
        def short_training(*args, **kwargs):
            kwargs['epochs'] = 2
            return train_model(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder) / 'study.json'
            study.write_text(json.dumps(original))
            out = Path(folder) / 'output'
            argv = ['run_budget_diagnostic', '--study', str(study), '--out', str(out)]
            with patch.object(sys, 'argv', argv), \
                 patch('src.data.load_rnd', return_value=frame), \
                 patch('src.data.load_qqq', return_value=extra), \
                 patch('src.train.train_model', side_effect=short_training) as train:
                run_budget_diagnostic.main()
                self.assertEqual(train.call_count, 15)
                run_budget_diagnostic.main()
                self.assertEqual(train.call_count, 15)
            self.assertTrue((out / 'seed-00-scores.npz').exists())
            self.assertEqual(json.loads((out / 'summary.json').read_text())['primary']['n_seeds'], 15)

    def test_provenance_and_summaries(self):
        audit()

    def test_generated_content(self):
        for path, expected in outputs().items():
            self.assertEqual(path.read_text(encoding='utf-8'), expected)

    def test_constant_pairs_have_no_t_test(self):
        result = paired_comparison([{'auc': .6}] * 15, [{'auc': .5}] * 15)
        self.assertTrue(result['degenerate'])
        self.assertNotEqual(result['p_value'], result['p_value'])

    def test_fixed_split_report_exposes_untestable_pair(self):
        result = json.loads((ROOT / 'results/final-v2/metrics.json').read_text())
        report = render(result)
        self.assertIn('constant across seeds', report)
        self.assertIn('not testable', report)
        self.assertNotIn('so it was under-trained', report)


if __name__ == '__main__':
    unittest.main()
