"""Regression checks for reporting a one-seed --no-ablation reproduction."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import aggregate
from src.report import render


class ReproductionReportTests(unittest.TestCase):
    def setUp(self):
        self.res = json.loads((ROOT / 'results/legacy-v1/metrics.json').read_text())
        self.res['config']['seeds'] = [0]
        for block in (self.res, self.res['generalisation_qqq']):
            for model, runs in block['per_seed'].items():
                block['per_seed'][model] = runs[:1]
                block['summary'][model] = aggregate(runs[:1])
        self.res['diagnostic_ae_dense_ablation'] = {
            'n_seeds': 0, 'description': 'Deliberately skipped', 'results': None}

    def test_skipped_diagnostic_is_reported_without_mutating_results(self):
        before = copy.deepcopy(self.res)
        text = render(self.res)
        self.assertIn('ablation was skipped', text)
        self.assertIn('not estimable with one seed', text)
        self.assertNotIn('34 paired tests are performed', text)
        self.assertNotIn('± 0.0000', text)
        self.assertNotIn('Two-sided paired t-tests across 1', text)
        self.assertEqual(before, self.res)

    def test_missing_nonempty_diagnostic_is_still_rejected(self):
        self.res['diagnostic_ae_dense_ablation']['n_seeds'] = 5
        with self.assertRaises(AssertionError):
            render(self.res)

    def test_legacy_empty_dictionary_is_also_supported(self):
        self.res['diagnostic_ae_dense_ablation']['results'] = {}
        self.assertIn('ablation was skipped', render(self.res))

    def test_missing_results_field_is_rejected(self):
        del self.res['diagnostic_ae_dense_ablation']['results']
        with self.assertRaises(AssertionError):
            render(self.res)

    def test_unknown_content_is_still_rejected(self):
        self.res['diagnostic_ae_dense_ablation']['unreported_results'] = [0.7]
        with self.assertRaises(AssertionError):
            render(self.res)


if __name__ == '__main__':
    unittest.main()
