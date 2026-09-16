"""Numerical and selection-isolation checks for final-v2."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import validation_loss
from src.run_study import selection_guard, load_selection
from src.data import make_splits, FEATURES
import pandas as pd


class FinalProtocolTests(unittest.TestCase):
    def test_released_selection_loads_with_final_protocol(self):
        reference = ROOT / 'results/final-v2/selection.json'
        guard = json.loads(reference.read_text())['guard']
        args = SimpleNamespace(protocol='final-v2', split_seed=0, epochs=600, patience=20,
            reps_candidates=guard['reps_candidates'], lr_candidates=guard['lr_candidates'],
            select_epochs=100, select_n=25000, batch_size=4096, lr=guard['default_lr'],
            n_train=100000, n_val=20000)
        selection = load_selection(reference, args)
        self.assertEqual(selection['digest'], 'f34c739c8982f99e')

    def test_event_weighting_matches_whole_dataset(self):
        x = torch.cat([torch.zeros(16384), torch.ones(3616)]).double()
        loss = lambda model, b: b.mean()
        for size in (1, 127, 8192, 20000):
            self.assertAlmostEqual(validation_loss(None, loss, x, size), float(x.mean()), places=13)
        self.assertAlmostEqual(validation_loss(None, loss, x, 8192, 'legacy-batches'), 1/3)

    def test_old_selection_is_rejected_by_new_protocol(self):
        reference = ROOT / 'results/legacy-v1/reference/selection.json'
        guard = json.loads(reference.read_text())['guard']
        args = SimpleNamespace(protocol='legacy-v1', split_seed=0, epochs=600, patience=20,
            reps_candidates=guard['reps_candidates'], lr_candidates=guard['lr_candidates'],
            select_epochs=guard['select_epochs'], select_n=guard['select_n'],
            batch_size=guard['batch_size'], lr=guard['default_lr'],
            n_train=guard['n_train'], n_val=guard['n_val'])
        self.assertEqual(selection_guard(args), guard)
        load_selection(reference, args)
        args.protocol = 'final-v2'
        with self.assertRaises(SystemExit):
            load_selection(reference, args)

    def test_fixed_partition_disjoint(self):
        frame = pd.DataFrame({f: np.arange(100) for f in FEATURES})
        frame['label'], frame['mjj'] = 0, 1000
        a = make_splits(frame, 20, 10, seed=0)
        b = make_splits(frame, 20, 10, seed=0)
        np.testing.assert_array_equal(a.test, b.test)
        train, val, test = [set(x[:, 0]) for x in (a.train, a.val, a.test)]
        self.assertFalse(train & val or train & test or val & test)


if __name__ == '__main__':
    unittest.main()
