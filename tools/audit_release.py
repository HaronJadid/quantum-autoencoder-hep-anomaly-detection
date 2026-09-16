"""Read-only provenance, summary and training-history checks for final-v2."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import aggregate


def without_timings(value):
    result = copy.deepcopy(value)
    result.pop('wall_seconds', None)
    result.pop('wall_seconds_summed', None)
    for models in result['histories'].values():
        for history in models.values():
            history.pop('seconds', None)
    return result


def main():
    base = ROOT / 'results/final-v2'
    load = lambda p: json.loads(p.read_text())
    result = load(base / 'metrics.json')
    seeds = result['config']['seeds']
    assert seeds == list(range(15))
    for block in (result, result['generalisation_qqq']):
        for name, runs in block['per_seed'].items():
            assert len(runs) == len(seeds)
            assert aggregate(runs) == block['summary'][name], name
    budget = base / 'budget-3000'
    manifest = load(budget / 'manifest.json')
    original = budget / 'source-study-metrics.json'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == manifest['original_metrics_sha256']
    assert without_timings(load(original)) == without_timings(result)
    snapshot = base / 'source_snapshot'
    combined = hashlib.sha256()
    for p in sorted((snapshot / 'src').glob('*.py')):
        combined.update(p.name.encode())
        combined.update(p.read_bytes())
        assert hashlib.sha256(p.read_bytes()).hexdigest() == manifest['source_hashes']['src/' + p.name]
    combined.update((snapshot / 'colab_run.py').read_bytes())
    assert combined.hexdigest() == load(base / 'run_manifest.json')['source_sha256']
    runs = [load(budget / f'seed-{seed:02d}.json') for seed in seeds]
    summary = load(budget / 'summary.json')
    assert summary['manifest'] == manifest
    assert manifest['selection_digest'] == result['config']['selection_digest']
    assert manifest['seeds'] == seeds
    for key in ('primary', 'three_prong'):
        assert aggregate([run[key] for run in runs]) == summary[key]
    for seed, run in zip(seeds, runs):
        assert run['seed'] == seed and run['max_epochs'] == manifest['max_epochs']
        assert run['matched_ae_spec'] == result['matched_ae_spec']
        history = run['history']
        assert not run['hit_epoch_cap'] and history['epochs_run'] < run['max_epochs']
        assert history['epochs_run'] - history['best_epoch'] == manifest['patience']
        np.testing.assert_array_equal(history['val_loss'][:600],
                                      result['histories'][str(seed)]['ae_matched']['val_loss'])
    assert summary['capped_seeds'] == []
    print('PASS: 18 study summaries; diagnostic summaries; 15 history prefixes;')
    print('      source hashes; original-file hash; timing-only provenance difference.')


if __name__ == '__main__':
    main()
