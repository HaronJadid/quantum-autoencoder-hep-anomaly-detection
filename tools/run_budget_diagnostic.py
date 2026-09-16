"""Reproduce the separately reported 3000-epoch matched-AE budget diagnostic.

No hyperparameter search or quantum training. Completed seeds are resumable
only under the same recorded settings, source and environment.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', type=Path, default=ROOT / 'results/final-v2/metrics.json')
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    if out == ROOT or out.is_relative_to(ROOT / 'results'):
        ap.error('Use a new run directory, not the repository root or published results.')

    import numpy as np
    import torch
    from src.baselines import matched_ae, count_params, ae_scores
    from src.data import load_rnd, load_qqq, make_splits, FEATURES
    from src.evaluate import evaluate, aggregate
    from src.run_study import scale, environment
    from src.train import train_model, ae_loss

    torch.set_num_threads(1)
    original_bytes = args.study.read_bytes()
    original = json.loads(original_bytes)
    cfg = original['config']
    assert cfg['protocol'] == 'final-v2' and cfg['split_policy'] == 'fixed'
    assert cfg['validation_weighting'] == 'events'
    assert cfg['seeds'] == list(range(15))
    assert original['parameter_counts']['ae_matched'] == 60
    manifest = {
        'purpose': 'Matched-AE training-budget sensitivity; not retuning',
        'original_metrics_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'selection_digest': cfg['selection_digest'],
        'max_epochs': 3000, 'patience': cfg['patience'],
        'batch_size': cfg['batch_size'], 'lr': cfg['lr_selected']['ae_matched'],
        'seeds': cfg['seeds'], 'environment': environment(), 'threads': 1,
        'source_hashes': {str(p.relative_to(ROOT)).replace('\\', '/'):
                          hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / 'src').glob('*.py'))},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Resume mismatch: use a fresh output directory; do not mix runs.')
    else:
        if list(out.iterdir()):
            raise ValueError('Nonempty output directory without a matching manifest.')
        path.write_text(json.dumps(manifest, indent=2))

    df = load_rnd(str(args.data_dir))
    extra = load_qqq(str(args.data_dir))
    splits = make_splits(df, cfg['n_train'], cfg['n_val'], seed=cfg['split_seed'])
    xtr, xva, xte, xextra = scale(splits.train, splits.val, splits.test,
                                 extra[FEATURES].to_numpy(), seed=cfg['split_seed'])
    background = splits.test_label == 0
    extra_labels = np.concatenate([np.zeros(int(background.sum()), dtype=np.int64),
                                   np.ones(len(extra), dtype=np.int64)])
    extra_mjj = np.concatenate([splits.test_mjj[background], extra['mjj'].to_numpy()])
    del df, extra
    for seed in cfg['seeds']:
        target = out / f'seed-{seed:02d}.json'
        if target.exists():
            saved = json.loads(target.read_text())
            assert saved['seed'] == seed and saved['max_epochs'] == 3000
            print(f'Seed {seed}: already complete.', flush=True)
            continue
        model, spec = matched_ae(len(FEATURES), cfg['n_latent'], 60, seed=seed)
        assert count_params(model) == 60 and spec == original['matched_ae_spec']
        print(f'Starting matched classical seed {seed}', flush=True)
        history = train_model(model, ae_loss, xtr, xva, epochs=3000,
                              batch_size=cfg['batch_size'], lr=manifest['lr'],
                              seed=seed, patience=cfg['patience'],
                              validation_weighting='events', verbose=False)
        primary_scores = ae_scores(model, xte)
        alternate_scores = np.concatenate([primary_scores[background], ae_scores(model, xextra)])
        result = {
            'seed': seed, 'max_epochs': 3000, 'matched_ae_spec': spec,
            'history': asdict(history), 'hit_epoch_cap': history.epochs_run >= 3000,
            'primary': evaluate(primary_scores, splits.test_label, splits.test_mjj),
            'three_prong': evaluate(alternate_scores, extra_labels, extra_mjj),
        }
        torch.save(model.state_dict(), out / f'seed-{seed:02d}-weights.pt')
        if seed == 0:
            np.savez_compressed(out / 'seed-00-scores.npz',
                                primary_scores=primary_scores, primary_labels=splits.test_label,
                                primary_mjj=splits.test_mjj, three_prong_scores=alternate_scores,
                                three_prong_labels=extra_labels, three_prong_mjj=extra_mjj)
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(target)
        print(f'Saved seed {seed}: best epoch {history.best_epoch}, '
              f'stopped at {history.epochs_run}.', flush=True)
    records = [json.loads((out / f'seed-{seed:02d}.json').read_text()) for seed in cfg['seeds']]
    summary = {'manifest': manifest,
               'primary': aggregate([r['primary'] for r in records]),
               'three_prong': aggregate([r['three_prong'] for r in records]),
               'capped_seeds': [r['seed'] for r in records if r['hit_epoch_cap']]}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Done. Separate diagnostic saved to {out}')


if __name__ == '__main__':
    main()
