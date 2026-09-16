"""Check saved results and generated prose without retraining models."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import aggregate
from src.report import BEGIN, END, render


def main():
    result = json.loads((ROOT / 'results/legacy-v1/metrics.json').read_text())
    for block in (result, result['generalisation_qqq']):
        for model, runs in block['per_seed'].items():
            assert len(runs) == len(result['config']['seeds']), model
            assert aggregate(runs) == block['summary'][model], model
    # Legacy measurements remain archived; the README now leads with final-v2.
    from build_release_report import outputs
    for path, expected in outputs().items():
        assert path.read_text(encoding='utf-8') == expected, path
    reproduction = ROOT / 'results/legacy-v1/parallel/seed0_scores/metrics.json'
    if reproduction.exists():
        rerun = json.loads(reproduction.read_text())
        for model, runs in result['per_seed'].items():
            assert runs[0]['auc'] == rerun['per_seed'][model][0]['auc'], model
        print('All nine seed-0 AUCs match the saved reproduction exactly.')
    else:
        print('Optional local seed-0 reproduction is unavailable.')
    # Complex states distinguish the density matrix from its conjugate.
    import pennylane as qml
    from src.encoding_analysis import encoded_density_matrix
    def phase_state(x, wires):
        qml.Hadamard(wires=0)
        qml.PhaseShift(x[..., 0], wires=0)
    rho = encoded_density_matrix(phase_state, np.array([[np.pi / 2]]), 1)
    expected = np.array([[1, -1j], [1j, 1]]) / 2
    np.testing.assert_allclose(rho, expected, atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(rho),
                               np.linalg.eigvalsh(rho.conj()), atol=1e-14)
    print('Legacy summaries, release reports and complex density matrix pass.')


if __name__ == '__main__':
    main()
