"""Exercise orchestration with mocked training; never claims to run Colab."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('colab_run', ROOT / 'tools/colab_run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class WorkflowTests(unittest.TestCase):
    def test_frozen_run_resume_and_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, 'run')
            frozen = Path(tmp, 'frozen.json')
            frozen.write_text('{"selection": "test fixture"}')
            commands = []
            def fake_run(cmd):
                commands.append(cmd)
                if 'src.run_study' in cmd:
                    self.assertNotIn('--selection-only', cmd)
                    dest = Path(cmd[cmd.index('--out') + 1])
                    dest.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()[:16]
                    (dest / 'metrics.json').write_text(json.dumps({
                        'config': {'seeds': [0], 'selection_digest': digest}}))
                    (dest / 'figures').mkdir(exist_ok=True)
                    for name in ('scores.png', 'sculpting.png'):
                        (dest / 'figures' / name).write_bytes(b'figure fixture')
                elif 'src.figures' in cmd:
                    Path(cmd[cmd.index('--outdir') + 1]).mkdir(exist_ok=True)
                return 0
            argv = ['colab_run', '--device', 'cpu', '--seeds', '1', '--chunk', '1',
                    '--selection-from', str(frozen), '--no-ablation',
                    '--save-scores', '--out', str(out)]
            with patch.object(sys, 'argv', argv), patch.object(runner, 'sh', fake_run):
                self.assertEqual(runner.main(), 0)
                self.assertEqual((out / 'parallel/selection.json').read_bytes(), frozen.read_bytes())
                self.assertTrue((out / 'figures/scores.png').exists())
                commands.clear()
                self.assertEqual(runner.main(), 0)
                self.assertFalse(any('src.run_study' in c for c in commands))
                commands.clear()
                changed = list(argv)
                changed[changed.index('--chunk') + 1] = '2'
                with patch.object(sys, 'argv', changed):
                    self.assertEqual(runner.main(), 2)
                self.assertEqual(commands, [])

    def test_bad_chunk_is_rejected(self):
        with patch.object(sys, 'argv', ['colab_run', '--chunk', '0']):
            with self.assertRaises(SystemExit) as caught:
                runner.main()
            self.assertEqual(caught.exception.code, 2)

    def test_notebook_inspect_cells(self):
        # Execute actual notebook code locally, suppressing rendered output.
        # No training, installation, data download or archive in inspect mode.
        notebook = json.loads((ROOT / 'notebooks/QAE_LHCO.ipynb').read_text())
        namespace = {}
        with patch('IPython.display.display'):
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    source = cell['source']
                    exec(compile(''.join(source), cell['id'], 'exec'), namespace)
        self.assertEqual(namespace['MODE'], 'inspect')
        self.assertEqual(len(namespace['res']['config']['seeds']), 15)


if __name__ == '__main__':
    unittest.main()
