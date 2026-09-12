"""Time a few real-scale QAE epochs at a given torch thread count."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
th = int(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else ""
torch.set_num_threads(th)
from src.data import load_rnd, make_splits, FEATURES
from src.run_study import scale, N_QUBITS, N_TRASH
from src.qae import QuantumAutoencoder
from src.train import train_model, qae_loss
df = load_rnd("data")
sp = make_splits(df, 100_000, 20_000, seed=0)
xtr, xva = scale(sp.train, sp.val, seed=0)
m = QuantumAutoencoder(N_QUBITS, N_TRASH, 7, seed=0, feature_map="ry")
t0 = time.time()
h = train_model(m, qae_loss, xtr, xva, epochs=5, batch_size=4096, lr=0.05,
                seed=0, patience=99, verbose=False)
dt = time.time() - t0
print(f"RESULT threads={th} {tag} {dt/5:.3f} s/epoch  (5 epochs in {dt:.1f}s)")
