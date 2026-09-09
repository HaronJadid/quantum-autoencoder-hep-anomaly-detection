"""One training loop, used by every torch model in this study.

The QAE and the classical autoencoders are trained by the *same* function with
the same optimiser, schedule, batching, early stopping and seeding. Only the
per-batch loss differs (trash-qubit infidelity vs reconstruction MSE), because
that is what distinguishes the models. This is what "trained identically"
means here, and it is why the comparison is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = 0
    seconds: float = 0.0


def train_model(model, loss_fn, train: np.ndarray, val: np.ndarray, *,
                epochs: int = 60, batch_size: int = 1024, lr: float = 0.05,
                seed: int = 0, patience: int = 10, verbose: bool = True,
                dtype=torch.float64) -> History:
    """Adam + early stopping on validation loss. Restores the best weights.

    `loss_fn(model, xb) -> scalar tensor`. Training data is background only.
    """
    import time

    torch.manual_seed(seed)
    xtr = torch.as_tensor(train, dtype=dtype)
    xva = torch.as_tensor(val, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    hist = History()
    best = (np.inf, 0, {k: v.detach().clone() for k, v in model.state_dict().items()})
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=gen)
        running, nb = 0.0, 0
        for i in range(0, len(xtr), batch_size):
            xb = xtr[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model, xb)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            nb += 1

        model.eval()
        with torch.no_grad():
            vl = float(np.mean([
                float(loss_fn(model, xva[i:i + 8192]))
                for i in range(0, len(xva), 8192)]))
        hist.train_loss.append(running / max(nb, 1))
        hist.val_loss.append(vl)
        hist.epochs_run = epoch + 1

        if vl < best[0] - 1e-9:
            best = (vl, epoch,
                    {k: v.detach().clone() for k, v in model.state_dict().items()})
        elif epoch - best[1] >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch+1} "
                      f"(no val improvement for {patience} epochs)")
            break

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(f"    epoch {epoch+1:3d}/{epochs}  "
                  f"train {hist.train_loss[-1]:.5f}  val {vl:.5f}")

    model.load_state_dict(best[2])
    hist.best_epoch = best[1] + 1
    hist.seconds = time.time() - t0
    if verbose:
        print(f"    done: {hist.epochs_run} epochs in {hist.seconds:.1f}s, "
              f"best val {best[0]:.5f} @ epoch {hist.best_epoch}")
    return hist


def qae_loss(model, xb):
    """Mean trash-qubit infidelity: 1 - P(trash = |0...0>)."""
    return model(xb).mean()


def ae_loss(model, xb):
    """Mean per-event reconstruction MSE."""
    return ((model(xb) - xb) ** 2).mean()
