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


def validation_loss(model, loss_fn, values, batch_size=8192, weighting="events"):
    """Evaluate event-mean loss; legacy equal-batch means remain reproducible."""
    if weighting not in {"events", "legacy-batches"}:
        raise ValueError(f"unknown validation weighting: {weighting}")
    if not len(values) or batch_size < 1:
        raise ValueError("validation data and batch size must be nonempty/positive")
    total, weight = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            loss = float(loss_fn(model, batch))
            if not np.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            n = len(batch) if weighting == "events" else 1
            total += n * loss
            weight += n
    return total / weight


def train_model(model, loss_fn, train: np.ndarray, val: np.ndarray, *,
                epochs: int = 60, batch_size: int = 1024, lr: float = 0.05,
                seed: int = 0, patience: int = 10, verbose: bool = True,
                dtype=torch.float64, validation_weighting="legacy-batches") -> History:
    """Adam + early stopping on validation loss. Restores the best weights.

    `loss_fn(model, xb) -> scalar tensor`. Training data is background only.
    """
    import time

    torch.manual_seed(seed)
    # Follow the model rather than taking a device argument: the classical
    # baselines are so cheap (0.03 s per training run, against 12 s per epoch
    # for a QAE) that moving them to an accelerator would cost more in
    # transfers than it saves, so only the quantum models are ever built off
    # the CPU and the data goes wherever the parameters already are.
    device = next(model.parameters()).device
    xtr = torch.as_tensor(train, dtype=dtype).to(device)
    xva = torch.as_tensor(val, dtype=dtype).to(device)
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
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            opt.step()
            running += float(loss.detach())
            nb += 1

        model.eval()
        if validation_weighting == "legacy-batches":
            # Preserve the historical floating-point reduction exactly.
            with torch.no_grad():
                vl = float(np.mean([
                    float(loss_fn(model, xva[i:i + 8192]))
                    for i in range(0, len(xva), 8192)]))
        else:
            vl = validation_loss(model, loss_fn, xva, weighting=validation_weighting)
        if not np.isfinite(vl):
            raise FloatingPointError("non-finite validation loss")
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
