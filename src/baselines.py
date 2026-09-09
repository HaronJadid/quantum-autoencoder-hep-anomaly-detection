"""Classical baselines, on identical inputs to the QAE.

Four of them, deliberately:

1. `TiedAE`      - matched parameter count (24, exactly the QAE's). Tied
                   weights, tanh latent, no biases: 6 -> 4 -> 6.
2. `DenseAE`     - the autoencoder a practitioner would actually build
                   (6-16-4-16-6, ~360 parameters).
3. PCA           - linear reconstruction error at the same latent dimension.
4. `mj1` alone   - single-feature threshold, no training at all.

Reporting only (1) would look like handicapping the classical side; reporting
only (2) would be unfair to the QAE. (4) is the control that matters most:
autoencoders on jet features are known to score largely as a proxy for jet
mass, so if the QAE cannot beat a plain cut on mj1 there is no result here.

Latent dimension matching: the QAE keeps 4 of 6 qubits, so the classical models
keep 4 of 6 dimensions. This matches the number of retained degrees of freedom.
It does NOT match Hilbert-space dimension (4 qubits span a 16-dimensional
space) -- no single notion of "same size" applies across the two model
families, which is exactly why several baselines are reported rather than one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TiedAE(nn.Module):
    """Tied-weight autoencoder: z = tanh(x W^T + b_z), x_hat = (z W) * s + b.

    Weight tying keeps the parameter count low and controllable:

        d * latent                      base
        + latent                        with latent_bias
        + 2 * d                         with output_affine

    The tanh keeps it from collapsing to plain PCA. The optional terms exist so
    the count can be matched exactly to the QAE's, which is not a fixed number:
    it is n_qubits * (reps + 1) for whatever depth gets selected.
    """

    def __init__(self, d: int = 6, latent: int = 4, seed: int = 0,
                 output_affine: bool = False, latent_bias: bool = False,
                 dtype=torch.float64):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(
            torch.randn(latent, d, generator=g, dtype=dtype) / np.sqrt(d))
        self.b_z = nn.Parameter(torch.zeros(latent, dtype=dtype)) if latent_bias else None
        if output_affine:
            self.scale = nn.Parameter(torch.ones(d, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(d, dtype=dtype))
        else:
            self.scale = self.bias = None

    def forward(self, x):
        h = x @ self.W.T
        if self.b_z is not None:
            h = h + self.b_z
        out = torch.tanh(h) @ self.W
        if self.scale is not None:
            out = out * self.scale + self.bias
        return out


def matched_ae(d: int, quantum_latent: int, target_params: int, seed: int = 0):
    """A tied AE with EXACTLY `target_params` trainable parameters.

    The QAE's parameter count depends on the ansatz depth chosen at run time,
    so the matched baseline has to be built to order rather than hard-coded.
    Candidates are searched preferring the latent dimension closest to the
    quantum model's, so the comparison holds compression fixed where it can.

    Raises if no exact match exists, rather than quietly reporting a baseline
    that is not actually matched.
    """
    candidates = []
    for latent in range(1, d + 1):
        for affine in (False, True):
            for lbias in (False, True):
                n = d * latent + (2 * d if affine else 0) + (latent if lbias else 0)
                if n == target_params:
                    candidates.append((abs(latent - quantum_latent), latent,
                                       affine, lbias))
    if not candidates:
        raise ValueError(
            f"no tied-AE configuration has exactly {target_params} parameters "
            f"for d={d}; adjust --reps-candidates or the baseline family "
            "rather than reporting an unmatched 'matched' baseline")
    _, latent, affine, lbias = min(candidates)
    model = TiedAE(d, latent, seed=seed, output_affine=affine, latent_bias=lbias)
    actual = count_params(model)
    assert actual == target_params, f"built {actual} params, wanted {target_params}"
    return model, {"latent": latent, "output_affine": affine,
                   "latent_bias": lbias, "params": actual}


class DenseAE(nn.Module):
    """Conventional dense autoencoder, 6-16-4-16-6 with ReLU."""

    def __init__(self, d: int = 6, hidden: int = 16, latent: int = 4,
                 seed: int = 0, dtype=torch.float64):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, d),
        ).to(dtype)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def ae_scores(model, x: np.ndarray, batch_size: int = 16384,
              dtype=torch.float64) -> np.ndarray:
    """Per-event reconstruction MSE = the classical anomaly score."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i:i + batch_size], dtype=dtype)
        out.append(((model(xb) - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


def pca_scores(train: np.ndarray, test: np.ndarray,
               n_components: int = 4) -> np.ndarray:
    """Reconstruction error of a PCA subspace fit on background only."""
    from sklearn.decomposition import PCA

    p = PCA(n_components=n_components).fit(train)
    recon = p.inverse_transform(p.transform(test))
    return ((test - recon) ** 2).mean(axis=1)


def single_feature_scores(test: np.ndarray, feature_index: int) -> np.ndarray:
    """Use one feature directly as the anomaly score (higher = more anomalous).

    No training. For mj1 this is the 'is the model just finding heavy jets?'
    control. AUC is invariant under the monotone per-feature scaling applied
    upstream, so this number is unaffected by the choice of scaler.
    """
    return test[:, feature_index].astype(np.float64)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
