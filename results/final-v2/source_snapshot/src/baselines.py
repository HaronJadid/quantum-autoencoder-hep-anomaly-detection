"""Classical baselines, on identical inputs to the QAE.

1. `TiedAE` via `matched_ae` - EXACTLY the QAE's parameter count, whatever
                   ansatz depth is selected at run time. Tied weights.
2. `UntiedAE`    - conventional untied autoencoders bracketing that count.
                   Weight tying is forced by the exact-match requirement: the
                   untied counts are 19/32/45/58/71/84 for latent 1..6, and the
                   QAE's target (6*(reps+1), e.g. 48) is not among them. Tying
                   is a capacity handicap that runs in the QAE's favour, so
                   untied models bracketing the target are reported too.
3. `DenseAE`     - the autoencoder a practitioner would actually build
                   (6-16-4-16-6, ~360 parameters).
4. PCA           - linear reconstruction error at the same latent dimension.
5. `mj1` alone   - single-feature threshold, no training at all.

Reporting only the matched model would look like handicapping the classical
side; reporting only the dense one would be unfair to the QAE. (5) is the
control that matters most: autoencoders on jet features are known to score
largely as a proxy for jet mass.

Latent dimension: the QAE keeps 4 of its 6 qubits, but the classical baselines
do NOT all keep 4 of 6 dimensions. PCA and DenseAE use latent 4; UntiedAE is
reported at latent 2, 3 and 4; and `matched_ae` uses whatever latent the exact
parameter count forces, which for a 48-parameter target is 6 -- no bottleneck
at all for 6-dimensional input. Matching the parameter count and matching the
retained degrees of freedom are different constraints and cannot both be
imposed; this code holds the parameter count fixed and reports the resulting
latent so the trade is visible. Neither matches Hilbert-space dimension (4
qubits span a 16-dimensional space).
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
        + latent                        with latent_gain
        + 2 * d                         with output_affine

    The tanh keeps it from collapsing to plain PCA. The optional terms are all
    ordinary autoencoder components (a latent bias, a per-latent decoder gain,
    an output affine), and they exist so the count can be matched exactly to
    the QAE's -- which is not a fixed number but n_qubits * (reps + 1) for
    whatever depth gets selected. Without `latent_gain` the reachable counts
    for d=6 top out at 54, which cannot match a reps=9 ansatz (60).
    """

    def __init__(self, d: int = 6, latent: int = 4, seed: int = 0,
                 output_affine: bool = False, latent_bias: bool = False,
                 latent_gain: bool = False, dtype=torch.float64):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(
            torch.randn(latent, d, generator=g, dtype=dtype) / np.sqrt(d))
        self.b_z = nn.Parameter(torch.zeros(latent, dtype=dtype)) if latent_bias else None
        self.g_z = nn.Parameter(torch.ones(latent, dtype=dtype)) if latent_gain else None
        if output_affine:
            self.scale = nn.Parameter(torch.ones(d, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(d, dtype=dtype))
        else:
            self.scale = self.bias = None

    def forward(self, x):
        h = x @ self.W.T
        if self.b_z is not None:
            h = h + self.b_z
        z = torch.tanh(h)
        if self.g_z is not None:
            z = z * self.g_z
        out = z @ self.W
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
                for lgain in (False, True):
                    n = (d * latent + (2 * d if affine else 0)
                         + (latent if lbias else 0) + (latent if lgain else 0))
                    if n == target_params:
                        # prefer the latent dimension closest to the quantum
                        # model's, then the simplest model achieving the count
                        candidates.append((abs(latent - quantum_latent),
                                           affine + lbias + lgain,
                                           latent, affine, lbias, lgain))
    if not candidates:
        raise ValueError(
            f"no tied-AE configuration has exactly {target_params} parameters "
            f"for d={d}; adjust --reps-candidates or the baseline family "
            "rather than reporting an unmatched 'matched' baseline")
    _, _, latent, affine, lbias, lgain = min(candidates)
    model = TiedAE(d, latent, seed=seed, output_affine=affine,
                   latent_bias=lbias, latent_gain=lgain)
    actual = count_params(model)
    assert actual == target_params, f"built {actual} params, wanted {target_params}"
    return model, {"latent": latent, "output_affine": affine,
                   "latent_bias": lbias, "latent_gain": lgain, "params": actual}


class _preserve_global_rng:
    """Restore global torch RNG state on exit.

    nn.Linear's constructor calls reset_parameters(), which draws from the
    global RNG before any local re-initialisation can run. Wrapping module
    construction in this keeps model building free of global side effects, so
    results do not depend on how many models were built beforehand.
    """

    def __enter__(self):
        self._state = torch.get_rng_state()
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self._state)
        return False


def _seeded_linear_init(module: nn.Module, gen: torch.Generator) -> None:
    """Re-initialise every nn.Linear from a LOCAL generator.

    torch.nn.Linear.reset_parameters() draws from the global RNG, so building a
    model would otherwise perturb global RNG state and make results depend on
    construction order. PyTorch's default is kaiming_uniform_(a=sqrt(5)) for the
    weight and uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for the bias; with
    a=sqrt(5) the weight bound reduces to the same 1/sqrt(fan_in), so both are
    reproduced exactly here.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            bound = 1.0 / np.sqrt(fan_in)
            with torch.no_grad():
                m.weight.uniform_(-bound, bound, generator=gen)
                if m.bias is not None:
                    m.bias.uniform_(-bound, bound, generator=gen)


class DenseAE(nn.Module):
    """Conventional dense autoencoder, 6-16-4-16-6.

    ReLU on the hidden layers only. The bottleneck layer is deliberately
    LINEAR: a ReLU applied directly to a 4-dimensional latent can drive latent
    units permanently negative and hence permanently zero, which at the larger
    learning rates in the sweep silently removes capacity and makes the
    reconstruction error a worse anomaly score for reasons unrelated to the
    comparison being made. Encoder and decoder hidden layers keep their ReLUs.

    Initialisation draws from a local torch.Generator (see
    `_seeded_linear_init`) rather than seeding the global RNG.
    """

    def __init__(self, d: int = 6, hidden: int = 16, latent: int = 4,
                 seed: int = 0, bottleneck_relu: bool = False,
                 dtype=torch.float64):
        super().__init__()
        # bottleneck_relu exists ONLY so the ablation in run_study can vary it;
        # the default (False) is the architecture used everywhere else.
        mid = [nn.ReLU()] if bottleneck_relu else []
        with _preserve_global_rng():
            self.net = nn.Sequential(
                nn.Linear(d, hidden), nn.ReLU(),
                nn.Linear(hidden, latent), *mid,       # no ReLU on the bottleneck
                nn.Linear(latent, hidden), nn.ReLU(),
                nn.Linear(hidden, d),
            ).to(dtype)
        _seeded_linear_init(self.net, torch.Generator().manual_seed(seed))

    def forward(self, x):
        return self.net(x)


class UntiedAE(nn.Module):
    """Plain untied autoencoder d -> latent -> d with biases, tanh latent.

    Included because the exact-parameter-count match to the QAE is only
    achievable with weight tying (see `matched_ae` and the README's Baselines
    section): an untied 6->latent->6 AE has 2*d*latent + latent + d parameters,
    which for d=6 gives 19, 32, 45, 58, 71, 84 for latent 1..6. The QAE's
    target is 6*(reps+1) for the selected depth -- 48 in the current run, 36 in
    an earlier one -- and neither is in that list. Tying is therefore forced by the matching requirement and is a real
    capacity handicap, so untied models bracketing the QAE's count are reported
    alongside it.
    """

    def __init__(self, d: int = 6, latent: int = 3, seed: int = 0,
                 dtype=torch.float64):
        super().__init__()
        with _preserve_global_rng():
            self.enc = nn.Linear(d, latent).to(dtype)
            self.dec = nn.Linear(latent, d, bias=False).to(dtype)
        self.out_bias = nn.Parameter(torch.zeros(d, dtype=dtype))
        _seeded_linear_init(self, torch.Generator().manual_seed(seed))

    def forward(self, x):
        return self.dec(torch.tanh(self.enc(x))) + self.out_bias


def untied_param_count(d: int, latent: int) -> int:
    """2*d*latent + latent + d -- encoder W+b, decoder W, output bias."""
    return 2 * d * latent + latent + d


@torch.no_grad()
def ae_scores(model, x: np.ndarray, batch_size: int = 16384,
              dtype=torch.float64) -> np.ndarray:
    """Per-event reconstruction MSE = the classical anomaly score."""
    model.eval()
    out = []
    device = next(model.parameters()).device
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i:i + batch_size], dtype=dtype).to(device)
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
