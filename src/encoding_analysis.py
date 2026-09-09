"""Is the encoded data compressible at all? An ansatz-independent bound.

A trash-qubit autoencoder applies one fixed unitary U and then asks for the
trash register to be |0...0>. Writing Pi_S for the projector onto the
2^n_latent-dimensional subspace that U maps the kept register onto,

    E_x[ P(trash = |0..0>) ] = E_x[ <psi(x)| U^dag Pi U |psi(x)> ]
                             = tr( rho  U^dag Pi U ),     rho = E_x |psi(x)><psi(x)|

Maximising over U is maximising tr(rho Pi_S) over all subspaces S of dimension
k = 2^n_latent, and by Ky Fan's theorem the maximum is the sum of the k largest
eigenvalues of rho:

    max_U E_x[P(trash = |0..0>)] = sum_{i=1..k} lambda_i(rho)

This is an UPPER BOUND on what any ansatz can reach with any depth and any
amount of training, and it depends only on the feature map and the data.

Reference value: for a maximally mixed rho on n qubits the bound is k/2^n,
which is exactly the random-guess probability. An encoding that puts rho near
maximally mixed is therefore incompressible in principle -- the autoencoder
cannot work, and no amount of tuning changes that.

Why this matters here: ZZFeatureMap starts with a Hadamard on every qubit and
then applies only phase gates, so every amplitude has modulus 2^(-n/2) for
every input. Distinct events differ only in phase and are close to mutually
orthogonal, so they span the whole space and rho is close to maximally mixed.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


def encoded_density_matrix(feature_map, x: np.ndarray, n_qubits: int,
                           batch_size: int = 4096) -> np.ndarray:
    """rho = E_x |psi(x)><psi(x)| for the given feature map, as a 2^n x 2^n array."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state(xb):
        feature_map(xb, range(n_qubits))
        return qml.state()

    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    total = 0
    for i in range(0, len(x), batch_size):
        psi = np.asarray(state(x[i:i + batch_size]))
        if psi.ndim == 1:
            psi = psi[None, :]
        rho += psi.conj().T @ psi          # sum_b |psi_b><psi_b|
        total += len(psi)
    return rho / total


def compressibility(feature_map, x: np.ndarray, n_qubits: int,
                    n_trash: int) -> dict:
    """Ansatz-independent ceiling on P(trash = |0...0>) for this encoding."""
    k = 2 ** (n_qubits - n_trash)
    dim = 2 ** n_qubits
    rho = encoded_density_matrix(feature_map, x, n_qubits)

    ev = np.linalg.eigvalsh(rho)[::-1].real       # descending
    ev = np.clip(ev, 0.0, None)
    bound = float(ev[:k].sum())
    random_ref = k / dim

    # participation ratio: effective number of dimensions rho occupies
    eff_dim = float(1.0 / np.sum(ev ** 2)) if np.sum(ev ** 2) > 0 else float("nan")

    # Absolute headroom is hard to read on its own: express it as a fraction of
    # the headroom a perfectly compressible encoding would have (bound = 1).
    usable = (bound - random_ref) / (1.0 - random_ref)

    return {
        "max_p_trash_zero": bound,
        "random_reference": random_ref,
        "headroom": bound - random_ref,
        "headroom_fraction": usable,
        "effective_dimension": eff_dim,
        "full_dimension": dim,
        "latent_dimension": k,
        "top_eigenvalues": ev[:8].tolist(),
        # A quarter of the available headroom is a generous bar: below it the
        # encoding leaves the autoencoder almost no room to separate anything.
        "compressible": usable > 0.25,
    }


def report(results: dict, name: str) -> str:
    r = results
    verdict = ("COMPRESSIBLE"
               if r["compressible"] else
               "NOT USEFULLY COMPRESSIBLE (ceiling is close to random guessing)")
    return (
        f"{name}\n"
        f"  max P(trash=|0..0>) over all ansaetze : {r['max_p_trash_zero']:.4f}\n"
        f"  random-guess reference               : {r['random_reference']:.4f}\n"
        f"  headroom                             : {r['headroom']:+.4f}"
        f"  ({r['headroom_fraction']:.0%} of what a fully\n"
        f"                                          compressible encoding would give)\n"
        f"  effective dimension of rho           : {r['effective_dimension']:.1f}"
        f" of {r['full_dimension']}\n"
        f"  verdict                              : {verdict}"
    )
