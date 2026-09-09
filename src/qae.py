"""Quantum autoencoder: circuit definition, PyTorch module, Qiskit cross-check.

Architecture (Romero, Olson & Aspuru-Guzik, arXiv:1612.02806; applied to HEP by
Ngairangbam, Spannowsky & Takeuchi, Phys. Rev. D 105, 095004, arXiv:2112.04958):

    |x>  --[ ZZFeatureMap ]--[ RealAmplitudes(theta) ]--  { latent  : keep
                                                         { trash   : measure

`n_qubits` qubits carry the encoded event. The ansatz is trained so that the
`n_trash` trash qubits are driven to |0...0>; the remaining `n_latent` qubits
then hold the compressed state. Training minimises

    L(theta) = 1 - P(trash = |0...0>)

averaged over BACKGROUND events only. The same quantity, per event, is the
anomaly score at test time: a background-like event compresses (score -> 0),
an anomalous one does not (score -> 1).

Why no SWAP test / reference register: measuring the trash qubits directly is
equivalent to the SWAP-test fidelity against a fresh |0...0> reference, and
costs n_trash + 1 fewer qubits. This is the standard simplification and is what
Ngairangbam et al. use.

Why PennyLane for training: Qiskit's gradients go through the parameter-shift
rule, which needs 2 circuit evaluations per parameter per step. At 24
parameters and O(10^5) training events that is ~10^6 circuit executions per
epoch -- infeasible on a free Colab CPU. PennyLane's `default.qubit` with
`diff_method="backprop"` differentiates through the statevector simulation
directly, giving the same exact (noiseless, infinite-shot) gradients in
seconds per epoch. `verify_against_qiskit` asserts the two implementations
produce the same statevector, so the Qiskit circuit remains the specification.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn

# --------------------------------------------------------------------------
# Circuit primitives, written to match qiskit.circuit.library exactly
# --------------------------------------------------------------------------


def zz_feature_map(x, wires):
    """ZZFeatureMap(feature_dimension=len(wires), reps=1, entanglement='linear').

    Qiskit emits, per repetition: H on every qubit; P(2*x_i) on qubit i; then
    for each linear pair (i, i+1): CX, P(2*(x_i - pi)*(x_{i+1} - pi)), CX.
    `x` may carry a leading batch dimension.
    """
    wires = list(wires)
    for i, w in enumerate(wires):
        qml.Hadamard(wires=w)
        qml.PhaseShift(2.0 * x[..., i], wires=w)
    for i in range(len(wires) - 1):
        a, b = wires[i], wires[i + 1]
        qml.CNOT(wires=[a, b])
        qml.PhaseShift(2.0 * (x[..., i] - np.pi) * (x[..., i + 1] - np.pi), wires=b)
        qml.CNOT(wires=[a, b])


def ry_feature_map(x, wires):
    """Angle encoding: RY(x_i) on qubit i, giving a real product state.

    Unlike ZZFeatureMap this puts the data in the amplitudes, not the phases.
    See `encoding_analysis.py`: ZZFeatureMap leaves every amplitude with the
    same modulus for every input, so encoded events are nearly mutually
    orthogonal and span the whole Hilbert space -- which caps how well ANY
    ansatz can compress them, close to the random-guess value. Angle encoding
    concentrates the background on a low-dimensional manifold instead.
    """
    for i, w in enumerate(wires):
        qml.RY(x[..., i], wires=w)


FEATURE_MAPS = {"zz": zz_feature_map, "ry": ry_feature_map}


def real_amplitudes(weights, wires, reps):
    """RealAmplitudes(num_qubits=len(wires), reps=reps, entanglement='linear').

    RY layer, then `reps` x (linear CX chain, RY layer).
    Parameter count is len(wires) * (reps + 1).
    """
    wires = list(wires)
    n = len(wires)
    k = 0
    for w in wires:
        qml.RY(weights[k], wires=w)
        k += 1
    for _ in range(reps):
        for i in range(n - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        for w in wires:
            qml.RY(weights[k], wires=w)
            k += 1


def n_params(n_qubits: int, reps: int) -> int:
    return n_qubits * (reps + 1)


def build_qiskit_feature_map(kind: str, n_qubits: int):
    """Qiskit version of the feature map named by `kind`."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import zz_feature_map as qk_zz_feature_map

    if kind == "zz":
        return qk_zz_feature_map(n_qubits, reps=1, entanglement="linear")
    if kind == "ry":
        x = ParameterVector("x", n_qubits)
        qc = QuantumCircuit(n_qubits, name="RYFeatureMap")
        for i in range(n_qubits):
            qc.ry(x[i], i)
        return qc
    raise ValueError(f"unknown feature map {kind!r}; expected one of {list(FEATURE_MAPS)}")


def build_qiskit_circuit(n_qubits: int = 6, reps: int = 3,
                         feature_map: str = "ry"):
    """The same circuit in Qiskit. Used for the figure and the cross-check."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import real_amplitudes as qk_real_amplitudes

    fm = build_qiskit_feature_map(feature_map, n_qubits)
    ans = qk_real_amplitudes(n_qubits, reps=reps, entanglement="linear")
    qc = QuantumCircuit(n_qubits)
    qc.compose(fm, range(n_qubits), inplace=True)
    qc.compose(ans, range(n_qubits), inplace=True)
    return qc, fm, ans


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class QuantumAutoencoder(nn.Module):
    """Trash-qubit quantum autoencoder. Input must already be scaled to [0, pi]."""

    def __init__(self, n_qubits: int = 6, n_trash: int = 2, reps: int = 3,
                 seed: int = 0, init_scale: float = 0.1,
                 feature_map: str = "ry"):
        super().__init__()
        if not 0 < n_trash < n_qubits:
            raise ValueError("n_trash must be strictly between 0 and n_qubits")
        if feature_map not in FEATURE_MAPS:
            raise ValueError(
                f"unknown feature map {feature_map!r}; expected one of "
                f"{list(FEATURE_MAPS)}")
        self.n_qubits, self.n_trash, self.reps = n_qubits, n_trash, reps
        self.feature_map = feature_map
        self.n_latent = n_qubits - n_trash
        self.trash_wires = list(range(self.n_latent, n_qubits))

        # Small initialisation keeps the ansatz near identity at step 0, which
        # trains more stably than a uniform-random start at this circuit size.
        g = torch.Generator().manual_seed(seed)
        self.weights = nn.Parameter(
            init_scale * torch.randn(n_params(n_qubits, reps), generator=g,
                                     dtype=torch.float64))

        dev = qml.device("default.qubit", wires=n_qubits)
        fm = FEATURE_MAPS[feature_map]

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            fm(x, range(n_qubits))
            real_amplitudes(w, range(n_qubits), reps)
            return qml.probs(wires=self.trash_wires)

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Anomaly score in [0, 1]: 1 - P(trash qubits measured all-zero)."""
        probs = self._circuit(x, self.weights)
        return 1.0 - probs[..., 0]

    @torch.no_grad()
    def score(self, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        out = []
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[i:i + batch_size], dtype=torch.float64)
            out.append(self.forward(xb).cpu().numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------
# Cross-check: PennyLane implementation == Qiskit specification
# --------------------------------------------------------------------------


def verify_against_qiskit(n_qubits: int = 6, reps: int = 3, n_trials: int = 5,
                          tol: float = 1e-10, seed: int = 0,
                          feature_map: str = "ry") -> float:
    """Assert the PennyLane and Qiskit circuits give identical statevectors.

    Qiskit's Statevector is little-endian (qubit 0 is the least significant bit)
    while PennyLane's is big-endian (wire 0 is the most significant), so the
    Qiskit state is compared after `reverse_qargs()`.

    Returns the largest absolute amplitude difference observed.
    """
    from qiskit.quantum_info import Statevector

    dev = qml.device("default.qubit", wires=n_qubits)
    pl_fm = FEATURE_MAPS[feature_map]

    @qml.qnode(dev)
    def pl_state(x, w):
        pl_fm(x, range(n_qubits))
        real_amplitudes(w, range(n_qubits), reps)
        return qml.state()

    qc, fm, ans = build_qiskit_circuit(n_qubits, reps, feature_map)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        x = rng.uniform(0.0, np.pi, size=n_qubits)
        w = rng.uniform(-np.pi, np.pi, size=n_params(n_qubits, reps))
        bound = qc.assign_parameters(
            {p: v for p, v in zip(list(fm.parameters), x)}
            | {p: v for p, v in zip(list(ans.parameters), w)})
        qk = Statevector(bound).reverse_qargs().data
        pl = np.asarray(pl_state(x, w))
        worst = max(worst, float(np.max(np.abs(qk - pl))))

    if worst > tol:
        raise AssertionError(
            f"PennyLane and Qiskit circuits disagree: max |dAmplitude| = {worst:.3e} "
            f"> {tol:.0e}. The PennyLane implementation is not a faithful copy "
            "of the Qiskit specification.")
    return worst
