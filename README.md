#  Quantum Autoencoder for Anomaly Detection in HEP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-purple)](https://qiskit.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)

##  Abstract
As High-Luminosity LHC (HL-LHC) data rates increase, classical trigger systems face bottlenecks. This project implements a **Hybrid Quantum-Classical Autoencoder (QAE)** to compress high-dimensional kinematic data ($p_T, \eta, \phi, m$) into a latent quantum state. The goal is to benchmark Quantum Machine Learning (QML) for **anomaly detection** (e.g., identifying rare decay channels like $H \to \mu\mu$ against QCD background).

##  Methodology
### 1. Data Simulation (Monte Carlo Approximation)
To benchmark the architecture without dependency on massive ROOT files, we generate **synthetic kinematic data** mimicking Standard Model background distributions:
- **Transverse Momentum ($p_T$):** Exponential decay (Soft QCD background).
- **Pseudorapidity ($\eta$):** Gaussian distribution centered at $\eta=0$ (Central region).
- **Azimuthal Angle ($\phi$):** Uniform distribution $[-\pi, \pi]$.
- **Invariant Mass ($m$):** Gaussian peak.

### 2. Quantum Architecture (PQC)
We utilize a **Parametrized Quantum Circuit (PQC)** as the encoder:
- **Encoding:** `ZZFeatureMap` (Maps classical data to Hilbert space via entanglement).
- **Ansatz:** `RealAmplitudes` (Variational form with rotation gates).
- **Compression:** The network trains to map normal background events to the ground state $|00\dots0\rangle$. Anomalies fail to compress, yielding high reconstruction error.

##  Results & Performance
- **Convergence:** The cost function (Fidelity Loss) converged within **100 epochs** using the Adam Optimizer.
- **Latency:** Training latency on local simulator benchmarks at ~400s for 700 events.

![Circuit Diagram](assets/circuit_diagram.png)
*Figure 1: The Parametrized Quantum Circuit (Ansatz) used for compression.*

![Loss Curve](assets/loss_curve.png)
*Figure 2: Training convergence minimizing the reconstruction error.*

##  Reproducibility (Docker)
This environment is fully containerized to ensure reproducibility across CERN's infrastructure.

```bash
# Build the container
docker build -t quantum-hep-agent .

# Run the training script
docker run quantum-hep-agent

## Technology Stack
Quantum: Qiskit, Qiskit Machine Learning

Classical ML: PyTorch, Scikit-learn

Infrastructure: Docker, Jupyter

Project developed for the CERN Summer Student Programme application 2026.
