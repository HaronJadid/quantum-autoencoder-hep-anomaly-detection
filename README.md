# Quantum Autoencoder for Anomaly Detection in High-Energy Physics

## Abstract
This project implements a **Hybrid Quantum-Classical Autoencoder** to compress and analyze kinematic data from the **ATLAS experiment** at the LHC. By leveraging a parametrized quantum circuit (PQC) with 4 qubits, the model learns a latent representation of particle jets ($p_T, \eta, \phi, m$) to distinguish between standard model background and potential anomalous signals.

## Methodology
* **Data Source:** ATLAS/CMS Open Data (Kinematic features normalized to $[0, 1]$).
* **Architecture:**
    * **Feature Map:** `ZZFeatureMap` (Linear Entanglement) for quantum embedding.
    * **Ansatz:** `RealAmplitudes` variational form (depth=1).
    * **Optimization:** Hybrid loop using **PyTorch** (Adam Optimizer) and **Qiskit** (SamplerQNN).
* **Performance:**
    * Demonstrated convergence of the cost function (MSE) over 15 epochs.
    * Benchmarked training latency on local simulator (~400s).

## Technology Stack
* **Quantum:** Qiskit, Qiskit Machine Learning
* **Classical ML:** PyTorch, Scikit-learn
* **Data:** Pandas, NumPy

## Results
<img width="931" height="461" alt="image" src="https://github.com/user-attachments/assets/da8c4353-70c2-4fb9-86d6-9204def61e59" />


*Figure 1: Training convergence showing the minimization of reconstruction error.*