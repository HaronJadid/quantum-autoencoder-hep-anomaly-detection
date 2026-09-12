"""Is this workload worth moving to a GPU? Paste into one Colab cell and run.

Answers one question and nothing else: how long does ONE training epoch of the
study's actual circuit take on a GPU versus this machine's CPU. It builds the
same circuit the study trains -- 6 qubits, RY angle encoding, a 7-repetition
RealAmplitudes-style ansatz, batch 4096 over 100k synthetic events -- but with
random inputs, so it needs neither the repo nor the LHCO files.

Reference numbers measured on the laptop this study currently runs on
(Ryzen 5 5600H, DDR4-3200, torch 2.14 CPU):

    one process                     several processes, 1 thread each
    1 thread   23.6 s/epoch         4 procs   63.4 s/epoch each
    2 threads  15.3 s/epoch         6 procs  101.8 s/epoch each
    3 threads  12.6 s/epoch
    6 threads  12.2 s/epoch

As aggregate throughput that is 0.082 epoch/s for one 6-thread process
against 0.063 (4 procs) and 0.059 (6 procs): running seeds in parallel on
this machine is about 28% SLOWER than running them one after another. Thread
scaling saturating at 3 says the same thing -- the cores are waiting on
memory, not computing -- which is also the reason to expect a GPU to help.

The full study is 15 seeds x 2 quantum models x up to 600 epochs with early
stopping, and took 6.5 h at 12.2 s/epoch. Divide 6.5 h by whatever speed-up
this prints to estimate the GPU run.

Verdict thresholds, to decide rather than admire the number:
    < 2x   not worth the move; run locally overnight.
    2-4x   worth it (6.5 h -> under 2 h).
    > 4x   clearly worth it.

    !pip -q install pennylane
    # then paste the rest
"""

import time

import pennylane as qml
import torch

N_QUBITS, REPS, BATCH, N_EVENTS = 6, 7, 4096, 100_000


def make(device):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    # default.qubit builds its initial state with qml.math.asarray(like="torch"),
    # which puts it on the CPU no matter where the parameters live; the first
    # parametrised gate then tries to combine a CPU state with CUDA angles and
    # raises "Expected all tensors to be on the same device". Preparing
    # |0...0> explicitly as a tensor on the target device pulls the whole
    # simulation onto it. Mathematically this is a no-op -- it is the state
    # the device would have started from -- and on CPU it changes nothing.
    zero = torch.zeros(2 ** N_QUBITS, dtype=torch.complex128, device=device)
    zero[0] = 1.0

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, w):
        qml.StatePrep(zero, wires=range(N_QUBITS))
        for i in range(N_QUBITS):
            qml.RY(x[..., i], wires=i)
        k = 0
        for _ in range(REPS):
            for i in range(N_QUBITS):
                qml.RY(w[k], wires=i)
                k += 1
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
        for i in range(N_QUBITS):
            qml.RY(w[k], wires=i)
            k += 1
        return qml.probs(wires=[4, 5])

    g = torch.Generator().manual_seed(0)
    n_w = N_QUBITS * (REPS + 1)
    w = (0.1 * torch.randn(n_w, generator=g, dtype=torch.float64)
         ).to(device).requires_grad_(True)
    return circuit, w


def bench(device, epochs=1):
    circuit, w = make(device)
    x = torch.rand(N_EVENTS, N_QUBITS, dtype=torch.float64,
                   generator=torch.Generator().manual_seed(1)) * 3.14159
    x = x.to(device)
    opt = torch.optim.Adam([w], lr=0.05)
    if device == "cuda":                      # warm up kernels and allocator
        for i in range(0, 2 * BATCH, BATCH):
            loss = (1.0 - circuit(x[i:i + BATCH], w)[..., 0]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(epochs):
        for i in range(0, len(x), BATCH):
            loss = (1.0 - circuit(x[i:i + BATCH], w)[..., 0]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / epochs


print("torch", torch.__version__, "| pennylane", qml.version())
print("cuda:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NOT AVAILABLE -- set Runtime > Change runtime type > T4 GPU")

cpu = bench("cpu")
print(f"\nCPU  {cpu:7.2f} s/epoch   (this Colab CPU, however many threads it has)")

if torch.cuda.is_available():
    try:
        gpu = bench("cuda")
    except RuntimeError as exc:
        print(f"\nGPU RUN FAILED: {exc}")
        print("\nIf this is still a device mismatch, the simulator is not "
              "following the input device and the GPU route is closed with "
              "this PennyLane version. Report the message and stop here.")
        raise SystemExit(0)
    print(f"GPU  {gpu:7.2f} s/epoch")
    print(f"\nGPU is {cpu / gpu:.1f}x this Colab's CPU, "
          f"and {12.2 / gpu:.1f}x the laptop's 12.2 s/epoch.")
    hours = 6.5 * gpu / 12.2
    print(f"Estimated full 15-seed study on this GPU: {hours:.1f} h "
          f"(it is 6.5 h on the laptop).")
    print("\n  <2x over the laptop: not worth moving."
          "\n  2-4x: worth it."
          "\n  >4x: clearly worth it.")
