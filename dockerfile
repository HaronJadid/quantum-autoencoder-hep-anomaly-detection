# Pinned environment for the LHCO quantum-autoencoder study.
#
# NOT a reproducible environment, and the distinction is the point.
#
# Four of the nine models reproduce exactly. Across two machines that differed
# in CPU vendor, Python minor version, torch minor version and compute device
# (laptop CPU vs Colab T4), these gave AUCs identical to all four decimal
# places the study reports, on every one of the 13 comparable seeds:
#
#     qae_ry      quantum autoencoder, RY encoding
#     qae_zz      quantum autoencoder, ZZFeatureMap
#     pca         PCA, 4 components          (no gradient training at all)
#     mj1_only    cut on m_J1                (no training at all)
#
# Five do not reproduce, and no container can make them:
#
#     ae_matched, ae_untied32, ae_untied45, ae_untied58, ae_dense
#
# Their optimisation is chaotic at the learning rates the selection procedure
# chose. Perturbing the training data by one part in 1e15 -- the size of a
# rounding difference between two machines -- and retraining from the same
# seed moves ae_matched's AUC by up to 0.169. Measured against an independent
# CPU run of the identical configuration, their per-seed AUCs moved by up to
# 0.157, which for ae_matched and the three untied AEs is LARGER than their
# across-seed spread. Run `docker run --rm qae-lhco python
# tools/diagnose_training_chaos.py` to see it happen.
#
# So this image pins what can be pinned -- library versions, and the thread
# count, which changes torch's reduction order and therefore its last bits --
# and is honest that the classical baselines will still move. It is a fixed
# starting point, not a guarantee of identical output.
#
# The versions installed here are the development environment, NOT the one
# that produced results/metrics.json; that environment was only partially
# recorded and cannot be reconstructed. See requirements.txt for exactly what
# is known, what is not, and where each version came from.
#
#   docker build -t qae-lhco .
#   docker run --rm qae-lhco                                   # usage
#   docker run --rm -v "$(pwd)/results:/app/results" \
#                   -v "$(pwd)/data:/app/data" qae-lhco smoke  # ~2 min check
#   docker run --rm -v "$(pwd)/results:/app/results" \
#                   -v "$(pwd)/data:/app/data" qae-lhco verify # 3-seed diff
#
# Mounting data/ caches the 74 MB LHCO feature file so Zenodo is hit once.
# Mounting results/ is what lets a run's output survive the container.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Behind a TLS-intercepting proxy or an antivirus that MITMs HTTPS (Avast,
# Zscaler, corporate middleboxes), this layer fails with CERTIFICATE_VERIFY_
# FAILED: the interceptor's root CA is trusted by the host but not by this
# image. Do NOT add --trusted-host here -- that disables verification for
# everyone. Either build with the interceptor off, or add its root CA to the
# image yourself and point pip at it:
#     COPY corp-root.pem /usr/local/share/ca-certificates/corp.crt
#     ENV PIP_CERT=/usr/local/share/ca-certificates/corp.crt
# torch first, from PyTorch's CPU wheel index. Installing it from PyPI
# instead drags in the CUDA runtime -- 553 MB of cuDNN alone, several GB in
# total -- into an image that never touches a GPU, and yields torch 2.14.0+cu*
# rather than the 2.14.0+cpu the pins were resolved against. Everything else
# comes from PyPI as normal; the second command sees torch already satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install --no-cache-dir -r requirements.txt

# OpenMP reads this before torch is imported, so it has to be an image-level
# environment variable; tools/entrypoint.py additionally pins torch's own
# intra-op and inter-op pools, which this variable does not always reach.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

COPY src/ ./src/
COPY tools/ ./tools/
COPY notebooks/ ./notebooks/
# The frozen snapshot of the reported run, and the selection it was trained
# under. Without these `verify` has nothing to compare against and would have
# to re-derive the selection, which takes ~20 minutes.
COPY results/legacy-v1/reference/ ./results/legacy-v1/reference/

ENTRYPOINT ["python", "tools/entrypoint.py"]
# Deliberately not `python -m src.run_study`: that silently started a
# multi-hour run, over the mounted results/, for anyone who typed
# `docker run qae-lhco` to see what the image did. The default now explains
# itself and exits.
CMD []
