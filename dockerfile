# Reproducible environment for the LHCO quantum-autoencoder study.
#
#   docker build -t qae-lhco .
#   docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/data:/app/data" qae-lhco
#
# The data volume is worth mounting: it caches the 74 MB LHCO feature file
# between runs so Zenodo is only hit once.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY notebooks/ ./notebooks/

CMD ["python", "-m", "src.run_study"]
