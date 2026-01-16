
FROM python:3.9-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

RUN pip install nbconvert
CMD ["jupyter", "nbconvert", "--to", "python", "--execute", "Quantum_Autoencoder_LHC_Analysis.ipynb"]
