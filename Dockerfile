FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl jq wget git \
    build-essential \
    libgomp1 \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /workspace

COPY requirements.txt .

# TORCH_INDEX передаётся через --build-arg:
#   CPU (default): docker compose up --build
#   GPU:           TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --extra-index-url "${TORCH_INDEX}" -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN jupyter lab --generate-config && \
    printf '%s\n' \
        "c.ServerApp.ip = '0.0.0.0'" \
        "c.ServerApp.port = 8888" \
        "c.ServerApp.open_browser = False" \
        "c.ServerApp.token = 'masterclass'" \
        "c.ServerApp.password = ''" \
        "c.ServerApp.allow_origin = '*'" \
        "c.ServerApp.root_dir = '/workspace'" \
    >> /root/.jupyter/jupyter_lab_config.py

EXPOSE 8888
ENTRYPOINT ["/entrypoint.sh"]
