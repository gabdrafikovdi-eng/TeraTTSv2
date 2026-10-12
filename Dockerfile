# syntax=docker/dockerfile:1
# Образ для бенчмарка задержки TeraTTSv2 и веб-демо.
# Сборка:  docker build -t terattsv2:latest .
# Запуск:  docker run --rm -v hf-cache:/models -v "$PWD/out:/app/out" terattsv2:latest bench.py --runs 3
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

WORKDIR /app

# libgomp1 нужен onnxruntime и torch, ca-certificates — для загрузки модели по HTTPS
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# torch ставим из CPU-индекса: с PyPI подтягиваются CUDA-колёса (+2–3 ГБ, в образе бесполезны).
# Для наших задач достаточно CPU-сборки (модель работает через onnxruntime).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py cli.py bench.py ./

# Непривилегированный пользователь: uid 1000 совпадает с типичным uid на VPS,
# поэтому bind-mount кэша модели с хоста не создаёт проблем с правами.
ARG UID=1000
RUN useradd --create-home --uid ${UID} runner \
    && mkdir -p /models /app/out \
    && chown -R runner:runner /models /app
USER runner

# Опционально «запечь» модель в образ: --build-arg PRELOAD=1 (+~1.2 ГБ к размеру).
# По умолчанию модель скачивается один раз в volume hf-cache.
ARG PRELOAD=0
RUN if [ "$PRELOAD" = "1" ]; then \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('TeraSpace/TeraTTSv2')"; \
    fi

ENTRYPOINT ["python"]
CMD ["bench.py", "--runs", "3", "--json", "out/bench.json"]
