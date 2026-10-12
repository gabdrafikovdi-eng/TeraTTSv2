# TeraTTSv2 — локальный запуск

Локальный запуск модели [TeraSpace/TeraTTSv2](https://huggingface.co/TeraSpace/TeraTTSv2)
(ONNX Runtime TTS, русский/английский): CLI-синтез в WAV и веб-демо на Gradio.

Официальное демо: <https://huggingface.co/spaces/sasfox4/TeraTTSv2-Demo> — наш `app.py`
это его локальная копия без зависимости `spaces` (ZeroGPU нужен только на HF Spaces).

## 1. Установка зависимостей

Проект управляется через [`uv`](https://docs.astral.sh/uv/) (в репозитории уже есть `.venv`):

```bash
uv pip install --python .venv/bin/python -r requirements.txt
# или полностью из pyproject:
uv sync --group demo
```

Альтернатива через обычный pip (если venv с pip):

```bash
pip install -r requirements.txt
```

> `sounddevice` (только для `--play`) в requirements закомментирован — поставьте отдельно:
> `uv pip install --python .venv/bin/python sounddevice`.

## 2. Быстрый локальный тест без веба (самый надёжный способ)

```bash
.venv/bin/python cli.py --text "<ru>Привет от TeraTTS.</ru>" --voice ru_f1 --out out/ru.wav
.venv/bin/python cli.py --text "<en>I have 42 apples.</en>" --voice eng_f3 --duration-scale 1.2
.venv/bin/python cli.py --text-file текст.txt --voice ru_m5 --play   # нужен sounddevice
```

Полезные флаги: `--diffusion-model distilled|teacher`, `--stress-mode full|dictionary|off`,
`--duration-scale` (>1 — медленнее), `--threads N`.

Первый запуск скачает модель (~1.17 ГБ) в кэш Hugging Face (`~/.cache/huggingface`).
Скрипт печатает нормализованный текст, время синтеза и RTF.

## 3. Веб-демо (Gradio) локально

```bash
.venv/bin/python app.py
# http://127.0.0.1:7860 — браузер откроется автоматически
```

Запуск на VPS (доступ снаружи по IP):

```bash
.venv/bin/python app.py --host 0.0.0.0 --port 7860
# не забудьте открыть порт: ufw allow 7860/tcp

# если снаружи порт закрыт — публичная ссылка через туннель gradio.live:
.venv/bin/python app.py --share
```

Пример systemd-юнита для VPS:

```ini
[Unit]
Description=TeraTTSv2 Gradio demo
After=network-online.target

[Service]
WorkingDirectory=/opt/TeraTTSv2
ExecStart=/opt/TeraTTSv2/.venv/bin/python app.py --host 0.0.0.0 --port 7860
Restart=always

[Install]
WantedBy=multi-user.target
```

## 4. Развёртывание и замер задержки на VPS (Docker)

Репозиторий публичный: <https://github.com/gabdrafikovdi-eng/TeraTTSv2.git> — на сервере
достаточно `git clone`, ничего копировать вручную не нужно.

### 4.1. Клонирование и установка Docker

```bash
# 1. Docker (Ubuntu/Debian, один раз)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # и перелогиниться, чтобы работать без sudo

# 2. Отдельная папка под проект
sudo mkdir -p /opt && sudo chown "$USER" /opt
cd /opt

# 3. Клонировать публичный репозиторий
git clone https://github.com/gabdrafikovdi-eng/TeraTTSv2.git
cd TeraTTSv2

# 4. Папка для отчётов: контейнер работает под uid 1000, а bind-mount на Linux
#    сохраняет владельца — на типичном VPS первый пользователь и есть uid 1000
mkdir -p out
id -u                      # если вывод не 1000 → sudo chown 1000:1000 out
```

Обновление до последней версии с GitHub:

```bash
cd /opt/TeraTTSv2 && git pull && docker compose build
```

### 4.2. Сборка и запуск

```bash
docker build -t terattsv2:latest .

# диалог диспетчера: 3 замера на реплику, отчёт в ./out/bench.json
docker run --rm -v hf-cache:/models -v "$PWD/out:/app/out" \
  terattsv2:latest bench.py --runs 3 --warmup 1 --json out/bench.json

# то же через compose
docker compose run --rm bench
```

Первый запуск скачивает модель (~1.2 ГБ) в volume `hf-cache`, дальше — мгновенно.
Повторные запуски можно делать без сети: добавьте `--offline` (`local_files_only`).
Размер образа без модели ~1.9 ГБ; можно «запечь» модель внутрь:
`docker build --build-arg PRELOAD=1 -t terattsv2:latest .`

Права на `./out`: контейнер работает под uid 1000, поэтому каталог должен принадлежать
uid 1000 (на типичном VPS это и есть ваш пользователь). Если uid другой — либо
`sudo chown 1000:1000 out`, либо пересоберите образ под себя:
`docker build --build-arg UID=$(id -u) -t terattsv2:latest .`.
В Docker Desktop (macOS/Windows) монтирование мапится на root — там запускайте
`TERATTS_UID=0 TERATTS_GID=0 docker compose run --rm bench`. Если JSON всё же не
записался, скрипт не падает: все цифры уже напечатаны в консоли.

### 4.3. Команды для разных вопросов

```bash
# 1. TTFC на диалоге + разбор бюджета задержки (бюджет 400 мс, сеть/телефония 80 мс)
docker compose run --rm bench bench.py --runs 5 --warmup 2 --budget-ms 400 --network-ms 80

# 2. Как chunk_frames влияет на TTFC и паузы (меньше чанк → быстрее старт воспроизведения)
docker compose run --rm bench bench.py --scenario none --sweep-chunk-frames 16,8,4,2

# 3. Сколько потоков onnxruntime держать на этом VPS
docker compose run --rm bench bench.py --scenario none --sweep-threads 1,2,4,8

# 4. Сколько одновременных звонков тянет сервер (каждая копия модели ~2 ГБ RAM)
docker compose run --rm --memory 12g bench bench.py --scenario none --concurrency 3 --runs 2

# 5. Общая латентность (RTF/p50/p90/p99) + round-trip к веб-демо
docker compose run --rm bench bench.py --scenario latency --stream --preset all
docker compose run --rm --network host bench bench.py --scenario latency --url http://127.0.0.1:7860
```

### 4.3. Как читать цифры

| Метрика | Что значит | Ориентир для телефонии |
| --- | --- | --- |
| TTFC p50 | задержка до первого звука | ≤ 300–500 мс (сеть/телефония добавляет 50–150 мс) |
| пауза max | время генерации следующего чанка | меньше длительности пакета (16 кадров ≈ 1.11 с) |
| запас min | накопленное аудио минус ожидание | ≥ 0 → речь без провалов |
| RTF | генерация/длительность аудио | < 1 обязательно, ≤ 0.2 комфортно |
| xRT | секунд речи на секунду CPU | напрямую ≈ число каналов при непрерывной речи |
| RSS | память процесса | 1.4–2.6 ГБ на копию модели (мы видели 1.36 ГБ в macOS-venv и 2.58 ГБ в Linux-контейнере) |

Важно: TTFC растёт с длиной реплики (диффузионная фаза считается до первого чанка),
поэтому для бота выгодны короткие фразы. `chunk_frames` влияет на паузы и на время
проигрывания, но **не** уменьшает TTFC — его определяет длина текста.
Абсолютные значения зависят от CPU: гоняйте бенчмарк на том же типе ядра, что и на VPS,
и с теми же `--cpus/--memory`, которые будут в продакшене (в Docker Desktop лимиты VM
особенно заметны: 2 vCPU дали TTFC 569–920 мс против 414–828 мс на 8-ядерном M2).

Первый прогон в контейнере включает скачивание модели, поэтому «загрузка» покажет
~100 с — на повторных запусках это 5–10 с.

## 5. API модели (напрямую, без наших скриптов)

```python
from transformers import AutoModel

tts = AutoModel.from_pretrained(
    "TeraSpace/TeraTTSv2",
    trust_remote_code=True,
    provider="CPUExecutionProvider",
    threads=6,
)
waveform = tts.generate_speech("<ru>Привет от TeraTTS.</ru>", voice="ru_f1", duration_scale=1)
tts.save_wav("teratts.wav", waveform)      # моно float32, 44 100 Гц
print(tts.normalize_text("<ru>У меня 21 яблоко.</ru>"))
```

Потоковая генерация: `for chunk in tts.generate_speech_stream(text, voice="eng_f3"): ...`

## 6. Важные нюансы модели

- **Теги обязательны**: оборачивайте текст в `<ru>…</ru>` или `<en>…</en>` — иначе ошибка.
- **Ударения в русском автоматические** (RUAccent). Ручные `+`-маркеры имеют приоритет.
  Режим `dictionary` — без нейросети (меньше памяти), `off` — отключить обработку.
- **Голоса**: рекомендуются `ru_f1` и `ru_m5`. Полный список — в `VOICES` в `app.py`.
- **Английский голос на русском тексте**: начинайте с `duration_scale ≈ 0.8`.
- **`trust_remote_code=True`**: код исполняется с Hub. Для продакшена зафиксируйте ревизию:
  `TERATTS_REVISION=<commit-sha> .venv/bin/python app.py`.
- **Переменные окружения**: `TERATTS_MODEL_ID`, `TERATTS_REVISION`, `TERATTS_THREADS`,
  `TERATTS_HOST`, `TERATTS_PORT`, `TERATTS_SHARE=1`, `HF_TOKEN` (если упёрлись в лимит загрузок).

## 7. Траблшутинг

| Симптом | Причина / решение |
| --- | --- |
| Медленно качает модель / обрывы | Проверьте `HF_HOME` (кэш) и повторите — докачка продолжится; при блокировке Hub нужен прокси/VPN |
| `No module named 'onnxruntime'` | Не выполнен шаг 1 в том же venv |
| Порт 7860 занят | `--port 7861` |
| Много RAM | `--stress-mode dictionary`, `TERATTS_THREADS=4`, не открывайте `teacher`-сэмплер |
| Долгая загрузка/низкий xRT в Docker на Mac/Windows | Лимиты Docker VM (`--cpus/--memory`), а не VPS: замеряйте на самом сервере |
| `Permission denied` при bind-mount кэша модели с хоста | Используйте named volume `hf-cache` или добавьте `user: "$(id -u):$(id -g)"` в compose |
| Сборка образа качает CUDA-колёса | Уберите `--index-url https://download.pytorch.org/whl/cpu` только если нужен CUDA (по умолчанию он уже стоит) |
| Ошибка о тегах | Текст не обёрнут в `<ru>`/`<en>` или теги не сбалансированы |

