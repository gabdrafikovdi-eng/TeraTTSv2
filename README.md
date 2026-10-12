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

# 4. Папка для отчётов (права выставит обёртка ./bench.sh — в том числе под root)
mkdir -p out
```

Обновление до последней версии с GitHub:

```bash
cd /opt/TeraTTSv2 && git pull && docker compose build
```

### 4.2. Сборка и запуск

```bash
docker build -t terattsv2:latest .

# Самый простой путь — обёртка: сама подставит uid/gid и создаст ./out
./bench.sh

# Вручную:
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

Права на `./out` (актуально, если вы заходите на VPS под root): внутри контейнера процесс
работает под uid 1000 (так кэш модели в volume всегда принадлежит одному и тому же uid).
Обёртка `./bench.sh` сама разбирается с правами, поэтому проще всего запускать через неё.
Если запускаете вручную:

| Кто вы на VPS | Что делать |
| --- | --- |
| **root (uid 0)** | `./bench.sh` — обёртка сама сделает `chown 1000:1000 out` |
| обычный пользователь (uid 1000) | `./bench.sh` — ничего дополнительно делать не нужно |
| uid, отличный и от 0, и от 1000 | `./bench.sh` — обёртка выдаст папке `out` права на запись; контейнер всё равно работает под uid 1000 (так кэш модели в volume всегда доступен) |

Почему uid именно 1000: named volume с кэшем модели Docker инициализирует правами из
образа (`chown 1000:1000 /models` в Dockerfile). Если запускать контейнер под другим uid,
он просто не увидит уже скачанную модель — поэтому обёртка держит uid 1000, а доступ к
`./out` выдаёт через chown/chmod. Пересобирать образ с `--build-arg UID=0` бессмысленно:
uid 0 занят в базовом образе.

В Docker Desktop (macOS/Windows) доступ к папке отображается на пользователя хоста,
поэтому настраивать ничего не нужно — но проект должен лежать в «расшаренном» пути
(`$HOME`), а не в `/tmp`. Если JSON всё же не записался, скрипт не падает: все цифры
уже напечатаны в консоли.

### 4.3. Команды для разных вопросов

```bash
# 1. TTFC на диалоге + разбор бюджета задержки (бюджет 400 мс, сеть/телефония 80 мс)
docker compose run --rm bench bench.py --runs 5 --warmup 2 --budget-ms 400 --network-ms 80

# 2. Как chunk_frames влияет на TTFC и паузы (меньше чанк → быстрее старт воспроизведения)
docker compose run --rm bench bench.py --scenario none --sweep-chunk-frames 16,8,4,2

# 3. Сколько потоков onnxruntime держать на этом VPS
docker compose run --rm bench bench.py --scenario none --sweep-threads 1,2,4,8

# 3б. Сколько TTFC «стоит» нейросеть ударений RUAccent (full / dictionary / off)
docker compose run --rm bench bench.py --scenario none --sweep-stress-modes

# 4. Сколько одновременных звонков тянет сервер (каждая копия модели ~2 ГБ RAM)
docker compose run --rm --memory 12g bench bench.py --scenario none --concurrency 3 --runs 2

# 5. Общая латентность (RTF/p50/p90/p99) + round-trip к веб-демо
docker compose run --rm bench bench.py --scenario latency --stream --preset all
docker compose run --rm --network host bench bench.py --scenario latency --url http://127.0.0.1:7860
```

### 4.4. Как читать цифры

| Метрика | Что значит | Ориентир для телефонии |
| --- | --- | --- |
| TTFC p50 | задержка до первого звука | ≤ 300–500 мс (сеть/телефония добавляет 50–150 мс) |
| пауза max | время генерации следующего чанка | меньше длительности пакета (16 кадров ≈ 1.11 с) |
| запас min | свободный буфер при воспроизведении (отсчёт с прихода первого чанка) | ≥ 0 → речь без провалов; проверяйте строку «Худший запас буфера» в итоге |
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

Ещё один практический момент: по умолчанию `--threads` = половина ядер, то есть на
2-ядерном VPS это 1 поток, и TTFC заметно хуже (в нашем тесте 1107 мс против 569 мс
при двух потоках). Задайте число потоков явно — вровень с количеством vCPU:
`TERATTS_THREADS=2 ./bench.sh` — и подберите оптимум через
`./bench.sh bench.py --scenario none --sweep-threads 1,2,4`.

### 4.5. Если TTFC выше бюджета

TTFC = препроцессинг (нормализация, ударения, энкодер текста) + диффузионная фаза
(8 шагов у distilled) + первый чанк вокодера, и он растёт с длиной реплики.
Рычаги, от самого дешёвого:

1. **Потоки ONNX** — по умолчанию берётся половина ядер, на 2 vCPU это 1 поток
   (в нашем тесте TTFC 1107 мс против 569 мс при двух). Задайте явно и подберите:
   `TERATTS_THREADS=4 ./bench.sh bench.py --scenario none --sweep-threads 2,3,4,6`.
2. **`duration-scale` < 1** — короче латент, меньше работы вокодеру: на M2 `0.8` дал TTFC
   431 мс против 506 мс при `1.0` (речь при этом примерно на 20% быстрее).
3. **Режим ударений** — `--sweep-stress-modes`: на M2 вышло `full` 563 мс, `dictionary` 548 мс,
   `off` 549 мс, то есть RUAccent — небольшая часть бюджета, и экономия не бесплатна
   (теряется автоматическая расстановка ударений).
4. **Дробить длинные реплики.** TTFC зависит от длины текста, поэтому первая фраза должна быть
   короткой («Здравствуйте!», «Откуда вас забрать?»), а остальное синтезируется, пока звучит
   предыдущая: внутри реплики запас буфера обычно >1 с, а паузы между чанками 50–90 мс.
5. **Свободный CPU.** Если рядом работают БД/приложение диспетчера, TTFC деградирует:
   проверьте `uptime` и `docker stats --no-stream`, сравните с
   `./bench.sh bench.py --scenario none --concurrency 2`.

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
| `Permission denied` при записи в `./out` | Папка принадлежит root (создана через `sudo`/Docker): `sudo chown 1000:1000 out` или запускайте `./bench.sh` |
| Модель уже скачана, но контейнер её «не видит» | Кэш в volume принадлежит другому uid (запускали то под root, то под пользователем): держитесь одного пользователя или удалите volume — `docker volume rm terattsv2_hf-cache` |
| Файлы bind-mount не видны в контейнере (macOS) | Путь вне «расшаренных» каталогов Docker Desktop (например `/tmp`) — перенесите проект в `$HOME` |
| Сборка образа качает CUDA-колёса | Уберите `--index-url https://download.pytorch.org/whl/cpu` только если нужен CUDA (по умолчанию он уже стоит) |
| Ошибка о тегах | Текст не обёрнут в `<ru>`/`<en>` или теги не сбалансированы |

