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

## 4. API модели (напрямую, без наших скриптов)

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

## 5. Важные нюансы модели

- **Теги обязательны**: оборачивайте текст в `<ru>…</ru>` или `<en>…</en>` — иначе ошибка.
- **Ударения в русском автоматические** (RUAccent). Ручные `+`-маркеры имеют приоритет.
  Режим `dictionary` — без нейросети (меньше памяти), `off` — отключить обработку.
- **Голоса**: рекомендуются `ru_f1` и `ru_m5`. Полный список — в `VOICES` в `app.py`.
- **Английский голос на русском тексте**: начинайте с `duration_scale ≈ 0.8`.
- **`trust_remote_code=True`**: код исполняется с Hub. Для продакшена зафиксируйте ревизию:
  `TERATTS_REVISION=<commit-sha> .venv/bin/python app.py`.
- **Переменные окружения**: `TERATTS_MODEL_ID`, `TERATTS_REVISION`, `TERATTS_THREADS`,
  `TERATTS_HOST`, `TERATTS_PORT`, `TERATTS_SHARE=1`, `HF_TOKEN` (если упёрлись в лимит загрузок).

## 6. Траблшутинг

| Симптом | Причина / решение |
| --- | --- |
| Медленно качает модель / обрывы | Проверьте `HF_HOME` (кэш) и повторите — докачка продолжится; при блокировке Hub нужен прокси/VPN |
| `No module named 'onnxruntime'` | Не выполнен шаг 1 в том же venv |
| Порт 7860 занят | `--port 7861` |
| Много RAM | `--stress-mode dictionary`, `TERATTS_THREADS=4`, не открывайте `teacher`-сэмплер |
| Ошибка о тегах | Текст не обёрнут в `<ru>`/`<en>` или теги не сбалансированы |

