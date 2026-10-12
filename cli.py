"""CLI-генерация речи TeraTTSv2 без веб-интерфейса (самый надёжный локальный тест).

Примеры:
    .venv/bin/python cli.py --text "<ru>Привет от TeraTTS.</ru>" --voice ru_f1 --out out/ru.wav
    .venv/bin/python cli.py --text "<en>I have 42 apples.</en>" --voice eng_f3 --duration-scale 1.2
    .venv/bin/python cli.py --text "<ru>Тест.</ru>" --play   # нужен sounddevice

Вывод: WAV 16-bit PCM, моно, 44 100 Гц (см. model card TeraSpace/TeraTTSv2).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from transformers import AutoModel

MODEL_ID = os.getenv("TERATTS_MODEL_ID", "TeraSpace/TeraTTSv2")
REVISION = os.getenv("TERATTS_REVISION") or None
SAMPLE_RATE = 44_100
LOGGER = logging.getLogger("teratts.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация речи TeraTTSv2 в WAV.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Текст для синтеза (обязательно теги <ru>…</ru>/<en>…</en>).")
    source.add_argument("--text-file", type=Path, help="Файл с текстом (UTF-8).")
    parser.add_argument("--voice", default="ru_f1", help="Идентификатор голоса, например ru_f1, ru_m5.")
    parser.add_argument("--out", type=Path, default=Path("out/teratts.wav"), help="Путь для WAV-файла.")
    parser.add_argument("--duration-scale", type=float, default=1.0, help=">1 — медленнее и длиннее.")
    parser.add_argument(
        "--diffusion-model",
        choices=("distilled", "teacher"),
        default="distilled",
        help="distilled — быстрый 8-шаговый сэмплер, teacher — с настраиваемым CFG.",
    )
    parser.add_argument(
        "--stress-mode",
        choices=("full", "dictionary", "off"),
        default="full",
        help="Обработка русских ударений: нейро+словарь / только словарь / выключена.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Число потоков onnxruntime.",
    )
    parser.add_argument("--play", action="store_true", help="Проиграть результат (требует sounddevice).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Пустой текст.")

    LOGGER.info("Загружаю %s (revision=%s, sampler=%s)…", MODEL_ID, REVISION or "main", args.diffusion_model)
    load_started = time.perf_counter()
    tts = AutoModel.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        provider="CPUExecutionProvider",
        threads=args.threads,
        diffusion_model=args.diffusion_model,
        russian_stress=args.stress_mode != "off",
        ruaccent_mode="dictionary" if args.stress_mode == "dictionary" else "full",
    )
    LOGGER.info("Модель загружена за %.1f с", time.perf_counter() - load_started)

    LOGGER.info("Нормализованный текст: %s", tts.normalize_text(text))

    synth_started = time.perf_counter()
    waveform = tts.generate_speech(text, voice=args.voice, duration_scale=args.duration_scale)
    elapsed = time.perf_counter() - synth_started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tts.save_wav(str(args.out), waveform)
    audio_seconds = len(waveform) / SAMPLE_RATE
    LOGGER.info(
        "Готово: %s (%.2f с аудио, синтез %.1f с, RTF %.2f)",
        args.out,
        audio_seconds,
        elapsed,
        elapsed / max(audio_seconds, 1e-6),
    )

    if args.play:
        import sounddevice as sd

        sd.play(waveform, SAMPLE_RATE)
        sd.wait()


if __name__ == "__main__":
    main()
