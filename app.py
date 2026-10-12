"""Локальное Gradio-демо для TeraSpace/TeraTTSv2 (ONNX Runtime TTS).

Отличия от официального Space (https://huggingface.co/spaces/sasfox4/TeraTTSv2-Demo):
  * убрана зависимость ``spaces`` (ZeroGPU) — локально модель считается на CPU
    через ``onnxruntime``;
  * добавлены параметры ``--host / --port / --share`` для запуска на локальной
    машине и на VPS;
  * добавлена возможность зафиксировать ревизию модели (``TERATTS_REVISION``).

Запуск:
    .venv/bin/python app.py
    .venv/bin/python app.py --host 0.0.0.0 --port 7860   # доступ снаружи (VPS)
    .venv/bin/python app.py --share                      # публичная ссылка gradio.live
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import unicodedata
from functools import lru_cache

import gradio as gr
import numpy as np
from transformers import AutoModel

MODEL_ID = os.getenv("TERATTS_MODEL_ID", "TeraSpace/TeraTTSv2")
REVISION = os.getenv("TERATTS_REVISION") or None
THREADS = int(os.getenv("TERATTS_THREADS", str(max(1, (os.cpu_count() or 1) // 2))))
SAMPLE_RATE = 44_100
LOGGER = logging.getLogger(__name__)

VOICES = [
    ("★ Best: Russian female 1", "ru_f1"),
    ("★ Best: Russian male 5", "ru_m5"),
    ("Russian female 2", "ru_f2"),
    ("Russian male 1", "ru_m1"),
    ("English female 3", "eng_f3"),
    ("English female 4 — whisper", "eng_f4_whisper"),
    ("English female 5", "eng_f5"),
    ("English male 2 — whisper", "eng_m2_whisper"),
    ("English male 3", "eng_m3"),
    ("English male 4", "eng_m4"),
]

CSS = """
#col-container { max-width: 900px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

# В Gradio 6 theme/css переехали из Blocks() в launch(); в Gradio 5 — наоборот.
# Определяем нужный вариант по сигнатуре, чтобы демо работало на обеих версиях.
STYLE = {"theme": gr.themes.Citrus(), "css": CSS}
STYLE_IN_LAUNCH = {"theme", "css"} <= set(inspect.signature(gr.Blocks.launch).parameters)



@lru_cache(maxsize=2)
def load_tts(diffusion_model: str, stress_mode: str) -> object:
    """Загрузить одну пару (sampler, режим ударений), чтобы ограничить память."""
    return AutoModel.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        provider="CPUExecutionProvider",
        threads=THREADS,
        diffusion_model=diffusion_model,
        russian_stress=stress_mode != "off",
        ruaccent_mode="dictionary" if stress_mode == "dictionary" else "full",
    )


def synthesize(
    text: str,
    voice: str = "ru_f1",
    diffusion_model: str = "distilled",
    stress_mode: str = "full",
    duration_scale: float = 1.0,
) -> tuple[tuple[int, np.ndarray], str]:
    """Синтез речи: возвращает аудио (44.1 кГц, моно) и нормализованный текст."""
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter text to synthesize.")
    if len(text) > 4_000:
        raise gr.Error("Text is limited to 4,000 characters.")
    try:
        model = load_tts(diffusion_model, stress_mode)
        normalized_text = model.normalize_text(text)
        waveform = model.generate_speech(
            text,
            voice=voice,
            duration_scale=float(duration_scale),
        )
    except Exception as error:
        LOGGER.exception("TeraTTS synthesis failed")
        raise gr.Error(f"Synthesis failed: {error}") from error

    # Словарь ONNX в NFKD; показываем NFC, чтобы й/ё оставались составными в UI.
    return (SAMPLE_RATE, waveform), unicodedata.normalize("NFC", normalized_text)


with gr.Blocks(title="TeraTTSv2", **(STYLE if not STYLE_IN_LAUNCH else {})) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
# TeraTTSv2
Русско-английский ONNX TTS для [TeraSpace/TeraTTSv2](https://huggingface.co/TeraSpace/TeraTTSv2).

Оборачивайте текст в `<ru>…</ru>` или `<en>…</en>`. Ударения в русских тегах расставляются
автоматически; при английском голосе на русском тексте попробуйте `duration_scale` ≈ `0.8`.
"""
        )
        with gr.Row():
            text = gr.Textbox(
                label="Text",
                value=(
                    "<ru>Сразу после этого слово взял директор по И И Яндекса. "
                    "Он согласился с Пашей, повторив его цитату слово в слово.</ru>"
                ),
                lines=5,
                scale=4,
            )
        with gr.Row():
            voice = gr.Dropdown(VOICES, value="ru_f1", label="Voice")
            duration_scale = gr.Slider(
                minimum=0.5,
                maximum=1.5,
                step=0.05,
                value=1.0,
                label="Duration scale",
                info="Higher = slower / longer speech.",
            )
        with gr.Accordion("Advanced", open=False):
            diffusion_model = gr.Dropdown(
                [("Distilled — fast", "distilled"), ("Teacher — adjustable CFG", "teacher")],
                value="distilled",
                label="Diffusion model",
            )
            stress_mode = gr.Dropdown(
                [
                    ("Full — neural + dictionary", "full"),
                    ("Dictionary only", "dictionary"),
                    ("Off", "off"),
                ],
                value="full",
                label="Russian stress",
            )
        run = gr.Button("Generate", variant="primary")
        audio = gr.Audio(label="Audio", type="numpy")
        normalized = gr.Textbox(label="Normalized text", lines=3, interactive=False)

        gr.Examples(
            examples=[
                ["<ru>Привет от TeraTTS.</ru>", "ru_f1"],
                ["<ru>У меня 21 яблоко.</ru>", "ru_m5"],
                ["<en>Streaming speech is ready.</en>", "eng_f3"],
                ["<ru>Это русский текст английским голосом.</ru>", "eng_f3"],
                ["<ru>У меня 21 яблоко.</ru> <en>I have 42 apples.</en>", "ru_f1"],
            ],
            inputs=[text, voice],
            outputs=[audio, normalized],
            fn=synthesize,
            cache_examples=False,
        )

        run.click(
            fn=synthesize,
            inputs=[text, voice, diffusion_model, stress_mode, duration_scale],
            outputs=[audio, normalized],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Локальное Gradio-демо TeraTTSv2.")
    parser.add_argument(
        "--host",
        default=os.getenv("TERATTS_HOST", "127.0.0.1"),
        help="Адрес прослушивания (0.0.0.0 — чтобы открыть доступ снаружи).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("TERATTS_PORT", "7860")),
        help="Порт веб-интерфейса.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.getenv("TERATTS_SHARE", "0") == "1",
        help="Создать публичную ссылку gradio.live (туннель).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    LOGGER.info("Model=%s revision=%s threads=%d", MODEL_ID, REVISION or "main", THREADS)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.host in {"127.0.0.1", "localhost"},
        **(STYLE if STYLE_IN_LAUNCH else {}),
    )


if __name__ == "__main__":
    main()
