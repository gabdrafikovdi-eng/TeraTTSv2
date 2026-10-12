"""Бенчмарк задержки TeraTTSv2 для телефонного бота (стриминг) и общая латентность.

Сценарии (--scenario):
  dispatch : реальный диалог такси-диспетчера по репликам + детальные метрики
             стриминга: TTFC (задержка до первого звука), паузы между чанками,
             запас реального времени, оценка ёмкости VPS, тест конкуренции,
             sweep по chunk_frames;
  latency  : общая латентность на пресетах short/medium/long, RTF, p50/p90/p99,
             холодная/повторная загрузка модели, подбор --threads, round-trip к
             запущенному веб-демо (gradio_client).

Что именно важно для телефонии:
  * TTFC — сколько миллисекунд молчит бот перед первой фразой;
  * «запас реального времени» — успевает ли генерация следующего чанка до того,
    как проиграется предыдущий (иначе в трубке будут провалы);
  * xRT — сколько секунд речи даёт одна секунда CPU (сколько каналов тянет VPS).

Примеры:
    python bench.py                                     # диалог диспетчера
    python bench.py --runs 5 --warmup 2                 # точнее статистика
    python bench.py --sweep-chunk-frames 16,8,4         # как снизить TTFC
    python bench.py --sweep-threads 1,2,4,8             # подбор потоков под VPS
    python bench.py --concurrency 2                     # 2 одновременных звонка
    python bench.py --scenario latency --url http://127.0.0.1:7860
    python bench.py --json out/bench.json               # машинночитаемый отчёт
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import time
import wave
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = os.getenv("TERATTS_MODEL_ID", "TeraSpace/TeraTTSv2")
REVISION = os.getenv("TERATTS_REVISION") or None
SAMPLE_RATE = 44_100

# Реплики, которые обычно генерирует LLM в системе телефонного диспетчера.
DISPATCH_LINES: list[str] = [
    "<ru>Здравствуйте! Скажите, откуда вас необходимо забрать?</ru>",
    "<ru>От Ленина, сорок до Шаймуратова, двенадцать. Триста рублей. Подтверждаете?</ru>",
    "<ru>Заказ подтверждён. Ищем свободную машину, как только водитель назначится, "
    "вам перезвоним. Что-то ещё?</ru>",
    "<ru>Здравствуйте! Меня зовут Ева, я виртуальный диспетчер такси. Подскажите, "
    "пожалуйста, откуда вас забрать?</ru>",
]

PRESETS: dict[str, str] = {
    "short": DISPATCH_LINES[0],
    "medium": DISPATCH_LINES[1],
    "long": DISPATCH_LINES[2] + " " + DISPATCH_LINES[3],
}


def env_int(name: str, default: int) -> int:
    """Целое из переменной окружения; пустое значение считаем «не задано»."""
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def percentile(values: list[float], quantile: float) -> float:
    """Линейная квантиль (как numpy.percentile с интерполяцией)."""
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def peak_rss_mb() -> float:
    """Пиковая RSS процесса в МБ (macOS отдаёт байты, Linux — килобайты)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 ** 2) if platform.system() == "Darwin" else usage / 1024


def wav_seconds(path: str | Path) -> float:
    """Длительность WAV в секундах (stdlib wave, без сторонних зависимостей)."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / handle.getframerate()
    except (wave.Error, OSError):
        return math.nan


def environment() -> dict:
    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "model_id": MODEL_ID,
        "revision": REVISION or "main",
    }


def load_model(*, threads: int, stress_mode: str, diffusion_model: str, local_files_only: bool = False):
    """Загрузить модель и вернуть (модель, секунды загрузки)."""
    from transformers import AutoModel

    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        provider="CPUExecutionProvider",
        threads=threads,
        diffusion_model=diffusion_model,
        russian_stress=stress_mode != "off",
        ruaccent_mode="dictionary" if stress_mode == "dictionary" else "full",
        local_files_only=local_files_only,
    )
    return model, time.perf_counter() - started


@dataclass
class StreamRun:
    """Один прогон стриминга: времена прихода чанков и их размеры."""

    marks: list[float]  # секунды от старта генерации до прихода чанка
    samples: list[int]  # сэмплов в каждом чанке (44.1 кГц, моно)

    @property
    def ttfc(self) -> float:
        """TTFC — задержка до первого звука (первый чанк)."""
        return self.marks[0]

    @property
    def total(self) -> float:
        return self.marks[-1]

    @property
    def audio_seconds(self) -> float:
        return sum(self.samples) / SAMPLE_RATE

    @property
    def gaps(self) -> list[float]:
        """Паузы между приходом чанков (время генерации очередного чанка)."""
        return [second - first for first, second in zip(self.marks, self.marks[1:])]

    def margins(self) -> list[float]:
        """Запас буфера в секундах: сколько аудио уже готово минус сколько уже играется.

        Воспроизведение стартует с первым чанком (t = marks[0]), поэтому отсчёт идёт
        от него: margin = накопленное аудио − (время прихода чанка − ttfc).
        Отрицательное значение = провал (underrun), если играть сразу после TTFC.
        """
        playback_start = self.marks[0]
        margins: list[float] = []
        accumulated = 0.0
        for index, (mark, samples) in enumerate(zip(self.marks, self.samples)):
            if index:
                margins.append(accumulated - (mark - playback_start))
            accumulated += samples / SAMPLE_RATE
        return margins


def synth_stream(
    model,
    text: str,
    voice: str,
    *,
    duration_scale: float,
    chunk_frames: int,
) -> StreamRun:
    """Один стриминговый прогон: замеряем время прихода каждого чанка."""
    started = time.perf_counter()
    marks: list[float] = []
    samples: list[int] = []
    for chunk in model.generate_speech_stream(
        text,
        voice=voice,
        duration_scale=duration_scale,
        chunk_frames=chunk_frames,
    ):
        marks.append(time.perf_counter() - started)
        samples.append(int(len(chunk)))
    return StreamRun(marks=marks, samples=samples)


def synth_full(model, text: str, voice: str, *, duration_scale: float) -> tuple[float, float]:
    """Обычный (нестриминговый) синтез → (секунды генерации, секунды аудио)."""
    started = time.perf_counter()
    waveform = model.generate_speech(text, voice=voice, duration_scale=duration_scale)
    return time.perf_counter() - started, len(waveform) / SAMPLE_RATE


def summarize_runs(
    label: str,
    text: str,
    runs: list[StreamRun],
    *,
    load_seconds: float,
    chunk_frames: int,
) -> dict:
    """Свести прогоны одной реплики в метрики для таблицы."""
    ttfc = [run.ttfc for run in runs]
    totals = [run.total for run in runs]
    audio = runs[-1].audio_seconds
    gaps = [gap for run in runs for gap in run.gaps]
    margins = [margin for run in runs for margin in run.margins()]
    p50 = percentile(totals, 0.5)
    return {
        "line": label,
        "text": text,
        "chars": len(text),
        "audio_seconds": round(audio, 3),
        "chunks": len(runs[-1].samples),
        "chunk_audio_seconds": round(runs[-1].samples[0] / SAMPLE_RATE, 3),
        "chunk_frames": chunk_frames,
        "ttfc_p50_ms": round(percentile(ttfc, 0.5) * 1000, 1),
        "ttfc_p90_ms": round(percentile(ttfc, 0.9) * 1000, 1),
        "ttfc_max_ms": round(max(ttfc) * 1000, 1),
        "synth_p50_s": round(p50, 3),
        "gap_p50_ms": round(percentile(gaps, 0.5) * 1000, 1) if gaps else None,
        "gap_max_ms": round(max(gaps) * 1000, 1) if gaps else None,
        "margin_min_ms": round(min(margins) * 1000, 1) if margins else None,
        "rtf": round(p50 / audio, 4) if audio else None,
        "xrt": round(audio / p50, 3) if p50 else None,
        "load_seconds": round(load_seconds, 3),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }


def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def print_replies(lines: list[str]) -> None:
    print("Реплики диалога диспетчера:")
    for index, line in enumerate(lines, start=1):
        print(f"  #{index}: {line}")
    print()


def print_stream_table(rows: list[dict]) -> None:
    print(
        f"{'реплика':<9}{'симв':>6}{'аудио,с':>8}{'чанков':>7}{'TTFC p50,мс':>12}{'p90,мс':>8}"
        f"{'синтез,с':>10}{'пауза max,мс':>13}{'запас min,мс':>13}{'RTF':>7}{'xRT':>7}"
    )
    for row in rows:
        print(
            f"{row['line']:<9}{row['chars']:>6}{row['audio_seconds']:>8.2f}{row['chunks']:>7}"
            f"{row['ttfc_p50_ms']:>12.0f}{row['ttfc_p90_ms']:>8.0f}{row['synth_p50_s']:>10.3f}"
            f"{_num(row['gap_max_ms'], 0):>13}{_num(row['margin_min_ms'], 0):>13}"
            f"{_num(row['rtf'], 3):>7}{_num(row['xrt'], 2):>7}"
        )
    print("TTFC — задержка до первого звука; «запас min» — свободный буфер при воспроизведении,")
    print("если начать его с первым чанком (≥0 → речь идёт без провалов);")
    print("RTF = генерация/аудио (<1 — быстрее реального времени); xRT = секунд речи на секунду CPU.")


def print_dispatch_summary(
    rows: list[dict],
    *,
    duty_cycle: float,
    budget_ms: float,
    network_ms: float,
) -> None:
    audio_total = sum(row["audio_seconds"] for row in rows)
    synth_total = sum(row["synth_p50_s"] for row in rows)
    ttfc = [row["ttfc_p50_ms"] for row in rows]
    xrt = audio_total / synth_total if synth_total else math.nan

    print()
    print("=== Итог по диалогу ===")
    print(f"Реплик: {len(rows)} | аудио: {audio_total:.2f} с | генерация: {synth_total:.2f} с")
    print(f"Общий RTF: {synth_total / audio_total:.3f} | xRT: {xrt:.2f}")
    print(f"TTFC, мс: min {min(ttfc):.0f} | p50 {percentile(ttfc, 0.5):.0f} | max {max(ttfc):.0f}")
    margins = [row["margin_min_ms"] for row in rows if row.get("margin_min_ms") is not None]
    if margins:
        worst = min(margins)
        verdict = "провалов в речи нет" if worst >= 0 else f"возможны паузы до {-worst:.0f} мс"
        print(f"Худший запас буфера: {worst:.0f} мс → {verdict}")

    print()
    print("=== Бюджет задержки до первого звука ===")
    for row in rows:
        total = row["ttfc_p50_ms"] + network_ms
        verdict = "ок" if total <= budget_ms else "ВЫШЕ БЮДЖЕТА"
        print(
            f"  {row['line']}: TTFC {row['ttfc_p50_ms']:.0f} мс + сеть/телефония {network_ms:.0f} мс"
            f" = {total:.0f} мс (бюджет {budget_ms:.0f} мс) → {verdict}"
        )

    print()
    print("=== Ёмкость VPS (оценка по измеренному xRT) ===")
    print(f"  Непрерывная речь: ~{xrt:.1f} канала (сервер занят только синтезом)")
    print(f"  Диалог с долей речи {duty_cycle:.0%}: ~{xrt / duty_cycle:.1f} каналов")
    print("  Это потолок: реальные цифры при нескольких звонках даёт --concurrency N.")


def print_latency_table(rows: list[dict]) -> None:
    print(
        f"{'пресет':<10}{'симв':>6}{'аудио,с':>9}{'p50,с':>8}{'p90,с':>8}{'p99,с':>8}"
        f"{'TTFC,мс':>9}{'RTF':>7}{'xRT':>7}{'симв/с':>8}"
    )
    for row in rows:
        print(
            f"{row['preset']:<10}{row['chars']:>6}{row['audio_seconds']:>9.2f}{row['synth_p50_s']:>8.3f}"
            f"{row['synth_p90_s']:>8.3f}{row['synth_p99_s']:>8.3f}"
            f"{_num(row['ttfc_p50_ms'], 0):>9}{_num(row['rtf'], 3):>7}"
            f"{_num(row['xrt'], 2):>7}{_num(row['chars_per_second'], 1):>8}"
        )


def print_sweep_table(title: str, rows: list[dict], key: str, key_title: str) -> None:
    """Таблица sweep-прогона; ширина колонки ключа подстраивается под значения."""
    width = max(8, len(key_title), *(len(str(row[key])) for row in rows))
    print()
    print(f"=== {title} ===")
    print(f"{key_title:>{width}}{'загрузка,с':>12}{'TTFC p50,мс':>13}{'синтез,p50,с':>13}{'xRT':>7}")
    for row in rows:
        print(
            f"{str(row[key]):>{width}}{row['load_seconds']:>12.1f}{_num(row['ttfc_p50_ms'], 0):>13}"
            f"{row['synth_p50_s']:>13.3f}{_num(row['xrt'], 2):>7}"
        )


def write_json(path: Path, payload: dict) -> None:
    """Сохранить отчёт; проблемы с правами не должны «убивать» уже собранные цифры."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as error:
        print(f"\nНе удалось записать JSON: {error}")
        print("В Docker с bind-mount './out' папку нужно создать заранее (иначе Docker")
        print("создаст её от root и uid 1000 внутри контейнера не сможет писать):")
        print("  mkdir -p out && sudo chown 1000:1000 out")
        return
    print(f"\nJSON-отчёт: {path}")


def resolve_presets(args) -> list[tuple[str, str]]:
    """Тексты для сценария latency: --text, --text-file или пресеты (--preset)."""
    if args.text:
        return [("custom", args.text)]
    text_file = getattr(args, "text_file", None)
    if text_file:
        lines = [
            line.strip()
            for line in Path(text_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            raise SystemExit(f"В файле {text_file} нет ни одной фразы (пустые строки и # пропускаются)")
        return [(f"L{index}", line) for index, line in enumerate(lines, start=1)]
    names = [name.strip() for name in args.preset.split(",") if name.strip()]
    if "all" in names:
        return list(PRESETS.items())
    unknown = [name for name in names if name not in PRESETS]
    if unknown:
        raise SystemExit(f"Неизвестные пресеты: {unknown}. Доступно: {', '.join(PRESETS)} или all")
    return [(name, PRESETS[name]) for name in names]


def load_configured_model(args):
    model, load_seconds = load_model(
        threads=args.threads,
        stress_mode=args.stress_mode,
        diffusion_model=args.diffusion_model,
        local_files_only=args.offline,
    )
    print(
        f"Модель загружена за {load_seconds:.1f} с | threads={args.threads}"
        f" | sampler={args.diffusion_model} | stress={args.stress_mode} | voice={args.voice}"
    )
    print(f"Пиковая RSS после загрузки: {peak_rss_mb():.0f} МБ")
    return model, load_seconds


def bench_dispatch(args) -> dict:
    """Сценарий dispatch: стриминг реального диалога такси-диспетчера."""
    lines = DISPATCH_LINES
    print_replies(lines)
    model, load_seconds = load_configured_model(args)
    print(f"chunk_frames={args.chunk_frames} (пакет ≈ {args.chunk_frames * 3072 / SAMPLE_RATE:.2f} с аудио)\n")

    rows: list[dict] = []
    for index, text in enumerate(lines, start=1):
        for _ in range(args.warmup):
            synth_stream(model, text, args.voice, duration_scale=args.duration_scale, chunk_frames=args.chunk_frames)
        runs = [
            synth_stream(model, text, args.voice, duration_scale=args.duration_scale, chunk_frames=args.chunk_frames)
            for _ in range(args.runs)
        ]
        rows.append(
            summarize_runs(f"#{index}", text, runs, load_seconds=load_seconds, chunk_frames=args.chunk_frames)
        )

    print_stream_table(rows)
    print_dispatch_summary(
        rows,
        duty_cycle=args.duty_cycle,
        budget_ms=args.budget_ms,
        network_ms=args.network_ms,
    )
    return {"load_seconds": round(load_seconds, 3), "rows": rows}


def bench_latency(args) -> dict:
    """Сценарий latency: RTF, p50/p90/p99 и TTFC на текстах разной длины."""
    cases = resolve_presets(args)
    model, load_seconds = load_configured_model(args)
    print()

    rows: list[dict] = []
    for name, text in cases:
        for _ in range(args.warmup):
            synth_full(model, text, args.voice, duration_scale=args.duration_scale)
        times: list[float] = []
        ttfc: list[float] = []
        audio = 0.0
        for _ in range(args.runs):
            if args.stream:
                run = synth_stream(
                    model, text, args.voice, duration_scale=args.duration_scale, chunk_frames=args.chunk_frames
                )
                times.append(run.total)
                ttfc.append(run.ttfc)
                audio = run.audio_seconds
            else:
                elapsed, audio = synth_full(model, text, args.voice, duration_scale=args.duration_scale)
                times.append(elapsed)
        p50 = percentile(times, 0.5)
        rows.append(
            {
                "preset": name,
                "chars": len(text),
                "audio_seconds": round(audio, 3),
                "synth_p50_s": round(p50, 3),
                "synth_p90_s": round(percentile(times, 0.9), 3),
                "synth_p99_s": round(percentile(times, 0.99), 3),
                "ttfc_p50_ms": round(percentile(ttfc, 0.5) * 1000, 1) if ttfc else None,
                "rtf": round(p50 / audio, 4) if audio else None,
                "xrt": round(audio / p50, 3) if p50 else None,
                "chars_per_second": round(len(text) / p50, 1) if p50 else None,
                "load_seconds": round(load_seconds, 3),
            }
        )

    print_latency_table(rows)
    result: dict = {"load_seconds": round(load_seconds, 3), "rows": rows}

    if args.url:
        http_rows = bench_http(args)
        result["http"] = http_rows
        print()
        print("=== Round-trip к веб-демо (весь путь: сеть + очередь + синтез) ===")
        print(f"{'пресет':<10}{'аудио,с':>9}{'p50,с':>9}{'p90,с':>9}{'p99,с':>9}{'xRT':>7}")
        for row in http_rows:
            print(
                f"{row['preset']:<10}{row['audio_seconds']:>9.2f}{row['roundtrip_p50_s']:>9.3f}"
                f"{row['roundtrip_p90_s']:>9.3f}{row['roundtrip_p99_s']:>9.3f}{_num(row['xrt'], 2):>7}"
            )
    return result


def bench_http(args) -> list[dict]:
    """Замер round-trip к запущенному app.py через gradio_client."""
    from gradio_client import Client

    started = time.perf_counter()
    client = Client(args.url, verbose=False)
    print(f"\n=== Подключение к {args.url}: {time.perf_counter() - started:.2f} с ===")

    rows: list[dict] = []
    for name, text in resolve_presets(args):
        times: list[float] = []
        result = None
        for attempt in range(args.warmup + args.runs):
            started = time.perf_counter()
            result = client.predict(
                text,
                args.voice,
                args.diffusion_model,
                args.stress_mode,
                args.duration_scale,
                api_name="/synthesize",
            )
            elapsed = time.perf_counter() - started
            if attempt >= args.warmup:
                times.append(elapsed)
        audio = wav_seconds(result[0]) if result else math.nan
        p50 = percentile(times, 0.5)
        rows.append(
            {
                "preset": name,
                "chars": len(text),
                "audio_seconds": None if math.isnan(audio) else round(audio, 3),
                "roundtrip_p50_s": round(p50, 3),
                "roundtrip_p90_s": round(percentile(times, 0.9), 3),
                "roundtrip_p99_s": round(percentile(times, 0.99), 3),
                "xrt": round(audio / p50, 3) if audio and p50 else None,
            }
        )
    return rows


def sweep_threads(args) -> list[dict]:
    """Подбор числа потоков onnxruntime: для каждого значения модель грузится заново."""
    text = PRESETS["medium"]
    rows: list[dict] = []
    for threads in args.sweep_threads:
        model, load_seconds = load_model(
            threads=threads,
            stress_mode=args.stress_mode,
            diffusion_model=args.diffusion_model,
            local_files_only=args.offline,
        )
        for _ in range(max(1, args.warmup)):
            synth_stream(model, text, args.voice, duration_scale=args.duration_scale, chunk_frames=args.chunk_frames)
        runs = [
            synth_stream(model, text, args.voice, duration_scale=args.duration_scale, chunk_frames=args.chunk_frames)
            for _ in range(args.runs)
        ]
        stats = summarize_runs("threads", text, runs, load_seconds=load_seconds, chunk_frames=args.chunk_frames)
        stats["threads"] = threads
        rows.append(stats)
        del model
        gc.collect()
    print_sweep_table(f"Подбор потоков onnxruntime (пресет medium, voice={args.voice})", rows, "threads", "потоки")
    print(f"Ядер в контейнере/на VPS: {os.cpu_count()}")
    return rows


def sweep_stress_modes(args) -> list[dict]:
    """Сравнить режимы русских ударений: сколько TTFC «стоит» нейросеть RUAccent."""
    text = max(DISPATCH_LINES, key=len)
    rows: list[dict] = []
    for mode in ("full", "dictionary", "off"):
        model, load_seconds = load_model(
            threads=args.threads,
            stress_mode=mode,
            diffusion_model=args.diffusion_model,
            local_files_only=args.offline,
        )
        for _ in range(max(1, args.warmup)):
            synth_stream(model, text, args.voice, duration_scale=args.duration_scale, chunk_frames=args.chunk_frames)
        runs = [
            synth_stream(model, text, args.voice, duration_scale=args.duration_scale, chunk_frames=args.chunk_frames)
            for _ in range(args.runs)
        ]
        stats = summarize_runs("stress", text, runs, load_seconds=load_seconds, chunk_frames=args.chunk_frames)
        stats["stress_mode"] = mode
        rows.append(stats)
        del model
        gc.collect()
    print_sweep_table(
        "Влияние режима русских ударений на TTFC (самая длинная реплика диалога)",
        rows,
        "stress_mode",
        "режим",
    )
    print("full — нейросеть RUAccent + словари; dictionary — только словари; off — выключено.")
    print("Загрузка ≈0 с означает переиспользование ONNX-сессий из кэша процесса")
    print("(в коде модели они кэшируются по пути, провайдеру и числу потоков).")
    return rows


def sweep_chunk_frames(args) -> list[dict]:
    """Как chunk_frames влияет на TTFC и паузы (на самой длинной реплике диалога)."""
    text = max(DISPATCH_LINES, key=len)
    model, load_seconds = load_model(
        threads=args.threads,
        stress_mode=args.stress_mode,
        diffusion_model=args.diffusion_model,
        local_files_only=args.offline,
    )
    rows: list[dict] = []
    for frames in args.sweep_chunk_frames:
        for _ in range(max(1, args.warmup)):
            synth_stream(model, text, args.voice, duration_scale=args.duration_scale, chunk_frames=frames)
        runs = [
            synth_stream(model, text, args.voice, duration_scale=args.duration_scale, chunk_frames=frames)
            for _ in range(args.runs)
        ]
        stats = summarize_runs("chunk_frames", text, runs, load_seconds=load_seconds, chunk_frames=frames)
        stats["chunk_frames"] = frames
        rows.append(stats)
    print_sweep_table(
        "Влияние chunk_frames на TTFC и паузы (самая длинная реплика диалога)",
        rows,
        "chunk_frames",
        "chunk_fr",
    )
    return rows


def _concurrency_worker(config: dict, text: str, queue, barrier) -> None:
    """Процесс-«звонок»: грузит свою модель, синхронно стартует и меряет стриминг."""
    try:
        model, load_seconds = load_model(
            threads=config["threads"],
            stress_mode=config["stress_mode"],
            diffusion_model=config["diffusion_model"],
        )
        for _ in range(config["warmup"]):
            synth_stream(
                model,
                text,
                config["voice"],
                duration_scale=config["duration_scale"],
                chunk_frames=config["chunk_frames"],
            )
        barrier.wait(timeout=900)
        runs = [
            synth_stream(
                model,
                text,
                config["voice"],
                duration_scale=config["duration_scale"],
                chunk_frames=config["chunk_frames"],
            )
            for _ in range(config["runs"])
        ]
        queue.put(
            {
                "load_seconds": load_seconds,
                "ttfc_ms": [run.ttfc * 1000 for run in runs],
                "synth_s": [run.total for run in runs],
                "audio_seconds": runs[0].audio_seconds,
                "peak_rss_mb": peak_rss_mb(),
            }
        )
    except Exception as error:  # noqa: BLE001 - воркер обязан сообщить родителю о любой ошибке
        queue.put({"error": f"{type(error).__name__}: {error}"})


def bench_concurrency(args) -> dict:
    """Сколько одновременных «звонков» тянет VPS: N процессов синтезируют параллельно."""
    import multiprocessing as mp

    text = PRESETS["medium"]
    context = mp.get_context("spawn")
    queue = context.Queue()
    barrier = context.Barrier(args.concurrency)
    config = {
        "threads": args.threads,
        "stress_mode": args.stress_mode,
        "diffusion_model": args.diffusion_model,
        "voice": args.voice,
        "chunk_frames": args.chunk_frames,
        "duration_scale": args.duration_scale,
        "runs": args.runs,
        "warmup": max(1, args.warmup),
    }

    print(f"\n=== Конкуренция: {args.concurrency} одновременных звонков ===")
    workers = [
        context.Process(target=_concurrency_worker, args=(config, text, queue, barrier))
        for _ in range(args.concurrency)
    ]
    started = time.perf_counter()
    for worker in workers:
        worker.start()
    payloads: list[dict] = []
    for _ in workers:
        try:
            payloads.append(queue.get(timeout=1800))
        except Exception:  # noqa: BLE001 - пустая очередь означает падение воркера
            payloads.append({"error": "воркер не ответил"})
    for worker in workers:
        worker.join(timeout=120)
    wall_seconds = time.perf_counter() - started
    for worker in workers:
        if worker.is_alive():
            worker.terminate()

    failed = [payload["error"] for payload in payloads if "error" in payload]
    if failed:
        print("Ошибки воркеров:")
        for message in failed:
            print(f"  {message}")
        return {"failed": failed}

    print(f"{'воркер':>7}{'загрузка,с':>12}{'TTFC p50,мс':>13}{'TTFC max,мс':>13}{'синтез p50,с':>13}{'RSS,МБ':>9}")
    for index, payload in enumerate(payloads, start=1):
        print(
            f"{index:>7}{payload['load_seconds']:>12.1f}"
            f"{percentile(payload['ttfc_ms'], 0.5):>13.0f}{max(payload['ttfc_ms']):>13.0f}"
            f"{percentile(payload['synth_s'], 0.5):>13.3f}{payload['peak_rss_mb']:>9.0f}"
        )

    audio_total = sum(payload["audio_seconds"] for payload in payloads)
    synth_total = sum(percentile(payload["synth_s"], 0.5) for payload in payloads)
    xrt_parallel = audio_total / synth_total if synth_total else 0.0
    all_ttfc = [value for payload in payloads for value in payload["ttfc_ms"]]
    print(f"\nПри {args.concurrency} звонках одновременно: суммарный xRT = {xrt_parallel:.2f}")
    print(f"TTFC p50 по всем каналам: {percentile(all_ttfc, 0.5):.0f} мс")
    print(f"Суммарная RSS всех воркеров: {sum(payload['peak_rss_mb'] for payload in payloads):.0f} МБ")
    return {
        "wall_seconds": round(wall_seconds, 2),
        "concurrency": args.concurrency,
        "xrt_parallel": round(xrt_parallel, 3),
        "workers": payloads,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Бенчмарк задержки TeraTTSv2 для телефонного диспетчера (стриминг, TTFC, RTF).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        choices=("dispatch", "latency", "all", "none"),
        default="dispatch",
        help="dispatch — диалог такси-диспетчера; latency — пресеты/произвольный текст; none — только sweep/конкуренция",
    )
    parser.add_argument("--text", default=None, help="Произвольный текст (вместо пресетов в latency)")
    parser.add_argument(
        "--text-file",
        default=None,
        help="Файл со списком фраз (по одной в строке, # — комментарий) для сценария latency",
    )
    parser.add_argument("--preset", default="all", help="Пресеты latency: short, medium, long, all")
    parser.add_argument("--url", default=os.getenv("TERATTS_URL"), help="URL веб-демо для round-trip, например http://127.0.0.1:7860")
    parser.add_argument("--runs", type=int, default=3, help="Измеряемых прогонов на текст")
    parser.add_argument("--warmup", type=int, default=1, help="Прогревочных прогонов (не учитываются)")
    parser.add_argument("--voice", default="ru_f1")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--chunk-frames", type=int, default=16, help="Кадров в чанке стрима (16 ≈ 1.11 с аудио)")
    parser.add_argument("--diffusion-model", choices=("distilled", "teacher"), default="distilled")
    parser.add_argument("--stress-mode", choices=("full", "dictionary", "off"), default="full")
    parser.add_argument(
        "--threads",
        type=int,
        default=env_int("TERATTS_THREADS", max(1, (os.cpu_count() or 1) // 2)),
        help="Потоков onnxruntime",
    )
    parser.add_argument("--stream", action="store_true", help="В latency измерять TTFC через стриминг")
    parser.add_argument("--offline", action="store_true", help="Только локальный кэш модели (local_files_only)")
    parser.add_argument("--concurrency", type=int, default=0, help="Число одновременных звонков, >1 включает тест")
    parser.add_argument("--sweep-threads", default=None, help="Подбор потоков, напр. 1,2,4,8")
    parser.add_argument("--sweep-chunk-frames", default=None, help="Подбор chunk_frames, напр. 16,8,4")
    parser.add_argument(
        "--sweep-stress-modes",
        action="store_true",
        help="Сравнить режимы ударений full/dictionary/off по TTFC",
    )
    parser.add_argument("--duty-cycle", type=float, default=0.5, help="Доля речи в звонке для оценки ёмкости VPS")
    parser.add_argument("--budget-ms", type=float, default=500.0, help="Бюджет задержки до первого звука, мс")
    parser.add_argument("--network-ms", type=float, default=50.0, help="Оценка сети+телефонии для бюджета, мс")
    parser.add_argument("--json", type=Path, default=None, help="Сохранить отчёт в JSON")
    return parser


def parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = build_parser().parse_args()
    args.sweep_threads = parse_int_list(args.sweep_threads)
    args.sweep_chunk_frames = parse_int_list(args.sweep_chunk_frames)

    print("=== Окружение ===")
    for key, value in environment().items():
        print(f"  {key}: {value}")
    print()

    report: dict = {
        "environment": environment(),
        "params": {
            "scenario": args.scenario,
            "voice": args.voice,
            "threads": args.threads,
            "chunk_frames": args.chunk_frames,
            "diffusion_model": args.diffusion_model,
            "stress_mode": args.stress_mode,
            "duration_scale": args.duration_scale,
            "runs": args.runs,
            "warmup": args.warmup,
        },
    }

    if args.concurrency > 1:
        report["concurrency"] = bench_concurrency(args)
    if args.sweep_threads:
        report["sweep_threads"] = sweep_threads(args)
    if args.sweep_chunk_frames:
        report["sweep_chunk_frames"] = sweep_chunk_frames(args)
    if args.sweep_stress_modes:
        report["sweep_stress_modes"] = sweep_stress_modes(args)

    if args.scenario == "dispatch":
        report["dispatch"] = bench_dispatch(args)
    elif args.scenario == "latency":
        report["latency"] = bench_latency(args)
    elif args.scenario == "all":
        report["dispatch"] = bench_dispatch(args)
        print()
        report["latency"] = bench_latency(args)

    if args.json:
        write_json(args.json, report)


if __name__ == "__main__":
    main()
