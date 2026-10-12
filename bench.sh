#!/usr/bin/env sh
# Запуск бенчмарка TeraTTSv2 в контейнере (VPS / локально).
#
# Контейнер всегда работает под uid 1000: кэш модели лежит в named volume, который
# Docker инициализирует правами из образа (uid 1000). Скрипт лишь открывает ./out
# на запись для этого uid:
#   * под root      — chown 1000:1000 out;
#   * под uid 1000  — ничего;
#   * другой uid    — chmod 777 out.
#
#   ./bench.sh                                  # диалог диспетчера → out/bench.json
#   ./bench.sh bench.py --runs 5 --warmup 2
#   ./bench.sh bench.py --scenario none --sweep-threads 1,2,4,8
#   TERATTS_THREADS=4 ./bench.sh
#   ./bench.sh bench.py --scenario none --concurrency 3 --runs 2   # 3 «звонка»

set -eu

cd "$(dirname "$0")"

IMAGE_UID=1000
IMAGE_GID=1000

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker не найден. Установка: curl -fsSL https://get.docker.com | sh" >&2
    exit 1
fi

# Плагин "docker compose" или отдельный docker-compose —
# используется то, что установлено в системе.
if docker compose version >/dev/null 2>&1; then
    compose="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    compose="docker-compose"
else
    echo "Не найден ни 'docker compose', ни 'docker-compose'." >&2
    exit 1
fi

mkdir -p out

if [ "$(id -u)" -eq 0 ]; then
    chown "${IMAGE_UID}:${IMAGE_GID}" out
    echo "root: папка out отдана uid ${IMAGE_UID}:${IMAGE_GID}"
elif [ "$(id -u)" -ne "$IMAGE_UID" ]; then
    chmod 777 out
    echo "uid $(id -u) ≠ ${IMAGE_UID}: папке out добавлены права на запись для контейнера"
fi

# Кэш модели в volume принадлежит uid ${IMAGE_UID} — контейнер запускаем под ним же,
# иначе он не увидит уже скачанную модель.
TERATTS_UID="$IMAGE_UID"
TERATTS_GID="$IMAGE_GID"
export TERATTS_UID TERATTS_GID

echo "Запуск: uid:gid=${TERATTS_UID}:${TERATTS_GID}, ${compose} run --rm bench $*"
# $compose без кавычек намеренно: нужно разбиение на слова ("docker compose").
# shellcheck disable=SC2086
if [ "$#" -gt 0 ]; then
    exec $compose run --rm bench "$@"
fi
exec $compose run --rm bench
