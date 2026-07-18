#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CPU_QUEUE="${CPU_QUEUE:-cpu}"
REMOTE_CPU="${REMOTE_CPU:-0}"

# По умолчанию используем существующие caches и готовые representations.
# Полная пересборка выполняется только с REBUILD_CACHE=1.
REBUILD="${REBUILD:-0}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"

cd "$PROJECT_ROOT"

CMD=(
  python
  final_exps/01_build_sequence_datasets.py
  --run-config
  configs/state_or_space_sequence_datasets.json
  --notebook-root
  .
  --enable-clearml
)

if [[ "$REBUILD_CACHE" == "1" ]]; then
  CMD+=(--rebuild --rebuild-cache)
elif [[ "$REBUILD" == "1" ]]; then
  CMD+=(--rebuild)
fi

if [[ "$REMOTE_CPU" == "1" ]]; then
  CMD+=(
    --execute-remotely
    --clearml-queue
    "$CPU_QUEUE"
  )
fi

POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-4}" \
PYTHONUNBUFFERED=1 \
"${CMD[@]}"
