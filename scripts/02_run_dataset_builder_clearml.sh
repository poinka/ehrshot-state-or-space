#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CPU_QUEUE="${CPU_QUEUE:-cpu}"
REMOTE_CPU="${REMOTE_CPU:-0}"
cd "$PROJECT_ROOT"

REMOTE_ARGS=()
if [[ "$REMOTE_CPU" == "1" ]]; then
  REMOTE_ARGS+=(--execute-remotely)
fi

POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-4}" \
PYTHONUNBUFFERED=1 \
python final_exps/01_build_sequence_datasets.py \
  --run-config configs/state_or_space_sequence_datasets.json \
  --notebook-root . \
  --rebuild \
  --rebuild-cache \
  --enable-clearml \
  "${REMOTE_ARGS[@]}"
