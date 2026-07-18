#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CPU_QUEUE="${CPU_QUEUE:-cpu}"
REMOTE_CPU="${REMOTE_CPU:-0}"
cd "$PROJECT_ROOT"

REMOTE_ARGS=()
if [[ "$REMOTE_CPU" == "1" ]]; then
  REMOTE_ARGS+=(--execute-remotely --clearml-queue "$CPU_QUEUE")
fi

PYTHONUNBUFFERED=1 python final_exps/00_build_train_only_persistent_whitelist.py \
  --ehrshot-root EHRSHOT_MEDS \
  --output-dir ehrshot_train_only_chronic_whitelist_50 \
  --dataset-version EHRSHOT_MEDS_local \
  --expected-code-count 239 \
  --overwrite \
  --enable-clearml \
  ${REMOTE_ARGS[@]+"${REMOTE_ARGS[@]}"}
