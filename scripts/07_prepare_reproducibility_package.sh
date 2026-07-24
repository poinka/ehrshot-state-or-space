#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTPUT_DIR="${REPRO_OUTPUT_DIR:-state_or_space_reproducibility_package}"
TASK_NAME="${REPRO_TASK_NAME:-state_or_space_prepare_reproducibility_package}"
SOURCE_ROOT="${REPRO_SOURCE_ROOT:-}"
SKIP_UPLOAD="${SKIP_REPRO_UPLOAD:-0}"

: "${CLEARML_PROJECT:?Set CLEARML_PROJECT}"
: "${CLEARML_OUTPUT_URI:?Set CLEARML_OUTPUT_URI}"
: "${EHRSHOT_S3_BASE:?Set EHRSHOT_S3_BASE}"

cd "$PROJECT_ROOT"
mkdir -p logs reproducibility

ARGS=(
  --repo-root "$PROJECT_ROOT"
  --storage-base-s3-prefix "$EHRSHOT_S3_BASE"
  --output-dir "$OUTPUT_DIR"
  --output-s3-prefix "$EHRSHOT_S3_BASE/state_or_space_reproducibility_package"
  --clearml-project "$CLEARML_PROJECT"
  --clearml-output-uri "$CLEARML_OUTPUT_URI"
  --clearml-task-name "$TASK_NAME"
  --enable-clearml
  --clearml-upload-artifacts
)

if [[ -n "$SOURCE_ROOT" ]]; then
  ARGS+=(--source-root "$SOURCE_ROOT")
fi
if [[ -f reproducibility/clearml_tasks.csv ]]; then
  ARGS+=(--clearml-tasks-file reproducibility/clearml_tasks.csv)
fi
if [[ "$SKIP_UPLOAD" == "1" ]]; then
  ARGS+=(--skip-upload)
fi

export CLEARML_TASK_NO_REUSE="${CLEARML_TASK_NO_REUSE:-1}"
export PYTHONUNBUFFERED=1

python final_exps/06_prepare_final_reproducibility_package.py \
  "${ARGS[@]}" \
  2>&1 | tee "logs/reproducibility_package_$(date +%Y%m%d_%H%M%S).log"
