#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

: "${CLEARML_PROJECT:?Set CLEARML_PROJECT}"
: "${CLEARML_OUTPUT_URI:?Set CLEARML_OUTPUT_URI}"
: "${EHRSHOT_S3_BASE:?Set EHRSHOT_S3_BASE}"
: "${GPU_QUEUE:?Set GPU_QUEUE}"
: "${RUN_CONFIG:?Set RUN_CONFIG}"

RUN_TAG="${RUN_TAG:-$(basename "$RUN_CONFIG" .json)}"

OUTPUT_DIR="$PROJECT_ROOT/ehrshot_state_or_space_final_sequence_results/$RUN_TAG"

SEQUENCE_S3_PREFIX="$EHRSHOT_S3_BASE/ehrshot_state_or_space_sequence_datasets"
RESULTS_S3_PREFIX="$EHRSHOT_S3_BASE/ehrshot_state_or_space_final_sequence_results/$RUN_TAG"
CHECKPOINT_S3_PREFIX="$EHRSHOT_S3_BASE/checkpoints"

mkdir -p "$OUTPUT_DIR" checkpoints logs

PYTHONUNBUFFERED=1 \
python final_exps/02_train_sequence_multiseed.py \
  --run-config "$RUN_CONFIG" \
  --sequence-data-dir ehrshot_state_or_space_sequence_datasets \
  --sequence-data-s3-prefix "$SEQUENCE_S3_PREFIX" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-dir checkpoints \
  --results-s3-prefix "$RESULTS_S3_PREFIX" \
  --checkpoint-s3-prefix "$CHECKPOINT_S3_PREFIX" \
  --device cuda \
  --num-workers 0 \
  --epochs 12 \
  --patience 3 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --grad-clip 1.0 \
  --emb-dim 64 \
  --hidden-dim 128 \
  --dropout 0.20 \
  --numeric-min-count 3 \
  --resume \
  --no-progress-bars \
  --progress-every-n-batches 10 \
  --enable-clearml \
  --execute-remotely \
  --clearml-queue "$GPU_QUEUE" \
  --clearml-project "$CLEARML_PROJECT" \
  --clearml-output-uri "$CLEARML_OUTPUT_URI" \
  --clearml-task-name "state_or_space_${RUN_TAG}"
