#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CPU_QUEUE="${CPU_QUEUE:-cpu}"
REMOTE_ANALYSIS="${REMOTE_ANALYSIS:-1}"
ANALYSIS_TASK_NAME="${ANALYSIS_TASK_NAME:-state_or_space_final_analysis_5seeds_wide}"
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-ehrshot_state_or_space_final_analysis_5seeds_wide}"

: "${CLEARML_PROJECT:?Set CLEARML_PROJECT}"
: "${CLEARML_OUTPUT_URI:?Set CLEARML_OUTPUT_URI}"
: "${EHRSHOT_S3_BASE:?Set EHRSHOT_S3_BASE}"

cd "$PROJECT_ROOT"
mkdir -p "$ANALYSIS_OUTPUT_DIR" \
  ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide \
  logs

REMOTE_ARGS=()
if [[ "$REMOTE_ANALYSIS" == "1" ]]; then
  REMOTE_ARGS+=(--execute-remotely --clearml-queue "$CPU_QUEUE")
fi

export CLEARML_TASK_NO_REUSE="${CLEARML_TASK_NO_REUSE:-1}"
export PYTHONUNBUFFERED=1

python final_exps/03_analyze_state_or_space.py \
  --predictions \
    ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/sequence_multiseed_heldout_predictions_wide.csv \
  --predictions-s3-url \
    "$EHRSHOT_S3_BASE/ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/sequence_multiseed_heldout_predictions_wide.csv" \
  --prediction-run-tags \
    core_4096_wide,context_16384_wide,icu_gap_extra_30_180_wide,additional_seeds_45_46_all_wide \
  --prediction-results-dir \
    ehrshot_state_or_space_final_sequence_results \
  --prediction-results-s3-root \
    "$EHRSHOT_S3_BASE/ehrshot_state_or_space_final_sequence_results" \
  --prediction-filename \
    sequence_multiseed_heldout_predictions_wide.csv \
  --combined-predictions-s3-url \
    "$EHRSHOT_S3_BASE/ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/sequence_multiseed_heldout_predictions_wide.csv" \
  --sequence-data-dir \
    ehrshot_state_or_space_sequence_datasets \
  --sequence-data-s3-prefix \
    "$EHRSHOT_S3_BASE/ehrshot_state_or_space_sequence_datasets" \
  --analysis-config \
    configs/state_or_space_analysis_5seeds.json \
  --output-dir \
    "$ANALYSIS_OUTPUT_DIR" \
  --output-s3-prefix \
    "$EHRSHOT_S3_BASE/ehrshot_state_or_space_final_analysis_5seeds_wide" \
  --n-bootstrap 10000 \
  --bootstrap-seed 42 \
  --enable-clearml \
  --clearml-project "$CLEARML_PROJECT" \
  --clearml-output-uri "$CLEARML_OUTPUT_URI" \
  --clearml-task-name "$ANALYSIS_TASK_NAME" \
  "${REMOTE_ARGS[@]}" \
  2>&1 | tee "logs/main_analysis_$(date +%Y%m%d_%H%M%S).log"