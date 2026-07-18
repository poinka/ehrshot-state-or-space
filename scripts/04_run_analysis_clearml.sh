#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CPU_QUEUE="${CPU_QUEUE:-cpu}"
REMOTE_ANALYSIS="${REMOTE_ANALYSIS:-1}"
cd "$PROJECT_ROOT"

REMOTE_ARGS=()
if [[ "$REMOTE_ANALYSIS" == "1" ]]; then
  REMOTE_ARGS+=(--execute-remotely --clearml-queue "$CPU_QUEUE")
fi

PYTHONUNBUFFERED=1 python final_exps/03_analyze_state_or_space.py \
  --predictions \
    ehrshot_state_or_space_final_sequence_results/sequence_multiseed_heldout_predictions.csv \
  --predictions-s3-url \
    's3://api.blackhole2.ai.innopolis.university:443/pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/ehrshot_state_or_space_final_sequence_results/sequence_multiseed_heldout_predictions.csv' \
  --sequence-data-dir ehrshot_state_or_space_sequence_datasets \
  --sequence-data-s3-prefix \
    's3://api.blackhole2.ai.innopolis.university:443/pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/ehrshot_state_or_space_sequence_datasets' \
  --analysis-config configs/state_or_space_analysis.json \
  --output-dir ehrshot_state_or_space_final_analysis \
  --n-bootstrap 10000 \
  --bootstrap-seed 42 \
  --enable-clearml \
  ${REMOTE_ARGS[@]+"${REMOTE_ARGS[@]}"}
