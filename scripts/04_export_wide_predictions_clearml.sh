#!/usr/bin/env bash
set -euo pipefail

: "${CLEARML_PROJECT:=pershin-medailab/EHR_Risk_Profiling/EHRSHOT}"
: "${CLEARML_OUTPUT_URI:=s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab}"
: "${EHRSHOT_S3_BASE:=s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT}"
: "${CPU_QUEUE:=cpu}"

export CLEARML_PROJECT CLEARML_OUTPUT_URI EHRSHOT_S3_BASE CPU_QUEUE
export CLEARML_TASK_NO_REUSE="1"
export POLARS_MAX_THREADS="4"

run_export() {
  local run_config="$1"
  local source_tag="$2"
  local destination_tag="$3"

  echo "===================================================================================================="
  echo "EXPORT WIDE PREDICTIONS: ${source_tag} -> ${destination_tag}"
  echo "RUN CONFIG: ${run_config}"
  echo "===================================================================================================="

  python final_exps/02_train_sequence_multiseed.py \
    --run-config "${run_config}" \
    --sequence-data-dir ehrshot_state_or_space_sequence_datasets \
    --sequence-data-s3-prefix "${EHRSHOT_S3_BASE}/ehrshot_state_or_space_sequence_datasets" \
    --output-dir "ehrshot_state_or_space_final_sequence_results/${destination_tag}" \
    --checkpoint-dir checkpoints \
    --checkpoint-s3-prefix "${EHRSHOT_S3_BASE}/checkpoints" \
    --source-results-s3-prefix "${EHRSHOT_S3_BASE}/ehrshot_state_or_space_final_sequence_results/${source_tag}" \
    --results-s3-prefix "${EHRSHOT_S3_BASE}/ehrshot_state_or_space_final_sequence_results/${destination_tag}" \
    --resume \
    --export-wide-only \
    --device cpu \
    --no-progress-bars \
    --enable-clearml \
    --execute-remotely \
    --clearml-queue "${CPU_QUEUE}" \
    --clearml-project "${CLEARML_PROJECT}" \
    --clearml-output-uri "${CLEARML_OUTPUT_URI}" \
    --clearml-task-name "state_or_space_export_wide_${destination_tag}"
}

CORE_CONFIG="configs/state_or_space_core_4096_runs.json"
CONTEXT_CONFIG="configs/state_or_space_context16384_runs.json"
if [[ ! -f "${CONTEXT_CONFIG}" && -f "configs/state_or_space_context_16384_runs.json" ]]; then
  CONTEXT_CONFIG="configs/state_or_space_context_16384_runs.json"
fi
GAP_CONFIG="configs/state_or_space_icu_gap_extra_runs.json"
ADDITIONAL_CONFIG="configs/state_or_space_additional_seeds_45_46_runs.json"

for required_config in "${CORE_CONFIG}" "${CONTEXT_CONFIG}" "${GAP_CONFIG}" "${ADDITIONAL_CONFIG}"; do
  if [[ ! -f "${required_config}" ]]; then
    echo "Missing run config: ${required_config}" >&2
    exit 1
  fi
done

run_export "${CORE_CONFIG}" "core_4096" "core_4096_wide"
run_export "${CONTEXT_CONFIG}" "context_16384" "context_16384_wide"
run_export "${GAP_CONFIG}" "icu_gap_extra_30_180" "icu_gap_extra_30_180_wide"
run_export "${ADDITIONAL_CONFIG}" "additional_seeds_45_46_all" "additional_seeds_45_46_all_wide"

cat <<'MSG'
All four ClearML export-only tasks were submitted.
They must not print "starting training loop" or any epoch lines.
Each destination folder must contain:
  sequence_multiseed_heldout_predictions_wide.csv
  sequence_multiseed_tuning_predictions_wide.csv
  sequence_multiseed_predictions_wide.csv
Only wide prediction CSVs are published to the new *_wide S3 prefixes.
MSG
