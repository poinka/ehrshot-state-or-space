#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

: "${EHRSHOT_S3_BASE:=s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT}"
: "${PUBLIC_RESULTS_OUTPUT_DIR:=public_results}"
: "${PUBLIC_RESULTS_S3_PREFIX:=}"
: "${PUBLIC_RESULTS_ENABLE_CLEARML:=1}"

mkdir -p logs

args=(
  --storage-base-s3-prefix "$EHRSHOT_S3_BASE"
  --output-dir "$PUBLIC_RESULTS_OUTPUT_DIR"
)

# For an offline/local test, point this to either an EHRSHOT artifact root or an
# extracted state_or_space_reproducibility_package directory.
if [[ -n "${PUBLIC_RESULTS_SOURCE_ROOT:-}" ]]; then
  args+=(--source-root "$PUBLIC_RESULTS_SOURCE_ROOT")
fi

# Public outputs may optionally be mirrored to a separate stable MinIO prefix.
if [[ -n "$PUBLIC_RESULTS_S3_PREFIX" ]]; then
  args+=(--output-s3-prefix "$PUBLIC_RESULTS_S3_PREFIX")
fi

if [[ "$PUBLIC_RESULTS_ENABLE_CLEARML" == "1" ]]; then
  args+=(
    --enable-clearml
    --clearml-project "${CLEARML_PROJECT:-pershin-medailab/EHR_Risk_Profiling/EHRSHOT}"
    --clearml-output-uri "${CLEARML_OUTPUT_URI:-s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab}"
    --clearml-task-name "state_or_space_prepare_public_results"
  )
fi

# The directory is intended for Git. A ZIP is usually unnecessary; set
# PUBLIC_RESULTS_MAKE_ZIP=1 only when a separate archive is required.
if [[ "${PUBLIC_RESULTS_MAKE_ZIP:-0}" == "1" ]]; then
  args+=(--make-zip)
fi

python final_exps/07_prepare_public_results.py "${args[@]}" \
  2>&1 | tee "logs/prepare_public_results_$(date +%Y%m%d_%H%M%S).log"
