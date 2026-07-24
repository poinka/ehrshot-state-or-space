#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTPUT_DIR="${COPY_FORWARD_OUTPUT_DIR:-ehrshot_copy_forward_perturbation_mps}"
TASK_NAME="${COPY_FORWARD_TASK_NAME:-state_or_space_artificial_copy_forward_mps_local}"
REUSE_PERTURBED="${REUSE_PERTURBED:-1}"
BASELINE_LOGIT_TOLERANCE="${BASELINE_LOGIT_TOLERANCE:-999}"
UPLOAD_STABLE_STORAGE="${UPLOAD_STABLE_STORAGE:-1}"

: "${CLEARML_PROJECT:?Set CLEARML_PROJECT}"
: "${CLEARML_OUTPUT_URI:?Set CLEARML_OUTPUT_URI}"
: "${EHRSHOT_S3_BASE:?Set EHRSHOT_S3_BASE}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" checkpoints logs

REUSE_ARGS=()
if [[ "$REUSE_PERTURBED" == "1" ]]; then
  REUSE_ARGS+=(--reuse-perturbed-sequences)
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export CLEARML_TASK_NO_REUSE="${CLEARML_TASK_NO_REUSE:-1}"
export PYTHONUNBUFFERED=1

python final_exps/04_artificial_copy_forward_inference.py \
  --dataset-config configs/state_or_space_sequence_datasets.json \
  --run-config configs/state_or_space_core_4096_runs.json \
  --run-config configs/state_or_space_additional_seeds_45_46_runs.json \
  --builder-script final_exps/01_build_sequence_datasets.py \
  --trainer-script final_exps/02_train_sequence_multiseed.py \
  --sequence-data-dir ehrshot_state_or_space_sequence_datasets \
  --sequence-data-s3-prefix \
    "$EHRSHOT_S3_BASE/ehrshot_state_or_space_sequence_datasets" \
  --storage-base-s3-prefix "$EHRSHOT_S3_BASE" \
  --ehrshot-s3-prefix "$EHRSHOT_S3_BASE/EHRSHOT_MEDS" \
  --whitelist-s3-prefix "$EHRSHOT_S3_BASE/state_or_space_whitelist" \
  --sequence-cache-s3-prefix \
    "$EHRSHOT_S3_BASE/ehrshot_state_or_space_sequence_datasets/_cache" \
  --checkpoint-dir checkpoints \
  --checkpoint-s3-prefix "$EHRSHOT_S3_BASE/checkpoints" \
  --baseline-predictions \
    ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/sequence_multiseed_heldout_predictions_wide.csv \
  --baseline-predictions-s3-url \
    "$EHRSHOT_S3_BASE/ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/sequence_multiseed_heldout_predictions_wide.csv" \
  --tasks guo_readmission,guo_icu \
  --compression-versions raw_4096,condition_era_90_backfill_4096 \
  --copy-fractions 0,0.25,0.5,1 \
  --min-existing-visits 2 \
  --selection-seed 20260722 \
  --output-dir "$OUTPUT_DIR" \
  "${REUSE_ARGS[@]}" \
  --device mps \
  --batch-size 8 \
  --num-workers 0 \
  --baseline-logit-tolerance "$BASELINE_LOGIT_TOLERANCE" \
  --allow-baseline-mismatch \
  --bootstrap 10000 \
  --bootstrap-seed 42 \
  --enable-clearml \
  --clearml-project "$CLEARML_PROJECT" \
  --clearml-output-uri "$CLEARML_OUTPUT_URI" \
  --clearml-task-name "$TASK_NAME" \
  --clearml-tags inference-only,copy-forward,stress-test,mps,local,wide-5seeds \
  --clearml-upload-artifacts \
  2>&1 | tee "logs/copy_forward_mps_$(date +%Y%m%d_%H%M%S).log"

if [[ "$UPLOAD_STABLE_STORAGE" == "1" ]]; then
  python - "$OUTPUT_DIR" "$EHRSHOT_S3_BASE/state_or_space_copy_forward/inference" <<'PY'
from pathlib import Path
import sys
from clearml import StorageManager

local_root = Path(sys.argv[1])
remote_root = sys.argv[2].rstrip("/")
allowed = {".csv", ".json"}
for path in sorted(local_root.iterdir()):
    if not path.is_file() or path.suffix.lower() not in allowed:
        continue
    remote = f"{remote_root}/{path.stem}/{path.name}"
    print(f"Upload stable artifact: {path} -> {remote}")
    StorageManager.upload_file(
        local_file=str(path),
        remote_url=remote,
        wait_for_upload=True,
    )
PY
fi
