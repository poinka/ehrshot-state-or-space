#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTPUT_DIR="${COPY_FORWARD_ANALYSIS_OUTPUT_DIR:-ehrshot_copy_forward_perturbation_mps/robustness_analysis}"
TASK_NAME="${COPY_FORWARD_ANALYSIS_TASK_NAME:-state_or_space_copy_forward_robustness_analysis_mps}"
UPLOAD_STABLE_STORAGE="${UPLOAD_STABLE_STORAGE:-1}"

: "${COPY_FORWARD_SOURCE_TASK_ID:?Set COPY_FORWARD_SOURCE_TASK_ID to the successful inference ClearML task ID}"
: "${CLEARML_PROJECT:?Set CLEARML_PROJECT}"
: "${CLEARML_OUTPUT_URI:?Set CLEARML_OUTPUT_URI}"
: "${EHRSHOT_S3_BASE:?Set EHRSHOT_S3_BASE}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" logs

export CLEARML_TASK_NO_REUSE="${CLEARML_TASK_NO_REUSE:-1}"
export PYTHONUNBUFFERED=1

python final_exps/05_analyze_copy_forward_robustness.py \
  --source-task-id "$COPY_FORWARD_SOURCE_TASK_ID" \
  --source-artifact-name copy_forward_ensemble_predictions \
  --output-dir "$OUTPUT_DIR" \
  --raw-version raw_4096 \
  --compressed-version condition_era_90_backfill_4096 \
  --top-fraction 0.10 \
  --bootstrap 10000 \
  --bootstrap-seed 42 \
  --enable-clearml \
  --clearml-project "$CLEARML_PROJECT" \
  --clearml-output-uri "$CLEARML_OUTPUT_URI" \
  --clearml-task-name "$TASK_NAME" \
  --clearml-tags analysis-only,copy-forward,robustness,mps,raw-vs-condition-era \
  --clearml-upload-artifacts \
  2>&1 | tee "logs/copy_forward_analysis_$(date +%Y%m%d_%H%M%S).log"

if [[ "$UPLOAD_STABLE_STORAGE" == "1" ]]; then
  python - "$OUTPUT_DIR" "$EHRSHOT_S3_BASE/state_or_space_copy_forward/analysis" <<'PY'
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
