#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUT_DIR="${PROVENANCE_DIR:-reproducibility}"
cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR"

{
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty=$(if git diff --quiet --ignore-submodules HEAD 2>/dev/null; then echo false; else echo true; fi)"
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/git_state.txt"

python - <<'PY' > "$OUT_DIR/environment_snapshot.txt"
import importlib
import platform
import sys

print("python=" + sys.version.replace("\n", " "))
print("platform=" + platform.platform())
for name in ["numpy", "pandas", "polars", "pyarrow", "scipy", "sklearn", "torch", "clearml", "boto3", "matplotlib"]:
    try:
        module = importlib.import_module(name)
        print(f"{name}={getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        print(f"{name}=ERROR:{exc!r}")
PY

python - "$OUT_DIR/clearml_tasks.csv" <<'PY'
import csv
import os
import sys
from pathlib import Path

rows = [
    ("whitelist", "CLEARML_TASK_WHITELIST", "state_or_space_whitelist"),
    ("sequence_dataset_builder", "CLEARML_TASK_DATASET_BUILDER", "ehrshot_state_or_space_sequence_datasets"),
    ("training_core_4096", "CLEARML_TASK_CORE_4096", "ehrshot_state_or_space_final_sequence_results/core_4096"),
    ("training_context_16384", "CLEARML_TASK_CONTEXT_16384", "ehrshot_state_or_space_final_sequence_results/context_16384"),
    ("training_icu_gap", "CLEARML_TASK_ICU_GAPS", "ehrshot_state_or_space_final_sequence_results/icu_gap_extra_30_180"),
    ("training_seeds_45_46", "CLEARML_TASK_SEEDS_45_46", "ehrshot_state_or_space_final_sequence_results/additional_seeds_45_46_all"),
    ("main_analysis", "CLEARML_TASK_MAIN_ANALYSIS", "ehrshot_state_or_space_final_analysis_5seeds_wide"),
    ("copy_forward_inference", "CLEARML_TASK_COPY_FORWARD_INFERENCE", "state_or_space_copy_forward/inference"),
    ("copy_forward_analysis", "CLEARML_TASK_COPY_FORWARD_ANALYSIS", "state_or_space_copy_forward/analysis"),
    ("reproducibility_package", "CLEARML_TASK_REPRO_PACKAGE", "state_or_space_reproducibility_package"),
]
path = Path(sys.argv[1])
with path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["stage", "task_id", "task_id_env", "storage_relative_path"])
    writer.writeheader()
    for stage, env_name, storage in rows:
        writer.writerow({
            "stage": stage,
            "task_id": os.environ.get(env_name, ""),
            "task_id_env": env_name,
            "storage_relative_path": storage,
        })
print(path)
PY

echo "Provenance written to $OUT_DIR"
