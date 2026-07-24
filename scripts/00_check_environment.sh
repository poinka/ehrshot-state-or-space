#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
REQUIRE_LOCAL_DATA="${REQUIRE_LOCAL_DATA:-0}"
CHECK_STORAGE="${CHECK_STORAGE:-0}"
cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
import json
import sys

required = [
    Path("final_exps/00_build_train_only_persistent_whitelist.py"),
    Path("final_exps/01_build_sequence_datasets.py"),
    Path("final_exps/02_train_sequence_multiseed.py"),
    Path("final_exps/03_analyze_state_or_space.py"),
    Path("final_exps/04_artificial_copy_forward_inference.py"),
    Path("final_exps/05_analyze_copy_forward_robustness.py"),
    Path("final_exps/06_prepare_final_reproducibility_package.py"),
    Path("final_exps/07_prepare_public_results.py"),
    Path("final_exps/common_ehrshot_eval.py"),
    Path("scripts/00_check_environment.sh"),
    Path("scripts/01_run_whitelist_clearml.sh"),
    Path("scripts/02_run_dataset_builder_clearml.sh"),
    Path("scripts/03_run_training_clearml.sh"),
    Path("scripts/04_run_analysis_clearml.sh"),
    Path("scripts/05_run_copy_forward_inference_mps.sh"),
    Path("scripts/06_run_copy_forward_analysis.sh"),
    Path("scripts/07_prepare_reproducibility_package.sh"),
    Path("scripts/08_record_provenance.sh"),
    Path("scripts/09_prepare_public_results.sh"),
    Path("configs/state_or_space_sequence_datasets.json"),
    Path("configs/state_or_space_core_4096_runs.json"),
    Path("configs/state_or_space_context_16384_runs.json"),
    Path("configs/state_or_space_icu_gap_extra_runs.json"),
    Path("configs/state_or_space_additional_seeds_45_46_runs.json"),
    Path("configs/state_or_space_analysis_5seeds.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing required project files:\n" + "\n".join(missing))

with Path("configs/state_or_space_sequence_datasets.json").open(encoding="utf-8") as fh:
    dataset_cfg = json.load(fh)
build = dataset_cfg["build"]
if build.get("include_prediction_time") is not True:
    raise SystemExit("Final protocol requires include_prediction_time=true")
if build.get("require_strict_before_prediction_time") is not False:
    raise SystemExit("Final protocol requires require_strict_before_prediction_time=false")
print("Prediction-time rule: event_time <= prediction_time")

run_configs = [
    "configs/state_or_space_core_4096_runs.json",
    "configs/state_or_space_context_16384_runs.json",
    "configs/state_or_space_icu_gap_extra_runs.json",
    "configs/state_or_space_additional_seeds_45_46_runs.json",
]
seen = set()
planned = 0
for raw_path in run_configs:
    path = Path(raw_path)
    with path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    seeds = [int(seed) for seed in cfg["seeds"]]
    for run in cfg["runs"]:
        task = str(run["task"])
        version = str(run["compression_version"])
        model = str(run["model_name"])
        for seed in seeds:
            key = (task, version, model, seed)
            if key in seen:
                raise SystemExit(f"Duplicate planned run across final configs: {key}")
            seen.add(key)
            planned += 1
if planned != 70:
    raise SystemExit(f"Expected 70 final model runs, found {planned}")
print("Final experiment matrix: 70 unique model runs")

with Path("configs/state_or_space_analysis_5seeds.json").open(encoding="utf-8") as fh:
    analysis_cfg = json.load(fh)
if [int(seed) for seed in analysis_cfg["seeds"]] != [42, 43, 44, 45, 46]:
    raise SystemExit("Final analysis config must use seeds 42,43,44,45,46")
print("Final analysis seeds: 42, 43, 44, 45, 46")
print("Python:", sys.version.replace("\n", " "))
PY

python - <<'PY'
import importlib
import platform

modules = [
    "numpy", "pandas", "polars", "pyarrow", "scipy", "sklearn",
    "torch", "tqdm", "matplotlib", "clearml", "boto3",
]
for name in modules:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'installed')}")

import torch
print("Platform:", platform.platform())
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA GPU:", torch.cuda.get_device_name(0))
print("MPS built:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()))
print("MPS available:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
PY

python -m py_compile \
  final_exps/00_build_train_only_persistent_whitelist.py \
  final_exps/01_build_sequence_datasets.py \
  final_exps/02_train_sequence_multiseed.py \
  final_exps/03_analyze_state_or_space.py \
  final_exps/04_artificial_copy_forward_inference.py \
  final_exps/05_analyze_copy_forward_robustness.py \
  final_exps/06_prepare_final_reproducibility_package.py \
  final_exps/07_prepare_public_results.py \
  final_exps/common_ehrshot_eval.py

for config in \
  configs/state_or_space_sequence_datasets.json \
  configs/state_or_space_core_4096_runs.json \
  configs/state_or_space_context_16384_runs.json \
  configs/state_or_space_icu_gap_extra_runs.json \
  configs/state_or_space_additional_seeds_45_46_runs.json \
  configs/state_or_space_analysis_5seeds.json; do
  python -m json.tool "$config" >/dev/null
done

for script in \
  scripts/00_check_environment.sh \
  scripts/01_run_whitelist_clearml.sh \
  scripts/02_run_dataset_builder_clearml.sh \
  scripts/03_run_training_clearml.sh \
  scripts/04_run_analysis_clearml.sh \
  scripts/05_run_copy_forward_inference_mps.sh \
  scripts/06_run_copy_forward_analysis.sh \
  scripts/07_prepare_reproducibility_package.sh \
  scripts/08_record_provenance.sh \
  scripts/09_prepare_public_results.sh; do
  bash -n "$script"
done

if [[ -d public_results ]]; then
  python - <<'PY_PUBLIC'
from pathlib import Path
import pandas as pd

root = Path("public_results")
required = [
    root / "PUBLICATION_SAFETY_MANIFEST.csv",
    root / "checks/publication_safety_and_integrity.csv",
    root / "figures/figure_1_primary_comparisons_forest.png",
    root / "figures/figure_2_copy_forward_probability_stability.png",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Incomplete public_results directory:\n" + "\n".join(missing))

safety = pd.read_csv(root / "PUBLICATION_SAFETY_MANIFEST.csv")
if not safety["safe"].astype(bool).all():
    bad = safety.loc[~safety["safe"].astype(bool)]
    raise SystemExit("Unsafe public result files:\n" + bad.to_string(index=False))

checks = pd.read_csv(root / "checks/publication_safety_and_integrity.csv")
if not checks["status"].eq("PASS").all():
    bad = checks.loc[~checks["status"].eq("PASS")]
    raise SystemExit("Failed public result checks:\n" + bad.to_string(index=False))
print(f"Public results: {len(safety)} publication-safe files; all integrity checks passed")
PY_PUBLIC
else
  echo "public_results/ is absent; publication export check skipped."
fi

if [[ "$REQUIRE_LOCAL_DATA" == "1" ]]; then
  for path in \
    EHRSHOT_MEDS/data/data.parquet \
    EHRSHOT_MEDS/metadata/subject_splits.parquet \
    EHRSHOT_MEDS/labels/guo_readmission/labels.parquet \
    EHRSHOT_MEDS/labels/guo_icu/labels.parquet; do
    [[ -f "$path" ]] || { echo "Missing local data file: $path" >&2; exit 1; }
  done
  echo "Local EHRSHOT MEDS inputs: OK"
else
  echo "Local EHRSHOT MEDS inputs were not required (REQUIRE_LOCAL_DATA=0)."
fi

if [[ "$CHECK_STORAGE" == "1" ]]; then
  : "${EHRSHOT_S3_BASE:?Set EHRSHOT_S3_BASE when CHECK_STORAGE=1}"
  python - <<'PY'
import os
from clearml import StorageManager
remote = os.environ["EHRSHOT_S3_BASE"].rstrip("/") + "/state_or_space_whitelist/build_metadata.json"
local = StorageManager.get_local_copy(remote_url=remote)
if not local:
    raise SystemExit(f"Could not read storage object: {remote}")
print("Storage access: OK", remote)
PY
fi

echo "Environment, final configs and protocol boundary: OK"