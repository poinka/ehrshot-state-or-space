#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
import sys

required = [
    Path("EHRSHOT_MEDS/data/data.parquet"),
    Path("EHRSHOT_MEDS/metadata/subject_splits.parquet"),
    Path("final_exps/00_build_train_only_persistent_whitelist.py"),
    Path("final_exps/01_build_sequence_datasets.py"),
    Path("final_exps/02_train_sequence_multiseed.py"),
    Path("final_exps/03_analyze_state_or_space.py"),
    Path("configs/state_or_space_sequence_datasets.json"),
    Path("configs/state_or_space_final_sequence_runs.json"),
    Path("configs/state_or_space_analysis.json"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing files:\n" + "\n".join(missing))
print("Required project files: OK")
print("Python:", sys.version)
PY

python - <<'PY'
import importlib
mods = ["numpy", "pandas", "polars", "pyarrow", "sklearn", "torch", "clearml"]
for name in mods:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'installed')}")

import torch
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -m py_compile \
  final_exps/00_build_train_only_persistent_whitelist.py \
  final_exps/01_build_sequence_datasets.py \
  final_exps/02_train_sequence_multiseed.py \
  final_exps/03_analyze_state_or_space.py \
  final_exps/common_ehrshot_eval.py

python -m json.tool configs/state_or_space_sequence_datasets.json >/dev/null
python -m json.tool configs/state_or_space_final_sequence_runs.json >/dev/null
python -m json.tool configs/state_or_space_analysis.json >/dev/null

echo "Environment and configs: OK"
