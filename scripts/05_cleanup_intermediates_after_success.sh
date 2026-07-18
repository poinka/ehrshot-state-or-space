#!/usr/bin/env bash
set -euo pipefail

# Run only after representation_invariants.csv contains no failed checks
# and all dataset files are uploaded to storage.

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("ehrshot_state_or_space_sequence_datasets/representation_invariants.csv")
if not path.exists():
    raise SystemExit(f"Missing {path}")
df = pd.read_csv(path)
if "passed" not in df.columns:
    raise SystemExit("representation_invariants.csv has no passed column")
failed = df[~df["passed"].astype(bool)]
if len(failed):
    raise SystemExit(f"Cannot clean: {len(failed)} invariant checks failed")
print(f"All {len(df)} invariant checks passed")
PY

find ehrshot_state_or_space_sequence_datasets \
  -type d -name '_long_parts' -prune -exec rm -rf {} +

rm -rf ehrshot_state_or_space_cache

echo "Intermediate long parts and local cache removed."
