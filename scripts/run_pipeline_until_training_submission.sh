#!/usr/bin/env bash
set -euo pipefail

bash scripts/00_check_environment.sh
bash scripts/01_run_whitelist_clearml.sh
bash scripts/02_run_dataset_builder_clearml.sh
bash scripts/03_run_training_clearml.sh

echo "Training task submitted to ClearML. Run scripts/04_run_analysis_clearml.sh after it finishes."
