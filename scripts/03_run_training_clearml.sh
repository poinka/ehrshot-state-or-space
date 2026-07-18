#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
GPU_QUEUE="${GPU_QUEUE:-gpu_40}"
cd "$PROJECT_ROOT"

mkdir -p checkpoints ehrshot_state_or_space_final_sequence_results logs

PYTHONUNBUFFERED=1 \
python final_exps/02_train_sequence_multiseed.py \
  --run-config configs/state_or_space_final_sequence_runs.json \
  --sequence-data-dir ehrshot_state_or_space_sequence_datasets \
  --sequence-data-s3-prefix \
    's3://api.blackhole2.ai.innopolis.university:443/pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/ehrshot_state_or_space_sequence_datasets' \
  --output-dir ehrshot_state_or_space_final_sequence_results \
  --checkpoint-dir checkpoints \
  --results-s3-prefix \
    's3://api.blackhole2.ai.innopolis.university:443/pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/ehrshot_state_or_space_final_sequence_results' \
  --checkpoint-s3-prefix \
    's3://api.blackhole2.ai.innopolis.university:443/pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT/checkpoints' \
  --device cuda \
  --num-workers 0 \
  --epochs 12 \
  --patience 3 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --grad-clip 1.0 \
  --emb-dim 64 \
  --hidden-dim 128 \
  --dropout 0.20 \
  --numeric-min-count 3 \
  --resume \
  --no-progress-bars \
  --progress-every-n-batches 10 \
  --enable-clearml \
  --execute-remotely \
  --clearml-queue "$GPU_QUEUE" \
  --clearml-task-name state_or_space_v1_final_sequence_models_strict
