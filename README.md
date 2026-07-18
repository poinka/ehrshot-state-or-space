# State or Space — восстановленный reproducibility pipeline

Пакет восстанавливает финальный пайплайн от исходного `EHRSHOT_MEDS` до
обучения моделей, checkpoints, calibrated predictions и paired patient-level
bootstrap.

## Структура

```text
EHRSHOT_MEDS/
  data/data.parquet
  metadata/subject_splits.parquet
  metadata/codes.parquet                  # желательно, но не обязательно
  benchmark/guo_readmission/labels.parquet
  benchmark/guo_icu/labels.parquet

final_exps/
  00_build_train_only_persistent_whitelist.py
  01_build_sequence_datasets.py
  02_train_sequence_multiseed.py
  03_analyze_state_or_space.py
  common_ehrshot_eval.py

configs/
  state_or_space_sequence_datasets.json
  state_or_space_final_sequence_runs.json
  state_or_space_analysis.json

scripts/
  00_check_environment.sh
  01_run_whitelist_clearml.sh
  02_run_dataset_builder_clearml.sh
  03_run_training_clearml.sh
  04_run_analysis_clearml.sh
  05_cleanup_intermediates_after_success.sh
```

## Зафиксированные решения

- Train-only whitelist устойчивых клинических концептов.
- Правило whitelist:

```python
n_subjects_with_code >= 50
and repeat_day_subject_share >= 0.50
and (
    persistent_365d_subject_share >= 0.10
    or p75_span_days >= 365
)
```

- Ожидаемое число концептов для использовавшегося snapshot: `239`.
- Главные задачи и архитектуры:
  - `guo_readmission` → `RETAIN_lite_numeric`;
  - `guo_icu` → `GRU_2L_numeric`.
- Seeds: `42, 43, 44`.
- Архитектура/обучение:
  - token embedding + Time2Vec + numeric projection;
  - embedding 64, hidden 128, dropout 0.20;
  - AdamW, lr `1e-3`, weight decay `1e-4`;
  - weighted BCE, gradient clipping `1.0`;
  - максимум 12 epochs, patience 3;
  - early stopping по tuning AUPRC;
  - Platt calibration на tuning logits.
- Контексты: `4096` и `16384`.
- ICU sensitivity: gap `30/90/180` дней.
- Main point estimate: среднее calibrated prediction по трём seeds.
- Paired bootstrap: patient-level, 10 000 повторов, seed 42.

## Важный cutoff

Текущий конфиг реализует последнее требование автоматических проверок:

```text
event_time < prediction_time
```

Поэтому в `configs/state_or_space_sequence_datasets.json` установлено:

```json
"include_prediction_time": false,
"require_strict_before_prediction_time": true
```

Это строже, чем ранее использованный creator-compatible вариант
`event_time <= prediction_time`. Не смешивайте результаты, построенные с разными
cutoff, в одной итоговой таблице.

## Хранилище

Файлы записываются как обычное дерево объектов в:

```text
.../EHRSHOT/state_or_space_whitelist/
.../EHRSHOT/ehrshot_state_or_space_sequence_datasets/
.../EHRSHOT/ehrshot_state_or_space_final_sequence_results/
.../EHRSHOT/checkpoints/
.../EHRSHOT/ehrshot_state_or_space_final_analysis/
```

Dataset builder не загружает `_cache` и `_long_parts`.
Checkpoints сохраняются локально в `checkpoints/` и после каждого seed
загружаются в storage.

## 1. Установка

```bash
python -m venv ehrshot
source ehrshot/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
clearml-init
```

## 2. Проверка

```bash
bash scripts/00_check_environment.sh
```

## 3. Whitelist

По умолчанию выполняется в текущем контейнере, но логируется как ClearML task и
загружает результаты в storage:

```bash
bash scripts/01_run_whitelist_clearml.sh
```

Ожидаемый итог:

```text
Selected codes: 239
```

Для remote CPU запуска MEDS должен быть доступен worker по тому же пути:

```bash
REMOTE_CPU=1 CPU_QUEUE=cpu bash scripts/01_run_whitelist_clearml.sh
```

## 4. Sequence datasets

```bash
bash scripts/02_run_dataset_builder_clearml.sh
```

Builder создаёт 14 task×representation datasets и завершает работу с ошибкой,
если нарушен хотя бы один invariant для `no_backfill` или `structure_null`.

Перед обучением проверьте:

```bash
python - <<'PY'
import pandas as pd
p = "ehrshot_state_or_space_sequence_datasets/representation_invariants.csv"
df = pd.read_csv(p)
print(df["passed"].value_counts(dropna=False))
assert df["passed"].astype(bool).all()
PY
```

## 5. Обучение в ClearML GPU queue

```bash
GPU_QUEUE=gpu_40 bash scripts/03_run_training_clearml.sh
```

Будет одна ClearML task с:

```text
14 model configs × 3 seeds = 42 обучения
```

На старте remote worker обязательно должен показать:

```text
DEVICE: cuda
N RUN CONFIGS: 14
SEEDS: [42, 43, 44]
```

Batch sizes в JSON:

- 4096 → 8;
- 16384 → 2.

При CUDA OOM уменьшайте одинаково внутри одного context:

- 4096: `8 → 4`;
- 16384: `2 → 1`.

Научные параметры модели при этом не меняются.

## 6. Финальный анализ

После завершения training task:

```bash
CPU_QUEUE=cpu bash scripts/04_run_analysis_clearml.sh
```

Скрипт скачает aggregate predictions из storage, если локального файла нет.
Основные сравнения:

1. `Full4096 = backfill4096 − raw4096`;
2. `RepresentationOnly4096 = no_backfill4096 − raw4096`;
3. `Backfill4096 = backfill4096 − no_backfill4096`;
4. `StateFeatures4096 = backfill4096 − structure_null4096`;
5. `ContextInteraction = Full16384 − Full4096`;
6. ICU gap `30/90/180`.

## 7. Очистка временных данных

Только после успешной загрузки datasets в storage и прохождения всех checks:

```bash
bash scripts/05_cleanup_intermediates_after_success.sh
```

## Полный запуск до отправки GPU task

```bash
bash scripts/run_pipeline_until_training_submission.sh
```

Analysis запускается отдельно после окончания обучения.
