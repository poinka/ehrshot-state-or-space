# State or Space? Persistence-aware compression for EHR risk prediction

Репозиторий содержит финальный воспроизводимый эксперимент по persistence-aware представлению продольной электронной медицинской карты в EHRSHOT.

Главный исследовательский вопрос:

> Модель получает пользу от явного представления продолжающегося клинического состояния или главным эффектом compression является освобождение ограниченного контекста?

Дополнительно проводится frozen-model stress test с искусственным copy-forward: уже известный устойчивый диагноз повторно добавляется в последующие визиты, после чего сравнивается устойчивость raw и `condition_era_90`.

## Зафиксированный протокол

- задачи:
  - `guo_readmission` — 30-дневная повторная госпитализация;
  - `guo_icu` — перевод в ICU;
- граница истории: **`event_time <= prediction_time`**;
- patient-level splits из официального EHRSHOT `subject_splits.parquet`;
- whitelist устойчивых диагнозов строится только по train split;
- модели:
  - `RETAIN_lite_numeric` для readmission;
  - `GRU_2L_numeric` для ICU;
- seeds: `42, 43, 44, 45, 46`;
- Platt calibration обучается только на tuning split;
- основной анализ выполняется только на held-out split;
- patient-cluster bootstrap: 10 000 повторений.

## Представления

Основной эксперимент включает 14 комбинаций task × representation:

- `raw_4096`;
- `condition_era_90_backfill_4096`;
- `condition_era_90_no_backfill_4096`;
- `condition_era_90_structure_null_4096`;
- `raw_16384`;
- `condition_era_90_backfill_16384`;
- для ICU дополнительно `condition_era_30_backfill_4096` и `condition_era_180_backfill_4096`.

Итого выполняется 70 обучений: 14 вариантов × 5 seed.

## Структура репозитория

```text
final_exps/
├── 00_build_train_only_persistent_whitelist.py
├── 01_build_sequence_datasets.py
├── 02_train_sequence_multiseed.py
├── 03_analyze_state_or_space.py
├── 04_artificial_copy_forward_inference.py
├── 05_analyze_copy_forward_robustness.py
├── 06_prepare_final_reproducibility_package.py
└── common_ehrshot_eval.py

configs/
├── state_or_space_sequence_datasets.json
├── state_or_space_core_4096_runs.json
├── state_or_space_context_16384_runs.json
├── state_or_space_icu_gap_extra_runs.json
├── state_or_space_additional_seeds_45_46_runs.json
└── state_or_space_analysis_5seeds.json

scripts/
├── 00_check_environment.sh
├── 01_run_whitelist_clearml.sh
├── 02_run_dataset_builder_clearml.sh
├── 03_run_training_clearml.sh
├── 04_run_analysis_clearml.sh
├── 05_run_copy_forward_inference_mps.sh
├── 06_run_copy_forward_analysis.sh
├── 07_prepare_reproducibility_package.sh
└── 08_record_provenance.sh
```

Устаревший общий pipeline намеренно отсутствует: этапы запускаются отдельно, а завершение каждого тяжёлого этапа проверяется по ClearML и MinIO.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
source .env
```

ClearML должен быть настроен на приватный сервер проекта. Секретные ключи не хранятся в репозитории.

## Проверка окружения

```bash
bash scripts/00_check_environment.sh
```

Для проверки локального EHRSHOT MEDS:

```bash
REQUIRE_LOCAL_DATA=1 bash scripts/00_check_environment.sh
```

Для проверки доступа к MinIO:

```bash
CHECK_STORAGE=1 bash scripts/00_check_environment.sh
```

Скрипт также проверяет:

- финальное правило `event_time <= prediction_time`;
- наличие шести финальных конфигов;
- отсутствие дублей в матрице из 70 запусков;
- seeds 42–46;
- импорт зависимостей и доступность CUDA/MPS;
- синтаксис всех финальных Python-скриптов.

## 1. Построение train-only whitelist

Локально с ClearML-логированием:

```bash
REMOTE_CPU=0 bash scripts/01_run_whitelist_clearml.sh
```

На CPU worker:

```bash
REMOTE_CPU=1 CPU_QUEUE=cpu bash scripts/01_run_whitelist_clearml.sh
```

Локальная папка сохраняется под именем `ehrshot_train_only_chronic_whitelist_50`, потому что этот путь зафиксирован в dataset config. В MinIO результат загружается в стабильный prefix:

```text
state_or_space_whitelist/
```

## 2. Построение sequence datasets

```bash
REMOTE_CPU=0 bash scripts/02_run_dataset_builder_clearml.sh
```

Полная пересборка cache:

```bash
REBUILD_CACHE=1 REMOTE_CPU=0 bash scripts/02_run_dataset_builder_clearml.sh
```

## 3. Обучение

`03_run_training_clearml.sh` отправляет один явно выбранный run config. Примеры:

```bash
RUN_CONFIG=configs/state_or_space_core_4096_runs.json \
RUN_TAG=core_4096 \
GPU_QUEUE=gpu \
bash scripts/03_run_training_clearml.sh
```

```bash
RUN_CONFIG=configs/state_or_space_context_16384_runs.json \
RUN_TAG=context_16384 \
GPU_QUEUE=gpu \
bash scripts/03_run_training_clearml.sh
```

```bash
RUN_CONFIG=configs/state_or_space_icu_gap_extra_runs.json \
RUN_TAG=icu_gap_extra_30_180 \
GPU_QUEUE=gpu \
bash scripts/03_run_training_clearml.sh
```

```bash
RUN_CONFIG=configs/state_or_space_additional_seeds_45_46_runs.json \
RUN_TAG=additional_seeds_45_46_all \
GPU_QUEUE=gpu \
bash scripts/03_run_training_clearml.sh
```

Каждая команда создаёт отдельную ClearML task. Это сделано намеренно: перезапуск одной группы не затрагивает остальные результаты.

## 4. Финальный 5-seed анализ в ClearML

```bash
REMOTE_ANALYSIS=1 \
CPU_QUEUE=cpu \
bash scripts/04_run_analysis_clearml.sh
```

Анализатор объединяет четыре wide run-группы:

```text
core_4096_wide
context_16384_wide
icu_gap_extra_30_180_wide
additional_seeds_45_46_all_wide
```

Перед расчётом он проверяет наличие всех 14 task/version пар и всех пяти seed. Канонический prediction-файл сохраняется в:

```text
ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/
sequence_multiseed_heldout_predictions_wide.csv
```

Основной результат анализа:

```text
ehrshot_state_or_space_final_analysis_5seeds_wide/
```

## 5. Artificial copy-forward inference на MPS

```bash
bash scripts/05_run_copy_forward_inference_mps.sh
```

Эксперимент не обучает модели. Он использует frozen checkpoints и добавляет один уже существующий устойчивый диагноз в 0%, 25%, 50% и 100% последующих подходящих визитов.

`zero_percent_baseline_agreement.csv` сохраняется как диагностическая проверка. Copy-forward deltas рассчитываются относительно 0% варианта, полученного в том же MPS-запуске.

После успешного запуска файлы дополнительно загружаются в стабильную структуру:

```text
state_or_space_copy_forward/inference/<artifact_name>/<filename>
```

## 6. Анализ copy-forward robustness

Сначала укажите ID успешной inference task:

```bash
export COPY_FORWARD_SOURCE_TASK_ID="<clearml-task-id>"
bash scripts/06_run_copy_forward_analysis.sh
```

Анализ сравнивает raw и `condition_era_90` по:

- изменению индивидуальных вероятностей;
- AUPRC, LogLoss и Brier;
- top-10% precision;
- retention, Jaccard и churn состава top-10%;
- patient-level bootstrap.

Стабильные артефакты сохраняются в:

```text
state_or_space_copy_forward/analysis/<artifact_name>/<filename>
```

## 7. Отслеживаемость и provenance

После завершения задач заполните task IDs в окружении и выполните:

```bash
bash scripts/08_record_provenance.sh
```

Будут созданы:

```text
reproducibility/git_state.txt
reproducibility/environment_snapshot.txt
reproducibility/clearml_tasks.csv
```

Task ID передаются через переменные из `.env.example`, а не зашиваются в Python-код.

## 8. Финальная таблица, два рисунка и reproducibility package

```bash
bash scripts/07_prepare_reproducibility_package.sh
```

Скрипт читает frozen artifacts из MinIO, выполняет финальные integrity checks и создаёт:

```text
state_or_space_reproducibility_package/
├── artifacts/
├── repository_snapshot/
├── tables/
│   ├── final_ensemble_results.csv
│   ├── final_primary_comparisons.csv
│   └── final_copy_forward_results.csv
├── figures/
│   ├── figure_1_primary_comparisons_forest.png
│   ├── figure_1_primary_comparisons_forest.pdf
│   ├── figure_2_copy_forward_probability_stability.png
│   └── figure_2_copy_forward_probability_stability.pdf
├── checks/
│   ├── final_integrity_summary.csv
│   ├── split_audit_status.csv
│   ├── representation_invariants.csv
│   └── zero_percent_baseline_agreement.csv
├── provenance/
├── storage_manifest.csv
├── package_manifest.json
└── SHA256SUMS.txt
```

Также создаётся ZIP и загружается в:

```text
state_or_space_reproducibility_package/
```

Для локальной проверки на ранее скачанном artifact root:

```bash
REPRO_SOURCE_ROOT=/path/to/EHRSHOT \
SKIP_REPRO_UPLOAD=1 \
bash scripts/07_prepare_reproducibility_package.sh
```

## Основные результаты

Полный persistence-aware вариант не показал статистически устойчивого общего превосходства над raw ни для readmission, ни для ICU.

Для ICU декомпозиция выявила два воспроизводимых механизма:

- state features улучшают AUROC, AUPRC, Brier и LogLoss относительно structure-null;
- backfill улучшает AUROC и LogLoss относительно no-backfill.

Copy-forward stress test оказался task-dependent:

- для readmission `condition_era_90` существенно устойчивее raw;
- для ICU в пятиseedовом ensemble raw устойчивее по изменению вероятностей и составу top-10%, хотя `condition_era_90` сохраняет более высокое абсолютное качество.

## Приватные данные

В публичный GitHub нельзя добавлять:

- `EHRSHOT_MEDS/`;
- sequence datasets и checkpoints;
- per-seed, wide и ensemble predictions;
- файлы с `subject_id` и `row_id`;
- `EHRSHOT.zip` и reproducibility package;
- ClearML/MinIO credentials.

Подробная карта приватных артефактов приведена в [`ARTIFACTS.md`](ARTIFACTS.md).
