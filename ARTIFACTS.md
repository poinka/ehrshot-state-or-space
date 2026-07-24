# Карта артефактов проекта State-or-Space

Все перечисленные ниже данные являются внутренними артефактами проекта и хранятся в приватном MinIO/ClearML. Папки с row-level predictions и `subject_id` не должны публиковаться в открытом GitHub.

## `state_or_space_whitelist/`

Train-only список устойчивых диагнозов и аудит его построения.

- `strong_empirical_chronic_like_diagnosis_codes_train_only.csv` — финальный whitelist из 239 кодов.
- `diagnosis_code_empirical_chronic_like_stats_train_only.csv` — статистика по всем кандидатам.
- `source_split_summary.csv` — подтверждение использования train split.
- `build_metadata.json` — пороги, версия датасета, параметры запуска.
- `output_sha256_manifest.csv` — контрольные суммы.

## `ehrshot_state_or_space_sequence_datasets/`

В приватном хранилище содержит построенные sequence datasets.

- `all_compression_version_summary.csv` — размеры и характеристики всех представлений.
- `representation_invariants.csv` — 132 автоматические проверки корректности.
- `resolved_run_config.json` — фактически использованная конфигурация.
- `resolved_run_matrix.csv` — матрица task × representation.
- `<task>/<version>/examples.parquet` — готовые последовательности.
- `<task>/<version>/episode_audit.parquet` — аудит длины и охвата истории.
- `<task>/<version>/vocab.json` — словарь токенов, необходимый для inference.

## `ehrshot_state_or_space_final_sequence_results/`

Результаты 70 обучений: 14 task/representation вариантов × 5 seed.

Финальные run-группы:

- `core_4096/` и `core_4096_wide/` — raw, full backfill, no-backfill и structure-null при 4096 для seed 42–44.
- `context_16384/` и `context_16384_wide/` — raw и full backfill при 16384 для seed 42–44.
- `icu_gap_extra_30_180/` и `icu_gap_extra_30_180_wide/` — ICU gap 30 и 180 для seed 42–44.
- `additional_seeds_45_46_all/` и `additional_seeds_45_46_all_wide/` — все 14 вариантов для seed 45–46.
- `combined_5seeds_wide/sequence_multiseed_heldout_predictions_wide.csv` — канонический wide-файл 5-seed held-out predictions.

В обычных run-папках сохраняются config, metrics, top-k, history, tuning/held-out predictions, Platt calibrator и numeric statistics. Wide-папки используются финальным анализатором для проверяемого объединения predictions.

## `ehrshot_state_or_space_final_analysis_5seeds_wide/`

Финальный основной анализ.

Критические файлы:

- `ensemble_metrics.csv` — финальные метрики 14 представлений.
- `ensemble_predictions.csv` — средние калиброванные вероятности по пяти seed.
- `paired_patient_bootstrap_deltas.csv` — основной patient-cluster bootstrap.
- `equal_patient_weight_paired_bootstrap_deltas.csv` — sensitivity с одинаковым суммарным весом пациентов.
- `metrics_by_seed.csv`, `metrics_mean_std.csv` — межseedовая устойчивость.
- `paired_seed_deltas.csv`, `seed_direction_summary.csv` — направление эффекта по seed.
- `paired_history_coverage_deltas.csv` — изменение доступной истории.
- `context_interaction_bootstrap.csv` — interaction compression × context length.
- `last_episode_*` — sensitivity по последнему эпизоду.
- `resolved_analysis_config.json` — frozen analysis settings.
- `prediction_merge_manifest.csv` — источник объединённого prediction-файла.

## `state_or_space_copy_forward/inference/`

Frozen-model stress test с искусственным copy-forward.

- `copy_forward_episode_plan/` — выбранный диагноз для каждого эпизода.
- `copy_forward_eligible_visits/` — подходящие последующие визиты.
- `copy_forward_cohort_summary/` — размер экспериментальных когорт.
- `copy_forward_injection_summary/` — фактическое число добавленных событий.
- `copy_forward_predictions/` — per-seed predictions.
- `copy_forward_ensemble_predictions/` — 5-seed ensemble predictions.
- `copy_forward_metrics_by_seed/`, `copy_forward_ensemble_metrics/` — метрики.
- `copy_forward_representation_robustness_bootstrap/` — прямой bootstrap raw vs condition era.
- `zero_percent_baseline_agreement/` — проверка реконструкции 0%.
- `resolved_copy_forward_config/` и `resolved_frozen_model_runs/` — настройки и checkpoints.

## `state_or_space_copy_forward/analysis/`

Пост-анализ устойчивости.

- `raw_vs_condition_era_robustness_summary/` — главная сводная таблица.
- `copy_forward_probability_stability/` — изменения вероятностей и Spearman.
- `copy_forward_metric_stability/` — AUPRC, LogLoss, Brier и top-10 precision.
- `copy_forward_top10_stability_vs_0/` — retention, Jaccard и churn top-10%.
- `copy_forward_top10_episode_membership/` — retained/entered/exited для каждого эпизода.
- `raw_vs_condition_era_probability_bootstrap/` — patient bootstrap.
- `raw_vs_condition_era_top10_composition/` — совпадение top-10 между representations.
- `resolved_robustness_analysis/` — источник и параметры анализа.

## `ehrshot_split_leakage_audit/`

Сводная проверка разделений и согласованности источников.

- `split_audit_status.csv` — 9 финальных проверок и число нарушений.
- `split_audit_summary.json/.md` — итоговое описание.
- `subject_split_overlap.csv`, `subject_multi_split_issues.csv`, `row_multi_split_issues.csv` — нарушения split, ожидаются пустыми.
- `official_split_mismatch.csv` — несовпадения с официальным `subject_splits`, ожидаются пустыми.
- `row_mapping_consistency_issues.csv` и `cross_source_subject_split_issues.csv` — межисточниковые нарушения, ожидаются пустыми.
- `dataset_file_manifest.csv` — какие источники проверялись и с каким правилом времени.
- `all_split_audit_core_rows.csv` — большой row-level технический файл; необязателен для минимального пакета и не должен публиковаться.

## Что хранить только приватно

- `EHRSHOT_MEDS/`;
- `examples.parquet` и другие row-level sequence datasets;
- checkpoints;
- wide, per-seed и ensemble predictions;
- `all_split_audit_core_rows.csv`;
- любые файлы с `subject_id` и `row_id`.
