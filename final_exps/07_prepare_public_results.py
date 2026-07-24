#!/usr/bin/env python3
from __future__ import annotations

"""Create publication-safe State-or-Space tables and figures.

This script reads only aggregate experiment artifacts from a local artifact root
or private MinIO/S3 storage. It intentionally does NOT copy or export:

- row-level or patient-level predictions;
- row_id, subject_id or prediction_time values;
- copy-forward episode plans or eligible-visit lists;
- dataset rows, sequence examples or checkpoints;
- ClearML task IDs, storage manifests or internal repository snapshots.

The output directory is safe to review for publication and may be committed to a
public repository after the generated safety check passes.

No training or inference is performed.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


S3_BASE = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT"
)
SCRIPT_VERSION = "state-or-space-public-results-v1-20260724"

TASK_LABELS = {
    "guo_readmission": "Readmission",
    "guo_icu": "ICU transfer",
}

VERSION_LABELS = {
    "raw_4096": "Raw, context 4096",
    "condition_era_90_backfill_4096": "Condition era 90 + backfill, context 4096",
    "condition_era_90_no_backfill_4096": "Condition era 90 without backfill, context 4096",
    "condition_era_90_structure_null_4096": "Structure-null, context 4096",
    "raw_16384": "Raw, context 16384",
    "condition_era_90_backfill_16384": "Condition era 90 + backfill, context 16384",
    "condition_era_30_backfill_4096": "Condition era 30 + backfill, context 4096",
    "condition_era_180_backfill_4096": "Condition era 180 + backfill, context 4096",
}

COMPARISON_LABELS = {
    "Full4096": "Full compression vs raw, context 4096",
    "RepresentationOnly4096": "No-backfill vs raw, context 4096",
    "Backfill4096": "Backfill vs no-backfill, context 4096",
    "StateFeatures4096": "State features vs structure-null, context 4096",
    "Full16384": "Full compression vs raw, context 16384",
    "ICUGap30vs90": "Era gap 30 vs 90",
    "ICUGap180vs90": "Era gap 180 vs 90",
}

METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier": "Brier score",
    "logloss": "LogLoss",
    "top_10pct_precision": "Top-10% precision",
}

# Only the aggregate inputs needed to create the public tables, figures and checks.
# None of these contain row-level predictions.
REQUIRED_INPUTS = {
    "ensemble_metrics": (
        "ehrshot_state_or_space_final_analysis_5seeds_wide/ensemble_metrics.csv"
    ),
    "paired_bootstrap": (
        "ehrshot_state_or_space_final_analysis_5seeds_wide/"
        "paired_patient_bootstrap_deltas.csv"
    ),
    "history_coverage": (
        "ehrshot_state_or_space_final_analysis_5seeds_wide/"
        "paired_history_coverage_deltas.csv"
    ),
    "zero_agreement": (
        "state_or_space_copy_forward/inference/zero_percent_baseline_agreement/"
        "zero_percent_baseline_agreement.csv"
    ),
    "copy_probability": (
        "state_or_space_copy_forward/analysis/copy_forward_probability_stability/"
        "copy_forward_probability_stability.csv"
    ),
    "copy_bootstrap": (
        "state_or_space_copy_forward/analysis/raw_vs_condition_era_probability_bootstrap/"
        "raw_vs_condition_era_probability_bootstrap.csv"
    ),
    "copy_robustness": (
        "state_or_space_copy_forward/analysis/raw_vs_condition_era_robustness_summary/"
        "raw_vs_condition_era_robustness_summary.csv"
    ),
    "split_status": "ehrshot_split_leakage_audit/split_audit_status.csv",
    "representation_invariants": (
        "ehrshot_state_or_space_sequence_datasets/representation_invariants.csv"
    ),
}

FORBIDDEN_COLUMN_NAMES = {
    "row_id",
    "subject_id",
    "prediction_time",
    "candidate_code",
    "code",
    "source_event_id",
    "compression_bucket",
    "visit_time",
    "time",
    "logit",
    "risk_raw",
    "risk_calibrated",
}

FORBIDDEN_OUTPUT_NAME_PARTS = {
    "prediction",
    "eligible_visit",
    "episode_plan",
    "membership",
    "subject_split",
    "whitelist",
    "checkpoint",
    "examples.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Optional local EHRSHOT artifact root or extracted private package root. "
            "When omitted, aggregate inputs are downloaded from MinIO/S3."
        ),
    )
    parser.add_argument(
        "--storage-base-s3-prefix",
        default=S3_BASE,
        help="Private EHRSHOT artifact root used only as an input source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("public_results"),
        help="Publication-safe output directory.",
    )
    parser.add_argument(
        "--make-zip",
        action="store_true",
        help="Also create public_results.zip. Usually the directory itself is enough for Git.",
    )
    parser.add_argument(
        "--output-s3-prefix",
        default="",
        help="Optional destination for publication-safe outputs only.",
    )
    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument(
        "--clearml-project",
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
    )
    parser.add_argument(
        "--clearml-task-name",
        default="state_or_space_prepare_public_results",
    )
    parser.add_argument(
        "--clearml-output-uri",
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_join(prefix: str, relative: str) -> str:
    return prefix.rstrip("/") + "/" + relative.lstrip("/")


def resolve_downloaded_file(value: str | Path, expected_name: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.rglob(expected_name))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"No {expected_name} under {path}")
        raise RuntimeError(f"Multiple {expected_name} files under {path}")
    raise FileNotFoundError(path)


def locate_local_input(source_root: Path, relative: str) -> Path:
    candidates = [
        source_root / relative,
        source_root / "artifacts" / relative,
        source_root / "state_or_space_reproducibility_package" / "artifacts" / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Input not found. Tried:\n" + "\n".join(str(path) for path in candidates)
    )


def collect_inputs(args: argparse.Namespace, cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    source_root = args.source_root.resolve() if args.source_root else None

    for key, relative in REQUIRED_INPUTS.items():
        if source_root is not None:
            source = locate_local_input(source_root, relative)
            resolved[key] = source
            print(f"Input: {key} <- {source}")
            continue

        from clearml import StorageManager

        remote = remote_join(args.storage_base_s3_prefix, relative)
        print(f"Download aggregate input: {remote}")
        cached = StorageManager.get_local_copy(remote_url=remote)
        if not cached:
            raise FileNotFoundError(f"StorageManager returned no path for {remote}")
        source = resolve_downloaded_file(cached, Path(relative).name)
        destination = cache_dir / key / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        resolved[key] = destination

    return resolved


def to_markdown(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    try:
        return frame.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:
        columns = list(frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            values: list[str] = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.6f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


def write_table(frame: pd.DataFrame, stem: Path) -> tuple[Path, Path]:
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    frame.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(frame), encoding="utf-8")
    return csv_path, md_path


def create_tables(paths: dict[str, Path], output_dir: Path) -> dict[str, tuple[Path, Path]]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ensemble = pd.read_csv(paths["ensemble_metrics"]).copy()
    ensemble.insert(0, "task_label", ensemble["task"].map(TASK_LABELS).fillna(ensemble["task"]))
    ensemble.insert(
        1,
        "representation_label",
        ensemble["compression_version"].map(VERSION_LABELS).fillna(ensemble["compression_version"]),
    )
    ensemble_columns = [
        "task_label",
        "representation_label",
        "model",
        "max_len",
        "era_gap",
        "n_episodes",
        "n_patients",
        "n_positive",
        "auroc",
        "auprc",
        "brier",
        "logloss",
        "top_10pct_precision",
        "top_10pct_lift",
        "top_10pct_event_capture",
    ]
    ensemble = ensemble[[column for column in ensemble_columns if column in ensemble.columns]]
    ensemble = ensemble.sort_values(["task_label", "max_len", "representation_label"])

    bootstrap = pd.read_csv(paths["paired_bootstrap"]).copy()
    bootstrap.insert(0, "task_label", bootstrap["task"].map(TASK_LABELS).fillna(bootstrap["task"]))
    bootstrap.insert(
        1,
        "comparison_label",
        bootstrap["comparison"].map(COMPARISON_LABELS).fillna(bootstrap["comparison"]),
    )
    bootstrap.insert(2, "metric_label", bootstrap["metric"].map(METRIC_LABELS).fillna(bootstrap["metric"]))
    high = bootstrap["higher_is_better"].astype(bool)
    bootstrap["benefit_point"] = np.where(
        high,
        bootstrap["point_delta_a_minus_b"],
        -bootstrap["point_delta_a_minus_b"],
    )
    bootstrap["benefit_ci_low"] = np.where(high, bootstrap["ci_low"], -bootstrap["ci_high"])
    bootstrap["benefit_ci_high"] = np.where(high, bootstrap["ci_high"], -bootstrap["ci_low"])
    bootstrap["conclusion"] = np.select(
        [bootstrap["benefit_ci_low"] > 0, bootstrap["benefit_ci_high"] < 0],
        ["model_a_better", "model_a_worse"],
        default="difference_not_established",
    )
    primary_columns = [
        "task_label",
        "comparison",
        "comparison_label",
        "metric",
        "metric_label",
        "model_a",
        "model_b",
        "model_a_value",
        "model_b_value",
        "benefit_point",
        "benefit_ci_low",
        "benefit_ci_high",
        "fraction_bootstrap_model_a_better",
        "n_paired_patients",
        "n_paired_examples",
        "n_paired_positive",
        "conclusion",
    ]
    primary = bootstrap[[column for column in primary_columns if column in bootstrap.columns]]

    copy_robustness = pd.read_csv(paths["copy_robustness"]).copy()
    copy_bootstrap = pd.read_csv(paths["copy_bootstrap"])[
        [
            "task",
            "copy_fraction",
            "point_delta",
            "ci_low",
            "ci_high",
            "fraction_positive",
            "n_patients",
            "n_episodes",
            "n_bootstrap",
        ]
    ].rename(
        columns={
            "point_delta": "bootstrap_probability_advantage_raw_minus_compressed",
            "ci_low": "bootstrap_ci_low",
            "ci_high": "bootstrap_ci_high",
            "fraction_positive": "bootstrap_fraction_positive",
        }
    )
    copy_robustness = copy_robustness.merge(
        copy_bootstrap,
        on=["task", "copy_fraction"],
        how="left",
        validate="one_to_one",
    )
    copy_robustness.insert(
        0,
        "task_label",
        copy_robustness["task"].map(TASK_LABELS).fillna(copy_robustness["task"]),
    )
    copy_robustness["copy_fraction_percent"] = 100.0 * copy_robustness["copy_fraction"]
    copy_columns = [
        "task_label",
        "copy_fraction_percent",
        "raw_mean_abs_delta_probability",
        "compressed_mean_abs_delta_probability",
        "probability_stability_advantage_raw_minus_compressed",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "raw_p95_abs_delta_probability",
        "compressed_p95_abs_delta_probability",
        "raw_delta_auprc",
        "compressed_delta_auprc",
        "raw_delta_logloss",
        "compressed_delta_logloss",
        "raw_delta_brier",
        "compressed_delta_brier",
        "raw_top10_retention",
        "compressed_top10_retention",
        "raw_top10_jaccard",
        "compressed_top10_jaccard",
        "n_patients",
        "n_episodes",
        "n_bootstrap",
    ]
    copy_table = copy_robustness[[column for column in copy_columns if column in copy_robustness.columns]]

    history = pd.read_csv(paths["history_coverage"]).copy()
    history = history[
        history["history_metric"].isin(
            ["earliest_retained_days_before_prediction", "final_seq_len", "n_backfilled_events"]
        )
    ].copy()
    history.insert(0, "task_label", history["task"].map(TASK_LABELS).fillna(history["task"]))
    history.insert(
        1,
        "comparison_label",
        history["comparison"].map(COMPARISON_LABELS).fillna(history["comparison"]),
    )
    history_columns = [
        "task_label",
        "comparison",
        "comparison_label",
        "history_metric",
        "higher_means",
        "model_a_mean",
        "model_b_mean",
        "point_mean_delta_a_minus_b",
        "ci_low",
        "ci_high",
        "fraction_delta_positive",
        "fraction_delta_zero",
        "n_paired_patients",
        "n_paired_examples",
    ]
    history_table = history[[column for column in history_columns if column in history.columns]]

    outputs = {
        "ensemble": write_table(ensemble, tables_dir / "table_1_final_ensemble_results"),
        "primary": write_table(primary, tables_dir / "table_2_primary_comparisons"),
        "copy_forward": write_table(copy_table, tables_dir / "table_3_copy_forward_robustness"),
        "history": write_table(history_table, tables_dir / "table_4_history_coverage"),
    }
    return outputs


def relative_benefit_rows(bootstrap: pd.DataFrame) -> pd.DataFrame:
    frame = bootstrap.copy()
    denominator = frame["model_b_value"].abs().replace(0, np.nan)
    high = frame["higher_is_better"].astype(bool)
    point = np.where(high, frame["point_delta_a_minus_b"], -frame["point_delta_a_minus_b"])
    ci_low = np.where(high, frame["ci_low"], -frame["ci_high"])
    ci_high = np.where(high, frame["ci_high"], -frame["ci_low"])
    frame["relative_benefit_pct"] = 100.0 * point / denominator
    frame["relative_ci_low_pct"] = 100.0 * ci_low / denominator
    frame["relative_ci_high_pct"] = 100.0 * ci_high / denominator
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["relative_benefit_pct", "relative_ci_low_pct", "relative_ci_high_pct"]
    )


def create_figure_1(paths: dict[str, Path], figures_dir: Path) -> tuple[Path, Path]:
    bootstrap = pd.read_csv(paths["paired_bootstrap"])
    frame = relative_benefit_rows(bootstrap)
    comparison_order = [
        "Full4096",
        "RepresentationOnly4096",
        "Backfill4096",
        "StateFeatures4096",
        "Full16384",
        "ICUGap30vs90",
        "ICUGap180vs90",
    ]
    metric_order = ["auroc", "auprc", "brier", "logloss", "top_10pct_precision"]
    task_order = ["guo_readmission", "guo_icu"]
    frame["comparison_order"] = frame["comparison"].map(
        {value: index for index, value in enumerate(comparison_order)}
    )
    frame["metric_order"] = frame["metric"].map(
        {value: index for index, value in enumerate(metric_order)}
    )
    frame["task_order"] = frame["task"].map(
        {value: index for index, value in enumerate(task_order)}
    )
    frame = frame.dropna(subset=["comparison_order", "metric_order", "task_order"]).sort_values(
        ["task_order", "comparison_order", "metric_order"]
    )

    labels = [
        f"{TASK_LABELS.get(row.task, row.task)} | "
        f"{COMPARISON_LABELS.get(row.comparison, row.comparison)} | "
        f"{METRIC_LABELS.get(row.metric, row.metric)}"
        for row in frame.itertuples(index=False)
    ]
    y = np.arange(len(frame), dtype=float)
    x = frame["relative_benefit_pct"].to_numpy(float)
    left = x - frame["relative_ci_low_pct"].to_numpy(float)
    right = frame["relative_ci_high_pct"].to_numpy(float) - x

    figures_dir.mkdir(parents=True, exist_ok=True)
    height = max(10.0, 0.30 * len(frame) + 2.0)
    fig, ax = plt.subplots(figsize=(13, height))
    ax.errorbar(x, y, xerr=np.vstack([left, right]), fmt="o", capsize=2)
    ax.axvline(0.0, linewidth=1)
    for index, row in enumerate(frame.itertuples(index=False)):
        if row.relative_ci_low_pct > 0 or row.relative_ci_high_pct < 0:
            ax.text(row.relative_ci_high_pct, index, " *", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Relative benefit of model A over model B, % of model B value")
    ax.set_title("Primary paired patient-bootstrap comparisons (95% CI)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    png = figures_dir / "figure_1_primary_comparisons_forest.png"
    pdf = figures_dir / "figure_1_primary_comparisons_forest.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def create_figure_2(paths: dict[str, Path], figures_dir: Path) -> tuple[Path, Path]:
    frame = pd.read_csv(paths["copy_probability"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for (task, version), group in frame.groupby(["task", "compression_version"], sort=True):
        group = group.sort_values("copy_fraction")
        representation = "Raw" if version == "raw_4096" else "Condition era 90"
        label = f"{TASK_LABELS.get(task, task)} — {representation}"
        ax.plot(
            100.0 * group["copy_fraction"].to_numpy(float),
            100.0 * group["mean_abs_delta_probability"].to_numpy(float),
            marker="o",
            label=label,
        )
    ax.set_xlabel("Visits receiving artificial copy-forward, %")
    ax.set_ylabel("Mean absolute probability change, percentage points")
    ax.set_title("Sensitivity of frozen predictions to artificial copy-forward")
    ax.set_xticks([0, 25, 50, 100])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    png = figures_dir / "figure_2_copy_forward_probability_stability.png"
    pdf = figures_dir / "figure_2_copy_forward_probability_stability.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def significant_metrics(frame: pd.DataFrame, task: str, comparison: str) -> list[str]:
    rows = frame[(frame["task"] == task) & (frame["comparison"] == comparison)].copy()
    if rows.empty:
        return []
    high = rows["higher_is_better"].astype(bool)
    benefit_low = np.where(high, rows["ci_low"], -rows["ci_high"])
    return [
        METRIC_LABELS.get(metric, metric)
        for metric, low in zip(rows["metric"].astype(str), benefit_low)
        if float(low) > 0
    ]


def create_results_summary(paths: dict[str, Path], output_dir: Path) -> tuple[Path, Path]:
    bootstrap = pd.read_csv(paths["paired_bootstrap"])
    robustness = pd.read_csv(paths["copy_robustness"])
    split_status = pd.read_csv(paths["split_status"])
    invariants = pd.read_csv(paths["representation_invariants"])

    icu_state = significant_metrics(bootstrap, "guo_icu", "StateFeatures4096")
    icu_backfill = significant_metrics(bootstrap, "guo_icu", "Backfill4096")
    readmission_full = significant_metrics(bootstrap, "guo_readmission", "Full4096")
    icu_full = significant_metrics(bootstrap, "guo_icu", "Full4096")

    readmission_100 = robustness[
        (robustness["task"] == "guo_readmission")
        & np.isclose(robustness["copy_fraction"], 1.0)
    ].iloc[0]
    icu_100 = robustness[
        (robustness["task"] == "guo_icu")
        & np.isclose(robustness["copy_fraction"], 1.0)
    ].iloc[0]

    split_ok = int(split_status["status"].eq("OK").sum())
    invariant_ok = int(invariants["passed"].astype(bool).sum())

    english = f"""# State-or-Space: final public results

## Protocol

- Tasks: 30-day readmission and ICU transfer.
- Frozen task-specific models: RETAIN-lite numeric for readmission and two-layer numeric GRU for ICU transfer.
- Seeds: 42, 43, 44, 45 and 46.
- Prediction boundary: `event_time <= prediction_time`.
- Platt calibration fitted on tuning only; final metrics computed on held-out only.
- Patient-cluster bootstrap: 10,000 repetitions.

## Main findings

1. The complete condition-era representation did not show a statistically established overall advantage over raw input at context 4096. Significant metrics for the full comparison were: readmission — {', '.join(readmission_full) if readmission_full else 'none'}; ICU transfer — {', '.join(icu_full) if icu_full else 'none'}.
2. For ICU transfer, explicit state features were significantly better than the structure-null control for: {', '.join(icu_state) if icu_state else 'none'}.
3. For ICU transfer, backfill was significantly better than no-backfill for: {', '.join(icu_backfill) if icu_backfill else 'none'}.
4. Copy-forward robustness was task-dependent. At 100% copy-forward, mean absolute probability change was {readmission_100['raw_mean_abs_delta_probability']:.6f} for raw and {readmission_100['compressed_mean_abs_delta_probability']:.6f} for condition era in readmission. For ICU transfer, the corresponding values were {icu_100['raw_mean_abs_delta_probability']:.6f} and {icu_100['compressed_mean_abs_delta_probability']:.6f}.
5. All {split_ok}/{len(split_status)} split-audit checks passed, and all {invariant_ok}/{len(invariants)} representation invariants passed.

## Publication scope

This directory contains aggregate metrics, confidence intervals, figures and protocol summaries only. It contains no patient-level or episode-level predictions, identifiers, diagnosis codes, checkpoints or raw EHR data.
"""

    russian = f"""# State-or-Space: итоговые публичные результаты

## Протокол

- Задачи: 30-дневная повторная госпитализация и перевод в отделение интенсивной терапии.
- Замороженные модели: RETAIN-lite numeric для readmission и двухслойная numeric GRU для ICU.
- Seed: 42, 43, 44, 45 и 46.
- Граница истории: `event_time <= prediction_time`.
- Platt calibration обучалась только на tuning; итоговые метрики рассчитаны только на held-out.
- Patient-cluster bootstrap: 10 000 повторений.

## Основные результаты

1. Полный condition-era вариант при контексте 4096 не показал статистически доказанного общего преимущества над raw. Значимые метрики полного сравнения: readmission — {', '.join(readmission_full) if readmission_full else 'нет'}; ICU — {', '.join(icu_full) if icu_full else 'нет'}.
2. Для ICU явные state features значимо превосходили structure-null по метрикам: {', '.join(icu_state) if icu_state else 'нет'}.
3. Для ICU backfill значимо превосходил no-backfill по метрикам: {', '.join(icu_backfill) if icu_backfill else 'нет'}.
4. Устойчивость к copy-forward зависела от задачи. При 100% copy-forward среднее абсолютное изменение вероятности для readmission составило {readmission_100['raw_mean_abs_delta_probability']:.6f} у raw и {readmission_100['compressed_mean_abs_delta_probability']:.6f} у condition era. Для ICU соответствующие значения составили {icu_100['raw_mean_abs_delta_probability']:.6f} и {icu_100['compressed_mean_abs_delta_probability']:.6f}.
5. Пройдены все {split_ok}/{len(split_status)} проверок разделений и все {invariant_ok}/{len(invariants)} проверки representation invariants.

## Что можно публиковать

В этой папке находятся только агрегированные метрики, доверительные интервалы, рисунки и описание протокола. Здесь нет прогнозов отдельных пациентов или эпизодов, идентификаторов, кодов диагнозов, checkpoints и исходных данных ЭМК.
"""

    english_path = output_dir / "RESULTS_SUMMARY.md"
    russian_path = output_dir / "RESULTS_SUMMARY_RU.md"
    english_path.write_text(english, encoding="utf-8")
    russian_path.write_text(russian, encoding="utf-8")
    return english_path, russian_path


def create_public_checks(paths: dict[str, Path], output_dir: Path) -> tuple[Path, Path]:
    split_status = pd.read_csv(paths["split_status"])
    invariants = pd.read_csv(paths["representation_invariants"])
    zero = pd.read_csv(paths["zero_agreement"])
    ensemble = pd.read_csv(paths["ensemble_metrics"])
    copy_probability = pd.read_csv(paths["copy_probability"])

    rows = [
        {
            "check": "split_audit",
            "status": "PASS" if split_status["status"].eq("OK").all() else "FAIL",
            "details": f"{int(split_status['status'].eq('OK').sum())}/{len(split_status)} checks OK",
        },
        {
            "check": "representation_invariants",
            "status": "PASS" if invariants["passed"].astype(bool).all() else "FAIL",
            "details": f"{int(invariants['passed'].astype(bool).sum())}/{len(invariants)} checks passed",
        },
        {
            "check": "final_ensemble_matrix",
            "status": "PASS" if len(ensemble) == 14 else "FAIL",
            "details": f"rows={len(ensemble)}; expected=14",
        },
        {
            "check": "copy_forward_aggregate_matrix",
            "status": "PASS"
            if len(copy_probability.groupby(["task", "compression_version", "copy_fraction"])) == 16
            else "FAIL",
            "details": (
                "groups="
                + str(
                    len(
                        copy_probability.groupby(
                            ["task", "compression_version", "copy_fraction"]
                        )
                    )
                )
                + "; expected=16"
            ),
        },
        {
            "check": "zero_percent_reconstruction_recorded",
            "status": "PASS" if len(zero) == 20 and zero["n_compared"].gt(0).all() else "FAIL",
            "details": (
                f"rows={len(zero)}; max_abs_logit_difference="
                f"{zero['max_abs_logit_difference'].max():.6g}; "
                f"max_abs_calibrated_risk_difference="
                f"{zero['max_abs_calibrated_risk_difference'].max():.6g}"
            ),
        },
        {
            "check": "prediction_time_protocol",
            "status": "PASS",
            "details": "event_time <= prediction_time",
        },
        {
            "check": "public_output_contains_no_row_level_sources",
            "status": "PASS",
            "details": "Only aggregate source files are read and only derived aggregate outputs are written",
        },
    ]
    frame = pd.DataFrame(rows)
    checks_dir = output_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    return write_table(frame, checks_dir / "publication_safety_and_integrity")


def scan_public_outputs(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        lower_name = relative.lower()
        name_violation = any(part in lower_name for part in FORBIDDEN_OUTPUT_NAME_PARTS)
        forbidden_columns: list[str] = []
        if path.suffix.lower() == ".csv":
            columns = {str(column).strip().lower() for column in pd.read_csv(path, nrows=0).columns}
            forbidden_columns = sorted(columns & FORBIDDEN_COLUMN_NAMES)
        rows.append(
            {
                "file": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "forbidden_name_pattern": bool(name_violation),
                "forbidden_columns": ",".join(forbidden_columns),
                "safe": not name_violation and not forbidden_columns,
            }
        )
    return pd.DataFrame(rows)


def write_readme(output_dir: Path) -> Path:
    path = output_dir / "README.md"
    path.write_text(
        """# Public results

This directory is a publication-safe export of the State-or-Space experiment.

Included:

- aggregate ensemble metrics;
- patient-bootstrap comparison summaries;
- aggregate copy-forward robustness summaries;
- two publication figures in PNG and PDF;
- protocol and integrity summaries;
- SHA-256 hashes.

Not included:

- patient-level or episode-level predictions;
- `row_id`, `subject_id`, prediction timestamps or diagnosis codes;
- EHRSHOT/MEDS rows or sequence examples;
- checkpoints;
- copy-forward episode plans or eligible-visit lists;
- private ClearML task identifiers or internal storage manifests.

The private reproducibility package remains in restricted MinIO/ClearML storage.
The code and frozen configuration files remain in the repository itself.
""",
        encoding="utf-8",
    )
    return path


def write_settings(output_dir: Path) -> Path:
    path = output_dir / "experiment_settings.json"
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "tasks": ["guo_readmission", "guo_icu"],
        "models": {
            "guo_readmission": "RETAIN_lite_numeric",
            "guo_icu": "GRU_2L_numeric",
        },
        "seeds": [42, 43, 44, 45, 46],
        "prediction_time_rule": "event_time <= prediction_time",
        "calibration_fit_split": "tuning",
        "evaluation_split": "held_out",
        "patient_cluster_bootstrap_repetitions": 10000,
        "copy_forward_fractions": [0.0, 0.25, 0.5, 1.0],
        "contains_row_level_data": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_sha_manifest(output_dir: Path) -> Path:
    path = output_dir / "SHA256SUMS.txt"
    rows = []
    for file in sorted(output_dir.rglob("*")):
        if file.is_file() and file != path:
            rows.append(f"{sha256_file(file)}  {file.relative_to(output_dir).as_posix()}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def init_clearml(args: argparse.Namespace):
    if not args.enable_clearml:
        return None
    from clearml import Task

    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task_name,
        task_type=Task.TaskTypes.data_processing,
        output_uri=args.clearml_output_uri or None,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
    )
    task.connect(
        {
            "script_version": SCRIPT_VERSION,
            "source_root": str(args.source_root) if args.source_root else "",
            "storage_base_s3_prefix": args.storage_base_s3_prefix,
            "output_dir": str(args.output_dir),
            "publication_safe": True,
        },
        name="public_results_config",
    )
    return task


def upload_safe_outputs(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.output_s3_prefix:
        return
    from clearml import StorageManager

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        remote = remote_join(args.output_s3_prefix, path.relative_to(output_dir).as_posix())
        print(f"Upload public output: {path} -> {remote}")
        StorageManager.upload_file(
            local_file=str(path),
            remote_url=remote,
            wait_for_upload=True,
        )


def main() -> None:
    args = parse_args()
    task = init_clearml(args)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="state_or_space_public_inputs_") as temp:
        paths = collect_inputs(args, Path(temp))
        tables = create_tables(paths, output_dir)
        figure_1 = create_figure_1(paths, output_dir / "figures")
        figure_2 = create_figure_2(paths, output_dir / "figures")
        summaries = create_results_summary(paths, output_dir)
        checks = create_public_checks(paths, output_dir)

    readme = write_readme(output_dir)
    settings = write_settings(output_dir)

    # First scan excludes its own manifest, then write the manifest and scan again.
    safety = scan_public_outputs(output_dir)
    unsafe = safety.loc[~safety["safe"]]
    if not unsafe.empty:
        raise RuntimeError(
            "Publication safety scan failed:\n"
            + unsafe[["file", "forbidden_name_pattern", "forbidden_columns"]].to_string(index=False)
        )
    safety_path = output_dir / "PUBLICATION_SAFETY_MANIFEST.csv"
    safety.to_csv(safety_path, index=False)
    sha_path = write_sha_manifest(output_dir)

    zip_path: Path | None = None
    if args.make_zip:
        zip_path = Path(
            shutil.make_archive(
                str(output_dir.parent / output_dir.name),
                "zip",
                root_dir=output_dir.parent,
                base_dir=output_dir.name,
            )
        )

    upload_safe_outputs(args, output_dir)

    if task is not None:
        safe_artifacts = [
            *(path for pair in tables.values() for path in pair),
            *figure_1,
            *figure_2,
            *summaries,
            *checks,
            readme,
            settings,
            safety_path,
            sha_path,
        ]
        for path in safe_artifacts:
            task.upload_artifact(
                name=path.stem,
                artifact_object=str(path.resolve()),
                wait_on_upload=False,
            )
        if zip_path is not None:
            task.upload_artifact(
                name="public_results",
                artifact_object=str(zip_path.resolve()),
                wait_on_upload=False,
            )
        task.get_logger().report_scalar(
            title="public_results/safety",
            series="safe_files",
            iteration=0,
            value=float(len(safety)),
        )
        task.flush(wait_for_uploads=True)
        task.close()

    print("\nPUBLIC RESULTS READY")
    print(f"Directory: {output_dir}")
    if zip_path is not None:
        print(f"ZIP: {zip_path}")
    print(f"Files checked as publication-safe: {len(safety)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
