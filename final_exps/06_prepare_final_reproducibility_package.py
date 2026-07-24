#!/usr/bin/env python3
from __future__ import annotations

"""Build the final reproducibility package for the State-or-Space experiment.

The script can read frozen artifacts either from a local EHRSHOT artifact root
(`--source-root`) or directly from MinIO/S3 through ClearML StorageManager.
It creates:
  * final tables;
  * two publication-ready figures;
  * split/prediction/invariant checks;
  * a stable artifact manifest with SHA-256 hashes;
  * git/environment/ClearML provenance;
  * a ZIP archive suitable for private storage.

No model training or inference is performed.
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


S3_BASE = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT"
)
SCRIPT_VERSION = "state-or-space-repro-package-v1-20260724"

MAIN_ANALYSIS_FILES = [
    "analysis_upload_manifest.csv",
    "context_interaction_bootstrap.csv",
    "ensemble_metrics.csv",
    "ensemble_predictions.csv",
    "equal_patient_weight_ensemble_metrics.csv",
    "equal_patient_weight_paired_bootstrap_deltas.csv",
    "last_episode_ensemble_metrics.csv",
    "last_episode_paired_bootstrap_deltas.csv",
    "metrics_by_seed.csv",
    "metrics_mean_std.csv",
    "paired_history_coverage_deltas.csv",
    "paired_patient_bootstrap_deltas.csv",
    "paired_seed_deltas.csv",
    "prediction_merge_manifest.csv",
    "resolved_analysis_config.json",
    "seed_direction_summary.csv",
]

SEQUENCE_METADATA_FILES = [
    "all_compression_version_summary.csv",
    "representation_invariants.csv",
    "resolved_run_config.json",
    "resolved_run_matrix.csv",
]

WHITELIST_FILES = [
    "build_metadata.json",
    "diagnosis_code_empirical_chronic_like_stats_train_only.csv",
    "output_sha256_manifest.csv",
    "source_split_summary.csv",
    "strong_empirical_chronic_like_diagnosis_codes_train_only.csv",
]

SPLIT_AUDIT_FILES = [
    "coverage_against_label_reference.csv",
    "cross_source_subject_split_issues.csv",
    "dataset_file_manifest.csv",
    "duplicate_prediction_example_issues.csv",
    "official_split_mismatch.csv",
    "official_subject_splits_used.csv",
    "row_mapping_consistency_issues.csv",
    "row_multi_split_issues.csv",
    "split_audit_status.csv",
    "split_audit_summary.json",
    "split_audit_summary.md",
    "split_summary.csv",
    "split_value_audit.csv",
    "subject_multi_split_issues.csv",
    "subject_split_overlap.csv",
]

COPY_FORWARD_INFERENCE_FILES = [
    "copy_forward_cohort_summary.csv",
    "copy_forward_eligible_visits.csv",
    "copy_forward_ensemble_metrics.csv",
    "copy_forward_ensemble_predictions.csv",
    "copy_forward_episode_plan.csv",
    "copy_forward_injection_summary.csv",
    "copy_forward_metrics_by_seed.csv",
    "copy_forward_predictions.csv",
    "copy_forward_representation_robustness_bootstrap.csv",
    "resolved_copy_forward_config.json",
    "resolved_frozen_model_runs.csv",
    "zero_percent_baseline_agreement.csv",
]

COPY_FORWARD_ANALYSIS_FILES = [
    "copy_forward_metric_stability.csv",
    "copy_forward_probability_stability.csv",
    "copy_forward_top10_episode_membership.csv",
    "copy_forward_top10_stability_vs_0.csv",
    "raw_vs_condition_era_probability_bootstrap.csv",
    "raw_vs_condition_era_robustness_summary.csv",
    "raw_vs_condition_era_top10_composition.csv",
    "resolved_robustness_analysis.json",
]

RUN_CONFIG_FILES = [
    "state_or_space_sequence_datasets.json",
    "state_or_space_core_4096_runs.json",
    "state_or_space_context_16384_runs.json",
    "state_or_space_icu_gap_extra_runs.json",
    "state_or_space_additional_seeds_45_46_runs.json",
    "state_or_space_analysis_5seeds.json",
]

TASK_LABELS = {
    "guo_readmission": "Readmission",
    "guo_icu": "ICU transfer",
}

VERSION_LABELS = {
    "raw_4096": "Raw, context 4096",
    "condition_era_90_backfill_4096": "Condition era 90 + backfill, 4096",
    "condition_era_90_no_backfill_4096": "Condition era 90 without backfill, 4096",
    "condition_era_90_structure_null_4096": "Structure-null, 4096",
    "raw_16384": "Raw, context 16384",
    "condition_era_90_backfill_16384": "Condition era 90 + backfill, 16384",
    "condition_era_30_backfill_4096": "Condition era 30 + backfill, 4096",
    "condition_era_180_backfill_4096": "Condition era 180 + backfill, 4096",
}

COMPARISON_LABELS = {
    "Full4096": "Full compression vs raw, 4096",
    "RepresentationOnly4096": "No-backfill vs raw, 4096",
    "Backfill4096": "Backfill vs no-backfill, 4096",
    "StateFeatures4096": "State features vs structure-null, 4096",
    "Full16384": "Full compression vs raw, 16384",
    "ICUGap30vs90": "Era gap 30 vs 90",
    "ICUGap180vs90": "Era gap 180 vs 90",
}

METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier": "Brier",
    "logloss": "LogLoss",
    "top_10pct_precision": "Top-10% precision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing final_exps/, configs/ and scripts/.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Optional local artifact root containing folders such as "
            "ehrshot_state_or_space_final_analysis_5seeds_wide/. When omitted, "
            "files are downloaded from --storage-base-s3-prefix."
        ),
    )
    parser.add_argument(
        "--storage-base-s3-prefix",
        default=S3_BASE,
        help="Private MinIO/S3 EHRSHOT artifact root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("state_or_space_reproducibility_package"),
    )
    parser.add_argument(
        "--output-s3-prefix",
        default=f"{S3_BASE}/state_or_space_reproducibility_package",
    )
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument(
        "--include-row-level-split-audit",
        action="store_true",
        help="Include the large all_split_audit_core_rows.csv file.",
    )
    parser.add_argument(
        "--skip-per-seed-copy-forward-predictions",
        action="store_true",
        help="Exclude copy_forward_predictions.csv to produce a smaller package.",
    )
    parser.add_argument(
        "--skip-wide-model-predictions",
        action="store_true",
        help="Exclude the 5-seed wide held-out prediction file.",
    )
    parser.add_argument(
        "--clearml-tasks-file",
        type=Path,
        default=None,
        help="Optional CSV prepared by scripts/08_record_provenance.sh.",
    )
    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument(
        "--clearml-project",
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
    )
    parser.add_argument(
        "--clearml-task-name",
        default="state_or_space_prepare_reproducibility_package",
    )
    parser.add_argument(
        "--clearml-output-uri",
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )
    parser.add_argument(
        "--clearml-tags",
        default="reproducibility,final-package,state-or-space",
    )
    parser.add_argument("--clearml-upload-artifacts", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def remote_join(prefix: str, relative: str) -> str:
    return prefix.rstrip("/") + "/" + relative.lstrip("/")


def copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(destination)


def resolve_downloaded_file(value: str | Path, expected_name: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.rglob(expected_name))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"No {expected_name} under downloaded path {path}")
        raise RuntimeError(f"Multiple {expected_name} files under downloaded path {path}")
    raise FileNotFoundError(path)


class ArtifactCollector:
    def __init__(
        self,
        source_root: Path | None,
        storage_base: str,
        package_artifact_root: Path,
    ) -> None:
        self.source_root = source_root.resolve() if source_root else None
        self.storage_base = storage_base.rstrip("/")
        self.package_artifact_root = package_artifact_root
        self.rows: list[dict[str, Any]] = []

    def collect(self, relative: str, required: bool = True) -> Path | None:
        relative = relative.strip("/")
        destination = self.package_artifact_root / relative
        source_uri = remote_join(self.storage_base, relative)
        try:
            if self.source_root is not None:
                source = self.source_root / relative
                if not source.exists():
                    raise FileNotFoundError(source)
            else:
                from clearml import StorageManager

                cached = StorageManager.get_local_copy(remote_url=source_uri)
                if not cached:
                    raise FileNotFoundError(f"StorageManager returned no path for {source_uri}")
                source = resolve_downloaded_file(cached, Path(relative).name)
            copy_file_atomic(source, destination)
            self.rows.append(
                {
                    "artifact_relative_path": relative,
                    "source_uri": str(source if self.source_root is not None else source_uri),
                    "package_relative_path": destination.relative_to(self.package_artifact_root.parent).as_posix(),
                    "size_bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "status": "included",
                }
            )
            print(f"Collected: {relative}")
            return destination
        except Exception as exc:
            self.rows.append(
                {
                    "artifact_relative_path": relative,
                    "source_uri": source_uri,
                    "package_relative_path": destination.relative_to(self.package_artifact_root.parent).as_posix(),
                    "size_bytes": None,
                    "sha256": None,
                    "status": f"missing: {exc!r}",
                }
            )
            if required:
                raise
            print(f"Optional artifact unavailable: {relative}: {exc!r}")
            return None


def init_clearml(args: argparse.Namespace):
    if not args.enable_clearml:
        return None
    from clearml import Task

    requirements = args.repo_root / "requirements.txt"
    if requirements.exists():
        Task.force_requirements_env_freeze(False, str(requirements))
    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task_name,
        task_type=Task.TaskTypes.data_processing,
        output_uri=args.clearml_output_uri or None,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["script_version"] = SCRIPT_VERSION
    task.connect(config, name="reproducibility_package")
    tags = [item.strip() for item in args.clearml_tags.split(",") if item.strip()]
    if tags:
        task.add_tags(tags)
    print(f"ClearML reproducibility task: {task.id}")
    return task


def collect_required_artifacts(args: argparse.Namespace, package_root: Path) -> tuple[ArtifactCollector, dict[str, Path]]:
    artifact_root = package_root / "artifacts"
    collector = ArtifactCollector(
        source_root=args.source_root,
        storage_base=args.storage_base_s3_prefix,
        package_artifact_root=artifact_root,
    )
    paths: dict[str, Path] = {}

    for name in MAIN_ANALYSIS_FILES:
        rel = f"ehrshot_state_or_space_final_analysis_5seeds_wide/{name}"
        path = collector.collect(rel, required=True)
        assert path is not None
        paths[f"main/{name}"] = path

    for name in SEQUENCE_METADATA_FILES:
        rel = f"ehrshot_state_or_space_sequence_datasets/{name}"
        path = collector.collect(rel, required=True)
        assert path is not None
        paths[f"sequence/{name}"] = path

    for name in WHITELIST_FILES:
        rel = f"state_or_space_whitelist/{name}"
        path = collector.collect(rel, required=True)
        assert path is not None
        paths[f"whitelist/{name}"] = path

    for name in SPLIT_AUDIT_FILES:
        rel = f"ehrshot_split_leakage_audit/{name}"
        path = collector.collect(rel, required=True)
        assert path is not None
        paths[f"split/{name}"] = path
    if args.include_row_level_split_audit:
        collector.collect(
            "ehrshot_split_leakage_audit/all_split_audit_core_rows.csv",
            required=True,
        )

    for name in COPY_FORWARD_INFERENCE_FILES:
        if name == "copy_forward_predictions.csv" and args.skip_per_seed_copy_forward_predictions:
            continue
        rel = f"state_or_space_copy_forward/inference/{Path(name).stem}/{name}"
        path = collector.collect(rel, required=True)
        assert path is not None
        paths[f"copy_forward_inference/{name}"] = path

    for name in COPY_FORWARD_ANALYSIS_FILES:
        rel = f"state_or_space_copy_forward/analysis/{Path(name).stem}/{name}"
        path = collector.collect(rel, required=True)
        assert path is not None
        paths[f"copy_forward_analysis/{name}"] = path

    if not args.skip_wide_model_predictions:
        rel = (
            "ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/"
            "sequence_multiseed_heldout_predictions_wide.csv"
        )
        path = collector.collect(rel, required=True)
        assert path is not None
        paths["predictions/wide_heldout"] = path

    return collector, paths


def copy_repository_snapshot(args: argparse.Namespace, package_root: Path) -> None:
    repo_root = args.repo_root.resolve()
    code_root = package_root / "repository_snapshot"
    code_root.mkdir(parents=True, exist_ok=True)

    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".DS_Store", "__MACOSX", "logs",
        "checkpoints", "EHRSHOT_MEDS", "*.zip",
    )
    for folder in ["final_exps", "configs", "scripts"]:
        source = repo_root / folder
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copytree(source, code_root / folder, dirs_exist_ok=True, ignore=ignore)

    for name in ["requirements.txt", "README.md", "ARTIFACTS.md", ".env.example", ".gitignore"]:
        source = repo_root / name
        if source.exists():
            copy_file_atomic(source, code_root / name)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_final_tables(paths: dict[str, Path], package_root: Path) -> dict[str, Path]:
    tables_dir = package_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ensemble = read_csv(paths["main/ensemble_metrics.csv"])
    ensemble = ensemble.copy()
    ensemble.insert(0, "task_label", ensemble["task"].map(TASK_LABELS).fillna(ensemble["task"]))
    ensemble.insert(
        2,
        "representation_label",
        ensemble["compression_version"].map(VERSION_LABELS).fillna(ensemble["compression_version"]),
    )
    ordered = [
        "task_label", "task", "representation_label", "compression_version",
        "model", "max_len", "era_gap", "n", "n_positive", "event_rate",
        "auroc", "auprc", "brier", "logloss", "top_10pct_precision",
        "top_10pct_lift", "top_10pct_event_capture", "n_patients", "n_seeds",
        "calibration",
    ]
    final_ensemble = ensemble[[col for col in ordered if col in ensemble.columns]].sort_values(
        ["task", "max_len", "compression_version"]
    )
    ensemble_csv = tables_dir / "final_ensemble_results.csv"
    final_ensemble.to_csv(ensemble_csv, index=False)
    markdown_path = tables_dir / "final_ensemble_results.md"
    try:
        markdown_text = final_ensemble.to_markdown(index=False, floatfmt=".6f")
    except ImportError:
        columns = list(final_ensemble.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in final_ensemble.itertuples(index=False, name=None):
            values = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.6f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        markdown_text = "\n".join(lines)
    markdown_path.write_text(markdown_text, encoding="utf-8")

    boot = read_csv(paths["main/paired_patient_bootstrap_deltas.csv"]).copy()
    boot["task_label"] = boot["task"].map(TASK_LABELS).fillna(boot["task"])
    boot["comparison_label"] = boot["comparison"].map(COMPARISON_LABELS).fillna(boot["comparison"])
    boot["metric_label"] = boot["metric"].map(METRIC_LABELS).fillna(boot["metric"])
    boot["ci_excludes_zero"] = (boot["ci_low"] > 0) | (boot["ci_high"] < 0)
    boot["model_a_significantly_better"] = np.where(
        boot["higher_is_better"].astype(bool),
        boot["ci_low"] > 0,
        boot["ci_high"] < 0,
    )
    boot["model_a_significantly_worse"] = np.where(
        boot["higher_is_better"].astype(bool),
        boot["ci_high"] < 0,
        boot["ci_low"] > 0,
    )
    boot["conclusion"] = np.select(
        [boot["model_a_significantly_better"], boot["model_a_significantly_worse"]],
        ["model_a_better", "model_a_worse"],
        default="difference_not_established",
    )
    primary_csv = tables_dir / "final_primary_comparisons.csv"
    boot.to_csv(primary_csv, index=False)

    robustness = read_csv(
        paths["copy_forward_analysis/raw_vs_condition_era_robustness_summary.csv"]
    ).copy()
    probability_bootstrap = read_csv(
        paths["copy_forward_analysis/raw_vs_condition_era_probability_bootstrap.csv"]
    )
    probability_bootstrap = probability_bootstrap.rename(
        columns={
            "point_delta": "probability_bootstrap_point_delta",
            "ci_low": "probability_bootstrap_ci_low",
            "ci_high": "probability_bootstrap_ci_high",
            "fraction_positive": "probability_bootstrap_fraction_positive",
        }
    )
    keep = [
        "task", "copy_fraction", "probability_bootstrap_point_delta",
        "probability_bootstrap_ci_low", "probability_bootstrap_ci_high",
        "probability_bootstrap_fraction_positive", "n_patients", "n_episodes",
        "n_bootstrap",
    ]
    robustness = robustness.merge(
        probability_bootstrap[keep], on=["task", "copy_fraction"], how="left", validate="one_to_one"
    )
    robustness.insert(0, "task_label", robustness["task"].map(TASK_LABELS).fillna(robustness["task"]))
    copy_csv = tables_dir / "final_copy_forward_results.csv"
    robustness.to_csv(copy_csv, index=False)

    return {
        "ensemble": ensemble_csv,
        "primary": primary_csv,
        "copy_forward": copy_csv,
    }


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
    bootstrap = read_csv(paths["main/paired_patient_bootstrap_deltas.csv"])
    frame = relative_benefit_rows(bootstrap)
    comparison_order = [
        "Full4096", "RepresentationOnly4096", "Backfill4096",
        "StateFeatures4096", "Full16384", "ICUGap30vs90", "ICUGap180vs90",
    ]
    metric_order = ["auroc", "auprc", "brier", "logloss", "top_10pct_precision"]
    task_order = ["guo_readmission", "guo_icu"]
    frame["comparison_order"] = frame["comparison"].map({v: i for i, v in enumerate(comparison_order)})
    frame["metric_order"] = frame["metric"].map({v: i for i, v in enumerate(metric_order)})
    frame["task_order"] = frame["task"].map({v: i for i, v in enumerate(task_order)})
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
        if row.ci_low > 0 or row.ci_high < 0:
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
    frame = read_csv(paths["copy_forward_analysis/copy_forward_probability_stability.csv"])
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


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def add_check(rows: list[dict[str, Any]], name: str, passed: bool, details: str) -> None:
    rows.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "passed": bool(passed),
            "details": details,
        }
    )


def create_integrity_checks(args: argparse.Namespace, paths: dict[str, Path], package_root: Path) -> Path:
    checks: list[dict[str, Any]] = []

    split_status = read_csv(paths["split/split_audit_status.csv"])
    split_ok = split_status["status"].eq("OK").all() and split_status["n_issues"].fillna(0).eq(0).all()
    add_check(
        checks,
        "split_leakage_audit_all_checks_ok",
        bool(split_ok),
        f"{int(split_status['status'].eq('OK').sum())}/{len(split_status)} checks have status OK",
    )

    invariants = read_csv(paths["sequence/representation_invariants.csv"])
    invariant_ok = invariants["passed"].astype(bool).all()
    add_check(
        checks,
        "all_representation_invariants_pass",
        bool(invariant_ok),
        f"{int(invariants['passed'].astype(bool).sum())}/{len(invariants)} checks passed",
    )

    config_path = args.repo_root / "configs/state_or_space_sequence_datasets.json"
    with config_path.open(encoding="utf-8") as fh:
        dataset_cfg = json.load(fh)
    build_cfg = dataset_cfg["build"]
    boundary_ok = (
        build_cfg.get("include_prediction_time") is True
        and build_cfg.get("require_strict_before_prediction_time") is False
    )
    add_check(
        checks,
        "prediction_time_boundary_is_less_or_equal",
        boundary_ok,
        "include_prediction_time=true; require_strict_before_prediction_time=false; event_time <= prediction_time",
    )

    analysis_cfg = json.loads(paths["main/resolved_analysis_config.json"].read_text(encoding="utf-8"))
    add_check(
        checks,
        "platt_calibration_fit_on_tuning",
        analysis_cfg.get("calibration_fit_split") == "tuning",
        f"calibration_fit_split={analysis_cfg.get('calibration_fit_split')}",
    )
    add_check(
        checks,
        "final_analysis_uses_held_out",
        analysis_cfg.get("analysis_split") == "held_out",
        f"analysis_split={analysis_cfg.get('analysis_split')}",
    )
    add_check(
        checks,
        "final_analysis_has_five_seeds",
        analysis_cfg.get("seeds") == [42, 43, 44, 45, 46],
        f"seeds={analysis_cfg.get('seeds')}",
    )

    wide_path = paths.get("predictions/wide_heldout")
    if wide_path is not None:
        predictions = read_csv(wide_path)
        key = [
            "task", "compression_version", "model", "seed", "split", "row_id", "subject_id"
        ]
        add_check(
            checks,
            "wide_predictions_have_no_duplicate_keys",
            not predictions.duplicated(key).any(),
            f"duplicate_rows={int(predictions.duplicated(key).sum())}",
        )
        pairs = predictions[["task", "compression_version"]].drop_duplicates()
        seed_counts = predictions.groupby(["task", "compression_version"])["seed"].nunique()
        add_check(
            checks,
            "wide_predictions_cover_14_task_versions_x_5_seeds",
            len(pairs) == 14 and seed_counts.eq(5).all(),
            f"task_version_pairs={len(pairs)}; seed_counts={sorted(seed_counts.unique().tolist())}",
        )
        add_check(
            checks,
            "wide_predictions_only_held_out",
            set(predictions["split"].astype(str).unique()) == {"held_out"},
            f"splits={sorted(predictions['split'].astype(str).unique().tolist())}",
        )
        raw_match = np.max(
            np.abs(
                predictions["risk_raw"].to_numpy(float)
                - sigmoid(predictions["logit"].to_numpy(float))
            )
        )
        add_check(
            checks,
            "risk_raw_matches_sigmoid_logit",
            raw_match <= 1e-10,
            f"max_abs_difference={raw_match:.3e}",
        )
        in_range = predictions[["risk_raw", "risk_calibrated"]].apply(
            lambda col: col.between(0, 1).all()
        ).all()
        add_check(checks, "prediction_probabilities_in_unit_interval", bool(in_range), "risk_raw and risk_calibrated")

    ensemble = read_csv(paths["main/ensemble_metrics.csv"])
    add_check(
        checks,
        "ensemble_metrics_have_14_final_rows",
        len(ensemble) == 14,
        f"rows={len(ensemble)}",
    )

    copy_predictions = read_csv(
        paths["copy_forward_inference/copy_forward_ensemble_predictions.csv"]
    )
    coverage = copy_predictions.groupby(
        ["task", "compression_version", "requested_copy_fraction"]
    ).size()
    add_check(
        checks,
        "copy_forward_has_all_16_task_version_fraction_groups",
        len(coverage) == 16,
        f"groups={len(coverage)}",
    )
    add_check(
        checks,
        "copy_forward_ensemble_uses_five_seeds",
        copy_predictions["n_seeds"].eq(5).all(),
        f"n_seeds_values={sorted(copy_predictions['n_seeds'].unique().tolist())}",
    )

    zero = read_csv(paths["copy_forward_inference/zero_percent_baseline_agreement.csv"])
    add_check(
        checks,
        "zero_percent_baseline_agreement_was_recorded",
        len(zero) == 20 and zero["n_compared"].gt(0).all(),
        (
            f"rows={len(zero)}; max_logit_diff={zero['max_abs_logit_difference'].max():.6g}; "
            f"max_calibrated_risk_diff={zero['max_abs_calibrated_risk_difference'].max():.6g}"
        ),
    )

    whitelist_meta = json.loads(paths["whitelist/build_metadata.json"].read_text(encoding="utf-8"))
    train_only = str(whitelist_meta.get("train_split_name", whitelist_meta.get("train_split", "train"))) == "train"
    add_check(
        checks,
        "persistent_whitelist_is_train_only",
        train_only,
        f"train_split={whitelist_meta.get('train_split_name', whitelist_meta.get('train_split', 'train'))}",
    )

    checks_dir = package_root / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    output = checks_dir / "final_integrity_summary.csv"
    frame = pd.DataFrame(checks)
    frame.to_csv(output, index=False)
    (checks_dir / "final_integrity_summary.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    copy_file_atomic(paths["split/split_audit_status.csv"], checks_dir / "split_audit_status.csv")
    copy_file_atomic(paths["sequence/representation_invariants.csv"], checks_dir / "representation_invariants.csv")
    copy_file_atomic(
        paths["copy_forward_inference/zero_percent_baseline_agreement.csv"],
        checks_dir / "zero_percent_baseline_agreement.csv",
    )
    if not frame["passed"].all():
        failed = frame.loc[~frame["passed"], ["check", "details"]]
        raise RuntimeError("Final integrity checks failed:\n" + failed.to_string(index=False))
    return output


def git_output(repo_root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_provenance(args: argparse.Namespace, package_root: Path) -> dict[str, Path]:
    provenance = package_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)

    git_state = {
        "commit": git_output(args.repo_root, ["rev-parse", "HEAD"]),
        "branch": git_output(args.repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_porcelain": git_output(args.repo_root, ["status", "--porcelain"]),
        "captured_at_utc": utc_now(),
    }
    git_path = provenance / "git_state.json"
    git_path.write_text(json.dumps(git_state, ensure_ascii=False, indent=2), encoding="utf-8")

    environment_rows = []
    for name in [
        "numpy", "pandas", "polars", "pyarrow", "scipy", "sklearn",
        "torch", "clearml", "boto3", "matplotlib",
    ]:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
        except Exception as exc:
            version = f"ERROR:{exc!r}"
        environment_rows.append({"package": name, "version": version})
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "captured_at_utc": utc_now(),
        "packages": environment_rows,
    }
    environment_path = provenance / "environment.json"
    environment_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")

    clearml_path = provenance / "clearml_tasks.csv"
    if args.clearml_tasks_file and args.clearml_tasks_file.exists():
        copy_file_atomic(args.clearml_tasks_file, clearml_path)
    else:
        mapping = [
            ("whitelist", "CLEARML_TASK_WHITELIST"),
            ("sequence_dataset_builder", "CLEARML_TASK_DATASET_BUILDER"),
            ("training_core_4096", "CLEARML_TASK_CORE_4096"),
            ("training_context_16384", "CLEARML_TASK_CONTEXT_16384"),
            ("training_icu_gap", "CLEARML_TASK_ICU_GAPS"),
            ("training_seeds_45_46", "CLEARML_TASK_SEEDS_45_46"),
            ("main_analysis", "CLEARML_TASK_MAIN_ANALYSIS"),
            ("copy_forward_inference", "CLEARML_TASK_COPY_FORWARD_INFERENCE"),
            ("copy_forward_analysis", "CLEARML_TASK_COPY_FORWARD_ANALYSIS"),
        ]
        with clearml_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["stage", "task_id", "task_id_env"])
            writer.writeheader()
            for stage, env_name in mapping:
                writer.writerow(
                    {
                        "stage": stage,
                        "task_id": os.environ.get(env_name, ""),
                        "task_id_env": env_name,
                    }
                )

    return {"git": git_path, "environment": environment_path, "clearml": clearml_path}


def write_package_readme(package_root: Path) -> Path:
    path = package_root / "README_REPRODUCIBILITY.md"
    text = """# State-or-Space reproducibility package

This private package contains the frozen outputs required to reproduce the final
State-or-Space tables, figures and integrity checks without retraining models.

## Protocol fixed in the final experiment

- prediction boundary: `event_time <= prediction_time`;
- tasks: `guo_readmission` and `guo_icu`;
- seeds: 42, 43, 44, 45 and 46;
- calibration: Platt scaling fitted on tuning only;
- final evaluation: held-out only;
- patient-cluster bootstrap: 10,000 repetitions;
- copy-forward stress test: 0%, 25%, 50% and 100% of eligible later visits.

## Package structure

- `artifacts/`: frozen private artifacts copied from MinIO or a local artifact root;
- `repository_snapshot/`: code, configs and command scripts used to build the package;
- `tables/`: final ensemble, paired-comparison and copy-forward tables;
- `figures/`: two final figures in PNG and PDF;
- `checks/`: split, invariant, prediction and copy-forward integrity checks;
- `provenance/`: git state, environment and ClearML task mapping;
- `storage_manifest.csv`: source URI, size and SHA-256 for every copied source artifact;
- `SHA256SUMS.txt`: hashes for the complete package.

## Privacy and licensing

The package contains row-level prediction identifiers and therefore must remain in
private project storage. It must not be committed to a public GitHub repository.
Raw EHRSHOT/MEDS event data and sequence `examples.parquet` files are intentionally
not duplicated in this package; their private storage location is referenced by the
configuration and manifests.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_sha_manifest(package_root: Path) -> Path:
    path = package_root / "SHA256SUMS.txt"
    rows = []
    for file in sorted(package_root.rglob("*")):
        if not file.is_file() or file == path:
            continue
        rows.append(f"{sha256_file(file)}  {file.relative_to(package_root).as_posix()}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def upload_outputs(args: argparse.Namespace, package_root: Path, zip_path: Path) -> pd.DataFrame:
    if args.skip_upload or not args.output_s3_prefix:
        return pd.DataFrame()
    from clearml import StorageManager

    selected = [
        zip_path,
        package_root / "storage_manifest.csv",
        package_root / "SHA256SUMS.txt",
        package_root / "checks/final_integrity_summary.csv",
        package_root / "tables/final_ensemble_results.csv",
        package_root / "tables/final_primary_comparisons.csv",
        package_root / "tables/final_copy_forward_results.csv",
        package_root / "figures/figure_1_primary_comparisons_forest.png",
        package_root / "figures/figure_1_primary_comparisons_forest.pdf",
        package_root / "figures/figure_2_copy_forward_probability_stability.png",
        package_root / "figures/figure_2_copy_forward_probability_stability.pdf",
    ]
    rows = []
    for local in selected:
        if not local.exists():
            continue
        remote = remote_join(args.output_s3_prefix, local.name)
        print(f"Upload reproducibility output: {local} -> {remote}")
        StorageManager.upload_file(
            local_file=str(local), remote_url=remote, wait_for_upload=True
        )
        rows.append(
            {
                "local_path": str(local),
                "remote_url": remote,
                "size_bytes": local.stat().st_size,
                "sha256": sha256_file(local),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(package_root / "output_upload_manifest.csv", index=False)
    return frame


def upload_clearml(task, package_root: Path, zip_path: Path, artifacts: Iterable[Path]) -> None:
    if task is None:
        return
    for path in artifacts:
        if not path.exists():
            continue
        task.upload_artifact(
            name=path.stem,
            artifact_object=str(path.resolve()),
            wait_on_upload=False,
        )
    task.upload_artifact(
        name="state_or_space_reproducibility_package",
        artifact_object=str(zip_path.resolve()),
        wait_on_upload=False,
    )


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    if args.source_root is not None:
        args.source_root = args.source_root.resolve()
    task = init_clearml(args)

    package_root = args.output_dir.resolve()
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    collector, paths = collect_required_artifacts(args, package_root)
    copy_repository_snapshot(args, package_root)
    tables = create_final_tables(paths, package_root)
    figure_1 = create_figure_1(paths, package_root / "figures")
    figure_2 = create_figure_2(paths, package_root / "figures")
    integrity = create_integrity_checks(args, paths, package_root)
    provenance = write_provenance(args, package_root)
    readme = write_package_readme(package_root)

    storage_manifest = pd.DataFrame(collector.rows)
    storage_manifest_path = package_root / "storage_manifest.csv"
    storage_manifest.to_csv(storage_manifest_path, index=False)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "source_root": str(args.source_root) if args.source_root else None,
        "storage_base_s3_prefix": args.storage_base_s3_prefix,
        "prediction_time_rule": "event_time <= prediction_time",
        "seeds": [42, 43, 44, 45, 46],
        "n_source_artifacts_included": int((storage_manifest["status"] == "included").sum()),
        "tables": {key: str(value.relative_to(package_root)) for key, value in tables.items()},
        "figures": [str(path.relative_to(package_root)) for path in (*figure_1, *figure_2)],
        "integrity_summary": str(integrity.relative_to(package_root)),
        "provenance": {key: str(value.relative_to(package_root)) for key, value in provenance.items()},
    }
    (package_root / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sha_path = write_sha_manifest(package_root)
    zip_base = package_root.parent / package_root.name
    zip_created = Path(shutil.make_archive(str(zip_base), "zip", root_dir=package_root.parent, base_dir=package_root.name))

    upload_frame = upload_outputs(args, package_root, zip_created)
    clearml_artifacts = [
        storage_manifest_path,
        sha_path,
        integrity,
        *tables.values(),
        *figure_1,
        *figure_2,
        readme,
    ]
    if args.clearml_upload_artifacts:
        upload_clearml(task, package_root, zip_created, clearml_artifacts)
    if task is not None:
        logger = task.get_logger()
        logger.report_scalar(
            title="reproducibility/integrity",
            series="passed_checks",
            iteration=0,
            value=float(pd.read_csv(integrity)["passed"].sum()),
        )
        logger.report_scalar(
            title="reproducibility/artifacts",
            series="included_source_artifacts",
            iteration=0,
            value=float((storage_manifest["status"] == "included").sum()),
        )
        task.flush(wait_for_uploads=True)
        task.close()

    print("\nREPRODUCIBILITY PACKAGE READY")
    print(f"Directory: {package_root}")
    print(f"ZIP: {zip_created}")
    print(f"Source artifacts: {(storage_manifest['status'] == 'included').sum()}")
    print(f"Uploaded outputs: {len(upload_frame)}")


if __name__ == "__main__":
    main()
