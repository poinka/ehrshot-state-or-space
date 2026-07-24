from __future__ import annotations

"""
Post-analysis for the artificial copy-forward experiment.

Compares raw_4096 and condition_era_90_backfill_4096 by:
  * calibrated probability changes relative to 0% copying;
  * AUPRC, LogLoss and Brier changes relative to 0%;
  * top-10% precision;
  * stability of the composition of the top-10% risk episodes.

No model training or inference is performed. The script consumes
copy_forward_ensemble_predictions.csv produced by the inference script.
"""

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


RAW_VERSION = "raw_4096"
COMPRESSED_VERSION = "condition_era_90_backfill_4096"
SCRIPT_VERSION = "copy-forward-robustness-clearml-v2-20260724"

REQUIRED_COLUMNS = {
    "task",
    "compression_version",
    "requested_copy_fraction",
    "row_id",
    "subject_id",
    "y_true",
    "risk_calibrated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw and condition-era robustness after copy-forward perturbation."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Local copy_forward_ensemble_predictions.csv/parquet. If it is absent, "
            "use --source-task-id or --predictions-s3-url."
        ),
    )
    parser.add_argument(
        "--predictions-s3-url",
        type=str,
        default="",
        help="Optional exact S3/MinIO URL of copy_forward_ensemble_predictions.csv.",
    )
    parser.add_argument(
        "--source-task-id",
        type=str,
        default="",
        help=(
            "ClearML task ID that produced the inference artifacts. The analyzer "
            "downloads the requested artifact through ClearML, without guessing its MinIO path."
        ),
    )
    parser.add_argument(
        "--source-artifact-name",
        type=str,
        default="copy_forward_ensemble_predictions",
        help="Artifact name in the source ClearML task.",
    )
    parser.add_argument(
        "--download-all-source-artifacts",
        action="store_true",
        help="Also download every artifact from the source inference task for provenance.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_copy_forward_perturbation_mps/robustness_analysis"),
    )
    parser.add_argument("--raw-version", type=str, default=RAW_VERSION)
    parser.add_argument(
        "--compressed-version", type=str, default=COMPRESSED_VERSION
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Fraction of episodes treated as the high-risk group.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help=(
            "Optional paired patient bootstrap repetitions for probability and metric "
            "stability advantages. Use 10000 for the final analysis; 0 disables it."
        ),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    parser.add_argument(
        "--enable-clearml",
        action="store_true",
        help="Log analysis parameters, scalars and output artifacts to ClearML.",
    )
    parser.add_argument(
        "--clearml-project",
        type=str,
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
    )
    parser.add_argument(
        "--clearml-task-name",
        type=str,
        default="state_or_space_copy_forward_robustness_analysis",
    )
    parser.add_argument(
        "--clearml-output-uri",
        type=str,
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )
    parser.add_argument(
        "--clearml-tags",
        type=str,
        default="analysis-only,copy-forward,robustness,raw-vs-condition-era",
    )
    parser.add_argument(
        "--clearml-upload-artifacts",
        action="store_true",
        help="Upload all analysis CSV/JSON outputs as ClearML artifacts.",
    )
    return parser.parse_args()



def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


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
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config["script_version"] = SCRIPT_VERSION
    task.connect(config, name="robustness_analysis")
    tags = parse_csv_list(args.clearml_tags)
    if args.source_task_id:
        tags.append(f"source-task:{args.source_task_id}")
    if tags:
        task.add_tags(sorted(set(tags)))
    print("ClearML analysis task initialized:")
    print(f"  task_id = {task.id}")
    print(f"  source_task_id = {args.source_task_id or '<not set>'}")
    return task


def _copy_downloaded_artifact(source: Path, destination_dir: Path, preferred_name: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        candidates = sorted(
            list(source.rglob("*.csv")) + list(source.rglob("*.parquet"))
        )
        exact = [path for path in candidates if path.stem == preferred_name]
        if exact:
            source = exact[0]
        elif len(candidates) == 1:
            source = candidates[0]
        else:
            raise FileNotFoundError(
                f"Downloaded artifact directory does not contain a unique prediction file: {source}. "
                f"Candidates: {[str(path) for path in candidates[:20]]}"
            )
    suffix = source.suffix or ".csv"
    destination = destination_dir / f"{preferred_name}{suffix}"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def download_source_task_artifacts(
    source_task_id: str,
    destination_dir: Path,
    required_artifact_name: str,
    download_all: bool,
) -> Path:
    from clearml import Task

    source_task = Task.get_task(task_id=source_task_id)
    available = sorted(source_task.artifacts.keys())
    if required_artifact_name not in source_task.artifacts:
        raise KeyError(
            f"Artifact {required_artifact_name!r} is absent in ClearML task {source_task_id}. "
            f"Available artifacts: {available}"
        )

    required_local = Path(
        source_task.artifacts[required_artifact_name].get_local_copy()
    )
    predictions_path = _copy_downloaded_artifact(
        required_local,
        destination_dir,
        required_artifact_name,
    )
    print(
        f"Downloaded source artifact {required_artifact_name}: "
        f"{predictions_path}"
    )

    if download_all:
        all_dir = destination_dir / "all_source_artifacts"
        all_dir.mkdir(parents=True, exist_ok=True)
        for artifact_name, artifact in source_task.artifacts.items():
            if artifact_name == required_artifact_name:
                continue
            try:
                local = Path(artifact.get_local_copy())
                target = all_dir / artifact_name
                if local.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(local, target)
                else:
                    target = target.with_suffix(local.suffix)
                    shutil.copy2(local, target)
                print(f"Downloaded source artifact {artifact_name}: {target}")
            except Exception as exc:
                print(f"WARNING: could not download source artifact {artifact_name}: {exc!r}")

    return predictions_path


def resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions is not None and args.predictions.exists():
        print(f"Using local predictions: {args.predictions}")
        return args.predictions

    input_dir = args.output_dir / "source_inputs"
    if args.source_task_id.strip():
        return download_source_task_artifacts(
            source_task_id=args.source_task_id.strip(),
            destination_dir=input_dir,
            required_artifact_name=args.source_artifact_name.strip(),
            download_all=bool(args.download_all_source_artifacts),
        )

    if args.predictions_s3_url.strip():
        from clearml import StorageManager

        cached = StorageManager.get_local_copy(
            remote_url=args.predictions_s3_url.strip()
        )
        if not cached:
            raise FileNotFoundError(
                f"Could not download predictions from {args.predictions_s3_url}"
            )
        return _copy_downloaded_artifact(
            Path(cached), input_dir, args.source_artifact_name.strip()
        )

    attempted = str(args.predictions) if args.predictions is not None else "<not supplied>"
    raise FileNotFoundError(
        "Predictions were not found locally and no remote source was provided. "
        f"Local path: {attempted}. Pass --source-task-id or --predictions-s3-url."
    )


def report_clearml_scalars(
    task,
    metric_table: pd.DataFrame,
    probability_table: pd.DataFrame,
    top10_table: pd.DataFrame,
    direct_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    if task is None:
        return
    logger = task.get_logger()

    for row in metric_table.to_dict("records"):
        iteration = int(round(float(row["copy_fraction"]) * 100))
        series = f'{row["task"]}/{row["compression_version"]}'
        for name in ["auprc", "logloss", "brier", "top10_precision"]:
            value = row.get(name)
            if value is not None and np.isfinite(value):
                logger.report_scalar(
                    title=f"copy_forward_metrics/{name}",
                    series=series,
                    iteration=iteration,
                    value=float(value),
                )

    for row in probability_table.to_dict("records"):
        iteration = int(round(float(row["copy_fraction"]) * 100))
        series = f'{row["task"]}/{row["compression_version"]}'
        for name in [
            "mean_abs_delta_probability",
            "p95_abs_delta_probability",
            "spearman_risk_vs_0",
        ]:
            value = row.get(name)
            if value is not None and np.isfinite(value):
                logger.report_scalar(
                    title=f"probability_stability/{name}",
                    series=series,
                    iteration=iteration,
                    value=float(value),
                )

    for row in top10_table.to_dict("records"):
        iteration = int(round(float(row["copy_fraction"]) * 100))
        series = f'{row["task"]}/{row["compression_version"]}'
        for name in ["retention_fraction", "jaccard", "churn_fraction"]:
            value = row.get(name)
            if value is not None and np.isfinite(value):
                logger.report_scalar(
                    title=f"top10_stability/{name}",
                    series=series,
                    iteration=iteration,
                    value=float(value),
                )

    advantage_names = [
        "probability_stability_advantage_raw_minus_compressed",
        "auprc_degradation_advantage_raw_minus_compressed",
        "logloss_degradation_advantage_raw_minus_compressed",
        "brier_degradation_advantage_raw_minus_compressed",
        "top10_retention_advantage_compressed_minus_raw",
        "top10_churn_advantage_raw_minus_compressed",
    ]
    for row in direct_summary.to_dict("records"):
        iteration = int(round(float(row["copy_fraction"]) * 100))
        series = str(row["task"])
        for name in advantage_names:
            value = row.get(name)
            if value is not None and np.isfinite(value):
                logger.report_scalar(
                    title=f"raw_vs_condition_era/{name}",
                    series=series,
                    iteration=iteration,
                    value=float(value),
                )

    if not bootstrap.empty:
        for row in bootstrap.to_dict("records"):
            iteration = int(round(float(row["copy_fraction"]) * 100))
            logger.report_scalar(
                title="bootstrap/probability_stability_advantage",
                series=str(row["task"]),
                iteration=iteration,
                value=float(row["point_delta"]),
            )


def upload_analysis_artifacts(task, output_dir: Path) -> None:
    if task is None:
        return
    for path in sorted(output_dir.glob("*.csv")) + sorted(output_dir.glob("*.json")):
        print(f"Uploading ClearML analysis artifact: {path.stem} <- {path}")
        task.upload_artifact(
            name=path.stem,
            artifact_object=str(path.resolve()),
            wait_on_upload=False,
        )

def read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["requested_copy_fraction"] = frame[
        "requested_copy_fraction"
    ].astype(float)
    frame["row_id"] = frame["row_id"].astype(int)
    frame["subject_id"] = frame["subject_id"].astype(int)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["risk_calibrated"] = frame["risk_calibrated"].astype(float)

    duplicate_key = [
        "task",
        "compression_version",
        "requested_copy_fraction",
        "row_id",
    ]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, duplicate_key].head(10)
        raise ValueError(
            "Expected one ensemble prediction per episode/version/fraction. "
            f"Duplicate examples:\n{sample.to_string(index=False)}"
        )
    return frame


def metric_values(frame: pd.DataFrame, top_fraction: float) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=int)
    p = np.clip(frame["risk_calibrated"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
    n_top = max(1, int(math.ceil(float(top_fraction) * len(frame))))
    order = np.lexsort((frame["row_id"].to_numpy(dtype=int), -p))
    top_idx = order[:n_top]
    return {
        "auprc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "top10_precision": float(y[top_idx].mean()),
        "n_episodes": int(len(frame)),
        "n_positive": int(y.sum()),
        "n_top10": int(n_top),
    }


def top_ids(frame: pd.DataFrame, top_fraction: float) -> set[int]:
    n_top = max(1, int(math.ceil(float(top_fraction) * len(frame))))
    ordered = frame.sort_values(
        ["risk_calibrated", "row_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    return set(ordered.head(n_top)["row_id"].astype(int))


def top_membership_frame(
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
    top_fraction: float,
) -> pd.DataFrame:
    merged = baseline[
        ["row_id", "subject_id", "y_true", "risk_calibrated"]
    ].rename(columns={"risk_calibrated": "risk_at_0"}).merge(
        perturbed[["row_id", "subject_id", "y_true", "risk_calibrated"]].rename(
            columns={"risk_calibrated": "risk_perturbed"}
        ),
        on=["row_id", "subject_id", "y_true"],
        how="inner",
        validate="one_to_one",
    )
    base_top = top_ids(
        merged.rename(columns={"risk_at_0": "risk_calibrated"}), top_fraction
    )
    pert_top = top_ids(
        merged.rename(columns={"risk_perturbed": "risk_calibrated"}), top_fraction
    )
    merged["in_top10_at_0"] = merged["row_id"].isin(base_top)
    merged["in_top10_perturbed"] = merged["row_id"].isin(pert_top)
    merged["top10_status"] = np.select(
        [
            merged["in_top10_at_0"] & merged["in_top10_perturbed"],
            merged["in_top10_at_0"] & ~merged["in_top10_perturbed"],
            ~merged["in_top10_at_0"] & merged["in_top10_perturbed"],
        ],
        ["retained", "exited", "entered"],
        default="outside",
    )
    return merged


def build_metric_and_probability_tables(
    frame: pd.DataFrame,
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []

    group_cols = ["task", "compression_version"]
    for (task, version), version_frame in frame.groupby(group_cols, sort=False):
        baseline = version_frame[
            version_frame["requested_copy_fraction"] == 0.0
        ].copy()
        if baseline.empty:
            raise ValueError(f"Missing 0% baseline for {task}/{version}")
        base_metrics = metric_values(baseline, top_fraction)

        for fraction, current in version_frame.groupby(
            "requested_copy_fraction", sort=True
        ):
            current = current.copy()
            metrics = metric_values(current, top_fraction)
            merged = baseline[
                ["row_id", "subject_id", "y_true", "risk_calibrated"]
            ].rename(columns={"risk_calibrated": "risk_at_0"}).merge(
                current[
                    ["row_id", "subject_id", "y_true", "risk_calibrated"]
                ].rename(columns={"risk_calibrated": "risk_perturbed"}),
                on=["row_id", "subject_id", "y_true"],
                how="inner",
                validate="one_to_one",
            )
            merged["delta_risk"] = (
                merged["risk_perturbed"] - merged["risk_at_0"]
            )
            merged["abs_delta_risk"] = merged["delta_risk"].abs()

            metric_rows.append(
                {
                    "task": task,
                    "compression_version": version,
                    "copy_fraction": float(fraction),
                    **metrics,
                    "baseline_auprc": base_metrics["auprc"],
                    "delta_auprc_vs_0": metrics["auprc"] - base_metrics["auprc"],
                    "auprc_degradation_vs_0": base_metrics["auprc"]
                    - metrics["auprc"],
                    "abs_auprc_shift_vs_0": abs(
                        metrics["auprc"] - base_metrics["auprc"]
                    ),
                    "baseline_logloss": base_metrics["logloss"],
                    "delta_logloss_vs_0": metrics["logloss"]
                    - base_metrics["logloss"],
                    "logloss_degradation_vs_0": metrics["logloss"]
                    - base_metrics["logloss"],
                    "abs_logloss_shift_vs_0": abs(
                        metrics["logloss"] - base_metrics["logloss"]
                    ),
                    "baseline_brier": base_metrics["brier"],
                    "delta_brier_vs_0": metrics["brier"]
                    - base_metrics["brier"],
                    "brier_degradation_vs_0": metrics["brier"]
                    - base_metrics["brier"],
                    "abs_brier_shift_vs_0": abs(
                        metrics["brier"] - base_metrics["brier"]
                    ),
                    "baseline_top10_precision": base_metrics["top10_precision"],
                    "delta_top10_precision_vs_0": metrics["top10_precision"]
                    - base_metrics["top10_precision"],
                    "top10_precision_degradation_vs_0": base_metrics[
                        "top10_precision"
                    ]
                    - metrics["top10_precision"],
                    "abs_top10_precision_shift_vs_0": abs(
                        metrics["top10_precision"]
                        - base_metrics["top10_precision"]
                    ),
                }
            )
            probability_rows.append(
                {
                    "task": task,
                    "compression_version": version,
                    "copy_fraction": float(fraction),
                    "n_episodes": int(len(merged)),
                    "n_patients": int(merged["subject_id"].nunique()),
                    "mean_delta_probability": float(merged["delta_risk"].mean()),
                    "mean_abs_delta_probability": float(
                        merged["abs_delta_risk"].mean()
                    ),
                    "median_abs_delta_probability": float(
                        merged["abs_delta_risk"].median()
                    ),
                    "p95_abs_delta_probability": float(
                        merged["abs_delta_risk"].quantile(0.95)
                    ),
                    "max_abs_delta_probability": float(
                        merged["abs_delta_risk"].max()
                    ),
                    "spearman_risk_vs_0": float(
                        merged[["risk_at_0", "risk_perturbed"]]
                        .corr(method="spearman")
                        .iloc[0, 1]
                    ),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(probability_rows)


def build_top10_stability_tables(
    frame: pd.DataFrame,
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    membership_parts: list[pd.DataFrame] = []

    for (task, version), version_frame in frame.groupby(
        ["task", "compression_version"], sort=False
    ):
        baseline = version_frame[
            version_frame["requested_copy_fraction"] == 0.0
        ].copy()
        for fraction, current in version_frame.groupby(
            "requested_copy_fraction", sort=True
        ):
            membership = top_membership_frame(baseline, current, top_fraction)
            membership.insert(0, "copy_fraction", float(fraction))
            membership.insert(0, "compression_version", version)
            membership.insert(0, "task", task)
            membership_parts.append(membership)

            base_top = set(
                membership.loc[membership["in_top10_at_0"], "row_id"].astype(int)
            )
            current_top = set(
                membership.loc[
                    membership["in_top10_perturbed"], "row_id"
                ].astype(int)
            )
            overlap = base_top & current_top
            union = base_top | current_top
            retained_positive = int(
                membership.loc[
                    membership["top10_status"] == "retained", "y_true"
                ].sum()
            )
            entered_positive = int(
                membership.loc[
                    membership["top10_status"] == "entered", "y_true"
                ].sum()
            )
            exited_positive = int(
                membership.loc[
                    membership["top10_status"] == "exited", "y_true"
                ].sum()
            )
            summary_rows.append(
                {
                    "task": task,
                    "compression_version": version,
                    "copy_fraction": float(fraction),
                    "top_fraction": float(top_fraction),
                    "n_top_at_0": int(len(base_top)),
                    "n_top_perturbed": int(len(current_top)),
                    "n_overlap": int(len(overlap)),
                    "retention_fraction": float(len(overlap) / max(1, len(base_top))),
                    "jaccard": float(len(overlap) / max(1, len(union))),
                    "n_entered": int(len(current_top - base_top)),
                    "n_exited": int(len(base_top - current_top)),
                    "churn_fraction": float(
                        len(current_top - base_top) / max(1, len(base_top))
                    ),
                    "n_retained_positive": retained_positive,
                    "n_entered_positive": entered_positive,
                    "n_exited_positive": exited_positive,
                    "top10_event_rate_at_0": float(
                        membership.loc[membership["in_top10_at_0"], "y_true"].mean()
                    ),
                    "top10_event_rate_perturbed": float(
                        membership.loc[
                            membership["in_top10_perturbed"], "y_true"
                        ].mean()
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    membership = pd.concat(membership_parts, ignore_index=True)
    return summary, membership


def build_direct_raw_vs_compressed_summary(
    metric_table: pd.DataFrame,
    probability_table: pd.DataFrame,
    top10_table: pd.DataFrame,
    raw_version: str,
    compressed_version: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fractions = sorted(
        set(metric_table["copy_fraction"].astype(float).tolist()) - {0.0}
    )
    tasks = sorted(metric_table["task"].unique())

    for task in tasks:
        for fraction in fractions:
            def one(table: pd.DataFrame, version: str) -> pd.Series:
                result = table[
                    (table["task"] == task)
                    & (table["compression_version"] == version)
                    & (table["copy_fraction"] == fraction)
                ]
                if len(result) != 1:
                    raise ValueError(
                        f"Expected one row for {task}/{version}/{fraction}, got {len(result)}"
                    )
                return result.iloc[0]

            raw_m = one(metric_table, raw_version)
            comp_m = one(metric_table, compressed_version)
            raw_p = one(probability_table, raw_version)
            comp_p = one(probability_table, compressed_version)
            raw_t = one(top10_table, raw_version)
            comp_t = one(top10_table, compressed_version)

            rows.append(
                {
                    "task": task,
                    "copy_fraction": float(fraction),
                    "raw_mean_abs_delta_probability": raw_p[
                        "mean_abs_delta_probability"
                    ],
                    "compressed_mean_abs_delta_probability": comp_p[
                        "mean_abs_delta_probability"
                    ],
                    "probability_stability_advantage_raw_minus_compressed": raw_p[
                        "mean_abs_delta_probability"
                    ]
                    - comp_p["mean_abs_delta_probability"],
                    "raw_p95_abs_delta_probability": raw_p[
                        "p95_abs_delta_probability"
                    ],
                    "compressed_p95_abs_delta_probability": comp_p[
                        "p95_abs_delta_probability"
                    ],
                    "p95_probability_stability_advantage_raw_minus_compressed": raw_p[
                        "p95_abs_delta_probability"
                    ]
                    - comp_p["p95_abs_delta_probability"],
                    "raw_delta_auprc": raw_m["delta_auprc_vs_0"],
                    "compressed_delta_auprc": comp_m["delta_auprc_vs_0"],
                    "raw_auprc_degradation": raw_m["auprc_degradation_vs_0"],
                    "compressed_auprc_degradation": comp_m[
                        "auprc_degradation_vs_0"
                    ],
                    "auprc_degradation_advantage_raw_minus_compressed": raw_m[
                        "auprc_degradation_vs_0"
                    ]
                    - comp_m["auprc_degradation_vs_0"],
                    "auprc_absolute_stability_advantage_raw_minus_compressed": raw_m[
                        "abs_auprc_shift_vs_0"
                    ]
                    - comp_m["abs_auprc_shift_vs_0"],
                    "raw_delta_logloss": raw_m["delta_logloss_vs_0"],
                    "compressed_delta_logloss": comp_m["delta_logloss_vs_0"],
                    "logloss_degradation_advantage_raw_minus_compressed": raw_m[
                        "logloss_degradation_vs_0"
                    ]
                    - comp_m["logloss_degradation_vs_0"],
                    "logloss_absolute_stability_advantage_raw_minus_compressed": raw_m[
                        "abs_logloss_shift_vs_0"
                    ]
                    - comp_m["abs_logloss_shift_vs_0"],
                    "raw_delta_brier": raw_m["delta_brier_vs_0"],
                    "compressed_delta_brier": comp_m["delta_brier_vs_0"],
                    "brier_degradation_advantage_raw_minus_compressed": raw_m[
                        "brier_degradation_vs_0"
                    ]
                    - comp_m["brier_degradation_vs_0"],
                    "brier_absolute_stability_advantage_raw_minus_compressed": raw_m[
                        "abs_brier_shift_vs_0"
                    ]
                    - comp_m["abs_brier_shift_vs_0"],
                    "raw_delta_top10_precision": raw_m[
                        "delta_top10_precision_vs_0"
                    ],
                    "compressed_delta_top10_precision": comp_m[
                        "delta_top10_precision_vs_0"
                    ],
                    "top10_precision_degradation_advantage_raw_minus_compressed": raw_m[
                        "top10_precision_degradation_vs_0"
                    ]
                    - comp_m["top10_precision_degradation_vs_0"],
                    "raw_top10_retention": raw_t["retention_fraction"],
                    "compressed_top10_retention": comp_t["retention_fraction"],
                    "top10_retention_advantage_compressed_minus_raw": comp_t[
                        "retention_fraction"
                    ]
                    - raw_t["retention_fraction"],
                    "raw_top10_jaccard": raw_t["jaccard"],
                    "compressed_top10_jaccard": comp_t["jaccard"],
                    "top10_jaccard_advantage_compressed_minus_raw": comp_t[
                        "jaccard"
                    ]
                    - raw_t["jaccard"],
                    "raw_top10_churn": raw_t["churn_fraction"],
                    "compressed_top10_churn": comp_t["churn_fraction"],
                    "top10_churn_advantage_raw_minus_compressed": raw_t[
                        "churn_fraction"
                    ]
                    - comp_t["churn_fraction"],
                }
            )
    return pd.DataFrame(rows)


def build_cross_representation_top10_composition(
    frame: pd.DataFrame,
    raw_version: str,
    compressed_version: str,
    top_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in sorted(frame["task"].unique()):
        fractions = sorted(
            frame.loc[frame["task"] == task, "requested_copy_fraction"].unique()
        )
        for fraction in fractions:
            raw = frame[
                (frame["task"] == task)
                & (frame["compression_version"] == raw_version)
                & (frame["requested_copy_fraction"] == fraction)
            ]
            compressed = frame[
                (frame["task"] == task)
                & (frame["compression_version"] == compressed_version)
                & (frame["requested_copy_fraction"] == fraction)
            ]
            raw_top = top_ids(raw, top_fraction)
            compressed_top = top_ids(compressed, top_fraction)
            overlap = raw_top & compressed_top
            union = raw_top | compressed_top
            rows.append(
                {
                    "task": task,
                    "copy_fraction": float(fraction),
                    "n_raw_top10": int(len(raw_top)),
                    "n_compressed_top10": int(len(compressed_top)),
                    "n_common": int(len(overlap)),
                    "common_fraction_of_raw_top10": float(
                        len(overlap) / max(1, len(raw_top))
                    ),
                    "common_fraction_of_compressed_top10": float(
                        len(overlap) / max(1, len(compressed_top))
                    ),
                    "jaccard_raw_vs_compressed": float(
                        len(overlap) / max(1, len(union))
                    ),
                    "n_raw_only": int(len(raw_top - compressed_top)),
                    "n_compressed_only": int(len(compressed_top - raw_top)),
                }
            )
    return pd.DataFrame(rows)


def paired_probability_bootstrap(
    frame: pd.DataFrame,
    raw_version: str,
    compressed_version: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    tasks = sorted(frame["task"].unique())
    fractions = sorted(set(frame["requested_copy_fraction"].unique()) - {0.0})

    for task in tasks:
        for fraction in fractions:
            def version_delta(version: str, name: str) -> pd.DataFrame:
                baseline = frame[
                    (frame["task"] == task)
                    & (frame["compression_version"] == version)
                    & (frame["requested_copy_fraction"] == 0.0)
                ][["row_id", "subject_id", "y_true", "risk_calibrated"]].rename(
                    columns={"risk_calibrated": f"{name}_risk_0"}
                )
                current = frame[
                    (frame["task"] == task)
                    & (frame["compression_version"] == version)
                    & (frame["requested_copy_fraction"] == fraction)
                ][["row_id", "subject_id", "y_true", "risk_calibrated"]].rename(
                    columns={"risk_calibrated": f"{name}_risk_f"}
                )
                return baseline.merge(
                    current,
                    on=["row_id", "subject_id", "y_true"],
                    how="inner",
                    validate="one_to_one",
                )

            raw = version_delta(raw_version, "raw")
            comp = version_delta(compressed_version, "compressed")
            merged = raw.merge(
                comp,
                on=["row_id", "subject_id", "y_true"],
                how="inner",
                validate="one_to_one",
            )
            merged["advantage"] = (
                (merged["raw_risk_f"] - merged["raw_risk_0"]).abs()
                - (
                    merged["compressed_risk_f"]
                    - merged["compressed_risk_0"]
                ).abs()
            )
            patient_groups = [
                g["advantage"].to_numpy(float)
                for _, g in merged.groupby("subject_id", sort=False)
            ]
            point = float(merged["advantage"].mean())
            sampled_values = np.empty(int(n_bootstrap), dtype=float)
            n_patients = len(patient_groups)
            for index in range(int(n_bootstrap)):
                sampled = rng.integers(0, n_patients, size=n_patients)
                sampled_values[index] = np.concatenate(
                    [patient_groups[i] for i in sampled]
                ).mean()
            rows.append(
                {
                    "task": task,
                    "copy_fraction": float(fraction),
                    "comparison": (
                        "mean_abs_probability_change_raw_minus_condition_era"
                    ),
                    "positive_means": "condition_era is more stable",
                    "point_delta": point,
                    "ci_low": float(np.quantile(sampled_values, 0.025)),
                    "ci_high": float(np.quantile(sampled_values, 0.975)),
                    "bootstrap_mean": float(sampled_values.mean()),
                    "bootstrap_std": float(sampled_values.std(ddof=1)),
                    "fraction_positive": float(np.mean(sampled_values > 0)),
                    "n_patients": int(n_patients),
                    "n_episodes": int(len(merged)),
                    "n_bootstrap": int(n_bootstrap),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not 0 < args.top_fraction < 1:
        raise ValueError("--top-fraction must be in (0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clearml_task = init_clearml(args)

    try:
        predictions_path = resolve_predictions_path(args)
        frame = read_predictions(predictions_path)
        selected_versions = {args.raw_version, args.compressed_version}
        frame = frame[frame["compression_version"].isin(selected_versions)].copy()
        missing_versions = selected_versions - set(frame["compression_version"].unique())
        if missing_versions:
            raise ValueError(f"Missing requested versions: {sorted(missing_versions)}")

        metric_table, probability_table = build_metric_and_probability_tables(
            frame, args.top_fraction
        )
        top10_table, top10_membership = build_top10_stability_tables(
            frame, args.top_fraction
        )
        direct_summary = build_direct_raw_vs_compressed_summary(
            metric_table=metric_table,
            probability_table=probability_table,
            top10_table=top10_table,
            raw_version=args.raw_version,
            compressed_version=args.compressed_version,
        )
        cross_representation = build_cross_representation_top10_composition(
            frame=frame,
            raw_version=args.raw_version,
            compressed_version=args.compressed_version,
            top_fraction=args.top_fraction,
        )
        bootstrap = paired_probability_bootstrap(
            frame=frame,
            raw_version=args.raw_version,
            compressed_version=args.compressed_version,
            n_bootstrap=args.bootstrap,
            seed=args.bootstrap_seed,
        )

        metric_table.to_csv(
            args.output_dir / "copy_forward_metric_stability.csv", index=False
        )
        probability_table.to_csv(
            args.output_dir / "copy_forward_probability_stability.csv", index=False
        )
        top10_table.to_csv(
            args.output_dir / "copy_forward_top10_stability_vs_0.csv", index=False
        )
        top10_membership.to_csv(
            args.output_dir / "copy_forward_top10_episode_membership.csv", index=False
        )
        direct_summary.to_csv(
            args.output_dir / "raw_vs_condition_era_robustness_summary.csv", index=False
        )
        cross_representation.to_csv(
            args.output_dir / "raw_vs_condition_era_top10_composition.csv", index=False
        )
        if not bootstrap.empty:
            bootstrap.to_csv(
                args.output_dir / "raw_vs_condition_era_probability_bootstrap.csv",
                index=False,
            )

        resolved = {
            "script_version": SCRIPT_VERSION,
            "predictions_resolved": str(predictions_path),
            "predictions_requested": (
                str(args.predictions) if args.predictions is not None else None
            ),
            "predictions_s3_url": args.predictions_s3_url,
            "source_task_id": args.source_task_id,
            "source_artifact_name": args.source_artifact_name,
            "raw_version": args.raw_version,
            "compressed_version": args.compressed_version,
            "top_fraction": args.top_fraction,
            "bootstrap": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "interpretation": {
                "probability_stability_advantage_raw_minus_compressed": (
                    "positive means condition_era changed probabilities less"
                ),
                "metric_absolute_stability_advantage_raw_minus_compressed": (
                    "positive means condition_era had a smaller absolute metric shift"
                ),
                "metric_degradation_advantage_raw_minus_compressed": (
                    "positive means condition_era had a smaller performance degradation"
                ),
                "top10_retention_advantage_compressed_minus_raw": (
                    "positive means condition_era retained more baseline high-risk episodes"
                ),
                "top10_churn_advantage_raw_minus_compressed": (
                    "positive means condition_era had less top-10% membership churn"
                ),
            },
        }
        (args.output_dir / "resolved_robustness_analysis.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        display_columns = [
            "task",
            "copy_fraction",
            "raw_mean_abs_delta_probability",
            "compressed_mean_abs_delta_probability",
            "probability_stability_advantage_raw_minus_compressed",
            "raw_delta_auprc",
            "compressed_delta_auprc",
            "raw_delta_logloss",
            "compressed_delta_logloss",
            "raw_delta_brier",
            "compressed_delta_brier",
            "raw_top10_retention",
            "compressed_top10_retention",
            "top10_retention_advantage_compressed_minus_raw",
        ]
        print("\nRAW VS CONDITION_ERA ROBUSTNESS")
        print(direct_summary[display_columns].to_string(index=False))
        print(f"\nSaved to: {args.output_dir}")

        if clearml_task is not None:
            report_clearml_scalars(
                clearml_task,
                metric_table=metric_table,
                probability_table=probability_table,
                top10_table=top10_table,
                direct_summary=direct_summary,
                bootstrap=bootstrap,
            )
            if args.clearml_upload_artifacts:
                upload_analysis_artifacts(clearml_task, args.output_dir)
            clearml_task.flush(wait_for_uploads=True)
    finally:
        if clearml_task is not None:
            clearml_task.close()


if __name__ == "__main__":
    main()