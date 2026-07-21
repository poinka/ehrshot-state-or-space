from __future__ import annotations

"""
Final analysis for the EHRSHOT State-or-Space experiment using wide predictions.

Expected held-out prediction schema (one row per example / seed):
    task, representation, compression_version, max_len, era_gap, model,
    model_family, numeric_on, seed, split, row_id, subject_id,
    prediction_time, y_true, logit, risk_raw, risk_calibrated

Main outputs:
    - metrics_by_seed.csv
    - metrics_mean_std.csv
    - paired_seed_deltas.csv
    - seed_direction_summary.csv
    - ensemble_predictions.csv
    - ensemble_metrics.csv
    - paired_patient_bootstrap_deltas.csv
    - equal_patient_weight_ensemble_metrics.csv
    - equal_patient_weight_paired_bootstrap_deltas.csv
    - paired_history_coverage_deltas.csv
    - context_interaction_bootstrap.csv
    - last_episode_ensemble_metrics.csv
    - last_episode_paired_bootstrap_deltas.csv
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from common_ehrshot_eval import binary_ranking_metrics, topk_metrics


S3_BASE = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT"
)

PREDICTIVE_METRICS = [
    "auroc",
    "auprc",
    "brier",
    "logloss",
    "top_10pct_precision",
]

LOWER_IS_BETTER_METRICS = {"brier", "logloss"}
DIRECTION_TOL = 1e-12

HISTORY_METRICS = [
    "earliest_retained_days_before_prediction",
    "covered_days",
    "final_seq_len",
    "n_backfill_events_added",
    "n_repeats_removed",
]

WIDE_REQUIRED_COLUMNS = {
    "task",
    "representation",
    "compression_version",
    "max_len",
    "era_gap",
    "model",
    "seed",
    "split",
    "row_id",
    "subject_id",
    "prediction_time",
    "y_true",
    "logit",
    "risk_raw",
    "risk_calibrated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "ehrshot_state_or_space_final_sequence_results/combined_5seeds/"
            "sequence_multiseed_heldout_predictions_wide.csv"
        ),
        help="Combined wide held-out predictions CSV.",
    )
    parser.add_argument(
        "--predictions-s3-url",
        default=(
            f"{S3_BASE}/ehrshot_state_or_space_final_sequence_results/"
            "combined_5seeds/sequence_multiseed_heldout_predictions_wide.csv"
        ),
        help="Fallback single combined wide prediction file.",
    )
    parser.add_argument(
        "--prediction-run-tags",
        default=(
            "core_4096_wide,context_16384_wide,icu_gap_extra_30_180_wide,"
            "additional_seeds_45_46_all_wide"
        ),
        help="Comma-separated result folders to merge before analysis.",
    )
    parser.add_argument(
        "--prediction-results-dir",
        type=Path,
        default=Path("ehrshot_state_or_space_final_sequence_results"),
    )
    parser.add_argument(
        "--prediction-results-s3-root",
        default=f"{S3_BASE}/ehrshot_state_or_space_final_sequence_results",
    )
    parser.add_argument(
        "--prediction-filename",
        default="sequence_multiseed_heldout_predictions_wide.csv",
    )
    parser.add_argument(
        "--combined-predictions-s3-url",
        default=(
            f"{S3_BASE}/ehrshot_state_or_space_final_sequence_results/"
            "combined_5seeds/sequence_multiseed_heldout_predictions_wide.csv"
        ),
    )
    parser.add_argument("--reuse-combined-predictions", action="store_true")
    parser.add_argument("--skip-combined-upload", action="store_true")

    parser.add_argument(
        "--sequence-data-dir",
        type=Path,
        default=Path("ehrshot_state_or_space_sequence_datasets"),
    )
    parser.add_argument(
        "--sequence-data-s3-prefix",
        default=f"{S3_BASE}/ehrshot_state_or_space_sequence_datasets",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("configs/state_or_space_analysis_5seeds.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_state_or_space_final_analysis_5seeds_wide"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-s3-prefix",
        default=f"{S3_BASE}/ehrshot_state_or_space_final_analysis_5seeds_wide",
    )
    parser.add_argument("--skip-upload", action="store_true")

    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--clearml-queue", default="cpu")
    parser.add_argument(
        "--clearml-project",
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
    )
    parser.add_argument(
        "--clearml-task-name",
        default="state_or_space_final_analysis_5seeds_wide",
    )
    parser.add_argument(
        "--clearml-output-uri",
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )
    return parser.parse_args()


def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def maybe_init_clearml(args: argparse.Namespace):
    remote = is_clearml_agent_run()
    if not args.enable_clearml and not remote:
        return None

    from clearml import Task

    task = Task.current_task() if remote else None
    if task is None:
        Task.force_requirements_env_freeze(False, "requirements.txt")
        task = Task.init(
            project_name=args.clearml_project,
            task_name=args.clearml_task_name,
            output_uri=args.clearml_output_uri or None,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
        )

    connected = dict(
        task.connect(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
        )
    )

    path_keys = {
        "predictions",
        "prediction_results_dir",
        "sequence_data_dir",
        "analysis_config",
        "output_dir",
    }
    int_keys = {"n_bootstrap", "bootstrap_seed"}
    bool_keys = {
        "reuse_combined_predictions",
        "skip_combined_upload",
        "skip_upload",
    }

    for key, value in connected.items():
        if not hasattr(args, key) or key in {"enable_clearml", "execute_remotely"}:
            continue
        if key in path_keys:
            setattr(args, key, Path(value))
        elif key in int_keys:
            setattr(args, key, int(value))
        elif key in bool_keys:
            setattr(args, key, _to_bool(value))
        else:
            setattr(args, key, value)

    print("Resolved ClearML parameters:")
    print(f"  remote_agent_run = {remote}")
    print(f"  predictions = {args.predictions}")
    print(f"  prediction_run_tags = {args.prediction_run_tags}")
    print(f"  sequence_data_dir = {args.sequence_data_dir}")
    print(f"  output_dir = {args.output_dir}")

    if args.execute_remotely and not remote:
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    return task


def parse_run_tags(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def expected_task_versions(config: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for comparison in config.get("paired_comparisons", []):
        task = str(comparison["task"])
        pairs.add((task, str(comparison["model_a"])))
        pairs.add((task, str(comparison["model_b"])))
    return pairs


def resolve_cached_file(cached_path: Path, filename: str) -> Path:
    cached_path = Path(cached_path)
    if cached_path.is_file():
        return cached_path
    if cached_path.is_dir():
        matches = sorted(cached_path.rglob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {filename!r} under {cached_path}, found {len(matches)}"
            )
        return matches[0]
    raise FileNotFoundError(cached_path)


def download_file(remote_url: str, local_path: Path, filename: str | None = None) -> Path:
    from clearml import StorageManager

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached_value = StorageManager.get_local_copy(remote_url=remote_url)
    if not cached_value:
        raise FileNotFoundError(f"StorageManager returned no path for {remote_url}")
    cached = resolve_cached_file(Path(cached_value), filename or local_path.name)
    shutil.copy2(cached, local_path)
    return local_path


def download_if_missing(local_path: Path, remote_url: str) -> Path:
    local_path = Path(local_path)
    if local_path.exists():
        return local_path
    if not remote_url:
        raise FileNotFoundError(local_path)
    print(f"Downloading: {remote_url}")
    return download_file(remote_url, local_path)


def upload_file(local_path: Path, remote_url: str) -> None:
    from clearml import StorageManager

    StorageManager.upload_file(
        local_file=str(local_path),
        remote_url=remote_url,
        wait_for_upload=True,
    )


def upload_tree(local_root: Path, remote_prefix: str) -> pd.DataFrame:
    if not remote_prefix:
        return pd.DataFrame()
    rows = []
    for path in sorted(Path(local_root).rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root).as_posix()
        remote = f"{remote_prefix.rstrip('/')}/{rel}"
        upload_file(path, remote)
        rows.append({"local_path": str(path), "remote_url": remote})
    return pd.DataFrame(rows)


def assert_duplicate_predictions_consistent(
    frame: pd.DataFrame,
    duplicate_key: list[str],
) -> None:
    duplicated = frame[frame.duplicated(duplicate_key, keep=False)]
    if duplicated.empty:
        return

    value_columns = [
        c for c in ["y_true", "logit", "risk_raw", "risk_calibrated", "prediction_time"]
        if c in duplicated.columns
    ]
    conflicts: list[dict[str, Any]] = []

    for key, part in duplicated.groupby(duplicate_key, dropna=False, sort=False):
        for column in value_columns:
            values = part[column].dropna().to_numpy()
            if len(values) <= 1:
                continue
            if column in {"logit", "risk_raw", "risk_calibrated"}:
                consistent = bool(np.allclose(values.astype(float), float(values[0]), rtol=1e-9, atol=1e-12))
            else:
                consistent = len(pd.unique(values)) == 1
            if not consistent:
                key_tuple = key if isinstance(key, tuple) else (key,)
                conflicts.append(
                    {
                        **dict(zip(duplicate_key, key_tuple)),
                        "conflicting_column": column,
                        "source_run_tags": sorted(part["source_run_tag"].astype(str).unique()),
                    }
                )
                break
        if len(conflicts) >= 10:
            break

    if conflicts:
        raise ValueError(
            "Split runs contain conflicting duplicate wide predictions. "
            f"Examples: {conflicts}"
        )


def load_prediction_run(
    run_tag: str,
    local_root: Path,
    remote_root: str,
    filename: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    local_path = Path(local_root) / run_tag / filename
    remote_url = f"{remote_root.rstrip('/')}/{run_tag}/{filename}" if remote_root else ""
    source = "local"

    if not local_path.exists():
        if not remote_url:
            raise FileNotFoundError(local_path)
        print(f"Downloading prediction run: {remote_url}")
        download_file(remote_url, local_path, filename=filename)
        source = "s3"

    frame = pd.read_csv(local_path)
    frame["source_run_tag"] = run_tag
    return frame, {
        "run_tag": run_tag,
        "source": source,
        "local_path": str(local_path),
        "remote_url": remote_url,
        "n_rows": int(len(frame)),
    }


def merge_prediction_runs(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[Path, pd.DataFrame]:
    run_tags = parse_run_tags(args.prediction_run_tags)
    if not run_tags:
        raise ValueError("No run tags supplied")

    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for tag in run_tags:
        frame, row = load_prediction_run(
            tag,
            args.prediction_results_dir,
            args.prediction_results_s3_root,
            args.prediction_filename,
        )
        print(f"Loaded {tag}: {len(frame)} rows")
        frames.append(frame)
        manifest_rows.append(row)

    combined = pd.concat(frames, ignore_index=True)
    key = [
        "task",
        "compression_version",
        "model",
        "seed",
        "split",
        "row_id",
        "subject_id",
    ]
    missing = [c for c in key if c not in combined.columns]
    if missing:
        raise ValueError(f"Cannot merge wide predictions; missing key columns: {missing}")

    assert_duplicate_predictions_consistent(combined, key)
    before = len(combined)
    combined = combined.drop_duplicates(key, keep="first").reset_index(drop=True)
    removed = before - len(combined)

    required_pairs = expected_task_versions(config)
    actual_pairs = set(
        combined[["task", "compression_version"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    missing_pairs = required_pairs - actual_pairs
    if missing_pairs:
        raise ValueError(f"Combined predictions miss required task/version pairs: {sorted(missing_pairs)}")

    expected_seeds = sorted(int(x) for x in config["seeds"])
    held = combined[combined["split"] == "held_out"]
    for task, version in sorted(required_pairs):
        part = held[(held["task"] == task) & (held["compression_version"] == version)]
        actual = sorted(part["seed"].astype(int).unique().tolist())
        if actual != expected_seeds:
            raise ValueError(f"{task}/{version}: expected seeds {expected_seeds}, got {actual}")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.predictions, index=False)
    manifest_rows.append(
        {
            "run_tag": "__combined__",
            "source": "generated",
            "local_path": str(args.predictions),
            "remote_url": args.combined_predictions_s3_url,
            "n_rows": int(len(combined)),
            "n_duplicate_rows_removed": int(removed),
        }
    )
    manifest = pd.DataFrame(manifest_rows)

    if not args.skip_combined_upload and args.combined_predictions_s3_url:
        print(f"Uploading combined wide predictions: {args.combined_predictions_s3_url}")
        upload_file(args.predictions, args.combined_predictions_s3_url)

    return args.predictions, manifest


def prepare_predictions(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[Path, pd.DataFrame]:
    tags = parse_run_tags(args.prediction_run_tags)
    if tags and not args.reuse_combined_predictions:
        return merge_prediction_runs(args, config)

    path = download_if_missing(args.predictions, args.predictions_s3_url)
    return path, pd.DataFrame(
        [{
            "run_tag": "__combined_reused__",
            "source": "existing",
            "local_path": str(path),
            "remote_url": args.predictions_s3_url,
            "n_rows": None,
        }]
    )


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def validate_wide_predictions(
    pred: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    missing = WIDE_REQUIRED_COLUMNS - set(pred.columns)
    if missing:
        raise ValueError(f"Wide prediction file is missing columns: {sorted(missing)}")

    pred = pred[pred["split"] == "held_out"].copy()
    if pred.empty:
        raise ValueError("No held_out rows in wide predictions")

    if "model_family" not in pred.columns:
        pred["model_family"] = "numeric_sequence"
    if "numeric_on" not in pred.columns:
        pred["numeric_on"] = True

    for col in ["seed", "row_id", "subject_id", "y_true", "max_len"]:
        pred[col] = pred[col].astype(int)
    pred["numeric_on"] = pred["numeric_on"].astype(bool)
    pred["logit"] = pred["logit"].astype(float)
    pred["risk_raw"] = pred["risk_raw"].astype(float)
    pred["risk_calibrated"] = pred["risk_calibrated"].astype(float)
    pred["prediction_time"] = pd.to_datetime(pred["prediction_time"], errors="raise")

    raw_expected = sigmoid_np(pred["logit"].to_numpy())
    if not np.allclose(raw_expected, pred["risk_raw"].to_numpy(), rtol=1e-7, atol=1e-9):
        diff = float(np.max(np.abs(raw_expected - pred["risk_raw"].to_numpy())))
        raise ValueError(f"risk_raw != sigmoid(logit); max_abs_diff={diff}")

    for risk_col in ["risk_raw", "risk_calibrated"]:
        if not pred[risk_col].between(0.0, 1.0).all():
            raise ValueError(f"{risk_col} contains values outside [0, 1]")

    key = ["task", "compression_version", "model", "seed", "row_id", "subject_id"]
    if pred.duplicated(key).any():
        raise ValueError("Duplicate held-out rows in wide predictions")

    expected_seeds = sorted(int(x) for x in config["seeds"])
    actual_seeds = sorted(pred["seed"].unique().tolist())
    if actual_seeds != expected_seeds:
        raise ValueError(f"Expected seeds {expected_seeds}, got {actual_seeds}")

    required_pairs = expected_task_versions(config)
    actual_pairs = set(
        pred[["task", "compression_version"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    missing_pairs = required_pairs - actual_pairs
    if missing_pairs:
        raise ValueError(f"Predictions miss task/version pairs: {sorted(missing_pairs)}")

    for task, version in sorted(required_pairs):
        part = pred[(pred["task"] == task) & (pred["compression_version"] == version)]
        counts = part.groupby("seed")["row_id"].nunique()
        if sorted(counts.index.tolist()) != expected_seeds or counts.nunique() != 1:
            raise ValueError(f"Unequal seed coverage for {task}/{version}: {counts.to_dict()}")

    # Compatibility aliases used only inside the analysis code and old-style outputs.
    pred["model_name"] = pred["model"]
    pred["example_id"] = pred["row_id"]
    pred["pred_proba"] = pred["risk_calibrated"]
    pred["calibration"] = "platt"
    return pred


def metric_bundle(y: np.ndarray, p: np.ndarray, tie: np.ndarray) -> dict[str, float]:
    base = binary_ranking_metrics(y, p)
    top10 = topk_metrics(y, p, top_fracs=[0.10], tie_breaker=tie).iloc[0]
    return {
        "auroc": float(base["auroc"]),
        "auprc": float(base["auprc"]),
        "brier": float(base["brier"]),
        "logloss": float(base["logloss"]),
        "top_10pct_precision": float(top10["top_k_event_rate"]),
        "top_10pct_lift": float(top10["top_k_lift"]),
        "top_10pct_event_capture": float(top10["event_capture"]),
        "n": int(base["n"]),
        "n_positive": int(base["n_positive"]),
        "event_rate": float(base["event_rate"]),
    }



def patient_equal_episode_weights(df: pd.DataFrame) -> np.ndarray:
    """Give every patient total weight 1, split equally across that patient's episodes."""
    counts = df.groupby("subject_id")["row_id"].transform("size").to_numpy(dtype=float)
    if np.any(counts <= 0):
        raise ValueError("Invalid episode count while building equal-patient weights")
    weights = 1.0 / counts

    check = pd.DataFrame(
        {
            "subject_id": df["subject_id"].to_numpy(),
            "weight": weights,
        }
    ).groupby("subject_id", sort=False)["weight"].sum()
    if not np.allclose(check.to_numpy(dtype=float), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("Equal-patient weights do not sum to 1 within every subject")
    return weights


def weighted_top_fraction_metrics(
    y: np.ndarray,
    p: np.ndarray,
    sample_weight: np.ndarray,
    tie: np.ndarray,
    frac: float = 0.10,
) -> dict[str, float]:
    """
    Weighted top-fraction metrics.

    Episodes are sorted by risk. The boundary episode may contribute a fractional
    amount of its weight so the selected group has exactly `frac` of total weight.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    tie = np.asarray(tie)

    if not (len(y) == len(p) == len(w) == len(tie)):
        raise ValueError("Weighted top-k arrays have inconsistent lengths")
    if len(y) == 0 or not np.isfinite(w).all() or np.any(w < 0):
        return {
            "top_10pct_precision": np.nan,
            "top_10pct_lift": np.nan,
            "top_10pct_event_capture": np.nan,
        }

    total_weight = float(w.sum())
    if total_weight <= 0:
        return {
            "top_10pct_precision": np.nan,
            "top_10pct_lift": np.nan,
            "top_10pct_event_capture": np.nan,
        }

    # Primary key: descending risk. Secondary key: deterministic row/tie id.
    order = np.lexsort((tie, -p))
    target_weight = float(frac * total_weight)
    remaining = target_weight
    selected_positive_weight = 0.0

    for idx in order:
        if remaining <= 0:
            break
        take = min(float(w[idx]), remaining)
        selected_positive_weight += take * float(y[idx])
        remaining -= take

    selected_weight = target_weight - max(remaining, 0.0)
    positive_weight = float(np.sum(w * y))
    base_rate = positive_weight / total_weight if total_weight > 0 else np.nan
    precision = (
        selected_positive_weight / selected_weight
        if selected_weight > 0
        else np.nan
    )
    lift = precision / base_rate if np.isfinite(base_rate) and base_rate > 0 else np.nan
    capture = (
        selected_positive_weight / positive_weight
        if positive_weight > 0
        else np.nan
    )
    return {
        "top_10pct_precision": float(precision),
        "top_10pct_lift": float(lift),
        "top_10pct_event_capture": float(capture),
    }


def metric_bundle_equal_patient_weight(
    y: np.ndarray,
    p: np.ndarray,
    tie: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, float]:
    """Predictive metrics where each patient's episodes have total weight 1."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    w = np.asarray(sample_weight, dtype=float)
    tie = np.asarray(tie)

    if not (len(y) == len(p) == len(w) == len(tie)):
        raise ValueError("Weighted metric arrays have inconsistent lengths")
    if len(y) == 0 or w.sum() <= 0:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "brier": np.nan,
            "logloss": np.nan,
            "top_10pct_precision": np.nan,
            "top_10pct_lift": np.nan,
            "top_10pct_event_capture": np.nan,
            "n": int(len(y)),
            "n_positive": int(y.sum()),
            "event_rate": np.nan,
            "total_weight": float(w.sum()),
        }

    positive_weight = float(w[y == 1].sum())
    negative_weight = float(w[y == 0].sum())
    if positive_weight > 0 and negative_weight > 0:
        auroc = float(roc_auc_score(y, p, sample_weight=w))
        auprc = float(average_precision_score(y, p, sample_weight=w))
    else:
        auroc = np.nan
        auprc = np.nan

    brier = float(np.average((y - p) ** 2, weights=w))
    logloss_value = float(log_loss(y, p, sample_weight=w, labels=[0, 1]))
    top = weighted_top_fraction_metrics(y, p, w, tie, frac=0.10)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "logloss": logloss_value,
        **top,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "event_rate": float(positive_weight / w.sum()),
        "total_weight": float(w.sum()),
    }


def bootstrap_direction_fields(
    raw_deltas: np.ndarray,
    point_raw_delta: float,
    higher_is_better: bool,
    tol: float = DIRECTION_TOL,
) -> dict[str, float | str]:
    """Summarize how often bootstrap replicates preserve the point-estimate direction."""
    raw_deltas = np.asarray(raw_deltas, dtype=float)
    raw_deltas = raw_deltas[np.isfinite(raw_deltas)]
    benefit = raw_deltas if higher_is_better else -raw_deltas
    point_benefit = point_raw_delta if higher_is_better else -point_raw_delta

    a_better = benefit > tol
    b_better = benefit < -tol
    equal = ~(a_better | b_better)

    if point_benefit > tol:
        matching = a_better
    elif point_benefit < -tol:
        matching = b_better
    else:
        matching = equal

    return {
        "point_effect_direction": direction_label(point_benefit, tol=tol),
        "fraction_bootstrap_model_a_better": float(np.mean(a_better)),
        "fraction_bootstrap_model_b_better": float(np.mean(b_better)),
        "fraction_bootstrap_equal": float(np.mean(equal)),
        "fraction_bootstrap_matching_point_direction": float(np.mean(matching)),
    }



def bootstrap_signed_direction_fields(
    raw_values: np.ndarray,
    point_value: float,
    tol: float = DIRECTION_TOL,
) -> dict[str, float | str]:
    """Direction summary for quantities that are not naturally model-A-better metrics."""
    raw_values = np.asarray(raw_values, dtype=float)
    raw_values = raw_values[np.isfinite(raw_values)]
    positive = raw_values > tol
    negative = raw_values < -tol
    equal = ~(positive | negative)

    if point_value > tol:
        matching = positive
        point_direction = "positive"
    elif point_value < -tol:
        matching = negative
        point_direction = "negative"
    else:
        matching = equal
        point_direction = "equal"

    return {
        "point_delta_direction": point_direction,
        "fraction_bootstrap_delta_positive": float(np.mean(positive)),
        "fraction_bootstrap_delta_negative": float(np.mean(negative)),
        "fraction_bootstrap_delta_equal": float(np.mean(equal)),
        "fraction_bootstrap_matching_point_direction": float(np.mean(matching)),
    }


def equal_patient_weight_ensemble_metrics(ens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "task", "model", "model_family", "representation", "compression_version",
        "max_len", "era_gap", "numeric_on",
    ]
    for key, part in ens.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        weights = patient_equal_episode_weights(part)
        row.update(
            metric_bundle_equal_patient_weight(
                part["y_true"].to_numpy(),
                part["risk_calibrated"].to_numpy(),
                part["row_id"].to_numpy(),
                weights,
            )
        )
        row["n_patients"] = int(part["subject_id"].nunique())
        row["n_seeds"] = int(part["n_seeds"].min())
        row["calibration"] = "platt"
        row["weighting"] = "equal_total_weight_per_subject"
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_seed(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "task", "model", "model_family", "representation",
        "compression_version", "max_len", "era_gap", "numeric_on", "seed",
    ]
    for key, part in pred.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            metric_bundle(
                part["y_true"].to_numpy(),
                part["risk_calibrated"].to_numpy(),
                part["row_id"].to_numpy(),
            )
        )
        row["calibration"] = "platt"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_seed_metrics(by_seed: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "task", "model", "model_family", "representation",
        "compression_version", "max_len", "era_gap", "numeric_on", "calibration",
    ]
    metric_cols = [
        "auroc", "auprc", "brier", "logloss", "top_10pct_precision",
        "top_10pct_lift", "top_10pct_event_capture",
    ]
    rows = []
    for key, part in by_seed.groupby(id_cols, dropna=False):
        row = dict(zip(id_cols, key))
        row["n_seeds"] = int(part["seed"].nunique())
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
            row[f"{metric}_min"] = float(part[metric].min())
            row[f"{metric}_max"] = float(part[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def make_ensemble(pred: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "task", "model", "model_family", "representation", "compression_version",
        "max_len", "era_gap", "numeric_on", "split", "row_id", "subject_id",
        "prediction_time", "y_true",
    ]
    ens = (
        pred.groupby(group_cols, dropna=False)
        .agg(
            risk_calibrated=("risk_calibrated", "mean"),
            risk_calibrated_std=("risk_calibrated", "std"),
            risk_raw=("risk_raw", "mean"),
            logit_mean=("logit", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    ens["risk_calibrated_std"] = ens["risk_calibrated_std"].fillna(0.0)
    if ens["n_seeds"].nunique() != 1:
        raise ValueError("Ensemble rows have inconsistent seed counts")
    ens["model_name"] = ens["model"]
    ens["example_id"] = ens["row_id"]
    ens["pred_proba"] = ens["risk_calibrated"]
    ens["calibration"] = "platt"
    return ens


def ensemble_metrics(ens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "task", "model", "model_family", "representation", "compression_version",
        "max_len", "era_gap", "numeric_on",
    ]
    for key, part in ens.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            metric_bundle(
                part["y_true"].to_numpy(),
                part["risk_calibrated"].to_numpy(),
                part["row_id"].to_numpy(),
            )
        )
        row["n_patients"] = int(part["subject_id"].nunique())
        row["n_seeds"] = int(part["n_seeds"].min())
        row["calibration"] = "platt"
        rows.append(row)
    return pd.DataFrame(rows)


def select_version(df: pd.DataFrame, task: str, version: str, seed: int | None = None) -> pd.DataFrame:
    out = df[(df["task"] == task) & (df["compression_version"] == version)].copy()
    if seed is not None:
        out = out[out["seed"] == int(seed)].copy()
    if out.empty:
        suffix = f" seed={seed}" if seed is not None else ""
        raise ValueError(f"No rows for {task}/{version}{suffix}")
    return out


def paired_prediction_merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    keep = ["row_id", "subject_id", "y_true", "risk_calibrated"]
    merged = (
        a[keep].rename(columns={"risk_calibrated": "pred_a"})
        .merge(
            b[keep].rename(columns={"risk_calibrated": "pred_b"}),
            on=["row_id", "subject_id", "y_true"],
            how="inner",
            validate="one_to_one",
        )
    )
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Paired prediction comparison does not cover all examples")
    return merged


def cluster_groups(df: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(idx, dtype=np.int64)
        for idx in df.groupby("subject_id", sort=False).indices.values()
    ]


def paired_bootstrap(
    merged: pd.DataFrame,
    comparison_name: str,
    task: str,
    model_a: str,
    model_b: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    y = merged["y_true"].to_numpy(dtype=int)
    pa = merged["pred_a"].to_numpy(dtype=float)
    pb = merged["pred_b"].to_numpy(dtype=float)
    tie = merged["row_id"].to_numpy(dtype=int)
    groups = cluster_groups(merged)
    rng = np.random.default_rng(seed)

    point_a = metric_bundle(y, pa, tie)
    point_b = metric_bundle(y, pb, tie)
    boot = {metric: [] for metric in PREDICTIVE_METRICS}

    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in sampled])
        sample_tie = np.arange(len(idx), dtype=np.int64)
        ma = metric_bundle(y[idx], pa[idx], sample_tie)
        mb = metric_bundle(y[idx], pb[idx], sample_tie)
        for metric in PREDICTIVE_METRICS:
            boot[metric].append(ma[metric] - mb[metric])

    rows = []
    for metric in PREDICTIVE_METRICS:
        values = np.asarray(boot[metric], dtype=float)
        values = values[np.isfinite(values)]
        raw_delta = point_a[metric] - point_b[metric]
        higher_is_better = metric not in LOWER_IS_BETTER_METRICS
        rows.append(
            {
                "comparison": comparison_name,
                "task": task,
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "model_a_value": point_a[metric],
                "model_b_value": point_b[metric],
                "point_delta_a_minus_b": raw_delta,
                "benefit_delta": raw_delta if higher_is_better else -raw_delta,
                "bootstrap_mean_delta": float(values.mean()),
                "bootstrap_std_delta": float(values.std(ddof=1)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                **bootstrap_direction_fields(values, raw_delta, higher_is_better),
                "n_bootstrap_valid": int(len(values)),
                "n_bootstrap_requested": int(n_bootstrap),
                "bootstrap_unit": "subject_id",
                "weighting": "episode_weighted_with_patient_cluster_resampling",
                "n_paired_examples": int(len(merged)),
                "n_paired_patients": int(merged["subject_id"].nunique()),
                "n_paired_positive": int(merged["y_true"].sum()),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_equal_patient_weight(
    merged: pd.DataFrame,
    comparison_name: str,
    task: str,
    model_a: str,
    model_b: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """
    Paired patient-cluster bootstrap with equal total patient weight.

    Every patient has total weight 1, split over their episodes. Patients are
    sampled with replacement and all episodes from each selected patient enter
    the replicate together. If a patient is sampled twice, their total weight is 2.
    """
    y = merged["y_true"].to_numpy(dtype=int)
    pa = merged["pred_a"].to_numpy(dtype=float)
    pb = merged["pred_b"].to_numpy(dtype=float)
    tie = merged["row_id"].to_numpy(dtype=int)
    base_weight = patient_equal_episode_weights(merged)
    groups = cluster_groups(merged)
    rng = np.random.default_rng(seed)

    point_a = metric_bundle_equal_patient_weight(y, pa, tie, base_weight)
    point_b = metric_bundle_equal_patient_weight(y, pb, tie, base_weight)
    boot = {metric: [] for metric in PREDICTIVE_METRICS}

    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in sampled])
        sample_tie = np.arange(len(idx), dtype=np.int64)
        sample_weight = base_weight[idx]
        ma = metric_bundle_equal_patient_weight(
            y[idx], pa[idx], sample_tie, sample_weight
        )
        mb = metric_bundle_equal_patient_weight(
            y[idx], pb[idx], sample_tie, sample_weight
        )
        for metric in PREDICTIVE_METRICS:
            boot[metric].append(ma[metric] - mb[metric])

    rows = []
    for metric in PREDICTIVE_METRICS:
        values = np.asarray(boot[metric], dtype=float)
        values = values[np.isfinite(values)]
        raw_delta = point_a[metric] - point_b[metric]
        higher_is_better = metric not in LOWER_IS_BETTER_METRICS
        rows.append(
            {
                "comparison": comparison_name,
                "task": task,
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "model_a_value": point_a[metric],
                "model_b_value": point_b[metric],
                "point_delta_a_minus_b": raw_delta,
                "benefit_delta": raw_delta if higher_is_better else -raw_delta,
                "bootstrap_mean_delta": float(values.mean()),
                "bootstrap_std_delta": float(values.std(ddof=1)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                **bootstrap_direction_fields(values, raw_delta, higher_is_better),
                "n_bootstrap_valid": int(len(values)),
                "n_bootstrap_requested": int(n_bootstrap),
                "bootstrap_unit": "subject_id",
                "weighting": "equal_total_weight_per_subject",
                "n_paired_examples": int(len(merged)),
                "n_paired_patients": int(merged["subject_id"].nunique()),
                "n_paired_positive": int(merged["y_true"].sum()),
                "total_patient_weight": float(base_weight.sum()),
            }
        )
    return pd.DataFrame(rows)



def run_comparisons(
    ens: pd.DataFrame,
    config: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for cfg in config["paired_comparisons"]:
        task = str(cfg["task"])
        a_version = str(cfg["model_a"])
        b_version = str(cfg["model_b"])
        merged = paired_prediction_merge(
            select_version(ens, task, a_version),
            select_version(ens, task, b_version),
        )
        rows.append(
            paired_bootstrap(
                merged,
                str(cfg["name"]),
                task,
                a_version,
                b_version,
                n_bootstrap,
                seed,
            )
        )
    return pd.concat(rows, ignore_index=True)



def run_comparisons_equal_patient_weight(
    ens: pd.DataFrame,
    config: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for cfg in config["paired_comparisons"]:
        task = str(cfg["task"])
        a_version = str(cfg["model_a"])
        b_version = str(cfg["model_b"])
        merged = paired_prediction_merge(
            select_version(ens, task, a_version),
            select_version(ens, task, b_version),
        )
        rows.append(
            paired_bootstrap_equal_patient_weight(
                merged,
                str(cfg["name"]),
                task,
                a_version,
                b_version,
                n_bootstrap,
                seed,
            )
        )
    return pd.concat(rows, ignore_index=True)


def direction_label(value: float, tol: float = 1e-12) -> str:
    if value > tol:
        return "model_a_better"
    if value < -tol:
        return "model_b_better"
    return "equal"


def seed_direction_tables(
    pred: pd.DataFrame,
    ens_paired: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensemble_lookup = {
        (str(row.task), str(row.comparison), str(row.metric)): float(row.benefit_delta)
        for row in ens_paired.itertuples(index=False)
    }

    rows = []
    for cfg in config["paired_comparisons"]:
        task = str(cfg["task"])
        name = str(cfg["name"])
        a_version = str(cfg["model_a"])
        b_version = str(cfg["model_b"])

        for seed in sorted(int(x) for x in config["seeds"]):
            a = select_version(pred, task, a_version, seed=seed)
            b = select_version(pred, task, b_version, seed=seed)
            merged = paired_prediction_merge(a, b)
            y = merged["y_true"].to_numpy(dtype=int)
            tie = merged["row_id"].to_numpy(dtype=int)
            ma = metric_bundle(y, merged["pred_a"].to_numpy(float), tie)
            mb = metric_bundle(y, merged["pred_b"].to_numpy(float), tie)

            for metric in PREDICTIVE_METRICS:
                raw_delta = float(ma[metric] - mb[metric])
                benefit_delta = raw_delta if metric not in {"brier", "logloss"} else -raw_delta
                ensemble_benefit = ensemble_lookup[(task, name, metric)]
                rows.append(
                    {
                        "task": task,
                        "comparison": name,
                        "model_a": a_version,
                        "model_b": b_version,
                        "seed": seed,
                        "metric": metric,
                        "higher_is_better": metric not in {"brier", "logloss"},
                        "model_a_value": float(ma[metric]),
                        "model_b_value": float(mb[metric]),
                        "raw_delta_a_minus_b": raw_delta,
                        "benefit_delta": benefit_delta,
                        "direction": direction_label(benefit_delta),
                        "model_a_better": benefit_delta > 1e-12,
                        "ensemble_benefit_delta": ensemble_benefit,
                        "ensemble_direction": direction_label(ensemble_benefit),
                        "matches_ensemble_direction": (
                            direction_label(benefit_delta) == direction_label(ensemble_benefit)
                        ),
                        "n_paired_examples": int(len(merged)),
                        "n_paired_patients": int(merged["subject_id"].nunique()),
                        "n_paired_positive": int(merged["y_true"].sum()),
                    }
                )

    detail = pd.DataFrame(rows)
    summary_rows = []
    group_cols = ["task", "comparison", "model_a", "model_b", "metric"]
    for key, part in detail.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        values = part["benefit_delta"].to_numpy(float)
        directions = part["direction"].tolist()
        ensemble_benefit = float(part["ensemble_benefit_delta"].iloc[0])
        ensemble_direction = str(part["ensemble_direction"].iloc[0])
        counts = pd.Series(directions).value_counts()
        row.update(
            {
                "n_seeds": int(len(part)),
                "n_seeds_model_a_better": int(counts.get("model_a_better", 0)),
                "n_seeds_model_b_better": int(counts.get("model_b_better", 0)),
                "n_seeds_equal": int(counts.get("equal", 0)),
                "fraction_seeds_model_a_better": float(np.mean(values > 1e-12)),
                "fraction_seeds_model_b_better": float(np.mean(values < -1e-12)),
                "mean_benefit_delta": float(np.mean(values)),
                "std_benefit_delta": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min_benefit_delta": float(np.min(values)),
                "max_benefit_delta": float(np.max(values)),
                "ensemble_benefit_delta": ensemble_benefit,
                "ensemble_direction": ensemble_direction,
                "all_seeds_same_direction": len(set(directions)) == 1,
                "n_seeds_matching_ensemble": int(part["matches_ensemble_direction"].sum()),
                "fraction_seeds_matching_ensemble": float(part["matches_ensemble_direction"].mean()),
                "majority_matches_ensemble": bool(part["matches_ensemble_direction"].mean() > 0.5),
            }
        )
        summary_rows.append(row)

    return detail, pd.DataFrame(summary_rows)


def ensure_episode_audit(
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    task: str,
    version: str,
) -> Path:
    local = Path(sequence_data_dir) / task / version / "episode_audit.parquet"
    if local.exists():
        return local
    if not sequence_data_s3_prefix:
        raise FileNotFoundError(local)
    remote = f"{sequence_data_s3_prefix.rstrip('/')}/{task}/{version}/episode_audit.parquet"
    print(f"Downloading episode audit: {remote}")
    return download_file(remote, local, filename="episode_audit.parquet")


def load_episode_audit(
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    task: str,
    version: str,
) -> pd.DataFrame:
    path = ensure_episode_audit(
        sequence_data_dir,
        sequence_data_s3_prefix,
        task,
        version,
    )
    requested = [
        "row_id", "subject_id", "prediction_time", "label", "split",
        *HISTORY_METRICS,
    ]
    audit = pd.read_parquet(path)
    missing = set(requested) - set(audit.columns)
    if missing:
        raise ValueError(f"{path} misses history audit columns: {sorted(missing)}")
    audit = audit[requested].copy()
    audit = audit[audit["split"] == "held_out"].copy()
    audit["row_id"] = audit["row_id"].astype(int)
    audit["subject_id"] = audit["subject_id"].astype(int)
    audit["label"] = audit["label"].astype(int)
    audit["prediction_time"] = pd.to_datetime(audit["prediction_time"], errors="raise")
    if audit.duplicated(["row_id", "subject_id"]).any():
        raise ValueError(f"Duplicate episode audit rows: {task}/{version}")
    return audit


def history_bootstrap_rows(
    merged: pd.DataFrame,
    task: str,
    comparison: str,
    model_a: str,
    model_b: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    groups = cluster_groups(merged)
    rng = np.random.default_rng(seed)
    rows = []

    for metric in HISTORY_METRICS:
        a = merged[f"{metric}_a"].to_numpy(dtype=float)
        b = merged[f"{metric}_b"].to_numpy(dtype=float)
        delta = a - b
        finite = np.isfinite(delta)
        if not finite.all():
            a = a[finite]
            b = b[finite]
            delta = delta[finite]
            reduced = merged.loc[finite].reset_index(drop=True)
            metric_groups = cluster_groups(reduced)
        else:
            metric_groups = groups

        values = []
        for _ in range(n_bootstrap):
            sampled = rng.integers(0, len(metric_groups), size=len(metric_groups))
            idx = np.concatenate([metric_groups[i] for i in sampled])
            values.append(float(np.mean(delta[idx])))
        values_arr = np.asarray(values, dtype=float)

        rows.append(
            {
                "task": task,
                "comparison": comparison,
                "model_a": model_a,
                "model_b": model_b,
                "history_metric": metric,
                "higher_means": {
                    "earliest_retained_days_before_prediction": "model sees further back",
                    "covered_days": "model covers a longer history period",
                    "final_seq_len": "model receives more events",
                    "n_backfill_events_added": "more earlier events were backfilled",
                    "n_repeats_removed": "more repeated mentions were removed",
                }[metric],
                "model_a_mean": float(np.mean(a)),
                "model_b_mean": float(np.mean(b)),
                "point_mean_delta_a_minus_b": float(np.mean(delta)),
                "median_paired_delta": float(np.median(delta)),
                "bootstrap_mean_delta": float(values_arr.mean()),
                "bootstrap_std_delta": float(values_arr.std(ddof=1)),
                "ci_low": float(np.quantile(values_arr, 0.025)),
                "ci_high": float(np.quantile(values_arr, 0.975)),
                **bootstrap_signed_direction_fields(
                    values_arr,
                    float(np.mean(delta)),
                ),
                "fraction_delta_positive": float(np.mean(delta > 0)),
                "fraction_delta_zero": float(np.mean(np.isclose(delta, 0.0))),
                "fraction_delta_negative": float(np.mean(delta < 0)),
                "n_bootstrap_valid": int(len(values_arr)),
                "n_paired_examples": int(len(delta)),
                "n_paired_patients": int(merged["subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def run_history_comparisons(
    config: dict[str, Any],
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []

    def get(task: str, version: str) -> pd.DataFrame:
        key = (task, version)
        if key not in cache:
            cache[key] = load_episode_audit(
                sequence_data_dir,
                sequence_data_s3_prefix,
                task,
                version,
            )
        return cache[key]

    for cfg in config["paired_comparisons"]:
        task = str(cfg["task"])
        name = str(cfg["name"])
        a_version = str(cfg["model_a"])
        b_version = str(cfg["model_b"])
        a = get(task, a_version)
        b = get(task, b_version)
        keys = ["row_id", "subject_id", "prediction_time", "label", "split"]
        merged = a.merge(
            b,
            on=keys,
            how="inner",
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        if len(merged) != len(a) or len(merged) != len(b):
            raise ValueError(f"History pairing incomplete for {task}/{name}")
        rows.append(
            history_bootstrap_rows(
                merged,
                task,
                name,
                a_version,
                b_version,
                n_bootstrap,
                seed,
            )
        )
    return pd.concat(rows, ignore_index=True)


def context_interaction_bootstrap(
    raw4096: pd.DataFrame,
    back4096: pd.DataFrame,
    raw16384: pd.DataFrame,
    back16384: pd.DataFrame,
    task: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    key = ["row_id", "subject_id", "y_true"]
    merged = (
        raw4096[key + ["risk_calibrated"]].rename(columns={"risk_calibrated": "raw4096"})
        .merge(back4096[key + ["risk_calibrated"]].rename(columns={"risk_calibrated": "back4096"}), on=key, validate="one_to_one")
        .merge(raw16384[key + ["risk_calibrated"]].rename(columns={"risk_calibrated": "raw16384"}), on=key, validate="one_to_one")
        .merge(back16384[key + ["risk_calibrated"]].rename(columns={"risk_calibrated": "back16384"}), on=key, validate="one_to_one")
    )
    y = merged["y_true"].to_numpy(dtype=int)
    tie = merged["row_id"].to_numpy(dtype=int)
    arrays = {
        c: merged[c].to_numpy(dtype=float)
        for c in ["raw4096", "back4096", "raw16384", "back16384"]
    }
    groups = cluster_groups(merged)
    rng = np.random.default_rng(seed)

    def interaction(idx: np.ndarray, sample_tie: np.ndarray) -> dict[str, float]:
        m = {name: metric_bundle(y[idx], arr[idx], sample_tie) for name, arr in arrays.items()}
        return {
            metric: (m["back16384"][metric] - m["raw16384"][metric])
            - (m["back4096"][metric] - m["raw4096"][metric])
            for metric in PREDICTIVE_METRICS
        }

    point = interaction(np.arange(len(merged)), tie)
    boot = {metric: [] for metric in PREDICTIVE_METRICS}
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in sampled])
        values = interaction(idx, np.arange(len(idx), dtype=np.int64))
        for metric in PREDICTIVE_METRICS:
            boot[metric].append(values[metric])

    rows = []
    for metric in PREDICTIVE_METRICS:
        values = np.asarray(boot[metric], dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "task": task,
                "comparison": "ContextInteraction=(Backfill16384-Raw16384)-(Backfill4096-Raw4096)",
                "metric": metric,
                "higher_is_better": metric not in {"brier", "logloss"},
                "point_interaction": point[metric],
                "bootstrap_mean": float(values.mean()),
                "bootstrap_std": float(values.std(ddof=1)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                **bootstrap_signed_direction_fields(
                    values,
                    point[metric],
                ),
                "n_bootstrap_valid": int(len(values)),
                "n_bootstrap_requested": int(n_bootstrap),
                "bootstrap_unit": "subject_id",
                "n_examples": int(len(merged)),
                "n_patients": int(merged["subject_id"].nunique()),
                "n_positive": int(merged["y_true"].sum()),
            }
        )
    return pd.DataFrame(rows)


def last_episode_per_patient(ens: pd.DataFrame) -> pd.DataFrame:
    out = ens.sort_values(
        ["task", "compression_version", "subject_id", "prediction_time", "row_id"]
    )
    return out.groupby(
        ["task", "compression_version", "subject_id"],
        as_index=False,
    ).tail(1)


def main() -> None:
    args = parse_args()
    clearml_task = maybe_init_clearml(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.analysis_config)
    predictions_path, merge_manifest = prepare_predictions(args, config)
    pred = validate_wide_predictions(pd.read_csv(predictions_path), config)

    by_seed = metrics_by_seed(pred)
    mean_std = summarize_seed_metrics(by_seed)
    ens = make_ensemble(pred)
    ens_metrics = ensemble_metrics(ens)
    paired = run_comparisons(
        ens,
        config,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )
    equal_patient_metrics = equal_patient_weight_ensemble_metrics(ens)
    equal_patient_paired = run_comparisons_equal_patient_weight(
        ens,
        config,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )
    paired_seed, seed_direction_summary = seed_direction_tables(pred, paired, config)

    history_deltas = run_history_comparisons(
        config,
        args.sequence_data_dir,
        args.sequence_data_s3_prefix,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )

    context = pd.concat(
        [
            context_interaction_bootstrap(
                select_version(ens, task_name, "raw_4096"),
                select_version(ens, task_name, "condition_era_90_backfill_4096"),
                select_version(ens, task_name, "raw_16384"),
                select_version(ens, task_name, "condition_era_90_backfill_16384"),
                task_name,
                args.n_bootstrap,
                args.bootstrap_seed,
            )
            for task_name in ["guo_readmission", "guo_icu"]
        ],
        ignore_index=True,
    )

    last_episode = last_episode_per_patient(ens)
    last_episode_metrics = ensemble_metrics(last_episode)
    last_episode_paired = run_comparisons(
        last_episode,
        config,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )

    outputs = {
        "prediction_merge_manifest.csv": merge_manifest,
        "metrics_by_seed.csv": by_seed,
        "metrics_mean_std.csv": mean_std,
        "paired_seed_deltas.csv": paired_seed,
        "seed_direction_summary.csv": seed_direction_summary,
        "ensemble_predictions.csv": ens,
        "ensemble_metrics.csv": ens_metrics,
        "paired_patient_bootstrap_deltas.csv": paired,
        "equal_patient_weight_ensemble_metrics.csv": equal_patient_metrics,
        "equal_patient_weight_paired_bootstrap_deltas.csv": equal_patient_paired,
        "paired_history_coverage_deltas.csv": history_deltas,
        "context_interaction_bootstrap.csv": context,
        "last_episode_ensemble_metrics.csv": last_episode_metrics,
        "last_episode_paired_bootstrap_deltas.csv": last_episode_paired,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    resolved = {
        "prediction_schema": "wide_v1",
        "predictions": str(predictions_path),
        "analysis_config": str(args.analysis_config),
        "seeds": [int(x) for x in config["seeds"]],
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "main_risk_column": "risk_calibrated",
        "calibration_fit_split": "tuning",
        "analysis_split": "held_out",
        "n_prediction_rows": int(len(pred)),
        "n_ensemble_rows": int(len(ens)),
        "history_metrics": HISTORY_METRICS,
        "bootstrap_direction_columns": [
            "fraction_bootstrap_model_a_better",
            "fraction_bootstrap_model_b_better",
            "fraction_bootstrap_equal",
            "fraction_bootstrap_matching_point_direction",
        ],
        "equal_patient_weighting": (
            "Each subject has total weight 1, divided equally across that "
            "subject's episodes. Top-10% is defined by cumulative patient weight."
        ),
    }
    with (args.output_dir / "resolved_analysis_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)

    upload_manifest = pd.DataFrame()
    if not args.skip_upload and args.output_s3_prefix:
        upload_manifest = upload_tree(args.output_dir, args.output_s3_prefix)
        upload_manifest.to_csv(args.output_dir / "analysis_upload_manifest.csv", index=False)

    if clearml_task is not None:
        for filename, frame in outputs.items():
            clearml_task.upload_artifact(filename.removesuffix(".csv"), frame)
        clearml_task.upload_artifact("resolved_analysis_config", resolved)
        if len(upload_manifest):
            clearml_task.upload_artifact("analysis_upload_manifest", upload_manifest)

    print("=" * 100)
    print("STATE-OR-SPACE WIDE FINAL ANALYSIS DONE")
    print(f"Output: {args.output_dir}")
    print(f"Prediction rows: {len(pred)}")
    print(f"Ensemble rows: {len(ens)}")
    print(f"Paired predictive rows: {len(paired)}")
    print(f"Equal-patient-weight paired rows: {len(equal_patient_paired)}")
    print(f"Paired seed rows: {len(paired_seed)}")
    print(f"History delta rows: {len(history_deltas)}")
    print(f"Context interaction rows: {len(context)}")
    print("=" * 100)


if __name__ == "__main__":
    main()