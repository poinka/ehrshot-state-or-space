#!/usr/bin/env python3
from __future__ import annotations

"""
Final analysis for the State-or-Space experiment.

Main outputs:
  - metrics by seed and mean±std;
  - calibrated seed ensemble predictions and metrics;
  - paired patient-level bootstrap deltas (10,000 by default);
  - difference-in-differences for context interaction;
  - ICU gap sensitivity;
  - supplemental last-episode-per-patient analysis.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common_ehrshot_eval import binary_ranking_metrics, topk_metrics


S3_BASE = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT"
)
METRICS = ["auroc", "auprc", "brier", "logloss", "top_10pct_precision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "ehrshot_state_or_space_final_sequence_results/"
            "sequence_multiseed_heldout_predictions.csv"
        ),
    )
    parser.add_argument(
        "--predictions-s3-url",
        default=(
            f"{S3_BASE}/ehrshot_state_or_space_final_sequence_results/"
            "sequence_multiseed_heldout_predictions.csv"
        ),
    )
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
        default=Path("configs/state_or_space_analysis.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_state_or_space_final_analysis"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-s3-prefix",
        default=f"{S3_BASE}/ehrshot_state_or_space_final_analysis",
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
        default="state_or_space_final_analysis",
    )
    parser.add_argument(
        "--clearml-output-uri",
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )
    return parser.parse_args()


def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


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

    cfg = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    connected = dict(task.connect(cfg))

    path_keys = {"predictions", "sequence_data_dir", "analysis_config", "output_dir"}
    int_keys = {"n_bootstrap", "bootstrap_seed"}
    bool_keys = {"skip_upload"}
    for key, value in connected.items():
        if not hasattr(args, key) or key in {"enable_clearml", "execute_remotely"}:
            continue
        if key in path_keys:
            setattr(args, key, Path(value))
        elif key in int_keys:
            setattr(args, key, int(value))
        elif key in bool_keys:
            setattr(args, key, str(value).lower() in {"1", "true", "yes", "y"})
        else:
            setattr(args, key, value)

    if args.execute_remotely and not remote:
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)
    return task


def download_if_missing(local_path: Path, remote_url: str) -> Path:
    local_path = Path(local_path)
    if local_path.exists():
        return local_path
    if not remote_url:
        raise FileNotFoundError(local_path)

    from clearml import StorageManager

    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(StorageManager.get_local_copy(remote_url=remote_url))
    if not cached.exists():
        raise FileNotFoundError(f"StorageManager returned missing path: {cached}")
    import shutil
    shutil.copy2(cached, local_path)
    return local_path


def upload_tree(local_root: Path, remote_prefix: str) -> pd.DataFrame:
    if not remote_prefix:
        return pd.DataFrame()
    from clearml import StorageManager

    rows = []
    for path in sorted(Path(local_root).rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root).as_posix()
        remote = f"{remote_prefix.rstrip('/')}/{rel}"
        StorageManager.upload_file(
            local_file=str(path),
            remote_url=remote,
            wait_for_upload=True,
        )
        rows.append({"local_path": str(path), "remote_url": remote})
    return pd.DataFrame(rows)


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


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


def validate_predictions(pred: pd.DataFrame, expected_seeds: list[int]) -> pd.DataFrame:
    required = {
        "task", "model_name", "compression_version", "calibration", "seed",
        "split", "example_id", "subject_id", "y_true", "pred_proba",
    }
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    pred = pred[(pred["split"] == "held_out") & (pred["calibration"] == "platt")].copy()
    pred["seed"] = pred["seed"].astype(int)
    pred["example_id"] = pred["example_id"].astype(int)
    pred["subject_id"] = pred["subject_id"].astype(int)
    pred["y_true"] = pred["y_true"].astype(int)
    pred["pred_proba"] = pred["pred_proba"].astype(float)

    dup_key = ["task", "compression_version", "seed", "example_id"]
    if pred.duplicated(dup_key).any():
        raise ValueError("Duplicate calibrated prediction rows detected")

    actual_seeds = sorted(pred["seed"].unique().tolist())
    if actual_seeds != sorted(expected_seeds):
        raise ValueError(f"Expected seeds {expected_seeds}, got {actual_seeds}")

    for (task, version), part in pred.groupby(["task", "compression_version"]):
        counts = part.groupby("seed")["example_id"].nunique()
        if counts.nunique() != 1:
            raise ValueError(f"Unequal example counts across seeds: {task}/{version}: {counts.to_dict()}")
    return pred


def metrics_by_seed(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "task", "model_name", "model_family", "representation",
        "compression_version", "numeric_on", "calibration", "seed",
    ]
    for key, part in pred.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            metric_bundle(
                part["y_true"].to_numpy(),
                part["pred_proba"].to_numpy(),
                part["example_id"].to_numpy(),
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_seed_metrics(by_seed: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "task", "model_name", "model_family", "representation",
        "compression_version", "numeric_on", "calibration",
    ]
    metric_cols = [
        c for c in [
            "auroc", "auprc", "brier", "logloss", "top_10pct_precision",
            "top_10pct_lift", "top_10pct_event_capture",
        ] if c in by_seed.columns
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
        "task", "model_name", "model_family", "representation",
        "compression_version", "numeric_on", "calibration", "split",
        "example_id", "subject_id", "y_true",
    ]
    ens = (
        pred.groupby(group_cols, dropna=False)
        .agg(
            pred_proba=("pred_proba", "mean"),
            pred_proba_std=("pred_proba", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    ens["pred_proba_std"] = ens["pred_proba_std"].fillna(0.0)
    if ens["n_seeds"].nunique() != 1:
        raise ValueError("Ensemble rows have inconsistent n_seeds")
    return ens


def ensemble_metrics(ens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "task", "model_name", "model_family", "representation",
        "compression_version", "numeric_on", "calibration",
    ]
    for key, part in ens.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            metric_bundle(
                part["y_true"].to_numpy(),
                part["pred_proba"].to_numpy(),
                part["example_id"].to_numpy(),
            )
        )
        row["n_patients"] = int(part["subject_id"].nunique())
        row["n_seeds"] = int(part["n_seeds"].min())
        rows.append(row)
    return pd.DataFrame(rows)


def select_version(ens: pd.DataFrame, task: str, version: str) -> pd.DataFrame:
    out = ens[(ens["task"] == task) & (ens["compression_version"] == version)].copy()
    if out.empty:
        raise ValueError(f"No ensemble predictions for {task}/{version}")
    return out


def paired_merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    keep = ["example_id", "subject_id", "y_true", "pred_proba"]
    merged = (
        a[keep].rename(columns={"pred_proba": "pred_a"})
        .merge(
            b[keep].rename(columns={"pred_proba": "pred_b"}),
            on=["example_id", "subject_id", "y_true"],
            how="inner",
            validate="one_to_one",
        )
    )
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Paired comparison does not cover all examples")
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
    tie = merged["example_id"].to_numpy(dtype=int)
    groups = cluster_groups(merged)
    rng = np.random.default_rng(seed)

    point_a = metric_bundle(y, pa, tie)
    point_b = metric_bundle(y, pb, tie)
    boot = {metric: [] for metric in METRICS}

    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in sampled])
        sample_tie = np.arange(len(idx), dtype=np.int64)
        ma = metric_bundle(y[idx], pa[idx], sample_tie)
        mb = metric_bundle(y[idx], pb[idx], sample_tie)
        for metric in METRICS:
            boot[metric].append(ma[metric] - mb[metric])

    rows = []
    for metric in METRICS:
        values = np.asarray(boot[metric], dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "comparison": comparison_name,
                "task": task,
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "higher_is_better": metric not in {"brier", "logloss"},
                "model_a_value": point_a[metric],
                "model_b_value": point_b[metric],
                "point_delta_a_minus_b": point_a[metric] - point_b[metric],
                "bootstrap_mean_delta": float(values.mean()),
                "bootstrap_std_delta": float(values.std(ddof=1)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "n_bootstrap_valid": int(len(values)),
                "n_bootstrap_requested": int(n_bootstrap),
                "bootstrap_unit": "subject_id",
                "n_paired_examples": int(len(merged)),
                "n_paired_patients": int(merged["subject_id"].nunique()),
                "n_paired_positive": int(merged["y_true"].sum()),
            }
        )
    return pd.DataFrame(rows)


def context_interaction_bootstrap(
    raw4096: pd.DataFrame,
    back4096: pd.DataFrame,
    raw16384: pd.DataFrame,
    back16384: pd.DataFrame,
    task: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    key = ["example_id", "subject_id", "y_true"]
    merged = (
        raw4096[key + ["pred_proba"]].rename(columns={"pred_proba": "raw4096"})
        .merge(back4096[key + ["pred_proba"]].rename(columns={"pred_proba": "back4096"}), on=key, validate="one_to_one")
        .merge(raw16384[key + ["pred_proba"]].rename(columns={"pred_proba": "raw16384"}), on=key, validate="one_to_one")
        .merge(back16384[key + ["pred_proba"]].rename(columns={"pred_proba": "back16384"}), on=key, validate="one_to_one")
    )
    y = merged["y_true"].to_numpy(dtype=int)
    tie = merged["example_id"].to_numpy(dtype=int)
    arrays = {c: merged[c].to_numpy(dtype=float) for c in ["raw4096", "back4096", "raw16384", "back16384"]}
    groups = cluster_groups(merged)
    rng = np.random.default_rng(seed)

    def interaction(idx: np.ndarray, sample_tie: np.ndarray) -> dict[str, float]:
        m = {name: metric_bundle(y[idx], arr[idx], sample_tie) for name, arr in arrays.items()}
        return {
            metric: (m["back16384"][metric] - m["raw16384"][metric])
            - (m["back4096"][metric] - m["raw4096"][metric])
            for metric in METRICS
        }

    point = interaction(np.arange(len(merged)), tie)
    boot = {metric: [] for metric in METRICS}
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in sampled])
        values = interaction(idx, np.arange(len(idx), dtype=np.int64))
        for metric in METRICS:
            boot[metric].append(values[metric])

    rows = []
    for metric in METRICS:
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
                "n_bootstrap_valid": int(len(values)),
                "n_examples": int(len(merged)),
                "n_patients": int(merged["subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def ensure_raw_examples_for_last_episode(
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    tasks: list[str],
) -> None:
    if not sequence_data_s3_prefix:
        return
    from clearml import StorageManager
    import shutil

    for task in tasks:
        local = Path(sequence_data_dir) / task / "raw_4096" / "examples.parquet"
        if local.exists():
            continue
        remote = (
            f"{sequence_data_s3_prefix.rstrip('/')}/"
            f"{task}/raw_4096/examples.parquet"
        )
        print(f"Downloading sequence examples for last-episode analysis: {remote}")
        cached = Path(StorageManager.get_local_copy(remote_url=remote))
        if not cached.exists():
            raise FileNotFoundError(f"StorageManager returned missing path: {cached}")
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, local)


def add_prediction_time_for_last_episode(
    ens: pd.DataFrame,
    sequence_data_dir: Path,
) -> pd.DataFrame:
    parts = []
    for task, task_part in ens.groupby("task"):
        examples_path = Path(sequence_data_dir) / task / "raw_4096" / "examples.parquet"
        if not examples_path.exists():
            continue
        ex = pd.read_parquet(examples_path, columns=["row_id", "prediction_time"])
        ex = ex.rename(columns={"row_id": "example_id"}).drop_duplicates("example_id")
        tmp = task_part.merge(ex, on="example_id", how="left", validate="many_to_one")
        parts.append(tmp)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["task", "compression_version", "subject_id", "prediction_time", "example_id"])
    return out.groupby(["task", "compression_version", "subject_id"], as_index=False).tail(1)


def run_comparisons(
    ens: pd.DataFrame,
    config: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for cfg in config["paired_comparisons"]:
        task = cfg["task"]
        a_version = cfg["model_a"]
        b_version = cfg["model_b"]
        a = select_version(ens, task, a_version)
        b = select_version(ens, task, b_version)
        merged = paired_merge(a, b)
        rows.append(
            paired_bootstrap(
                merged=merged,
                comparison_name=cfg["name"],
                task=task,
                model_a=a_version,
                model_b=b_version,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    task = maybe_init_clearml(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = download_if_missing(args.predictions, args.predictions_s3_url)
    config = load_config(args.analysis_config)
    pred = pd.read_csv(predictions_path)
    pred = validate_predictions(pred, expected_seeds=[int(x) for x in config["seeds"]])

    by_seed = metrics_by_seed(pred)
    mean_std = summarize_seed_metrics(by_seed)
    ens = make_ensemble(pred)
    ens_metrics = ensemble_metrics(ens)
    paired = run_comparisons(
        ens,
        config=config,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )

    context_rows = []
    for task_name in ["guo_readmission", "guo_icu"]:
        context_rows.append(
            context_interaction_bootstrap(
                raw4096=select_version(ens, task_name, "raw_4096"),
                back4096=select_version(ens, task_name, "condition_era_90_backfill_4096"),
                raw16384=select_version(ens, task_name, "raw_16384"),
                back16384=select_version(ens, task_name, "condition_era_90_backfill_16384"),
                task=task_name,
                n_bootstrap=args.n_bootstrap,
                seed=args.bootstrap_seed,
            )
        )
    context = pd.concat(context_rows, ignore_index=True)

    ensure_raw_examples_for_last_episode(
        args.sequence_data_dir,
        args.sequence_data_s3_prefix,
        tasks=["guo_readmission", "guo_icu"],
    )
    last_episode = add_prediction_time_for_last_episode(ens, args.sequence_data_dir)
    if len(last_episode):
        last_episode_metrics = ensemble_metrics(last_episode)
        last_episode_paired = run_comparisons(
            last_episode,
            config=config,
            n_bootstrap=args.n_bootstrap,
            seed=args.bootstrap_seed,
        )
    else:
        last_episode_metrics = pd.DataFrame()
        last_episode_paired = pd.DataFrame()

    outputs = {
        "metrics_by_seed.csv": by_seed,
        "metrics_mean_std.csv": mean_std,
        "ensemble_predictions.csv": ens,
        "ensemble_metrics.csv": ens_metrics,
        "paired_patient_bootstrap_deltas.csv": paired,
        "context_interaction_bootstrap.csv": context,
        "last_episode_ensemble_metrics.csv": last_episode_metrics,
        "last_episode_paired_bootstrap_deltas.csv": last_episode_paired,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    resolved = {
        "predictions": str(predictions_path),
        "analysis_config": str(args.analysis_config),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "n_prediction_rows": int(len(pred)),
        "n_ensemble_rows": int(len(ens)),
    }
    with (args.output_dir / "resolved_analysis_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)

    upload_manifest = pd.DataFrame()
    if not args.skip_upload and args.output_s3_prefix:
        upload_manifest = upload_tree(args.output_dir, args.output_s3_prefix)
        upload_manifest.to_csv(args.output_dir / "analysis_upload_manifest.csv", index=False)

    if task is not None:
        for name, frame in outputs.items():
            task.upload_artifact(name.removesuffix(".csv"), frame)
        task.upload_artifact("resolved_analysis_config", resolved)
        if len(upload_manifest):
            task.upload_artifact("analysis_upload_manifest", upload_manifest)

    print("=" * 100)
    print("STATE-OR-SPACE FINAL ANALYSIS DONE")
    print(f"Output: {args.output_dir}")
    print(f"Prediction rows: {len(pred)}")
    print(f"Ensemble rows: {len(ens)}")
    print(f"Paired bootstrap rows: {len(paired)}")
    print(f"Context interaction rows: {len(context)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
