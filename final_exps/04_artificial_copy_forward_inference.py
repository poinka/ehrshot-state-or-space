from __future__ import annotations

"""
Inference-only artificial copy-forward stress test for the frozen EHRSHOT models.

For each held-out prediction episode:
  1. deterministically select one already observed persistent diagnosis;
  2. require that it appears in at least two reconstructed visits;
  3. find later reconstructed visits before prediction_time where the code is absent;
  4. copy the same diagnosis into 0%, 25%, 50%, or 100% of those visits;
  5. rebuild raw_4096 and condition_era_90_backfill_4096 inputs;
  6. run the already trained checkpoints without any fitting or fine-tuning.

The same selected diagnosis and the same copied visits are used for raw and compressed
representations. Visit subsets are nested: the 25% subset is contained in the 50%
subset, which is contained in the 100% subset.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader


DEFAULT_FRACTIONS = [0.0, 0.25, 0.50, 1.0]
DEFAULT_VERSIONS = ["raw_4096", "condition_era_90_backfill_4096"]
EXAMPLE_KEYS = [
    "task",
    "row_id",
    "subject_id",
    "prediction_time",
    "label",
    "split",
]
FINAL_LONG_COLS = EXAMPLE_KEYS + [
    "time",
    "code",
    "numeric_value",
    "text_value",
    "days_before_prediction",
    "is_compression_token",
    "order_anchor_id",
    "role_order",
    "event_position",
]
UNK_ID = 1

HISTORY_COLS = EXAMPLE_KEYS + [
    "time",
    "code",
    "numeric_value",
    "text_value",
    "days_before_prediction",
    "compression_bucket",
    "source_event_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-only artificial copy-forward stress test."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="Config used to build the sequence datasets and history cache.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        action="append",
        required=True,
        help=(
            "Training run config containing run_set, seeds and runs. "
            "Pass more than once when seeds 42-44 and 45-46 are in different configs."
        ),
    )
    parser.add_argument(
        "--builder-script",
        type=Path,
        default=Path("final_exps/01_build_sequence_datasets.py"),
    )
    parser.add_argument(
        "--trainer-script",
        type=Path,
        default=Path("final_exps/02_train_sequence_multiseed.py"),
    )
    parser.add_argument(
        "--sequence-data-dir",
        type=Path,
        default=Path("ehrshot_state_or_space_sequence_datasets"),
    )
    parser.add_argument(
        "--sequence-data-s3-prefix",
        type=str,
        default="",
        help="S3 prefix used to download missing vocab.json files on a remote worker.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument(
        "--checkpoint-s3-prefix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        required=True,
        help="Combined wide held-out predictions from the final five-seed analysis.",
    )
    parser.add_argument(
        "--baseline-predictions-s3-url",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_copy_forward_perturbation"),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="guo_readmission,guo_icu",
    )
    parser.add_argument(
        "--compression-versions",
        type=str,
        default=",".join(DEFAULT_VERSIONS),
    )
    parser.add_argument(
        "--copy-fractions",
        type=str,
        default="0,0.25,0.5,1",
    )
    parser.add_argument(
        "--min-existing-visits",
        type=int,
        default=2,
        help="A selected diagnosis must already occur in at least this many visits.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=20260722,
        help="Used only for deterministic ranking of eligible visits; no model training.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--baseline-logit-tolerance",
        type=float,
        default=2e-4,
        help="Maximum allowed |new 0%% logit - saved baseline logit|.",
    )
    parser.add_argument(
        "--allow-baseline-mismatch",
        action="store_true",
        help="Warn instead of failing when the rebuilt 0%% input does not reproduce baseline logits.",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--reuse-perturbed-sequences",
        action="store_true",
        help="Reuse cached perturbed sequence parquet files when present.",
    )
    parser.add_argument(
        "--enable-clearml",
        action="store_true",
        help="Initialize ClearML logging for this inference-only task.",
    )
    parser.add_argument(
        "--execute-remotely",
        action="store_true",
        help="Enqueue the task to a ClearML agent and stop the local launcher process.",
    )
    parser.add_argument(
        "--clearml-queue",
        type=str,
        default="",
        help="ClearML queue name or queue ID used with --execute-remotely.",
    )
    parser.add_argument(
        "--clearml-docker-image",
        type=str,
        default="",
        help="Optional base Docker image for the ClearML agent task.",
    )
    parser.add_argument(
        "--clearml-project",
        type=str,
        default="pershin-medailab/EHR_Risk_Profiling/EHRSHOT",
    )
    parser.add_argument(
        "--clearml-task-name",
        type=str,
        default="state_or_space_artificial_copy_forward_inference",
    )
    parser.add_argument(
        "--clearml-output-uri",
        type=str,
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )
    parser.add_argument(
        "--clearml-tags",
        type=str,
        default="inference-only,copy-forward,stress-test",
        help="Comma-separated ClearML task tags.",
    )
    parser.add_argument(
        "--clearml-upload-artifacts",
        action="store_true",
        help="Upload output CSV/JSON files as ClearML artifacts after successful completion.",
    )
    return parser.parse_args()


def init_clearml_local(args: argparse.Namespace):
    """Initialize ClearML locally or enqueue the same task to a remote agent."""
    remote_agent_run = bool(
        os.environ.get("CLEARML_TASK_ID")
        or os.environ.get("TRAINS_TASK_ID")
    )

    if not args.enable_clearml and not remote_agent_run:
        return None

    from clearml import Task

    if not remote_agent_run:
        # Keep repository requirements instead of freezing the local macOS environment.
        requirements_path = Path("requirements.txt")
        if requirements_path.exists():
            Task.force_requirements_env_freeze(False, str(requirements_path))

        task = Task.init(
            project_name=args.clearml_project,
            task_name=args.clearml_task_name,
            task_type=Task.TaskTypes.inference,
            output_uri=args.clearml_output_uri or None,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
        )

        if args.clearml_docker_image.strip():
            task.set_base_docker(args.clearml_docker_image.strip())
    else:
        task = Task.current_task()
        if task is None:
            task = Task.init(
                project_name=args.clearml_project,
                task_name=args.clearml_task_name,
                task_type=Task.TaskTypes.inference,
                output_uri=args.clearml_output_uri or None,
                auto_connect_arg_parser=False,
                auto_connect_frameworks=False,
            )

    tags = parse_csv_list(args.clearml_tags)
    if tags:
        task.add_tags(tags)

    connected = {
        "dataset_config": str(args.dataset_config),
        "run_configs": [str(x) for x in args.run_config],
        "builder_script": str(args.builder_script),
        "trainer_script": str(args.trainer_script),
        "sequence_data_dir": str(args.sequence_data_dir),
        "sequence_data_s3_prefix": args.sequence_data_s3_prefix,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_s3_prefix": args.checkpoint_s3_prefix,
        "baseline_predictions": str(args.baseline_predictions),
        "baseline_predictions_s3_url": args.baseline_predictions_s3_url,
        "output_dir": str(args.output_dir),
        "tasks": args.tasks,
        "compression_versions": args.compression_versions,
        "copy_fractions": args.copy_fractions,
        "min_existing_visits": args.min_existing_visits,
        "selection_seed": args.selection_seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "baseline_logit_tolerance": args.baseline_logit_tolerance,
        "allow_baseline_mismatch": args.allow_baseline_mismatch,
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "reuse_perturbed_sequences": args.reuse_perturbed_sequences,
        "clearml_queue": args.clearml_queue,
        "clearml_docker_image": args.clearml_docker_image,
        "no_training": True,
        "execution_mode": "remote" if remote_agent_run else "local_launcher",
    }
    task.connect(connected, name="copy_forward_inference")

    print("ClearML inference task initialized:")
    print(f"  task_id = {task.id}")
    print(f"  project = {args.clearml_project}")
    print(f"  task_name = {args.clearml_task_name}")
    print(f"  remote_agent_run = {remote_agent_run}")
    print(f"  device = {args.device}")
    print("  no_training = True")

    if args.execute_remotely and not remote_agent_run:
        if not args.clearml_queue.strip():
            raise ValueError("--clearml-queue is required with --execute-remotely")
        print(f"Enqueueing ClearML task to queue: {args.clearml_queue}")
        task.execute_remotely(
            queue_name=args.clearml_queue.strip(),
            exit_process=True,
        )

    return task


def upload_clearml_outputs(task, output_dir: Path) -> None:
    """Upload produced CSV/JSON summaries and predictions as ClearML artifacts."""
    if task is None:
        return

    artifact_files = [
        "resolved_frozen_model_runs.csv",
        "copy_forward_episode_plan.csv",
        "copy_forward_eligible_visits.csv",
        "copy_forward_cohort_summary.csv",
        "copy_forward_injection_summary.csv",
        "zero_percent_baseline_agreement.csv",
        "copy_forward_metrics_by_seed.csv",
        "copy_forward_ensemble_metrics.csv",
        "copy_forward_representation_robustness_bootstrap.csv",
        "resolved_copy_forward_config.json",
        "copy_forward_predictions.csv",
        "copy_forward_ensemble_predictions.csv",
    ]

    for filename in artifact_files:
        path = output_dir / filename
        if not path.exists():
            print(f"ClearML artifact skipped, file not found: {path}")
            continue
        artifact_name = path.stem
        print(f"Uploading ClearML artifact: {artifact_name} <- {path}")
        task.upload_artifact(
            name=artifact_name,
            artifact_object=str(path.resolve()),
            wait_on_upload=False,
        )


def report_clearml_metrics(task, ensemble_metrics: pd.DataFrame, robustness: pd.DataFrame) -> None:
    """Report compact scalar series to the ClearML plots tab."""
    if task is None:
        return

    logger = task.get_logger()
    metric_names = [
        "auroc",
        "auprc",
        "brier",
        "logloss",
        "top_10pct_precision",
        "mean_abs_delta_risk_vs_0",
        "p95_abs_delta_risk_vs_0",
        "spearman_risk_vs_0",
    ]

    for row in ensemble_metrics.to_dict("records"):
        fraction_pct = int(round(float(row["copy_fraction"]) * 100))
        series = f'{row["task"]}/{row["compression_version"]}'
        for metric in metric_names:
            value = row.get(metric)
            if value is None or not np.isfinite(value):
                continue
            logger.report_scalar(
                title=f"copy_forward/{metric}",
                series=series,
                iteration=fraction_pct,
                value=float(value),
            )

    for row in robustness.to_dict("records"):
        fraction_pct = int(round(float(row["copy_fraction"]) * 100))
        logger.report_scalar(
            title="copy_forward/robustness_raw_minus_compressed",
            series=str(row["task"]),
            iteration=fraction_pct,
            value=float(row["point_mean"]),
        )


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_fraction_list(value: str) -> list[float]:
    fractions = sorted({float(x) for x in parse_csv_list(value)})
    if not fractions or fractions[0] != 0.0:
        raise ValueError("copy fractions must include 0")
    if any(x < 0 or x > 1 for x in fractions):
        raise ValueError("copy fractions must be in [0, 1]")
    return fractions


def load_module(path: Path, module_name: str):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(path.parent.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def stable_score(
    selection_seed: int,
    task: str,
    row_id: int,
    code: str,
    bucket: str,
) -> int:
    payload = f"{selection_seed}|{task}|{row_id}|{code}|{bucket}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def maybe_download_file(remote_url: str, local_path: Path) -> Path:
    if local_path.exists():
        return local_path
    if not remote_url:
        raise FileNotFoundError(local_path)
    from clearml import StorageManager

    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached = StorageManager.get_local_copy(remote_url=remote_url)
    if not cached:
        raise FileNotFoundError(f"Could not download {remote_url}")
    cached_path = Path(cached)
    if cached_path.is_dir():
        matches = list(cached_path.rglob(local_path.name))
        if not matches:
            raise FileNotFoundError(
                f"StorageManager returned directory without {local_path.name}: {cached_path}"
            )
        cached_path = matches[0]
    shutil.copy2(cached_path, local_path)
    return local_path


def ensure_sequence_vocabs(
    sequence_data_dir: Path,
    sequence_data_s3_prefix: str,
    tasks: Iterable[str],
    versions: Iterable[str],
) -> None:
    """Download only the vocab files required for frozen inference."""
    for task in tasks:
        for version in versions:
            local_path = sequence_data_dir / task / version / "vocab.json"
            if local_path.exists():
                continue
            if not sequence_data_s3_prefix.strip():
                raise FileNotFoundError(
                    f"Missing vocab file: {local_path}. "
                    "Pass --sequence-data-s3-prefix for remote execution."
                )
            remote_url = (
                sequence_data_s3_prefix.rstrip("/")
                + f"/{task}/{version}/vocab.json"
            )
            print(f"Download vocab: {remote_url} -> {local_path}")
            maybe_download_file(remote_url, local_path)


def ensure_history_cache(builder, tasks: Iterable[str]) -> None:
    """Build the strict pre-prediction history cache when it is absent."""
    missing: list[str] = []
    for task in tasks:
        manifest = builder.task_history_dir(task) / "manifest.json"
        if not manifest.exists():
            missing.append(task)
            continue
        try:
            if builder.validate_history_manifest(manifest) is None:
                missing.append(task)
        except Exception:
            missing.append(task)

    if not missing:
        print("History cache preflight: all requested task caches are available")
        return

    print(f"History cache missing for tasks={missing}; rebuilding from EHRSHOT_MEDS")
    builder.build_base_cache()
    for task in missing:
        labels = builder.load_labels(task)
        builder.build_task_history_parts(task, labels)


def read_run_sources(
    paths: list[Path],
    trainer_module,
    tasks: set[str],
    versions: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_set = str(payload["run_set"])
        seeds = [int(x) for x in payload.get("seeds", [])]
        for raw_run in payload["runs"]:
            run = trainer_module.normalize_run_cfg(raw_run)
            if run["task"] not in tasks:
                continue
            if run["compression_version"] not in versions:
                continue
            for seed in seeds:
                key = (run["task"], run["compression_version"], seed)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"run_set": run_set, "seed": seed, "run": run})
    if not rows:
        raise ValueError("No matching task/version/seed combinations were found in run configs")
    return rows


def resolve_history_parts(builder, task: str) -> list[Path]:
    task_dir = builder.task_history_dir(task)
    manifest_path = task_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"History cache manifest not found: {manifest_path}. "
            "Run the sequence dataset builder with keep_history_cache=true first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts: list[Path] = []
    for raw_path in manifest.get("parts", []):
        path = Path(raw_path)
        if not path.exists():
            path = task_dir / path.name
        if not path.exists():
            raise FileNotFoundError(f"History cache part not found: {path}")
        parts.append(path)
    if not parts:
        raise ValueError(f"No history parts listed in {manifest_path}")
    return parts


def build_candidate_plan_for_part(
    history: pl.DataFrame,
    persistent_codes: list[str],
    min_existing_visits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one selected diagnosis per episode and all eligible later visits."""
    history = history.filter(pl.col("split") == "held_out")
    if history.is_empty():
        return pd.DataFrame(), pd.DataFrame()

    reconstructed = history.filter(
        pl.col("compression_bucket").str.starts_with("reconstructed_visit=")
    )
    persistent = reconstructed.filter(pl.col("code").is_in(persistent_codes))
    if persistent.is_empty():
        return pd.DataFrame(), pd.DataFrame()

    code_stats = (
        persistent.group_by(EXAMPLE_KEYS + ["code"])
        .agg(
            pl.col("compression_bucket").n_unique().alias("n_existing_visits"),
            pl.len().alias("n_existing_mentions"),
            pl.col("time").min().alias("first_diagnosis_time"),
        )
        .filter(pl.col("n_existing_visits") >= int(min_existing_visits))
    )
    if code_stats.is_empty():
        return pd.DataFrame(), pd.DataFrame()

    visits = reconstructed.group_by(EXAMPLE_KEYS + ["compression_bucket"]).agg(
        pl.col("time").min().alias("visit_time")
    )
    existing_code_visits = persistent.select(
        EXAMPLE_KEYS + ["code", "compression_bucket"]
    ).unique()

    eligible_all = (
        code_stats.join(visits, on=EXAMPLE_KEYS, how="inner")
        .filter(pl.col("visit_time") > pl.col("first_diagnosis_time"))
        .join(
            existing_code_visits,
            on=EXAMPLE_KEYS + ["code", "compression_bucket"],
            how="anti",
        )
    )
    if eligible_all.is_empty():
        return pd.DataFrame(), pd.DataFrame()

    eligible_counts = eligible_all.group_by(EXAMPLE_KEYS + ["code"]).agg(
        pl.len().alias("n_eligible_visits")
    )
    ranked = (
        code_stats.join(eligible_counts, on=EXAMPLE_KEYS + ["code"], how="inner")
        .sort(
            [
                "row_id",
                "n_existing_visits",
                "n_existing_mentions",
                "n_eligible_visits",
                "first_diagnosis_time",
                "code",
            ],
            descending=[False, True, True, True, False, False],
        )
        .group_by(EXAMPLE_KEYS, maintain_order=True)
        .agg(
            pl.col("code").first().alias("candidate_code"),
            pl.col("n_existing_visits").first(),
            pl.col("n_existing_mentions").first(),
            pl.col("n_eligible_visits").first(),
            pl.col("first_diagnosis_time").first(),
        )
    )

    selected_eligible = (
        eligible_all.rename({"code": "candidate_code"})
        .join(
            ranked.select(
                EXAMPLE_KEYS
                + [
                    "candidate_code",
                    "n_eligible_visits",
                ]
            ),
            on=EXAMPLE_KEYS + ["candidate_code"],
            how="inner",
        )
        .select(
            EXAMPLE_KEYS
            + [
                "candidate_code",
                "n_eligible_visits",
                "compression_bucket",
                "visit_time",
            ]
        )
        .sort(["row_id", "visit_time", "compression_bucket"])
    )
    return ranked.to_pandas(), selected_eligible.to_pandas()


def build_candidate_plan(
    builder,
    tasks: Iterable[str],
    min_existing_visits: int,
    selection_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[Path]]]:
    episode_parts: list[pd.DataFrame] = []
    visit_parts: list[pd.DataFrame] = []
    history_parts_by_task: dict[str, list[Path]] = {}

    for task in tasks:
        history_parts = resolve_history_parts(builder, task)
        history_parts_by_task[task] = history_parts
        for index, path in enumerate(history_parts):
            history = pl.read_parquet(path).select(HISTORY_COLS)
            episodes, visits = build_candidate_plan_for_part(
                history=history,
                persistent_codes=builder.persistent_codes_list,
                min_existing_visits=min_existing_visits,
            )
            if not episodes.empty:
                episodes["history_part"] = str(path)
                episode_parts.append(episodes)
            if not visits.empty:
                visits["history_part"] = str(path)
                visit_parts.append(visits)
            print(
                f"[{task}] candidate plan part {index + 1}/{len(history_parts)}: "
                f"episodes={len(episodes)}, eligible_visits={len(visits)}",
                flush=True,
            )

    episode_plan = pd.concat(episode_parts, ignore_index=True) if episode_parts else pd.DataFrame()
    visit_plan = pd.concat(visit_parts, ignore_index=True) if visit_parts else pd.DataFrame()
    if episode_plan.empty or visit_plan.empty:
        raise ValueError("No eligible held-out episodes for artificial copying")

    visit_plan["selection_score"] = [
        stable_score(
            selection_seed=selection_seed,
            task=str(row.task),
            row_id=int(row.row_id),
            code=str(row.candidate_code),
            bucket=str(row.compression_bucket),
        )
        for row in visit_plan.itertuples(index=False)
    ]
    visit_plan = visit_plan.sort_values(
        ["task", "row_id", "selection_score", "visit_time", "compression_bucket"]
    ).reset_index(drop=True)
    visit_plan["eligible_rank_zero_based"] = visit_plan.groupby(
        ["task", "row_id"], sort=False
    ).cumcount()
    counts = visit_plan.groupby(["task", "row_id"], sort=False).size().rename("n_eligible_check")
    visit_plan = visit_plan.merge(counts, on=["task", "row_id"], validate="many_to_one")
    if not np.array_equal(
        visit_plan["n_eligible_check"].to_numpy(),
        visit_plan["n_eligible_visits"].to_numpy(),
    ):
        raise ValueError("Eligible visit count mismatch")
    visit_plan = visit_plan.drop(columns="n_eligible_check")
    return episode_plan, visit_plan, history_parts_by_task


def selected_visits_for_fraction(visit_plan: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if fraction <= 0:
        return visit_plan.iloc[0:0].copy()
    selected = visit_plan.copy()
    selected["n_to_copy"] = np.ceil(
        selected["n_eligible_visits"].astype(float) * float(fraction)
    ).astype(int)
    selected = selected[
        selected["eligible_rank_zero_based"] < selected["n_to_copy"]
    ].copy()
    selected["copied_visit_rank"] = selected.groupby(
        ["task", "row_id"], sort=False
    ).cumcount()
    return selected


def inject_copy_rows(
    history: pl.DataFrame,
    selected_visits: pd.DataFrame,
) -> pl.DataFrame:
    history = history.filter(pl.col("split") == "held_out").select(HISTORY_COLS)
    if selected_visits.empty:
        return history.sort(["row_id", "time", "source_event_id", "code"])

    row_ids = set(history["row_id"].to_list())
    selected = selected_visits[selected_visits["row_id"].isin(row_ids)].copy()
    if selected.empty:
        return history.sort(["row_id", "time", "source_event_id", "code"])

    selected["time"] = pd.to_datetime(selected["visit_time"])
    selected["code"] = selected["candidate_code"].astype(str)
    selected["numeric_value"] = np.nan
    selected["text_value"] = None
    selected["days_before_prediction"] = (
        pd.to_datetime(selected["prediction_time"]) - selected["time"]
    ).dt.total_seconds() / 86400.0
    selected["source_event_id"] = (
        7_000_000_000_000_000
        + selected["row_id"].astype(np.int64) * 100_000
        + selected["copied_visit_rank"].astype(np.int64)
    )
    synthetic = pl.from_pandas(selected[HISTORY_COLS]).with_columns(
        pl.col("numeric_value").cast(pl.Float32),
        pl.col("days_before_prediction").cast(pl.Float32),
        pl.col("source_event_id").cast(pl.Int64),
    )
    return pl.concat([history, synthetic], how="vertical_relaxed").sort(
        ["row_id", "time", "source_event_id", "code"]
    )


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        try:
            return lf.collect(streaming=True)
        except TypeError:
            return lf.collect()


def build_long_representation(
    builder,
    history: pl.DataFrame,
    compression_version: str,
) -> pl.DataFrame:
    if compression_version == "raw_4096":
        transformed = builder.raw_long(history.lazy())
        max_len = 4096
    elif compression_version == "condition_era_90_backfill_4096":
        transformed = builder.condition_era(history.lazy(), gap_days=90)
        max_len = 4096
    else:
        raise ValueError(f"Unsupported perturbation representation: {compression_version}")

    return collect_streaming(
        builder.assign_event_position(
            builder.truncate_transformed_last_n(transformed, max_len)
        )
    ).select(FINAL_LONG_COLS)


def aggregate_sequence_df(
    builder,
    final_long: pl.DataFrame,
    vocab: dict[str, int],
) -> pd.DataFrame:
    vocab_df = pl.DataFrame(
        {"code": list(vocab.keys()), "token_id": list(vocab.values())}
    )
    lf = (
        final_long.lazy()
        .join(vocab_df.lazy(), on="code", how="left")
        .with_columns(
            pl.col("token_id").fill_null(UNK_ID).cast(pl.Int32)
        )
        .sort(["row_id", "event_position"])
        .with_columns(
            pl.col("time")
            .diff()
            .over("row_id")
            .dt.total_days()
            .fill_null(0)
            .cast(pl.Float32)
            .alias("delta_days")
        )
    )
    agg = [
        pl.col("token_id").alias("token_ids"),
        pl.col("code").alias("codes"),
        pl.col("days_before_prediction").cast(pl.Float32).alias("days_before_prediction"),
        pl.col("delta_days").cast(pl.Float32).alias("delta_days"),
        pl.col("numeric_value").cast(pl.Float32).alias("numeric_values"),
        pl.len().cast(pl.Int32).alias("seq_len"),
    ]
    return collect_streaming(
        lf.group_by(EXAMPLE_KEYS, maintain_order=True).agg(agg)
    ).to_pandas()


def build_perturbed_sequences(
    builder,
    tasks: list[str],
    versions: list[str],
    fractions: list[float],
    episode_plan: pd.DataFrame,
    visit_plan: pd.DataFrame,
    history_parts_by_task: dict[str, list[Path]],
    sequence_data_dir: Path,
    output_dir: Path,
    reuse: bool,
) -> dict[tuple[str, str, float], Path]:
    cache_root = output_dir / "perturbed_sequences"
    paths: dict[tuple[str, str, float], Path] = {}

    for task in tasks:
        task_episode_ids = set(
            episode_plan.loc[episode_plan["task"] == task, "row_id"].astype(int)
        )
        if not task_episode_ids:
            continue
        for version in versions:
            vocab_path = sequence_data_dir / task / version / "vocab.json"
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            for fraction in fractions:
                label = f"{int(round(fraction * 100)):03d}"
                out_path = cache_root / task / version / f"copy_{label}.parquet"
                paths[(task, version, fraction)] = out_path
                if reuse and out_path.exists():
                    print(f"Reuse perturbed sequences: {out_path}")
                    continue

                selected = selected_visits_for_fraction(visit_plan, fraction)
                selected = selected[selected["task"] == task]
                seq_parts: list[pd.DataFrame] = []
                for part_index, history_path in enumerate(history_parts_by_task[task]):
                    history = pl.read_parquet(history_path).select(HISTORY_COLS)
                    history = history.filter(
                        (pl.col("split") == "held_out")
                        & pl.col("row_id").is_in(list(task_episode_ids))
                    )
                    if history.is_empty():
                        continue
                    perturbed = inject_copy_rows(history, selected)
                    final_long = build_long_representation(
                        builder=builder,
                        history=perturbed,
                        compression_version=version,
                    )
                    seq_parts.append(
                        aggregate_sequence_df(builder, final_long, vocab)
                    )
                    print(
                        f"[{task} | {version} | {fraction:.2f}] "
                        f"part {part_index + 1}/{len(history_parts_by_task[task])}",
                        flush=True,
                    )

                if not seq_parts:
                    raise ValueError(f"No sequences built for {task}/{version}/{fraction}")
                seq = pd.concat(seq_parts, ignore_index=True)
                seq = seq.merge(
                    episode_plan[
                        [
                            "task",
                            "row_id",
                            "candidate_code",
                            "n_existing_visits",
                            "n_existing_mentions",
                            "n_eligible_visits",
                        ]
                    ],
                    on=["task", "row_id"],
                    how="inner",
                    validate="one_to_one",
                )
                copied_counts = (
                    selected.groupby(["task", "row_id"]).size().rename("n_copied_visits")
                    if not selected.empty
                    else pd.Series(dtype=int, name="n_copied_visits")
                )
                if len(copied_counts):
                    seq = seq.merge(
                        copied_counts.reset_index(),
                        on=["task", "row_id"],
                        how="left",
                        validate="one_to_one",
                    )
                else:
                    seq["n_copied_visits"] = 0
                seq["n_copied_visits"] = seq["n_copied_visits"].fillna(0).astype(int)
                seq["requested_copy_fraction"] = float(fraction)
                seq["realized_copy_fraction"] = (
                    seq["n_copied_visits"] / seq["n_eligible_visits"].clip(lower=1)
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                seq.to_parquet(out_path, index=False)
                print(f"Saved perturbed sequences: {out_path} rows={len(seq)}")
    return paths


def load_baseline_predictions(args: argparse.Namespace) -> pd.DataFrame:
    path = maybe_download_file(
        args.baseline_predictions_s3_url,
        args.baseline_predictions,
    )
    df = pd.read_csv(path)
    required = {
        "task",
        "compression_version",
        "seed",
        "split",
        "row_id",
        "subject_id",
        "y_true",
        "logit",
        "risk_raw",
        "risk_calibrated",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Baseline wide predictions missing columns: {missing}")
    return df[df["split"] == "held_out"].copy()


def recover_platt_parameters(rows: pd.DataFrame) -> tuple[float, float, float]:
    p = np.clip(rows["risk_calibrated"].to_numpy(float), 1e-8, 1 - 1e-8)
    target = np.log(p / (1 - p))
    x = rows["logit"].to_numpy(float)
    design = np.column_stack([x, np.ones_like(x)])
    coef, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    reconstructed = coef * x + intercept
    max_abs_error = float(np.max(np.abs(reconstructed - target)))
    return float(coef), float(intercept), max_abs_error


def checkpoint_path_and_url(
    trainer,
    checkpoint_dir: Path,
    checkpoint_s3_prefix: str,
    run_set: str,
    run: dict[str, Any],
    seed: int,
) -> tuple[Path, str]:
    local = trainer.build_local_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        run_set=run_set,
        run_cfg=run,
        seed=seed,
    )
    remote = trainer.build_sequence_checkpoint_remote_url(
        checkpoint_s3_prefix=checkpoint_s3_prefix,
        run_set=run_set,
        run_cfg=run,
        seed=seed,
    )
    return local, remote


def load_frozen_model(
    trainer,
    source: dict[str, Any],
    checkpoint_dir: Path,
    checkpoint_s3_prefix: str,
    device: torch.device,
):
    run = source["run"]
    seed = int(source["seed"])
    local_path, remote_url = checkpoint_path_and_url(
        trainer=trainer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_s3_prefix=checkpoint_s3_prefix,
        run_set=source["run_set"],
        run=run,
        seed=seed,
    )
    maybe_download_file(remote_url, local_path)
    checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)

    model_args = SimpleNamespace(
        emb_dim=int(checkpoint.get("emb_dim", 64)),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        dropout=float(checkpoint.get("dropout", 0.20)),
    )
    model = trainer.make_model(
        run_cfg=run,
        vocab_size=int(checkpoint["vocab_size"]),
        args=model_args,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, local_path


def inference_one(
    trainer,
    model,
    checkpoint: dict[str, Any],
    run: dict[str, Any],
    sequence_df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    token_mean = np.asarray(checkpoint["token_numeric_mean"], dtype=np.float32)
    token_std = np.asarray(checkpoint["token_numeric_std"], dtype=np.float32)
    dataset = trainer.EHRSequenceDataset(sequence_df, max_len=int(run["max_len"]))
    collate = trainer.make_collate_fn(
        token_numeric_mean=token_mean,
        token_numeric_std=token_std,
        numeric_on=bool(run["numeric_on"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate,
    )
    output = trainer.run_inference(
        model=model,
        loader=loader,
        device=device,
        progress_enabled=True,
        progress_desc="copy-forward inference",
    )
    return pd.DataFrame(
        {
            "row_id": output["example_id"].astype(int),
            "subject_id": output["subject_id"].astype(int),
            "y_true": output["y_true"].astype(int),
            "logit": output["logits"].astype(float),
        }
    )


def compute_metrics(trainer, frame: pd.DataFrame, risk_col: str) -> dict[str, float]:
    y = frame["y_true"].to_numpy(int)
    p = frame[risk_col].to_numpy(float)
    metrics = trainer.binary_ranking_metrics(y, p)
    topk = trainer.topk_metrics(y, p).set_index("top_frac")
    metrics["top_10pct_precision"] = float(topk.loc[0.10, "top_k_event_rate"])
    metrics["top_10pct_lift"] = float(topk.loc[0.10, "top_k_lift"])
    metrics["top_10pct_capture"] = float(topk.loc[0.10, "event_capture"])
    return metrics


def patient_bootstrap_mean_delta(
    frame: pd.DataFrame,
    value_col: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    groups = [g[value_col].to_numpy(float) for _, g in frame.groupby("subject_id", sort=False)]
    if not groups:
        raise ValueError("No patients for bootstrap")
    point = float(frame[value_col].mean())
    rng = np.random.default_rng(seed)
    values = []
    n = len(groups)
    for _ in range(int(n_bootstrap)):
        sampled = rng.integers(0, n, size=n)
        sample = np.concatenate([groups[i] for i in sampled])
        values.append(float(sample.mean()))
    arr = np.asarray(values, dtype=float)
    return {
        "point_mean": point,
        "bootstrap_mean": float(arr.mean()),
        "bootstrap_std": float(arr.std(ddof=1)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "fraction_bootstrap_positive": float(np.mean(arr > 0)),
        "n_patients": int(frame["subject_id"].nunique()),
        "n_episodes": int(len(frame)),
    }


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clearml_task = init_clearml_local(args)

    builder_module = load_module(args.builder_script, "state_or_space_builder")
    trainer_module = load_module(args.trainer_script, "state_or_space_trainer")

    tasks = parse_csv_list(args.tasks)
    versions = parse_csv_list(args.compression_versions)
    fractions = parse_fraction_list(args.copy_fractions)
    if set(versions) != set(DEFAULT_VERSIONS):
        raise ValueError(
            "This script currently supports exactly raw_4096 and "
            "condition_era_90_backfill_4096."
        )

    dataset_config_path = args.dataset_config.resolve()
    dataset_cfg = builder_module.load_config(dataset_config_path)
    builder = builder_module.CachedSequenceBuilder(
        cfg=dataset_cfg,
        notebook_root=project_root,
        run_config_path=dataset_config_path,
        rebuild=False,
        rebuild_cache=False,
    )

    ensure_history_cache(builder, tasks)
    ensure_sequence_vocabs(
        sequence_data_dir=args.sequence_data_dir,
        sequence_data_s3_prefix=args.sequence_data_s3_prefix,
        tasks=tasks,
        versions=versions,
    )

    sources = read_run_sources(
        paths=args.run_config,
        trainer_module=trainer_module,
        tasks=set(tasks),
        versions=set(versions),
    )
    source_table = pd.DataFrame(
        [
            {
                "run_set": x["run_set"],
                "task": x["run"]["task"],
                "compression_version": x["run"]["compression_version"],
                "model_name": x["run"]["model_name"],
                "seed": x["seed"],
            }
            for x in sources
        ]
    ).sort_values(["task", "compression_version", "seed"])
    source_table.to_csv(args.output_dir / "resolved_frozen_model_runs.csv", index=False)
    print(source_table.to_string(index=False))

    episode_plan, visit_plan, history_parts_by_task = build_candidate_plan(
        builder=builder,
        tasks=tasks,
        min_existing_visits=args.min_existing_visits,
        selection_seed=args.selection_seed,
    )
    episode_plan.to_csv(args.output_dir / "copy_forward_episode_plan.csv", index=False)
    visit_plan.to_csv(args.output_dir / "copy_forward_eligible_visits.csv", index=False)

    plan_summary = (
        episode_plan.groupby("task")
        .agg(
            n_episodes=("row_id", "size"),
            n_patients=("subject_id", "nunique"),
            n_positive=("label", "sum"),
            mean_existing_visits=("n_existing_visits", "mean"),
            mean_eligible_visits=("n_eligible_visits", "mean"),
            median_eligible_visits=("n_eligible_visits", "median"),
        )
        .reset_index()
    )
    plan_summary.to_csv(args.output_dir / "copy_forward_cohort_summary.csv", index=False)
    print("\nPerturbation cohort:")
    print(plan_summary.to_string(index=False))

    sequence_paths = build_perturbed_sequences(
        builder=builder,
        tasks=tasks,
        versions=versions,
        fractions=fractions,
        episode_plan=episode_plan,
        visit_plan=visit_plan,
        history_parts_by_task=history_parts_by_task,
        sequence_data_dir=args.sequence_data_dir,
        output_dir=args.output_dir,
        reuse=args.reuse_perturbed_sequences,
    )

    baseline = load_baseline_predictions(args)
    device = get_device(args.device)
    print(f"DEVICE: {device}")

    prediction_parts: list[pd.DataFrame] = []
    agreement_rows: list[dict[str, Any]] = []

    for source in sources:
        run = source["run"]
        task = run["task"]
        version = run["compression_version"]
        seed = int(source["seed"])
        sequence_vocabulary = json.loads(
            (args.sequence_data_dir / task / version / "vocab.json").read_text(encoding="utf-8")
        )

        baseline_rows = baseline[
            (baseline["task"] == task)
            & (baseline["compression_version"] == version)
            & (baseline["seed"].astype(int) == seed)
        ].copy()
        if baseline_rows.empty:
            raise ValueError(f"No baseline wide predictions for {task}/{version}/seed{seed}")
        platt_coef, platt_intercept, platt_recovery_error = recover_platt_parameters(
            baseline_rows
        )

        model, checkpoint, checkpoint_path = load_frozen_model(
            trainer=trainer_module,
            source=source,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_s3_prefix=args.checkpoint_s3_prefix,
            device=device,
        )
        if int(checkpoint["vocab_size"]) != len(sequence_vocabulary):
            raise ValueError(
                f"Vocab size mismatch for {task}/{version}/seed{seed}: "
                f"checkpoint={checkpoint['vocab_size']}, vocab={len(sequence_vocabulary)}"
            )
        print(
            f"Frozen checkpoint: {task}/{version}/seed{seed} -> {checkpoint_path}"
        )

        for fraction in fractions:
            seq = pd.read_parquet(sequence_paths[(task, version, fraction)])
            inferred = inference_one(
                trainer=trainer_module,
                model=model,
                checkpoint=checkpoint,
                run=run,
                sequence_df=seq,
                device=device,
                batch_size=min(int(args.batch_size), int(run["batch_size"])),
                num_workers=args.num_workers,
            )
            inferred["risk_raw"] = sigmoid_np(inferred["logit"].to_numpy())
            inferred["risk_calibrated"] = sigmoid_np(
                platt_coef * inferred["logit"].to_numpy() + platt_intercept
            )
            inferred = inferred.merge(
                seq[
                    [
                        "task",
                        "row_id",
                        "prediction_time",
                        "candidate_code",
                        "n_existing_visits",
                        "n_existing_mentions",
                        "n_eligible_visits",
                        "n_copied_visits",
                        "requested_copy_fraction",
                        "realized_copy_fraction",
                        "seq_len",
                    ]
                ],
                on="row_id",
                how="left",
                validate="one_to_one",
            )
            inferred["compression_version"] = version
            inferred["representation"] = run["representation"]
            inferred["model"] = run["model_name"]
            inferred["seed"] = seed
            inferred["platt_coef_recovered"] = platt_coef
            inferred["platt_intercept_recovered"] = platt_intercept
            inferred["platt_recovery_max_abs_error"] = platt_recovery_error
            prediction_parts.append(inferred)

            if fraction == 0.0:
                check = inferred.merge(
                    baseline_rows[
                        [
                            "row_id",
                            "subject_id",
                            "y_true",
                            "logit",
                            "risk_calibrated",
                        ]
                    ].rename(
                        columns={
                            "logit": "saved_logit",
                            "risk_calibrated": "saved_risk_calibrated",
                        }
                    ),
                    on=["row_id", "subject_id", "y_true"],
                    how="inner",
                    validate="one_to_one",
                )
                max_logit = float(np.max(np.abs(check["logit"] - check["saved_logit"])))
                max_risk = float(
                    np.max(
                        np.abs(
                            check["risk_calibrated"] - check["saved_risk_calibrated"]
                        )
                    )
                )
                agreement_rows.append(
                    {
                        "task": task,
                        "compression_version": version,
                        "seed": seed,
                        "n_compared": len(check),
                        "max_abs_logit_difference": max_logit,
                        "max_abs_calibrated_risk_difference": max_risk,
                        "platt_recovery_max_abs_logit_error": platt_recovery_error,
                    }
                )
                if max_logit > args.baseline_logit_tolerance:
                    message = (
                        f"0% baseline mismatch for {task}/{version}/seed{seed}: "
                        f"max |delta logit|={max_logit:.6g} > "
                        f"{args.baseline_logit_tolerance:.6g}"
                    )
                    if args.allow_baseline_mismatch:
                        print("WARNING:", message)
                    else:
                        raise RuntimeError(message)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    key = ["task", "compression_version", "seed", "row_id"]
    baseline_zero = predictions[predictions["requested_copy_fraction"] == 0.0][
        key + ["logit", "risk_raw", "risk_calibrated"]
    ].rename(
        columns={
            "logit": "logit_at_0",
            "risk_raw": "risk_raw_at_0",
            "risk_calibrated": "risk_calibrated_at_0",
        }
    )
    predictions = predictions.merge(baseline_zero, on=key, how="left", validate="many_to_one")
    predictions["delta_logit_vs_0"] = predictions["logit"] - predictions["logit_at_0"]
    predictions["delta_risk_raw_vs_0"] = predictions["risk_raw"] - predictions["risk_raw_at_0"]
    predictions["delta_risk_calibrated_vs_0"] = (
        predictions["risk_calibrated"] - predictions["risk_calibrated_at_0"]
    )
    predictions["abs_delta_risk_calibrated_vs_0"] = predictions[
        "delta_risk_calibrated_vs_0"
    ].abs()
    predictions.to_csv(args.output_dir / "copy_forward_predictions.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(
        args.output_dir / "zero_percent_baseline_agreement.csv", index=False
    )

    metrics_seed_rows: list[dict[str, Any]] = []
    for group_key, group in predictions.groupby(
        [
            "task",
            "compression_version",
            "representation",
            "model",
            "seed",
            "requested_copy_fraction",
        ],
        sort=False,
    ):
        task, version, representation, model_name, seed, fraction = group_key
        metrics_seed_rows.append(
            {
                "task": task,
                "compression_version": version,
                "representation": representation,
                "model": model_name,
                "seed": int(seed),
                "copy_fraction": float(fraction),
                **compute_metrics(trainer_module, group, "risk_calibrated"),
                "mean_delta_risk_vs_0": float(group["delta_risk_calibrated_vs_0"].mean()),
                "mean_abs_delta_risk_vs_0": float(group["abs_delta_risk_calibrated_vs_0"].mean()),
                "median_abs_delta_risk_vs_0": float(group["abs_delta_risk_calibrated_vs_0"].median()),
                "p95_abs_delta_risk_vs_0": float(
                    group["abs_delta_risk_calibrated_vs_0"].quantile(0.95)
                ),
                "spearman_risk_vs_0": float(
                    group[["risk_calibrated", "risk_calibrated_at_0"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
            }
        )
    metrics_by_seed = pd.DataFrame(metrics_seed_rows)
    metrics_by_seed.to_csv(args.output_dir / "copy_forward_metrics_by_seed.csv", index=False)

    ensemble = (
        predictions.groupby(
            [
                "task",
                "compression_version",
                "representation",
                "model",
                "requested_copy_fraction",
                "row_id",
                "subject_id",
                "y_true",
                "candidate_code",
                "n_existing_visits",
                "n_eligible_visits",
                "n_copied_visits",
                "realized_copy_fraction",
            ],
            as_index=False,
        )
        .agg(
            risk_calibrated=("risk_calibrated", "mean"),
            risk_std_across_seeds=("risk_calibrated", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    ensemble_zero = ensemble[ensemble["requested_copy_fraction"] == 0.0][
        ["task", "compression_version", "row_id", "risk_calibrated"]
    ].rename(columns={"risk_calibrated": "risk_calibrated_at_0"})
    ensemble = ensemble.merge(
        ensemble_zero,
        on=["task", "compression_version", "row_id"],
        how="left",
        validate="many_to_one",
    )
    ensemble["delta_risk_vs_0"] = (
        ensemble["risk_calibrated"] - ensemble["risk_calibrated_at_0"]
    )
    ensemble["abs_delta_risk_vs_0"] = ensemble["delta_risk_vs_0"].abs()
    ensemble.to_csv(args.output_dir / "copy_forward_ensemble_predictions.csv", index=False)

    ensemble_metric_rows: list[dict[str, Any]] = []
    for group_key, group in ensemble.groupby(
        [
            "task",
            "compression_version",
            "representation",
            "model",
            "requested_copy_fraction",
        ],
        sort=False,
    ):
        task, version, representation, model_name, fraction = group_key
        ensemble_metric_rows.append(
            {
                "task": task,
                "compression_version": version,
                "representation": representation,
                "model": model_name,
                "copy_fraction": float(fraction),
                **compute_metrics(trainer_module, group, "risk_calibrated"),
                "mean_delta_risk_vs_0": float(group["delta_risk_vs_0"].mean()),
                "mean_abs_delta_risk_vs_0": float(group["abs_delta_risk_vs_0"].mean()),
                "median_abs_delta_risk_vs_0": float(group["abs_delta_risk_vs_0"].median()),
                "p95_abs_delta_risk_vs_0": float(group["abs_delta_risk_vs_0"].quantile(0.95)),
                "spearman_risk_vs_0": float(
                    group[["risk_calibrated", "risk_calibrated_at_0"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
            }
        )
    ensemble_metrics = pd.DataFrame(ensemble_metric_rows)
    ensemble_metrics.to_csv(args.output_dir / "copy_forward_ensemble_metrics.csv", index=False)

    robustness_rows: list[dict[str, Any]] = []
    for task in tasks:
        for fraction in [x for x in fractions if x > 0]:
            raw = ensemble[
                (ensemble["task"] == task)
                & (ensemble["compression_version"] == "raw_4096")
                & (ensemble["requested_copy_fraction"] == fraction)
            ][
                ["row_id", "subject_id", "y_true", "abs_delta_risk_vs_0"]
            ].rename(columns={"abs_delta_risk_vs_0": "abs_delta_raw"})
            compressed = ensemble[
                (ensemble["task"] == task)
                & (
                    ensemble["compression_version"]
                    == "condition_era_90_backfill_4096"
                )
                & (ensemble["requested_copy_fraction"] == fraction)
            ][
                ["row_id", "subject_id", "y_true", "abs_delta_risk_vs_0"]
            ].rename(columns={"abs_delta_risk_vs_0": "abs_delta_compressed"})
            merged = raw.merge(
                compressed,
                on=["row_id", "subject_id", "y_true"],
                how="inner",
                validate="one_to_one",
            )
            merged["raw_minus_compressed_abs_delta"] = (
                merged["abs_delta_raw"] - merged["abs_delta_compressed"]
            )
            stats = patient_bootstrap_mean_delta(
                frame=merged,
                value_col="raw_minus_compressed_abs_delta",
                n_bootstrap=args.bootstrap,
                seed=args.bootstrap_seed,
            )
            robustness_rows.append(
                {
                    "task": task,
                    "copy_fraction": float(fraction),
                    "comparison": "raw_abs_change_minus_compressed_abs_change",
                    "positive_means": "compressed representation is less sensitive",
                    "mean_abs_delta_raw": float(merged["abs_delta_raw"].mean()),
                    "mean_abs_delta_compressed": float(
                        merged["abs_delta_compressed"].mean()
                    ),
                    **stats,
                }
            )
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(
        args.output_dir / "copy_forward_representation_robustness_bootstrap.csv",
        index=False,
    )

    injection_rows: list[dict[str, Any]] = []
    for task in tasks:
        for fraction in fractions:
            selected = selected_visits_for_fraction(
                visit_plan[visit_plan["task"] == task], fraction
            )
            n_episodes = int(episode_plan[episode_plan["task"] == task]["row_id"].nunique())
            injection_rows.append(
                {
                    "task": task,
                    "requested_copy_fraction": float(fraction),
                    "n_eligible_episodes": n_episodes,
                    "n_copied_visits_total": int(len(selected)),
                    "n_episodes_with_at_least_one_copy": int(selected["row_id"].nunique()),
                    "mean_copied_visits_per_episode": float(len(selected) / n_episodes),
                    "mean_realized_fraction": float(
                        (
                            selected.groupby("row_id").size()
                            / episode_plan[episode_plan["task"] == task]
                            .set_index("row_id")["n_eligible_visits"]
                        ).fillna(0).mean()
                    )
                    if fraction > 0
                    else 0.0,
                }
            )
    pd.DataFrame(injection_rows).to_csv(
        args.output_dir / "copy_forward_injection_summary.csv", index=False
    )

    resolved = {
        "dataset_config": str(args.dataset_config),
        "run_configs": [str(x) for x in args.run_config],
        "tasks": tasks,
        "compression_versions": versions,
        "copy_fractions": fractions,
        "min_existing_visits": args.min_existing_visits,
        "selection_seed": args.selection_seed,
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "device": str(device),
        "no_training": True,
        "candidate_rule": (
            "persistent whitelist code observed in >= min_existing_visits reconstructed "
            "visits; choose highest visit count, then mentions, eligible later visits, "
            "earliest first occurrence, lexicographic code"
        ),
        "visit_selection_rule": (
            "eligible later reconstructed visits where the code is absent; stable hash ranking; "
            "select ceil(fraction * n_eligible), producing nested subsets"
        ),
    }
    (args.output_dir / "resolved_copy_forward_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if clearml_task is not None:
        report_clearml_metrics(clearml_task, ensemble_metrics, robustness)
        if args.clearml_upload_artifacts:
            upload_clearml_outputs(clearml_task, args.output_dir)
        clearml_task.flush(wait_for_uploads=True)
        clearml_task.close()

    print("\nARTIFICIAL COPY-FORWARD INFERENCE TEST DONE")
    print(f"Output: {args.output_dir}")
    print("\nRepresentation robustness:")
    print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()