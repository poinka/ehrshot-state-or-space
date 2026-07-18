#!/usr/bin/env python3
from __future__ import annotations

"""
Build a train-only empirical persistence whitelist from EHRSHOT MEDS.

Selection rule (frozen protocol):
    n_subjects_with_code >= 50
    and repeat_day_subject_share >= 0.50
    and (
        persistent_365d_subject_share >= 0.10
        or p75_span_days >= 365
    )

Only condition_occurrence events from TRAIN subjects are used.
The final file is:
    strong_empirical_chronic_like_diagnosis_codes_train_only.csv
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


S3_BASE = (
    "s3://api.blackhole2.ai.innopolis.university:443/"
    "pershin-medailab/pershin-medailab/EHR_Risk_Profiling/EHRSHOT"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ehrshot-root", type=Path, default=Path("EHRSHOT_MEDS"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehrshot_train_only_chronic_whitelist_50"),
    )
    parser.add_argument("--dataset-version", default="EHRSHOT_MEDS_local")
    parser.add_argument("--train-split-name", default="train")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-code-count", type=int, default=0)

    parser.add_argument("--strong-min-subjects", type=int, default=50)
    parser.add_argument("--strong-min-repeat-share", type=float, default=0.50)
    parser.add_argument("--strong-min-persistent-365-share", type=float, default=0.10)
    parser.add_argument("--strong-min-p75-span-days", type=float, default=365.0)

    parser.add_argument("--output-s3-prefix", default=f"{S3_BASE}/state_or_space_whitelist")
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
        default="state_or_space_build_train_only_whitelist",
    )
    parser.add_argument(
        "--clearml-output-uri",
        default="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        try:
            return lf.collect(streaming=True)
        except TypeError:
            return lf.collect()


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

    connected = dict(
        task.connect(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
        )
    )

    path_keys = {"ehrshot_root", "output_dir"}
    bool_keys = {"overwrite", "skip_upload"}
    int_keys = {"expected_code_count", "strong_min_subjects"}
    float_keys = {
        "strong_min_repeat_share",
        "strong_min_persistent_365_share",
        "strong_min_p75_span_days",
    }

    for key, value in connected.items():
        if not hasattr(args, key) or key in {"enable_clearml", "execute_remotely"}:
            continue
        if key in path_keys:
            setattr(args, key, Path(value))
        elif key in bool_keys:
            setattr(args, key, str(value).lower() in {"1", "true", "yes", "y"})
        elif key in int_keys:
            setattr(args, key, int(value))
        elif key in float_keys:
            setattr(args, key, float(value))
        else:
            setattr(args, key, value)

    if args.execute_remotely and not remote:
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)
    return task


def upload_tree(local_root: Path, remote_prefix: str) -> pd.DataFrame:
    if not remote_prefix:
        return pd.DataFrame()
    from clearml import StorageManager

    rows: list[dict[str, str]] = []
    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_root).as_posix()
        remote = f"{remote_prefix.rstrip('/')}/{relative}"
        StorageManager.upload_file(
            local_file=str(path),
            remote_url=remote,
            wait_for_upload=True,
        )
        rows.append({"local_path": str(path), "remote_url": remote})
    return pd.DataFrame(rows)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in path.iterdir() if p.name != ".DS_Store"]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to rebuild."
        )
    if overwrite:
        for p in existing:
            if p.is_file() or p.is_symlink():
                p.unlink()
            else:
                import shutil
                shutil.rmtree(p)


def build_stats(
    data_path: Path,
    splits_path: Path,
    train_split_name: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    splits = pl.read_parquet(splits_path).select(
        pl.col("subject_id"), pl.col("split").cast(pl.Utf8)
    )
    if splits.select(pl.col("subject_id").is_duplicated().any()).item():
        raise ValueError("subject_splits.parquet contains duplicate subject_id values")

    split_summary = (
        splits.group_by("split")
        .agg(pl.len().alias("n_subjects"))
        .sort("split")
        .to_pandas()
    )

    train_subjects = (
        splits.filter(pl.col("split") == train_split_name)
        .select("subject_id")
        .unique()
    )
    if train_subjects.height == 0:
        raise ValueError(f"No subjects found for split={train_split_name!r}")

    schema = set(pl.scan_parquet(data_path).collect_schema().names())
    required = {"subject_id", "time", "code", "omop_table"}
    missing = required - schema
    if missing:
        raise ValueError(f"MEDS data.parquet is missing columns: {sorted(missing)}")

    diagnosis = (
        pl.scan_parquet(data_path)
        .filter(pl.col("omop_table") == "condition_occurrence")
        .select(
            pl.col("subject_id"),
            pl.col("time"),
            pl.col("code").cast(pl.Utf8).alias("code"),
        )
        .join(train_subjects.lazy(), on="subject_id", how="semi")
        .filter(pl.col("time").is_not_null() & pl.col("code").is_not_null())
        .with_columns(pl.col("time").dt.date().alias("event_date"))
    )

    scope = collect_streaming(
        diagnosis.select(
            pl.len().alias("n_train_condition_events"),
            pl.col("subject_id").n_unique().alias("n_train_subjects_with_diagnosis"),
            pl.col("code").n_unique().alias("n_train_diagnosis_codes"),
            pl.col("time").min().alias("first_train_diagnosis_time"),
            pl.col("time").max().alias("last_train_diagnosis_time"),
        )
    ).to_dicts()[0]

    n_diagnosis_subjects = int(scope["n_train_subjects_with_diagnosis"])
    if n_diagnosis_subjects == 0:
        raise ValueError("No train diagnosis events found")

    by_patient_code = (
        diagnosis.group_by(["subject_id", "code"])
        .agg(
            pl.len().alias("n_events_for_code"),
            pl.col("event_date").n_unique().alias("n_days_with_code"),
            pl.col("time").min().alias("first_code_time"),
            pl.col("time").max().alias("last_code_time"),
        )
        .with_columns(
            (pl.col("last_code_time") - pl.col("first_code_time"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("code_span_days"),
            (pl.col("n_events_for_code") - 1).clip(lower_bound=0).alias(
                "n_duplicate_events_for_code"
            ),
            (pl.col("n_days_with_code") - 1).clip(lower_bound=0).alias(
                "n_duplicate_days_for_code"
            ),
            (pl.col("n_days_with_code") >= 2).alias("is_repeated_across_days"),
        )
        .with_columns(
            (
                (pl.col("n_days_with_code") >= 2)
                & (pl.col("code_span_days") >= 90)
            ).alias("is_persistent_90d"),
            (
                (pl.col("n_days_with_code") >= 2)
                & (pl.col("code_span_days") >= 180)
            ).alias("is_persistent_180d"),
            (
                (pl.col("n_days_with_code") >= 2)
                & (pl.col("code_span_days") >= 365)
            ).alias("is_persistent_365d"),
        )
    )

    stats = (
        by_patient_code.group_by("code")
        .agg(
            pl.col("subject_id").n_unique().alias("n_subjects_with_code"),
            pl.col("n_events_for_code").sum().alias("n_events"),
            pl.col("n_duplicate_events_for_code")
            .sum()
            .alias("n_duplicate_events_over_first"),
            pl.col("n_duplicate_days_for_code")
            .sum()
            .alias("n_duplicate_days_over_first"),
            pl.col("n_events_for_code").median().alias("median_events_per_subject"),
            pl.col("n_events_for_code").quantile(0.90).alias("p90_events_per_subject"),
            pl.col("n_days_with_code").median().alias("median_days_with_code"),
            pl.col("n_days_with_code").quantile(0.90).alias("p90_days_with_code"),
            pl.col("code_span_days").median().alias("median_span_days"),
            pl.col("code_span_days").quantile(0.75).alias("p75_span_days"),
            pl.col("code_span_days").quantile(0.90).alias("p90_span_days"),
            pl.col("is_repeated_across_days").mean().alias("repeat_day_subject_share"),
            pl.col("is_persistent_90d").mean().alias("persistent_90d_subject_share"),
            pl.col("is_persistent_180d").mean().alias("persistent_180d_subject_share"),
            pl.col("is_persistent_365d").mean().alias("persistent_365d_subject_share"),
        )
        .with_columns(
            (pl.col("n_subjects_with_code") / pl.lit(n_diagnosis_subjects)).alias(
                "subject_prevalence"
            ),
            (
                pl.col("n_duplicate_events_over_first") / pl.col("n_events")
            ).alias("duplicate_event_share"),
        )
        .with_columns(
            (
                0.20 * pl.col("repeat_day_subject_share")
                + 0.20 * pl.col("persistent_90d_subject_share")
                + 0.25 * pl.col("persistent_180d_subject_share")
                + 0.25 * pl.col("persistent_365d_subject_share")
                + 0.10
                * (
                    pl.col("p90_span_days")
                    / (pl.col("p90_span_days") + 365.0)
                )
            ).alias("empirical_persistence_score")
        )
        .with_columns(
            (
                (pl.col("n_subjects_with_code") >= args.strong_min_subjects)
                & (
                    pl.col("repeat_day_subject_share")
                    >= args.strong_min_repeat_share
                )
                & (
                    (
                        pl.col("persistent_365d_subject_share")
                        >= args.strong_min_persistent_365_share
                    )
                    | (pl.col("p75_span_days") >= args.strong_min_p75_span_days)
                )
            ).alias("strong_chronic_like")
        )
        .sort(["strong_chronic_like", "empirical_persistence_score", "code"], descending=[True, True, False])
    )

    stats_pd = collect_streaming(stats).to_pandas()
    scope["n_train_subjects_in_split_file"] = int(train_subjects.height)
    return stats_pd, split_summary, scope


def add_code_descriptions(stats: pd.DataFrame, codes_path: Path) -> pd.DataFrame:
    out = stats.copy()
    if not codes_path.exists():
        out.insert(1, "diagnosis_name", out["code"])
        return out

    codes = pd.read_parquet(codes_path)
    if "code" not in codes.columns:
        out.insert(1, "diagnosis_name", out["code"])
        return out

    description_col = "description" if "description" in codes.columns else None
    if description_col is None:
        out.insert(1, "diagnosis_name", out["code"])
        return out

    meta = (
        codes[["code", description_col]]
        .drop_duplicates("code")
        .rename(columns={description_col: "diagnosis_name"})
    )
    out = out.merge(meta, on="code", how="left", validate="one_to_one")
    out["diagnosis_name"] = out["diagnosis_name"].fillna(out["code"])
    cols = ["code", "diagnosis_name"] + [
        c for c in out.columns if c not in {"code", "diagnosis_name"}
    ]
    return out[cols]


def write_sha_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != "output_sha256_manifest.csv":
            rows.append(
                {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "output_sha256_manifest.csv", index=False)
    return manifest


def main() -> None:
    args = parse_args()
    task = maybe_init_clearml(args)

    ehrshot_root = args.ehrshot_root.resolve()
    output_dir = args.output_dir.resolve()
    data_path = ehrshot_root / "data" / "data.parquet"
    splits_path = ehrshot_root / "metadata" / "subject_splits.parquet"
    codes_path = ehrshot_root / "metadata" / "codes.parquet"

    for path in [data_path, splits_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    stats, split_summary, scope = build_stats(
        data_path=data_path,
        splits_path=splits_path,
        train_split_name=args.train_split_name,
        args=args,
    )
    stats = add_code_descriptions(stats, codes_path)
    strong = stats[stats["strong_chronic_like"]].copy().reset_index(drop=True)

    if args.expected_code_count > 0 and len(strong) != args.expected_code_count:
        raise RuntimeError(
            f"Expected {args.expected_code_count} persistent codes, got {len(strong)}. "
            "Check MEDS snapshot, split file and thresholds before continuing."
        )

    stats_path = output_dir / "diagnosis_code_empirical_chronic_like_stats_train_only.csv"
    strong_path = output_dir / "strong_empirical_chronic_like_diagnosis_codes_train_only.csv"
    split_summary_path = output_dir / "source_split_summary.csv"

    stats.to_csv(stats_path, index=False)
    strong.to_csv(strong_path, index=False)
    split_summary.to_csv(split_summary_path, index=False)

    thresholds = {
        "n_subjects_with_code_min": args.strong_min_subjects,
        "repeat_day_subject_share_min": args.strong_min_repeat_share,
        "persistent_365d_subject_share_min": args.strong_min_persistent_365_share,
        "p75_span_days_min_alternative": args.strong_min_p75_span_days,
    }
    metadata = {
        "artifact": "train_only_empirical_persistence_whitelist",
        "build_time_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": args.dataset_version,
        "ehrshot_root": str(ehrshot_root),
        "train_split_name": args.train_split_name,
        "event_filter": "omop_table == condition_occurrence",
        "selection_rule": (
            "n_subjects_with_code >= 50 and repeat_day_subject_share >= 0.50 "
            "and (persistent_365d_subject_share >= 0.10 or p75_span_days >= 365)"
        ),
        "thresholds": thresholds,
        "scope": scope,
        "counts": {
            "n_all_train_diagnosis_codes": int(len(stats)),
            "n_selected_codes": int(len(strong)),
        },
        "selected_whitelist_sha256": sha256_file(strong_path),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "polars": pl.__version__,
        },
    }
    with (output_dir / "build_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(metadata), f, ensure_ascii=False, indent=2)

    manifest = write_sha_manifest(output_dir)

    upload_manifest = pd.DataFrame()
    if not args.skip_upload and args.output_s3_prefix:
        upload_manifest = upload_tree(output_dir, args.output_s3_prefix)
        upload_manifest.to_csv(output_dir / "minio_upload_manifest.csv", index=False)

    if task is not None:
        task.upload_artifact("whitelist_build_metadata", metadata)
        task.upload_artifact("whitelist_selected_codes", strong)
        task.upload_artifact("whitelist_full_stats", stats)
        task.upload_artifact("whitelist_sha_manifest", manifest)
        if len(upload_manifest):
            task.upload_artifact("whitelist_upload_manifest", upload_manifest)

    print("=" * 100)
    print("TRAIN-ONLY PERSISTENCE WHITELIST BUILT")
    print(f"Output: {output_dir}")
    print(f"Train subjects in split file: {scope['n_train_subjects_in_split_file']}")
    print(f"Train subjects with diagnoses: {scope['n_train_subjects_with_diagnosis']}")
    print(f"All train diagnosis codes: {len(stats)}")
    print(f"Selected codes: {len(strong)}")
    print(f"Whitelist SHA-256: {sha256_file(strong_path)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
