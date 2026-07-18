from __future__ import annotations

"""
Cache-based, config-driven builder for the frozen "State or space?" experiment.

Why this version exists
-----------------------
The previous low-memory implementation split prediction examples into small
batches and rebuilt the complete MEDS -> visit-anchor -> history pipeline for
every batch and every representation. That avoided OOM, but repeatedly scanned
and sorted the 41M-event MEDS table.

This implementation separates the work into reusable stages:

1. Build one global base-event cache from MEDS:
       MEDS -> event fields + reconstructed visit/day compression bucket.
2. Build task-specific history parts once, batching by SUBJECT, not by example:
       base cache + all labels for those subjects -> legal history rows.
3. Build every requested representation from those small history parts.
4. Build one train-only vocabulary per representation and aggregate sequence
   parts without collecting all long rows into RAM.

The JSON run config remains the source of truth for paths, cutoff, whitelist,
context lengths, gaps and the representation matrix.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl


SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
PAD_ID = SPECIAL_TOKENS["<PAD>"]
UNK_ID = SPECIAL_TOKENS["<UNK>"]

BUILDER_SCHEMA_VERSION = "state_or_space_episode_audit_v2"

EXAMPLE_KEY_COLS = [
    "task",
    "row_id",
    "subject_id",
    "prediction_time",
    "label",
    "split",
]

HISTORY_COLS = EXAMPLE_KEY_COLS + [
    "time",
    "code",
    "numeric_value",
    "text_value",
    "days_before_prediction",
    "compression_bucket",
    "source_event_id",
]

# Internal transformed-event columns before final position assignment.
TRANSFORM_LONG_COLS = EXAMPLE_KEY_COLS + [
    "time",
    "code",
    "numeric_value",
    "text_value",
    "days_before_prediction",
    "is_compression_token",
    "order_anchor_id",
    "role_order",
    "audit_persistent_events",
    "audit_repeats_removed",
    "audit_visit_day_duplicates_removed",
    "audit_era_repeat_mentions_removed",
    "audit_condition_eras",
    "audit_compressed_eras",
]

# Final long representation persisted in _long_parts_v2.
FINAL_LONG_COLS = EXAMPLE_KEY_COLS + [
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

CODE_SUMMARY_COLS = EXAMPLE_KEY_COLS + ["code"]
COMPRESSION_GROUP_COLS = CODE_SUMMARY_COLS + ["compression_bucket"]
ORDER_COLS = ["row_id", "time", "order_anchor_id", "role_order", "code"]

REQUIRED_EPISODE_AUDIT_COLUMNS = [
    "original_history_len",
    "transformation_input_len",
    "transformed_history_len_before_truncation",
    "transformed_history_len",
    "final_seq_len",
    "n_repeats_removed",
    "n_visit_day_duplicates_removed",
    "n_era_repeat_mentions_removed",
    "n_net_events_removed_by_transform",
    "was_raw_window_truncated",
    "was_transformed_truncated",
    "was_truncated_by_max_len",
    "was_truncated",
    "earliest_retained_time",
    "latest_retained_time",
    "earliest_retained_days_before_prediction",
    "covered_days",
    "retained_calendar_span_days",
    "n_backfill_events_added",
    "n_unique_persistent_codes_full_history",
    "n_persistent_diagnoses",
    "n_persistent_events_original",
    "persistent_code_list_id",
    "persistent_code_list_sha256",
]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EHRSHOT sequence datasets using reusable caches."
    )
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--notebook-root", type=Path, default=Path("."))
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild final representations. Caches are reused unless --rebuild-cache is set.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the global base cache and task history caches.",
    )
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--execute-remotely", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def as_python_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and math.isnan(value):
        return []
    if hasattr(value, "__iter__") and not isinstance(value, str):
        return list(value)
    return []


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
    return value


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        try:
            return lf.collect(streaming=True)
        except TypeError:
            return lf.collect()


def sink_parquet_compat(
    lf: pl.LazyFrame,
    path: Path,
    compression: str = "lz4",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lf.sink_parquet(str(path), compression=compression)
    except (AttributeError, TypeError):
        collect_streaming(lf).write_parquet(path, compression=compression)


def chunked(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("Batch size must be positive.")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def ensure_empty_or_remove(path: Path, rebuild: bool) -> None:
    if rebuild and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------

def expected_representation_name(spec: dict[str, Any]) -> str:
    context = int(spec["context_length"])
    if spec["kind"] == "raw":
        return f"raw_{context}"
    return (
        f"condition_era_{int(spec['gap_days'])}_"
        f"{spec['window_mode']}_{context}"
    )


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(raw)
    if not str(cfg.get("run_set_id", "")).strip():
        raise ValueError("run_set_id is required.")

    paths = dict(cfg.get("paths", {}))
    for key in ["ehrshot_root", "persistent_code_list", "output_dir"]:
        if not str(paths.get(key, "")).strip():
            raise ValueError(f"paths.{key} is required.")
    paths.setdefault("audit_dir", str(Path(paths["persistent_code_list"]).parent))
    paths.setdefault("output_s3_prefix", "")
    paths.setdefault("cache_dir", str(Path(paths["output_dir"]) / "_cache"))
    cfg["paths"] = paths

    build = dict(cfg.get("build", {}))
    build.setdefault("include_prediction_time", True)
    build.setdefault("visit_anchor_max_days", 1)
    build.setdefault("keep_code_strings", True)
    build.setdefault("min_code_count", 1)
    build.setdefault("max_vocab_size", 0)
    build.setdefault("rebuild", False)
    build.setdefault("upload", False)
    build.setdefault("subject_batch_size", 256)
    build.setdefault("cache_compression", "lz4")
    build.setdefault("output_compression", "zstd")
    build.setdefault("keep_long_parts", True)
    build.setdefault("keep_history_cache", True)
    build.setdefault("fail_on_invariant_violation", True)
    build.setdefault("require_strict_before_prediction_time", True)
    build.setdefault(
        "persistent_code_list_id",
        Path(paths["persistent_code_list"]).stem,
    )
    cfg["build"] = build

    if int(build["visit_anchor_max_days"]) < 0:
        raise ValueError("build.visit_anchor_max_days must be >= 0")
    if int(build["subject_batch_size"]) < 1:
        raise ValueError("build.subject_batch_size must be >= 1")
    if int(build["min_code_count"]) < 1:
        raise ValueError("build.min_code_count must be >= 1")
    if int(build["max_vocab_size"]) < 0:
        raise ValueError("build.max_vocab_size must be >= 0")

    if (
        bool(build["require_strict_before_prediction_time"])
        and bool(build["include_prediction_time"])
    ):
        raise ValueError(
            "Strict representation invariants require event_time < prediction_time, "
            "but build.include_prediction_time=true. Set include_prediction_time=false "
            "and rebuild the task-history cache and representations."
        )

    if (
        bool(build["require_strict_before_prediction_time"])
        and not bool(build["keep_code_strings"])
    ):
        raise ValueError(
            "Strict structure_null sequence checks require build.keep_code_strings=true."
        )

    reps = cfg.get("representations")
    if not isinstance(reps, list) or not reps:
        raise ValueError("representations must be a non-empty list")

    names: set[str] = set()
    pair_keys: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []

    for idx, raw_spec in enumerate(reps):
        spec = dict(raw_spec)
        name = str(spec.get("name", "")).strip()
        if not name:
            raise ValueError(f"representations[{idx}].name is required")
        if name in names:
            raise ValueError(f"Duplicate representation name: {name}")
        names.add(name)

        kind = str(spec.get("kind", "")).strip()
        if kind not in {"raw", "condition_era"}:
            raise ValueError(f"Unsupported kind for {name}: {kind}")
        spec["kind"] = kind

        tasks = spec.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"{name}.tasks must be a non-empty list")
        spec["tasks"] = [str(x) for x in tasks]

        context = int(spec.get("context_length", 0))
        if context <= 0:
            raise ValueError(f"{name}.context_length must be positive")
        spec["context_length"] = context

        spec.setdefault("analysis_groups", [])
        if kind == "raw":
            spec["window_mode"] = "raw"
            spec["gap_days"] = None
            spec.setdefault("raw_reference", None)
            spec.setdefault("source_representation", None)
        else:
            mode = str(spec.get("window_mode", "")).strip()
            if mode not in {"backfill", "no_backfill", "structure_null"}:
                raise ValueError(f"Unsupported window_mode for {name}: {mode}")
            spec["window_mode"] = mode
            spec["gap_days"] = int(spec.get("gap_days", 0))
            if spec["gap_days"] <= 0:
                raise ValueError(f"{name}.gap_days must be positive")
            if not spec.get("raw_reference"):
                spec["raw_reference"] = f"raw_{context}"
            if mode == "structure_null" and not spec.get("source_representation"):
                raise ValueError(f"{name} requires source_representation")

        expected = expected_representation_name(spec)
        if name != expected:
            raise ValueError(
                f"Representation name mismatch: name={name}, expected={expected}"
            )

        for task in spec["tasks"]:
            key = (task, name)
            if key in pair_keys:
                raise ValueError(f"Duplicate task/representation pair: {key}")
            pair_keys.add(key)

        normalized.append(spec)

    # A structure-null representation must be built after its source because it
    # reuses the exact final long positions of the source backfill variant.
    name_to_position = {spec["name"]: i for i, spec in enumerate(normalized)}
    has_structure_null = False
    for spec in normalized:
        if spec.get("window_mode") != "structure_null":
            continue
        has_structure_null = True
        source = str(spec["source_representation"])
        if source not in name_to_position:
            raise ValueError(
                f"{spec['name']}: source_representation not found: {source}"
            )
        if name_to_position[source] >= name_to_position[spec["name"]]:
            raise ValueError(
                f"{spec['name']}: source_representation must appear earlier in config"
            )
    if has_structure_null and not bool(build["keep_long_parts"]):
        raise ValueError(
            "build.keep_long_parts must be true because structure_null reuses "
            "the exact source backfill long parts."
        )

    cfg["representations"] = normalized
    cfg.setdefault("dataset_version", "EHRSHOT_MEDS_local")
    cfg.setdefault("clearml", {})
    return cfg


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return normalize_config(json.load(f))


def expand_run_matrix(cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for build_order, spec in enumerate(cfg["representations"], start=1):
        for task in spec["tasks"]:
            rows.append(
                {
                    "build_order": build_order,
                    "run_set_id": cfg["run_set_id"],
                    "task": task,
                    "version": spec["name"],
                    "analysis_groups": "|".join(spec.get("analysis_groups", [])),
                    "kind": spec["kind"],
                    "window_mode": spec.get("window_mode"),
                    "gap_days": spec.get("gap_days"),
                    "context_length": spec["context_length"],
                    "raw_reference": spec.get("raw_reference"),
                    "source_representation": spec.get("source_representation"),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

class CachedSequenceBuilder:
    """
    Cache-based sequence builder with exact per-episode audit fields.

    Key contracts
    -------------
    * MEDS is scanned and visit/day buckets are built once in the global cache.
    * Task histories are cached by unique subject and reused by all variants.
    * Every final event has a deterministic event_position.
    * structure_null is derived from the already materialized backfill events,
      so event count, timestamps and order are exactly preserved.
    * examples.parquet and episode_audit.parquet contain all requested audit
      quantities for every prediction episode.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        notebook_root: Path,
        run_config_path: Path,
        rebuild: bool,
        rebuild_cache: bool,
    ) -> None:
        self.cfg = cfg
        self.notebook_root = notebook_root.resolve()
        self.run_config_path = run_config_path.resolve()
        self.run_config_sha256 = sha256_file(self.run_config_path)

        paths = cfg["paths"]
        build = cfg["build"]

        self.ehrshot_root = resolve_path(self.notebook_root, paths["ehrshot_root"])
        self.output_dir = resolve_path(self.notebook_root, paths["output_dir"])
        self.cache_dir = resolve_path(self.notebook_root, paths["cache_dir"])
        self.persistent_code_list_path = resolve_path(
            self.notebook_root, paths["persistent_code_list"]
        )

        self.data_path = self.ehrshot_root / "data" / "data.parquet"
        self.splits_path = self.ehrshot_root / "metadata" / "subject_splits.parquet"
        self.labels_dir = self.ehrshot_root / "labels"

        for path in [
            self.data_path,
            self.splits_path,
            self.labels_dir,
            self.persistent_code_list_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(path)

        self.include_prediction_time = bool(build["include_prediction_time"])
        self.visit_anchor_max_days = int(build["visit_anchor_max_days"])
        self.keep_code_strings = bool(build["keep_code_strings"])
        self.min_code_count = int(build["min_code_count"])
        max_vocab = int(build["max_vocab_size"])
        self.max_vocab_size = max_vocab if max_vocab > 0 else None
        self.subject_batch_size = int(build["subject_batch_size"])
        self.cache_compression = str(build["cache_compression"])
        self.output_compression = str(build["output_compression"])
        self.keep_long_parts = bool(build["keep_long_parts"])
        self.keep_history_cache = bool(build["keep_history_cache"])
        self.fail_on_invariant_violation = bool(
            build["fail_on_invariant_violation"]
        )
        self.require_strict_before_prediction_time = bool(
            build["require_strict_before_prediction_time"]
        )
        self.rebuild = bool(rebuild or build.get("rebuild", False))
        self.rebuild_cache = bool(rebuild_cache)
        self.persistent_code_list_id = str(
            build.get(
                "persistent_code_list_id",
                self.persistent_code_list_path.stem,
            )
        )

        self.splits = pl.read_parquet(self.splits_path)
        self.task_to_label_file = {
            path.parent.name: path
            for path in sorted(self.labels_dir.glob("*/labels.parquet"))
        }

        whitelist = pd.read_csv(self.persistent_code_list_path, dtype={"code": str})
        if "code" not in whitelist.columns:
            raise ValueError("Persistent whitelist must contain a 'code' column")
        self.persistent_codes = set(whitelist["code"].dropna().astype(str))
        if not self.persistent_codes:
            raise ValueError("Persistent whitelist is empty")
        self.persistent_code_list_sha256 = sha256_file(
            self.persistent_code_list_path
        )
        self.persistent_codes_list = sorted(self.persistent_codes)

        self.events_lf = pl.scan_parquet(str(self.data_path))
        self.event_schema = self.events_lf.collect_schema().names()

        # Versioned cache paths avoid silently reusing the older cache that did
        # not contain source_event_id and exact ordering fields.
        self.base_cache_path = (
            self.cache_dir / "base_events_with_buckets_episode_audit_v2.parquet"
        )
        self.base_cache_metadata_path = (
            self.cache_dir / "base_events_with_buckets_episode_audit_v2.json"
        )
        self.history_cache_root = (
            self.cache_dir / "task_history_parts_episode_audit_v2"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Labels and global base cache
    # ------------------------------------------------------------------

    def detect_label_col(self, label_path: Path) -> str:
        lf = pl.scan_parquet(str(label_path))
        candidates = [
            "boolean_value",
            "integer_value",
            "float_value",
            "categorical_value",
            "label",
            "value",
        ]
        existing = [c for c in candidates if c in lf.collect_schema().names()]
        if not existing:
            raise ValueError(f"No known label column in {label_path}")
        counts = collect_streaming(
            lf.select([pl.col(c).is_not_null().sum().alias(c) for c in existing])
        ).to_dicts()[0]
        for col in existing:
            if int(counts[col]) > 0:
                return col
        raise ValueError(f"All candidate label columns are null in {label_path}")

    def load_labels(self, task: str) -> pl.DataFrame:
        if task not in self.task_to_label_file:
            raise FileNotFoundError(f"No labels.parquet for task={task}")
        path = self.task_to_label_file[task]
        label_col = self.detect_label_col(path)
        labels = (
            pl.read_parquet(path)
            .with_row_index("row_id")
            .with_columns(
                pl.col(label_col).cast(pl.Int8).alias("label"),
                pl.lit(task).alias("task"),
            )
            .join(self.splits, on="subject_id", how="left")
            .select(EXAMPLE_KEY_COLS)
            .sort(["subject_id", "prediction_time", "row_id"])
        )
        if labels.filter(pl.col("split").is_null()).height:
            raise ValueError(f"{task}: labels with missing split")
        return labels

    def build_visit_anchors_lf(self) -> pl.LazyFrame:
        visits = self.events_lf.filter(
            pl.col("code").cast(pl.Utf8).str.starts_with("Visit/")
        )
        if "omop_table" in self.event_schema:
            visits = visits.filter(pl.col("omop_table") == "visit_occurrence")
        return (
            visits.select(
                pl.col("subject_id"),
                pl.col("time").alias("visit_time"),
            )
            .unique(subset=["subject_id", "visit_time"])
            .sort(["subject_id", "visit_time"])
            .with_columns(
                (
                    pl.col("visit_time").cum_count().over("subject_id") - 1
                ).cast(pl.Int32).alias("reconstructed_visit_idx")
            )
        )

    def validate_base_cache(self) -> bool:
        if not self.base_cache_path.exists() or not self.base_cache_metadata_path.exists():
            return False
        try:
            metadata = json.loads(
                self.base_cache_metadata_path.read_text(encoding="utf-8")
            )
            if metadata.get("builder_schema_version") != BUILDER_SCHEMA_VERSION:
                return False
            schema = pl.scan_parquet(str(self.base_cache_path)).collect_schema().names()
            required = {
                "subject_id",
                "time",
                "code",
                "numeric_value",
                "text_value",
                "compression_bucket",
                "source_event_id",
            }
            return required.issubset(schema)
        except Exception:
            return False

    def build_base_cache(self) -> None:
        if not self.rebuild_cache and self.validate_base_cache():
            print(f"Base cache exists, reuse: {self.base_cache_path}")
            return

        self.base_cache_path.unlink(missing_ok=True)
        self.base_cache_metadata_path.unlink(missing_ok=True)

        print("=" * 100)
        print("BUILD GLOBAL BASE EVENT CACHE (one MEDS pass)")
        print(f"source: {self.data_path}")
        print(f"target: {self.base_cache_path}")

        # The row index is assigned before sorting and remains a stable unique
        # tie-breaker for events with identical subject/time/code.
        source = self.events_lf.with_row_index("source_event_id")
        exprs: list[pl.Expr] = [
            pl.col("source_event_id").cast(pl.Int64),
            pl.col("subject_id"),
            pl.col("time"),
            pl.col("code").cast(pl.Utf8).alias("code"),
        ]
        if "numeric_value" in self.event_schema:
            exprs.append(
                pl.col("numeric_value").cast(pl.Float32).alias("numeric_value")
            )
        else:
            exprs.append(pl.lit(None).cast(pl.Float32).alias("numeric_value"))
        if "text_value" in self.event_schema:
            exprs.append(pl.col("text_value").cast(pl.Utf8).alias("text_value"))
        else:
            exprs.append(pl.lit(None).cast(pl.Utf8).alias("text_value"))

        base = (
            source.select(exprs)
            .filter(pl.col("subject_id").is_not_null())
            .filter(pl.col("time").is_not_null())
            .filter(pl.col("code").is_not_null())
            .with_columns(
                pl.col("time").dt.date().cast(pl.Utf8).alias("event_day")
            )
            .sort(["subject_id", "time", "code", "source_event_id"])
        )

        anchors = self.build_visit_anchors_lf()
        base_with_bucket = (
            base.join_asof(
                anchors,
                left_on="time",
                right_on="visit_time",
                by="subject_id",
                strategy="backward",
            )
            .with_columns(
                (
                    (pl.col("time") - pl.col("visit_time")).dt.total_hours() / 24
                ).cast(pl.Float32).alias("days_since_visit_anchor")
            )
            .with_columns(
                (
                    pl.col("reconstructed_visit_idx").is_not_null()
                    & pl.col("days_since_visit_anchor").is_not_null()
                    & (pl.col("days_since_visit_anchor") >= 0)
                    & (
                        pl.col("days_since_visit_anchor")
                        <= self.visit_anchor_max_days
                    )
                ).alias("has_valid_reconstructed_visit")
            )
            .with_columns(
                pl.when(pl.col("has_valid_reconstructed_visit"))
                .then(
                    pl.concat_str(
                        pl.lit("reconstructed_visit="),
                        pl.col("reconstructed_visit_idx").cast(pl.Utf8),
                    )
                )
                .otherwise(
                    pl.concat_str(pl.lit("day="), pl.col("event_day"))
                )
                .alias("compression_bucket")
            )
            .select(
                "subject_id",
                "time",
                "code",
                "numeric_value",
                "text_value",
                "compression_bucket",
                "source_event_id",
            )
            .sort(["subject_id", "time", "code", "source_event_id"])
        )

        sink_parquet_compat(
            base_with_bucket,
            self.base_cache_path,
            compression=self.cache_compression,
        )
        self.base_cache_metadata_path.write_text(
            json.dumps(
                {
                    "builder_schema_version": BUILDER_SCHEMA_VERSION,
                    "data_path": str(self.data_path.resolve()),
                    "visit_anchor_max_days": self.visit_anchor_max_days,
                    "cache_compression": self.cache_compression,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved base cache: {self.base_cache_path}")

    # ------------------------------------------------------------------
    # Task history cache, batched by unique subject
    # ------------------------------------------------------------------

    def task_history_dir(self, task: str) -> Path:
        return self.history_cache_root / task

    def validate_history_manifest(self, manifest_path: Path) -> list[Path] | None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("builder_schema_version") != BUILDER_SCHEMA_VERSION:
                return None
            if bool(manifest.get("include_prediction_time")) != self.include_prediction_time:
                return None
            if int(manifest.get("visit_anchor_max_days", -1)) != self.visit_anchor_max_days:
                return None
            parts = [Path(x) for x in manifest["parts"]]
            if not all(path.exists() for path in parts):
                return None
            schema = pl.scan_parquet(str(parts[0])).collect_schema().names() if parts else []
            if not set(HISTORY_COLS).issubset(schema):
                return None
            return parts
        except Exception:
            return None

    def build_task_history_parts(self, task: str, labels: pl.DataFrame) -> list[Path]:
        out_dir = self.task_history_dir(task)
        manifest_path = out_dir / "manifest.json"

        if manifest_path.exists() and not self.rebuild_cache:
            parts = self.validate_history_manifest(manifest_path)
            if parts is not None:
                print(f"[{task}] task history cache exists: {len(parts)} parts")
                return parts

        ensure_empty_or_remove(out_dir, rebuild=True)
        subjects = sorted(labels["subject_id"].unique().to_list())
        subject_batches = list(chunked(subjects, self.subject_batch_size))
        parts: list[Path] = []

        print("=" * 100)
        print(f"BUILD TASK HISTORY CACHE: task={task}")
        print(
            f"subjects={len(subjects)}, subject_batch_size={self.subject_batch_size}, "
            f"parts={len(subject_batches)}"
        )

        for index, subject_ids in enumerate(subject_batches):
            part_path = out_dir / f"history_part_{index:04d}.parquet"
            labels_part = labels.filter(pl.col("subject_id").is_in(subject_ids))

            base_part = pl.scan_parquet(str(self.base_cache_path)).filter(
                pl.col("subject_id").is_in(subject_ids)
            )
            cutoff = (
                pl.col("time") <= pl.col("prediction_time")
                if self.include_prediction_time
                else pl.col("time") < pl.col("prediction_time")
            )
            history = (
                labels_part.lazy()
                .join(base_part, on="subject_id", how="inner")
                .filter(cutoff)
                .with_columns(
                    (
                        pl.col("prediction_time") - pl.col("time")
                    ).dt.total_days().cast(pl.Float32).alias(
                        "days_before_prediction"
                    )
                )
                .select(HISTORY_COLS)
                .sort(["row_id", "time", "code", "source_event_id"])
            )
            sink_parquet_compat(
                history,
                part_path,
                compression=self.cache_compression,
            )
            parts.append(part_path.resolve())
            print(
                f"[{task}] history part {index + 1}/{len(subject_batches)}: "
                f"subjects={len(subject_ids)} -> {part_path.name}"
            )

        manifest = {
            "builder_schema_version": BUILDER_SCHEMA_VERSION,
            "task": task,
            "n_labels": labels.height,
            "n_subjects": len(subjects),
            "subject_batch_size": self.subject_batch_size,
            "include_prediction_time": self.include_prediction_time,
            "visit_anchor_max_days": self.visit_anchor_max_days,
            "base_cache": str(self.base_cache_path.resolve()),
            "parts": [str(path) for path in parts],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return parts

    # ------------------------------------------------------------------
    # Ordering and representation transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _history_order_cols() -> list[str]:
        return ["row_id", "time", "source_event_id", "code"]

    @staticmethod
    def truncate_history_last_n(
        history: pl.LazyFrame,
        max_events: int,
    ) -> pl.LazyFrame:
        order_cols = ["row_id", "time", "source_event_id", "code"]
        return (
            history.sort(order_cols)
            .with_columns(
                pl.col("time")
                .cum_count()
                .over("row_id")
                .cast(pl.Int64)
                .alias("_position_1based"),
                pl.len()
                .over("row_id")
                .cast(pl.Int64)
                .alias("_group_len"),
            )
            .filter(
                pl.col("_position_1based")
                > (
                    pl.col("_group_len")
                    - pl.lit(int(max_events), dtype=pl.Int64)
                )
            )
            .drop("_position_1based", "_group_len")
            .sort(order_cols)
        )

    @staticmethod
    def truncate_transformed_last_n(
        transformed: pl.LazyFrame,
        max_events: int,
    ) -> pl.LazyFrame:
        return (
            transformed.sort(ORDER_COLS)
            .with_columns(
                pl.col("time")
                .cum_count()
                .over("row_id")
                .cast(pl.Int64)
                .alias("_position_1based"),
                pl.len()
                .over("row_id")
                .cast(pl.Int64)
                .alias("_group_len"),
            )
            .filter(
                pl.col("_position_1based")
                > (
                    pl.col("_group_len")
                    - pl.lit(int(max_events), dtype=pl.Int64)
                )
            )
            .drop("_position_1based", "_group_len")
            .sort(ORDER_COLS)
        )

    @staticmethod
    def assign_event_position(transformed: pl.LazyFrame) -> pl.LazyFrame:
        return (
            transformed.sort(ORDER_COLS)
            .with_columns(
                pl.col("time")
                .cum_count()
                .over("row_id")
                .cast(pl.Int32)
                .alias("event_position")
            )
            .sort(["row_id", "event_position"])
        )

    @staticmethod
    def zero_audit_markers() -> list[pl.Expr]:
        return [
            pl.lit(0).cast(pl.Int32).alias("audit_persistent_events"),
            pl.lit(0).cast(pl.Int32).alias("audit_repeats_removed"),
            pl.lit(0).cast(pl.Int32).alias(
                "audit_visit_day_duplicates_removed"
            ),
            pl.lit(0).cast(pl.Int32).alias(
                "audit_era_repeat_mentions_removed"
            ),
            pl.lit(0).cast(pl.Int32).alias("audit_condition_eras"),
            pl.lit(0).cast(pl.Int32).alias("audit_compressed_eras"),
        ]

    def raw_long(self, history: pl.LazyFrame) -> pl.LazyFrame:
        return (
            history.with_columns(
                pl.lit(False).alias("is_compression_token"),
                pl.col("source_event_id").cast(pl.Int64).alias(
                    "order_anchor_id"
                ),
                pl.lit(0).cast(pl.Int8).alias("role_order"),
                *self.zero_audit_markers(),
            )
            .select(TRANSFORM_LONG_COLS)
            .sort(ORDER_COLS)
        )

    def prepare_dedup_from_history(
        self,
        history: pl.LazyFrame,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        marked = history.with_columns(
            pl.col("code")
            .is_in(self.persistent_codes_list)
            .fill_null(False)
            .alias("is_compressible")
        )
        non_compressed = (
            marked.filter(~pl.col("is_compressible"))
            .with_columns(
                pl.lit(False).alias("is_compression_token"),
                pl.col("source_event_id").cast(pl.Int64).alias(
                    "order_anchor_id"
                ),
                pl.lit(0).cast(pl.Int8).alias("role_order"),
                *self.zero_audit_markers(),
            )
            .select(TRANSFORM_LONG_COLS)
        )
        dedup = (
            marked.filter(pl.col("is_compressible"))
            .sort(COMPRESSION_GROUP_COLS + ["time", "source_event_id", "code"])
            .group_by(COMPRESSION_GROUP_COLS)
            .agg(
                pl.len().cast(pl.Int32).alias("bucket_original_event_count"),
                pl.col("time").sort_by(["time", "source_event_id"]).first().alias(
                    "time"
                ),
                pl.col("numeric_value")
                .sort_by(["time", "source_event_id"])
                .first()
                .alias("numeric_value"),
                pl.col("text_value")
                .sort_by(["time", "source_event_id"])
                .first()
                .alias("text_value"),
                pl.col("source_event_id")
                .sort_by(["time", "source_event_id"])
                .first()
                .cast(pl.Int64)
                .alias("order_anchor_id"),
            )
            .with_columns(
                (
                    pl.col("prediction_time") - pl.col("time")
                ).dt.total_days().cast(pl.Float32).alias(
                    "days_before_prediction"
                ),
                (
                    pl.col("bucket_original_event_count") - 1
                ).cast(pl.Int32).alias("bucket_duplicates_removed"),
                pl.lit(False).alias("is_compression_token"),
                pl.lit(0).cast(pl.Int8).alias("role_order"),
            )
        )
        return non_compressed, dedup

    @staticmethod
    def count_bin_expr(n: pl.Expr) -> pl.Expr:
        return (
            pl.when(n <= 1).then(pl.lit("1"))
            .when(n == 2).then(pl.lit("2"))
            .when(n <= 5).then(pl.lit("3_5"))
            .when(n <= 10).then(pl.lit("6_10"))
            .otherwise(pl.lit("gt10"))
        )

    def compression_token(
        self,
        summary: pl.LazyFrame,
        code_expr: pl.Expr,
        time_col: str,
        anchor_col: str,
        role_order: int,
        numeric_expr: pl.Expr,
        carry_audit: bool,
    ) -> pl.LazyFrame:
        audit_exprs: list[pl.Expr]
        if carry_audit:
            audit_exprs = [
                pl.col("n_original_persistent_events")
                .cast(pl.Int32)
                .alias("audit_persistent_events"),
                pl.col("n_repeats_removed")
                .cast(pl.Int32)
                .alias("audit_repeats_removed"),
                pl.col("n_visit_day_duplicates_removed")
                .cast(pl.Int32)
                .alias("audit_visit_day_duplicates_removed"),
                pl.col("n_era_repeat_mentions_removed")
                .cast(pl.Int32)
                .alias("audit_era_repeat_mentions_removed"),
                pl.lit(1).cast(pl.Int32).alias("audit_condition_eras"),
                pl.lit(1).cast(pl.Int32).alias("audit_compressed_eras"),
            ]
        else:
            audit_exprs = self.zero_audit_markers()

        return (
            summary.with_columns(
                pl.col(time_col).alias("time"),
                code_expr.alias("code"),
                numeric_expr.cast(pl.Float32).alias("numeric_value"),
                pl.lit(None).cast(pl.Utf8).alias("text_value"),
                (
                    pl.col("prediction_time") - pl.col(time_col)
                ).dt.total_days().cast(pl.Float32).alias(
                    "days_before_prediction"
                ),
                pl.lit(True).alias("is_compression_token"),
                pl.col(anchor_col).cast(pl.Int64).alias("order_anchor_id"),
                pl.lit(role_order).cast(pl.Int8).alias("role_order"),
                *audit_exprs,
            )
            .select(TRANSFORM_LONG_COLS)
        )

    def condition_era(
        self,
        history: pl.LazyFrame,
        gap_days: int,
    ) -> pl.LazyFrame:
        non_compressed, dedup = self.prepare_dedup_from_history(history)
        with_era = (
            dedup.sort(CODE_SUMMARY_COLS + ["time", "order_anchor_id", "code"])
            .with_columns(
                pl.col("time")
                .shift(1)
                .over(CODE_SUMMARY_COLS)
                .alias("prev_dx_time")
            )
            .with_columns(
                (
                    pl.col("time") - pl.col("prev_dx_time")
                ).dt.total_days().cast(pl.Float32).alias("gap_from_prev_dx_days")
            )
            .with_columns(
                (
                    pl.col("prev_dx_time").is_null()
                    | (pl.col("gap_from_prev_dx_days") > gap_days)
                ).cast(pl.Int32).alias("new_era_flag")
            )
            .with_columns(
                (
                    pl.col("new_era_flag")
                    .cum_sum()
                    .over(CODE_SUMMARY_COLS)
                    - 1
                ).cast(pl.Int32).alias("era_idx")
            )
        )
        era_keys = CODE_SUMMARY_COLS + ["era_idx"]
        summary = (
            with_era.group_by(era_keys)
            .agg(
                pl.len().cast(pl.Int32).alias("n_dx_points"),
                pl.col("bucket_original_event_count")
                .sum()
                .cast(pl.Int32)
                .alias("n_original_persistent_events"),
                pl.col("bucket_duplicates_removed")
                .sum()
                .cast(pl.Int32)
                .alias("n_visit_day_duplicates_removed"),
                pl.col("time").min().alias("first_time"),
                pl.col("time").max().alias("last_time"),
                pl.col("order_anchor_id")
                .sort_by(["time", "order_anchor_id"])
                .first()
                .cast(pl.Int64)
                .alias("first_anchor_id"),
                pl.col("order_anchor_id")
                .sort_by(["time", "order_anchor_id"])
                .last()
                .cast(pl.Int64)
                .alias("last_anchor_id"),
            )
            .with_columns(
                (
                    pl.col("last_time") - pl.col("first_time")
                ).dt.total_days().cast(pl.Float32).alias("span_days"),
                (
                    pl.col("n_original_persistent_events") - 1
                ).cast(pl.Int32).alias("n_repeats_removed"),
                (
                    pl.col("n_dx_points") - 1
                ).cast(pl.Int32).alias("n_era_repeat_mentions_removed"),
            )
        )

        normal_summary = summary.filter(pl.col("n_dx_points") <= 1)
        compressed = summary.filter(pl.col("n_dx_points") >= 2)

        normal_dx = (
            with_era.join(normal_summary, on=era_keys, how="inner")
            .with_columns(
                pl.col("n_original_persistent_events")
                .cast(pl.Int32)
                .alias("audit_persistent_events"),
                pl.col("n_repeats_removed")
                .cast(pl.Int32)
                .alias("audit_repeats_removed"),
                pl.col("n_visit_day_duplicates_removed")
                .cast(pl.Int32)
                .alias("audit_visit_day_duplicates_removed"),
                pl.col("n_era_repeat_mentions_removed")
                .cast(pl.Int32)
                .alias("audit_era_repeat_mentions_removed"),
                pl.lit(1).cast(pl.Int32).alias("audit_condition_eras"),
                pl.lit(0).cast(pl.Int32).alias("audit_compressed_eras"),
            )
            .select(TRANSFORM_LONG_COLS)
        )

        prefix = f"DX_ERA{gap_days}"
        start = self.compression_token(
            compressed,
            pl.concat_str(pl.lit(f"{prefix}_START/"), pl.col("code")),
            "first_time",
            "first_anchor_id",
            10,
            pl.lit(0.0),
            carry_audit=True,
        )
        end = self.compression_token(
            compressed,
            pl.concat_str(pl.lit(f"{prefix}_END/"), pl.col("code")),
            "last_time",
            "last_anchor_id",
            20,
            pl.col("span_days"),
            carry_audit=False,
        )
        count = self.compression_token(
            compressed,
            pl.concat_str(
                pl.lit(f"{prefix}_COUNT_BIN/"),
                self.count_bin_expr(pl.col("n_dx_points")),
                pl.lit("/"),
                pl.col("code"),
            ),
            "last_time",
            "last_anchor_id",
            30,
            pl.col("n_dx_points"),
            carry_audit=False,
        )
        return pl.concat(
            [non_compressed, normal_dx, start, end, count],
            how="vertical_relaxed",
        ).sort(ORDER_COLS)

    @staticmethod
    def structure_null(source: pl.LazyFrame, gap_days: int) -> pl.LazyFrame:
        prefix = f"DX_ERA{gap_days}"
        is_start = pl.col("code").str.starts_with(f"{prefix}_START/")
        is_end = pl.col("code").str.starts_with(f"{prefix}_END/")
        is_count = pl.col("code").str.starts_with(f"{prefix}_COUNT_BIN/")
        structural = is_start | is_end | is_count

        base_code = (
            pl.when(is_start)
            .then(pl.col("code").str.replace(rf"^{prefix}_START/", ""))
            .when(is_end)
            .then(pl.col("code").str.replace(rf"^{prefix}_END/", ""))
            .when(is_count)
            .then(
                pl.col("code").str.replace(
                    rf"^{prefix}_COUNT_BIN/[^/]+/", ""
                )
            )
            .otherwise(pl.col("code"))
        )
        # event_position is preserved and is the only ordering key used later.
        return (
            source.with_columns(
                base_code.alias("code"),
                pl.when(structural)
                .then(pl.lit(None).cast(pl.Float32))
                .otherwise(pl.col("numeric_value"))
                .alias("numeric_value"),
                pl.when(structural)
                .then(pl.lit(None).cast(pl.Utf8))
                .otherwise(pl.col("text_value"))
                .alias("text_value"),
                pl.lit(False).alias("is_compression_token"),
            )
            .select(FINAL_LONG_COLS)
            .sort(["row_id", "event_position"])
        )

    # ------------------------------------------------------------------
    # Episode audit construction
    # ------------------------------------------------------------------

    def persistent_stats(
        self,
        frame: pl.DataFrame,
        prefix: str,
    ) -> pl.DataFrame:
        if frame.height == 0:
            return pl.DataFrame(schema={
                **{c: frame.schema[c] for c in EXAMPLE_KEY_COLS if c in frame.schema},
                f"n_persistent_events_{prefix}": pl.Int32,
                f"n_unique_persistent_codes_{prefix}": pl.Int32,
            })
        return (
            frame.lazy()
            .filter(pl.col("code").is_in(self.persistent_codes_list))
            .group_by(EXAMPLE_KEY_COLS)
            .agg(
                pl.len().cast(pl.Int32).alias(
                    f"n_persistent_events_{prefix}"
                ),
                pl.col("code").n_unique().cast(pl.Int32).alias(
                    f"n_unique_persistent_codes_{prefix}"
                ),
            )
            .collect()
        )

    def build_history_audit(
        self,
        history: pl.DataFrame,
        raw_window: pl.DataFrame,
        transform_input: pl.DataFrame,
        context: int,
    ) -> pl.DataFrame:
        full = (
            history.lazy()
            .group_by(EXAMPLE_KEY_COLS)
            .agg(
                pl.len().cast(pl.Int32).alias("original_history_len"),
                pl.col("time").min().alias("original_earliest_time"),
                pl.col("time").max().alias("original_latest_time"),
            )
            .collect()
        )
        full_persistent = self.persistent_stats(history, "original")

        raw_stats = (
            raw_window.lazy()
            .sort(["row_id", "time", "source_event_id", "code"])
            .group_by(EXAMPLE_KEY_COLS, maintain_order=True)
            .agg(
                pl.len().cast(pl.Int32).alias("raw_window_len"),
                pl.col("time").first().alias("raw_window_boundary_time"),
                pl.col("source_event_id")
                .first()
                .cast(pl.Int64)
                .alias("raw_window_boundary_source_event_id"),
                pl.col("time").last().alias("raw_window_latest_time"),
            )
            .collect()
        )
        raw_persistent = self.persistent_stats(raw_window, "raw_window")

        input_stats = (
            transform_input.lazy()
            .group_by(EXAMPLE_KEY_COLS)
            .agg(
                pl.len().cast(pl.Int32).alias("transformation_input_len")
            )
            .collect()
        )
        input_persistent = self.persistent_stats(transform_input, "transform_input")

        audit = (
            full.join(full_persistent, on=EXAMPLE_KEY_COLS, how="left")
            .join(raw_stats, on=EXAMPLE_KEY_COLS, how="left")
            .join(raw_persistent, on=EXAMPLE_KEY_COLS, how="left")
            .join(input_stats, on=EXAMPLE_KEY_COLS, how="left")
            .join(input_persistent, on=EXAMPLE_KEY_COLS, how="left")
            .with_columns(
                pl.col("n_persistent_events_original")
                .fill_null(0)
                .cast(pl.Int32),
                pl.col("n_unique_persistent_codes_original")
                .fill_null(0)
                .cast(pl.Int32),
                pl.col("n_persistent_events_raw_window")
                .fill_null(0)
                .cast(pl.Int32),
                pl.col("n_unique_persistent_codes_raw_window")
                .fill_null(0)
                .cast(pl.Int32),
                pl.col("n_persistent_events_transform_input")
                .fill_null(0)
                .cast(pl.Int32),
                pl.col("n_unique_persistent_codes_transform_input")
                .fill_null(0)
                .cast(pl.Int32),
                pl.col("raw_window_len").fill_null(0).cast(pl.Int32),
                pl.col("transformation_input_len")
                .fill_null(0)
                .cast(pl.Int32),
                (pl.col("original_history_len") > int(context)).alias(
                    "was_raw_window_truncated"
                ),
            )
        )
        return audit

    @staticmethod
    def build_transform_audit(transformed_pre: pl.DataFrame) -> pl.DataFrame:
        return (
            transformed_pre.lazy()
            .group_by(EXAMPLE_KEY_COLS)
            .agg(
                pl.len().cast(pl.Int32).alias(
                    "transformed_history_len_before_truncation"
                ),
                pl.col("audit_persistent_events")
                .sum()
                .cast(pl.Int32)
                .alias("persistent_events_accounted_by_transform"),
                pl.col("audit_repeats_removed")
                .sum()
                .cast(pl.Int32)
                .alias("n_repeats_removed"),
                pl.col("audit_visit_day_duplicates_removed")
                .sum()
                .cast(pl.Int32)
                .alias("n_visit_day_duplicates_removed"),
                pl.col("audit_era_repeat_mentions_removed")
                .sum()
                .cast(pl.Int32)
                .alias("n_era_repeat_mentions_removed"),
                pl.col("audit_condition_eras")
                .sum()
                .cast(pl.Int32)
                .alias("n_condition_eras"),
                pl.col("audit_compressed_eras")
                .sum()
                .cast(pl.Int32)
                .alias("n_compressed_eras"),
            )
            .collect()
        )

    def build_final_audit(
        self,
        final_long: pl.DataFrame,
        history_audit: pl.DataFrame,
        transform_audit: pl.DataFrame,
        spec: dict[str, Any],
    ) -> pl.DataFrame:
        context = int(spec["context_length"])
        final_stats = (
            final_long.lazy()
            .group_by(EXAMPLE_KEY_COLS)
            .agg(
                pl.len().cast(pl.Int32).alias("final_seq_len"),
                pl.col("time").min().alias("earliest_retained_time"),
                pl.col("time").max().alias("latest_retained_time"),
                pl.col("days_before_prediction")
                .max()
                .cast(pl.Float32)
                .alias("earliest_retained_days_before_prediction"),
            )
            .with_columns(
                (
                    pl.col("prediction_time") - pl.col("earliest_retained_time")
                ).dt.total_days().cast(pl.Float32).alias("covered_days"),
                (
                    pl.col("latest_retained_time")
                    - pl.col("earliest_retained_time")
                ).dt.total_days().cast(pl.Float32).alias(
                    "retained_calendar_span_days"
                ),
            )
            .collect()
        )

        audit = (
            history_audit.join(transform_audit, on=EXAMPLE_KEY_COLS, how="left")
            .join(final_stats, on=EXAMPLE_KEY_COLS, how="left")
            .with_columns(
                pl.col("transformed_history_len_before_truncation")
                .fill_null(0)
                .cast(pl.Int32),
                pl.col("final_seq_len").fill_null(0).cast(pl.Int32),
                pl.col("n_repeats_removed").fill_null(0).cast(pl.Int32),
                pl.col("n_visit_day_duplicates_removed")
                .fill_null(0)
                .cast(pl.Int32),
                pl.col("n_era_repeat_mentions_removed")
                .fill_null(0)
                .cast(pl.Int32),
                pl.col("n_condition_eras").fill_null(0).cast(pl.Int32),
                pl.col("n_compressed_eras").fill_null(0).cast(pl.Int32),
                pl.col("persistent_events_accounted_by_transform")
                .fill_null(0)
                .cast(pl.Int32),
            )
            .with_columns(
                pl.col("transformed_history_len_before_truncation").alias(
                    "transformed_history_len"
                ),
                (
                    pl.col("transformation_input_len")
                    - pl.col("transformed_history_len_before_truncation")
                ).cast(pl.Int32).alias("n_net_events_removed_by_transform"),
                (
                    pl.col("transformed_history_len_before_truncation")
                    > int(context)
                ).alias("was_transformed_truncated"),
                pl.col("n_unique_persistent_codes_original").alias(
                    "n_unique_persistent_codes_full_history"
                ),
                pl.col("n_unique_persistent_codes_original").alias(
                    "n_persistent_diagnoses"
                ),
                pl.lit(self.persistent_code_list_id).alias(
                    "persistent_code_list_id"
                ),
                pl.lit(self.persistent_code_list_sha256).alias(
                    "persistent_code_list_sha256"
                ),
            )
        )

        mode = str(spec["window_mode"])
        if mode == "no_backfill":
            audit = audit.with_columns(
                (
                    pl.col("was_raw_window_truncated")
                    | pl.col("was_transformed_truncated")
                ).alias("was_truncated_by_max_len")
            )
        else:
            audit = audit.with_columns(
                pl.col("was_transformed_truncated").alias(
                    "was_truncated_by_max_len"
                )
            )
        audit = audit.with_columns(
            pl.col("was_truncated_by_max_len").alias("was_truncated")
        )

        # Count final positions that are strictly earlier than the first raw
        # position retained by the corresponding raw context. This is the
        # operational definition of an event/position added by backfill.
        if mode in {"backfill", "structure_null"} and final_long.height:
            boundaries = audit.select(
                EXAMPLE_KEY_COLS
                + [
                    "original_history_len",
                    "raw_window_boundary_time",
                    "raw_window_boundary_source_event_id",
                ]
            )
            backfill = (
                final_long.lazy()
                .join(boundaries.lazy(), on=EXAMPLE_KEY_COLS, how="left")
                .with_columns(
                    (
                        (pl.col("original_history_len") > int(context))
                        & (
                            (pl.col("time") < pl.col("raw_window_boundary_time"))
                            | (
                                (pl.col("time") == pl.col("raw_window_boundary_time"))
                                & (
                                    pl.col("order_anchor_id")
                                    < pl.col("raw_window_boundary_source_event_id")
                                )
                            )
                        )
                    ).cast(pl.Int32).alias("_is_backfill_position")
                )
                .group_by(EXAMPLE_KEY_COLS)
                .agg(
                    pl.col("_is_backfill_position")
                    .sum()
                    .cast(pl.Int32)
                    .alias("n_backfill_events_added")
                )
                .collect()
            )
            audit = audit.join(backfill, on=EXAMPLE_KEY_COLS, how="left")
        else:
            audit = audit.with_columns(
                pl.lit(0).cast(pl.Int32).alias("n_backfill_events_added")
            )

        return audit.with_columns(
            pl.col("n_backfill_events_added").fill_null(0).cast(pl.Int32)
        )

    # ------------------------------------------------------------------
    # Representation part materialization
    # ------------------------------------------------------------------

    def representation_dir(self, task: str, version: str) -> Path:
        return self.output_dir / task / version

    def long_parts_dir(self, task: str, version: str) -> Path:
        return self.representation_dir(task, version) / "_long_parts_v2"

    def audit_parts_dir(self, task: str, version: str) -> Path:
        return self.representation_dir(task, version) / "_audit_parts_v2"

    def materialize_regular_part(
        self,
        history_path: Path,
        spec: dict[str, Any],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        history = pl.read_parquet(history_path).select(HISTORY_COLS)
        context = int(spec["context_length"])
        raw_window = collect_streaming(
            self.truncate_history_last_n(history.lazy(), context)
        )

        if spec["kind"] == "raw" or spec["window_mode"] == "backfill":
            transform_input = history
        elif spec["window_mode"] == "no_backfill":
            transform_input = raw_window
        else:
            raise ValueError(spec["window_mode"])

        history_audit = self.build_history_audit(
            history=history,
            raw_window=raw_window,
            transform_input=transform_input,
            context=context,
        )

        if spec["kind"] == "raw":
            transformed_pre = collect_streaming(self.raw_long(transform_input.lazy()))
        else:
            transformed_pre = collect_streaming(
                self.condition_era(
                    transform_input.lazy(),
                    int(spec["gap_days"]),
                )
            )

        transform_audit = self.build_transform_audit(transformed_pre)
        final_long = collect_streaming(
            self.assign_event_position(
                self.truncate_transformed_last_n(
                    transformed_pre.lazy(),
                    context,
                )
            )
        ).select(FINAL_LONG_COLS)

        audit = self.build_final_audit(
            final_long=final_long,
            history_audit=history_audit,
            transform_audit=transform_audit,
            spec=spec,
        )
        return final_long, audit

    def materialize_structure_null_part(
        self,
        source_long_path: Path,
        source_audit_path: Path,
        spec: dict[str, Any],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if not source_long_path.exists():
            raise FileNotFoundError(source_long_path)
        if not source_audit_path.exists():
            raise FileNotFoundError(source_audit_path)
        final_long = collect_streaming(
            self.structure_null(
                pl.scan_parquet(str(source_long_path)),
                int(spec["gap_days"]),
            )
        )
        audit = pl.read_parquet(source_audit_path).with_columns(
            pl.lit(self.persistent_code_list_id).alias("persistent_code_list_id"),
            pl.lit(self.persistent_code_list_sha256).alias(
                "persistent_code_list_sha256"
            ),
        )
        return final_long, audit

    def materialize_representation_parts(
        self,
        task: str,
        spec: dict[str, Any],
        history_parts: list[Path],
    ) -> tuple[list[Path], list[Path]]:
        version = spec["name"]
        long_dir = self.long_parts_dir(task, version)
        audit_dir = self.audit_parts_dir(task, version)
        ensure_empty_or_remove(long_dir, rebuild=self.rebuild)
        ensure_empty_or_remove(audit_dir, rebuild=self.rebuild)

        long_parts: list[Path] = []
        audit_parts: list[Path] = []

        for index, history_path in enumerate(history_parts):
            long_path = long_dir / f"long_part_{index:04d}.parquet"
            audit_path = audit_dir / f"audit_part_{index:04d}.parquet"
            if long_path.exists() and audit_path.exists() and not self.rebuild:
                long_parts.append(long_path)
                audit_parts.append(audit_path)
                continue

            if spec["window_mode"] == "structure_null":
                source_version = str(spec["source_representation"])
                source_long_path = (
                    self.long_parts_dir(task, source_version)
                    / f"long_part_{index:04d}.parquet"
                )
                source_audit_path = (
                    self.audit_parts_dir(task, source_version)
                    / f"audit_part_{index:04d}.parquet"
                )
                final_long, audit = self.materialize_structure_null_part(
                    source_long_path,
                    source_audit_path,
                    spec,
                )
            else:
                final_long, audit = self.materialize_regular_part(
                    history_path,
                    spec,
                )

            final_long.write_parquet(
                long_path,
                compression=self.cache_compression,
            )
            audit.write_parquet(
                audit_path,
                compression=self.cache_compression,
            )
            long_parts.append(long_path)
            audit_parts.append(audit_path)
            print(
                f"[{task} | {version}] part {index + 1}/{len(history_parts)} "
                f"rows={final_long.height} episodes={audit.height}"
            )
        return long_parts, audit_parts

    # ------------------------------------------------------------------
    # Long parts -> vocabulary -> sequence parts
    # ------------------------------------------------------------------

    def build_vocab(
        self,
        long_parts: list[Path],
    ) -> tuple[dict[str, int], pd.DataFrame]:
        scans = [
            pl.scan_parquet(str(path))
            .filter(pl.col("split") == "train")
            .group_by("code")
            .agg(pl.len().alias("n"))
            for path in long_parts
        ]
        counts = (
            pl.concat(scans, how="vertical_relaxed")
            .group_by("code")
            .agg(pl.col("n").sum().alias("n"))
            .filter(pl.col("n") >= self.min_code_count)
            .sort(["n", "code"], descending=[True, False])
        )
        counts_pd = collect_streaming(counts).to_pandas()
        if self.max_vocab_size is not None:
            counts_pd = counts_pd.head(
                max(0, self.max_vocab_size - len(SPECIAL_TOKENS))
            )
        vocab = dict(SPECIAL_TOKENS)
        for code in counts_pd["code"].astype(str):
            if code not in vocab:
                vocab[code] = len(vocab)
        return vocab, counts_pd

    def aggregate_sequence_part(
        self,
        long_path: Path,
        vocab: dict[str, int],
    ) -> pd.DataFrame:
        vocab_df = pl.DataFrame(
            {"code": list(vocab.keys()), "token_id": list(vocab.values())}
        )
        lf = (
            pl.scan_parquet(str(long_path))
            .join(vocab_df.lazy(), on="code", how="left")
            .with_columns(
                pl.col("token_id")
                .fill_null(UNK_ID)
                .cast(pl.Int32)
                .alias("token_id")
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
        agg: list[pl.Expr] = [
            pl.col("token_id").alias("token_ids"),
            pl.col("days_before_prediction")
            .cast(pl.Float32)
            .alias("days_before_prediction"),
            pl.col("delta_days").cast(pl.Float32).alias("delta_days"),
            pl.col("numeric_value").cast(pl.Float32).alias("numeric_values"),
            pl.len().cast(pl.Int32).alias("seq_len"),
            pl.col("token_id").n_unique().cast(pl.Int32).alias("n_unique_tokens"),
            pl.col("is_compression_token")
            .sum()
            .cast(pl.Int32)
            .alias("n_compression_events"),
        ]
        if self.keep_code_strings:
            agg.insert(1, pl.col("code").alias("codes"))
        return collect_streaming(
            lf.group_by(EXAMPLE_KEY_COLS, maintain_order=True).agg(agg)
        ).to_pandas()

    @staticmethod
    def default_audit_values() -> dict[str, Any]:
        return {
            "original_history_len": 0,
            "transformation_input_len": 0,
            "transformed_history_len_before_truncation": 0,
            "transformed_history_len": 0,
            "final_seq_len": 0,
            "n_repeats_removed": 0,
            "n_visit_day_duplicates_removed": 0,
            "n_era_repeat_mentions_removed": 0,
            "n_net_events_removed_by_transform": 0,
            "was_raw_window_truncated": False,
            "was_transformed_truncated": False,
            "was_truncated_by_max_len": False,
            "was_truncated": False,
            "earliest_retained_days_before_prediction": np.nan,
            "covered_days": np.nan,
            "retained_calendar_span_days": np.nan,
            "n_backfill_events_added": 0,
            "n_unique_persistent_codes_original": 0,
            "n_unique_persistent_codes_full_history": 0,
            "n_persistent_diagnoses": 0,
            "n_persistent_events_original": 0,
            "n_persistent_events_raw_window": 0,
            "n_unique_persistent_codes_raw_window": 0,
            "n_persistent_events_transform_input": 0,
            "n_unique_persistent_codes_transform_input": 0,
            "n_condition_eras": 0,
            "n_compressed_eras": 0,
            "persistent_events_accounted_by_transform": 0,
        }

    def add_legacy_raw_reference_columns(
        self,
        out: pd.DataFrame,
        task: str,
        spec: dict[str, Any],
    ) -> pd.DataFrame:
        result = out.copy()
        if spec["kind"] == "raw":
            result["raw_seq_len"] = result["seq_len"].astype(np.int32)
            result["n_events_removed_vs_raw"] = 0
        else:
            raw_reference = str(spec["raw_reference"])
            raw_path = self.representation_dir(task, raw_reference) / "examples.parquet"
            if not raw_path.exists():
                raise FileNotFoundError(f"Raw reference not built: {raw_path}")
            raw = pd.read_parquet(raw_path, columns=["row_id", "seq_len"]).rename(
                columns={"seq_len": "raw_seq_len"}
            )
            result = result.merge(raw, on="row_id", how="left", validate="one_to_one")
            result["raw_seq_len"] = result["raw_seq_len"].fillna(0).astype(np.int32)
            result["n_events_removed_vs_raw"] = (
                result["raw_seq_len"].astype(int) - result["seq_len"].astype(int)
            ).astype(np.int32)
        result["n_compressible_chronic_events_raw"] = result[
            "n_persistent_events_raw_window"
        ].fillna(0).astype(np.int32)
        result["n_synthetic_events"] = result[
            "n_compression_events"
        ].fillna(0).astype(np.int32)
        return result

    def representation_is_complete(
        self,
        examples_path: Path,
        metadata_path: Path,
        vocab_path: Path,
        audit_path: Path,
    ) -> bool:
        if not all(path.exists() for path in [examples_path, metadata_path, vocab_path, audit_path]):
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return metadata.get("builder_schema_version") == BUILDER_SCHEMA_VERSION
        except Exception:
            return False

    def build_representation(
        self,
        task: str,
        spec: dict[str, Any],
        labels: pl.DataFrame,
        history_parts: list[Path],
    ) -> dict[str, Any]:
        version = spec["name"]
        out_dir = self.representation_dir(task, version)
        examples_path = out_dir / "examples.parquet"
        episode_audit_path = out_dir / "episode_audit.parquet"
        metadata_path = out_dir / "metadata.json"
        vocab_path = out_dir / "vocab.json"
        counts_path = out_dir / "code_counts_train.csv"

        if (
            not self.rebuild
            and self.representation_is_complete(
                examples_path,
                metadata_path,
                vocab_path,
                episode_audit_path,
            )
        ):
            print(f"[{task} | {version}] already exists, skip")
            return json.loads(metadata_path.read_text(encoding="utf-8"))

        out_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 100)
        print(f"BUILD REPRESENTATION: task={task} | version={version}")

        long_parts, audit_parts = self.materialize_representation_parts(
            task,
            spec,
            history_parts,
        )
        vocab, counts = self.build_vocab(long_parts)
        vocab_path.write_text(
            json.dumps(vocab, ensure_ascii=False),
            encoding="utf-8",
        )
        counts.to_csv(counts_path, index=False)

        seq_parts: list[pd.DataFrame] = []
        for index, long_path in enumerate(long_parts):
            seq_parts.append(self.aggregate_sequence_part(long_path, vocab))
            print(
                f"[{task} | {version}] sequence part "
                f"{index + 1}/{len(long_parts)}"
            )
        seqs = pd.concat(seq_parts, ignore_index=True) if seq_parts else pd.DataFrame()
        audits = pd.concat(
            [pd.read_parquet(path) for path in audit_parts],
            ignore_index=True,
        ) if audit_parts else pd.DataFrame()

        label_pd = labels.to_pandas()
        out = label_pd.merge(
            seqs,
            on=EXAMPLE_KEY_COLS,
            how="left",
            validate="one_to_one",
        )
        if not audits.empty:
            out = out.merge(
                audits,
                on=EXAMPLE_KEY_COLS,
                how="left",
                validate="one_to_one",
            )

        list_cols = [
            "token_ids",
            "days_before_prediction",
            "delta_days",
            "numeric_values",
        ]
        if self.keep_code_strings:
            list_cols.append("codes")
        for col in list_cols:
            if col not in out.columns:
                out[col] = [[] for _ in range(len(out))]
            else:
                out[col] = out[col].apply(as_python_list)

        out["seq_len"] = out["token_ids"].apply(len).astype(np.int32)
        out["n_unique_tokens"] = out["token_ids"].apply(
            lambda values: len(set(values))
        ).astype(np.int32)
        out["n_compression_events"] = out[
            "n_compression_events"
        ].fillna(0).astype(np.int32)

        for col, default in self.default_audit_values().items():
            if col not in out.columns:
                out[col] = default
            else:
                out[col] = out[col].fillna(default)
        if "persistent_code_list_id" not in out.columns:
            out["persistent_code_list_id"] = self.persistent_code_list_id
        else:
            out["persistent_code_list_id"] = out[
                "persistent_code_list_id"
            ].fillna(self.persistent_code_list_id)
        if "persistent_code_list_sha256" not in out.columns:
            out["persistent_code_list_sha256"] = self.persistent_code_list_sha256
        else:
            out["persistent_code_list_sha256"] = out[
                "persistent_code_list_sha256"
            ].fillna(self.persistent_code_list_sha256)

        int_audit_cols = [
            "original_history_len",
            "transformation_input_len",
            "transformed_history_len_before_truncation",
            "transformed_history_len",
            "final_seq_len",
            "n_repeats_removed",
            "n_visit_day_duplicates_removed",
            "n_era_repeat_mentions_removed",
            "n_net_events_removed_by_transform",
            "n_backfill_events_added",
            "n_unique_persistent_codes_original",
            "n_unique_persistent_codes_full_history",
            "n_persistent_diagnoses",
            "n_persistent_events_original",
            "n_persistent_events_raw_window",
            "n_unique_persistent_codes_raw_window",
            "n_persistent_events_transform_input",
            "n_unique_persistent_codes_transform_input",
            "n_condition_eras",
            "n_compressed_eras",
            "persistent_events_accounted_by_transform",
        ]
        for col in int_audit_cols:
            out[col] = out[col].astype(np.int32)
        for col in [
            "was_raw_window_truncated",
            "was_transformed_truncated",
            "was_truncated_by_max_len",
            "was_truncated",
        ]:
            out[col] = out[col].astype(bool)

        # final_seq_len is an independently generated audit value; equality is
        # enforced so the audit cannot silently diverge from the sequence.
        if not np.array_equal(
            out["final_seq_len"].to_numpy(),
            out["seq_len"].to_numpy(),
        ):
            bad = out.loc[
                out["final_seq_len"] != out["seq_len"],
                ["row_id", "final_seq_len", "seq_len"],
            ].head(20)
            raise ValueError(
                f"{task}/{version}: final_seq_len audit mismatch:\n"
                + bad.to_string(index=False)
            )

        out = self.add_legacy_raw_reference_columns(out, task, spec)

        context = int(spec["context_length"])
        if int(out["seq_len"].max()) > context:
            raise ValueError(f"{task}/{version}: seq_len exceeds {context}")
        if (out["n_backfill_events_added"] < 0).any():
            raise ValueError(f"{task}/{version}: negative backfill count")
        if spec["window_mode"] in {"raw", "no_backfill"} and (
            out["n_backfill_events_added"] != 0
        ).any():
            raise ValueError(f"{task}/{version}: backfill count must be zero")

        all_days = [
            value
            for values in out["days_before_prediction"]
            for value in as_python_list(values)
        ]
        min_days = float(np.min(all_days)) if all_days else np.nan
        if np.isfinite(min_days) and min_days < -1e-6:
            raise ValueError(f"{task}/{version}: future event detected")

        audit_output_cols = EXAMPLE_KEY_COLS + [
            col for col in out.columns
            if col in REQUIRED_EPISODE_AUDIT_COLUMNS
            or col in {
                "original_earliest_time",
                "original_latest_time",
                "raw_window_len",
                "raw_window_boundary_time",
                "raw_window_boundary_source_event_id",
                "raw_window_latest_time",
                "n_persistent_events_raw_window",
                "n_unique_persistent_codes_raw_window",
                "n_persistent_events_transform_input",
                "n_unique_persistent_codes_transform_input",
                "n_condition_eras",
                "n_compressed_eras",
                "persistent_events_accounted_by_transform",
            }
        ]
        audit_output_cols = list(dict.fromkeys(audit_output_cols))
        out[audit_output_cols].to_parquet(
            episode_audit_path,
            index=False,
            compression=self.output_compression,
        )
        out.to_parquet(
            examples_path,
            index=False,
            compression=self.output_compression,
        )

        split_summary = (
            out.groupby("split")
            .agg(
                n_examples=("label", "size"),
                n_patients=("subject_id", "nunique"),
                n_positive=("label", "sum"),
                event_rate=("label", "mean"),
                mean_original_history_len=("original_history_len", "mean"),
                mean_transformed_history_len=("transformed_history_len", "mean"),
                mean_seq_len=("seq_len", "mean"),
                median_seq_len=("seq_len", "median"),
                p90_seq_len=("seq_len", lambda x: float(np.quantile(x, 0.90))),
                max_seq_len=("seq_len", "max"),
                truncated_share=("was_truncated", "mean"),
                mean_repeats_removed=("n_repeats_removed", "mean"),
                mean_backfill_events_added=("n_backfill_events_added", "mean"),
                mean_covered_days=("covered_days", "mean"),
                mean_persistent_diagnoses=("n_persistent_diagnoses", "mean"),
                mean_compression_events=("n_compression_events", "mean"),
            )
            .reset_index()
        )

        metadata = {
            "builder_schema_version": BUILDER_SCHEMA_VERSION,
            "run_set_id": self.cfg["run_set_id"],
            "dataset_version": self.cfg["dataset_version"],
            "task": task,
            "version": version,
            "representation_spec": spec,
            "n_examples": len(out),
            "n_patients": int(out["subject_id"].nunique()),
            "n_positive": int(out["label"].sum()),
            "event_rate": float(out["label"].mean()),
            "vocab_size": len(vocab),
            "mean_seq_len": float(out["seq_len"].mean()),
            "median_seq_len": float(out["seq_len"].median()),
            "p90_seq_len": float(np.quantile(out["seq_len"], 0.90)),
            "max_seq_len": int(out["seq_len"].max()),
            "mean_original_history_len": float(out["original_history_len"].mean()),
            "mean_transformed_history_len": float(out["transformed_history_len"].mean()),
            "truncated_share": float(out["was_truncated"].mean()),
            "mean_repeats_removed": float(out["n_repeats_removed"].mean()),
            "mean_backfill_events_added": float(
                out["n_backfill_events_added"].mean()
            ),
            "min_days_before_prediction": min_days,
            "include_prediction_time": self.include_prediction_time,
            "visit_anchor_max_days": self.visit_anchor_max_days,
            "persistent_code_count": len(self.persistent_codes),
            "persistent_code_list_id": self.persistent_code_list_id,
            "persistent_code_list": str(self.persistent_code_list_path),
            "persistent_code_list_sha256": self.persistent_code_list_sha256,
            "run_config": str(self.run_config_path),
            "run_config_sha256": self.run_config_sha256,
            "episode_audit_definitions": {
                "original_history_len": (
                    "all legal raw events before/at prediction_time before any truncation"
                ),
                "transformed_history_len": (
                    "representation length after transformation and before final max_len"
                ),
                "n_repeats_removed": (
                    "persistent input mentions minus one retained state per condition era; "
                    "includes visit/day duplicates and repeated era confirmations"
                ),
                "was_truncated": (
                    "any max_len truncation used by the representation; for no_backfill "
                    "includes the initial raw-window truncation and final transformed truncation"
                ),
                "covered_days": (
                    "days from earliest final retained event to prediction_time"
                ),
                "n_backfill_events_added": (
                    "final representation positions strictly earlier than the earliest raw "
                    "event retained by the corresponding raw context"
                ),
                "n_persistent_diagnoses": (
                    "unique whitelist codes in the full legal episode history"
                ),
            },
            "cache": {
                "base_cache": str(self.base_cache_path),
                "history_parts": [str(path) for path in history_parts],
                "subject_batch_size": self.subject_batch_size,
            },
            "split_summary": split_summary.to_dict(orient="records"),
            "files": {
                "examples": str(examples_path),
                "episode_audit": str(episode_audit_path),
                "vocab": str(vocab_path),
                "code_counts_train": str(counts_path),
                "metadata": str(metadata_path),
            },
        }
        metadata_path.write_text(
            json.dumps(jsonable(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved: {examples_path}")
        print(f"Saved: {episode_audit_path}")
        print(split_summary.to_string(index=False))

        # Keep source long/audit parts until structure_null has been built.
        if not self.keep_long_parts and spec["window_mode"] != "backfill":
            shutil.rmtree(self.long_parts_dir(task, version), ignore_errors=True)
            shutil.rmtree(self.audit_parts_dir(task, version), ignore_errors=True)
        return metadata

    # ------------------------------------------------------------------
    # Invariants and execution
    # ------------------------------------------------------------------

    @staticmethod
    def strip_structure_code(code: str, gap_days: int) -> str:
        prefix = f"DX_ERA{gap_days}"
        value = str(code)
        value = re.sub(rf"^{prefix}_START/", "", value)
        value = re.sub(rf"^{prefix}_END/", "", value)
        value = re.sub(rf"^{prefix}_COUNT_BIN/[^/]+/", "", value)
        return value

    @staticmethod
    def structural_code_pattern(gap_days: int) -> str:
        return rf"^DX_ERA{int(gap_days)}_(?:START|END|COUNT_BIN)/"

    @staticmethod
    def _count_rows(lf: pl.LazyFrame) -> int:
        result = collect_streaming(lf.select(pl.len().alias("n")))
        return int(result["n"][0])

    @staticmethod
    def _null_safe_equal(left: pl.Expr, right: pl.Expr) -> pl.Expr:
        return (left == right) | (left.is_null() & right.is_null())

    @staticmethod
    def _numeric_equal(left: pl.Expr, right: pl.Expr) -> pl.Expr:
        return (
            (left == right)
            | (left.is_null() & right.is_null())
            | (left.is_nan() & right.is_nan())
        )

    @staticmethod
    def _invariant_row(
        task: str,
        version: str,
        check: str,
        passed: bool,
        details: str = "",
    ) -> dict[str, Any]:
        return {
            "task": task,
            "version": version,
            "check": check,
            "passed": bool(passed),
            "details": str(details),
        }

    def persist_and_raise_invariant_failures(
        self,
        invariant_rows: list[dict[str, Any]],
        only_new_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        """Save partial invariant results and stop on the first violation."""
        invariants = pd.DataFrame(invariant_rows)
        invariants.to_csv(
            self.output_dir / "representation_invariants.csv",
            index=False,
        )

        checked = (
            pd.DataFrame(only_new_rows)
            if only_new_rows is not None
            else invariants
        )
        if checked.empty or "passed" not in checked.columns:
            return

        failed = checked[~checked["passed"].astype(bool)]
        if len(failed) and self.fail_on_invariant_violation:
            raise RuntimeError(
                "Representation invariant failure:\n"
                + failed.to_string(index=False)
            )

    def check_no_backfill(
        self,
        task: str,
        spec: dict[str, Any],
        history_parts: list[Path],
    ) -> list[dict[str, Any]]:
        """Event-level invariants for the no_backfill representation."""
        if spec["window_mode"] != "no_backfill":
            return []

        version = str(spec["name"])
        context = int(spec["context_length"])
        counters = {
            "n_events_before_raw_window": 0,
            "n_events_without_raw_window_anchor": 0,
            "n_anchor_time_mismatch": 0,
            "n_event_order_mismatch": 0,
            "n_noncontiguous_positions": 0,
            "n_after_prediction_time": 0,
        }

        for index, history_path in enumerate(history_parts):
            long_path = (
                self.long_parts_dir(task, version)
                / f"long_part_{index:04d}.parquet"
            )
            if not long_path.exists():
                raise FileNotFoundError(long_path)

            history = pl.scan_parquet(str(history_path)).select(HISTORY_COLS)
            raw_window = self.truncate_history_last_n(history, context)

            boundary = (
                raw_window
                .sort(["row_id", "time", "source_event_id", "code"])
                .group_by("row_id", maintain_order=True)
                .agg(
                    pl.col("time").first().alias("raw_boundary_time"),
                    pl.col("source_event_id")
                    .first()
                    .cast(pl.Int64)
                    .alias("raw_boundary_source_event_id"),
                )
            )
            anchors = raw_window.select(
                "row_id",
                pl.col("source_event_id")
                .cast(pl.Int64)
                .alias("order_anchor_id"),
                pl.col("time").alias("anchor_time"),
            )

            final = pl.scan_parquet(str(long_path)).select(FINAL_LONG_COLS)
            joined = (
                final
                .join(boundary, on="row_id", how="left")
                .join(
                    anchors,
                    on=["row_id", "order_anchor_id"],
                    how="left",
                )
            )

            counters["n_events_before_raw_window"] += self._count_rows(
                joined.filter(
                    (pl.col("time") < pl.col("raw_boundary_time"))
                    | (
                        (pl.col("time") == pl.col("raw_boundary_time"))
                        & (
                            pl.col("order_anchor_id")
                            < pl.col("raw_boundary_source_event_id")
                        )
                    )
                )
            )
            counters["n_events_without_raw_window_anchor"] += self._count_rows(
                joined.filter(pl.col("anchor_time").is_null())
            )
            counters["n_anchor_time_mismatch"] += self._count_rows(
                joined.filter(
                    pl.col("anchor_time").is_not_null()
                    & (pl.col("time") != pl.col("anchor_time"))
                )
            )
            counters[
                "n_after_prediction_time"
            ] += self._count_rows(
                final.filter(pl.col("time") > pl.col("prediction_time"))
            )

            counters["n_event_order_mismatch"] += self._count_rows(
                final
                .sort(ORDER_COLS)
                .with_columns(
                    pl.col("time")
                    .cum_count()
                    .over("row_id")
                    .cast(pl.Int32)
                    .alias("expected_event_position")
                )
                .filter(
                    pl.col("event_position")
                    != pl.col("expected_event_position")
                )
            )
            counters["n_noncontiguous_positions"] += self._count_rows(
                final
                .group_by("row_id")
                .agg(
                    pl.len().cast(pl.Int32).alias("n"),
                    pl.col("event_position").min().alias("min_position"),
                    pl.col("event_position").max().alias("max_position"),
                    pl.col("event_position")
                    .n_unique()
                    .alias("n_unique_positions"),
                )
                .filter(
                    (pl.col("min_position") != 1)
                    | (pl.col("max_position") != pl.col("n"))
                    | (pl.col("n_unique_positions") != pl.col("n"))
                )
            )

        return [
            self._invariant_row(
                task,
                version,
                "no_backfill_no_event_before_raw_window_start",
                counters["n_events_before_raw_window"] == 0,
                f"violating_events={counters['n_events_before_raw_window']}",
            ),
            self._invariant_row(
                task,
                version,
                "no_backfill_no_event_from_earlier_history",
                counters["n_events_without_raw_window_anchor"] == 0,
                (
                    "events_without_anchor_in_raw_window="
                    f"{counters['n_events_without_raw_window_anchor']}"
                ),
            ),
            self._invariant_row(
                task,
                version,
                "no_backfill_anchor_time_matches_raw_window",
                counters["n_anchor_time_mismatch"] == 0,
                f"violating_events={counters['n_anchor_time_mismatch']}",
            ),
            self._invariant_row(
                task,
                version,
                "no_backfill_event_order_preserved",
                (
                    counters["n_event_order_mismatch"] == 0
                    and counters["n_noncontiguous_positions"] == 0
                ),
                (
                    f"order_mismatches={counters['n_event_order_mismatch']}; "
                    "episodes_with_noncontiguous_positions="
                    f"{counters['n_noncontiguous_positions']}"
                ),
            ),
            self._invariant_row(
                task,
                version,
                "no_backfill_all_events_strictly_before_prediction_time",
                counters["n_after_prediction_time"] == 0,
                (
                    "violating_events="
                    f"{counters['n_after_prediction_time']}"
                ),
            ),
        ]

    def check_structure_null(
        self,
        task: str,
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Exact long-event, vocabulary and sequence checks for structure_null."""
        if spec["window_mode"] != "structure_null":
            return []

        version = str(spec["name"])
        source_version = str(spec["source_representation"])
        gap = int(spec["gap_days"])
        pattern = self.structural_code_pattern(gap)

        source_parts = sorted(
            self.long_parts_dir(task, source_version).glob("*.parquet")
        )
        null_parts = sorted(self.long_parts_dir(task, version).glob("*.parquet"))
        if len(source_parts) != len(null_parts):
            return [
                self._invariant_row(
                    task,
                    version,
                    "structure_null_same_number_of_long_parts",
                    False,
                    f"source_parts={len(source_parts)}; null_parts={len(null_parts)}",
                )
            ]

        counters = {
            "n_source_events": 0,
            "n_null_events": 0,
            "n_source_only_positions": 0,
            "n_null_only_positions": 0,
            "n_time_mismatch": 0,
            "n_order_mismatch": 0,
            "n_base_code_mismatch": 0,
            "n_null_structural_codes": 0,
            "n_null_compression_flags": 0,
            "n_structural_features_not_removed": 0,
            "n_nonstructural_feature_mismatch": 0,
            "n_invariant_field_mismatch": 0,
        }

        join_keys = ["row_id", "event_position"]
        compare_cols = [
            "task",
            "subject_id",
            "prediction_time",
            "label",
            "split",
            "time",
            "code",
            "numeric_value",
            "text_value",
            "days_before_prediction",
            "is_compression_token",
            "order_anchor_id",
            "role_order",
        ]

        for source_path, null_path in zip(source_parts, null_parts):
            source_raw = pl.scan_parquet(str(source_path)).select(FINAL_LONG_COLS)
            null_raw = pl.scan_parquet(str(null_path)).select(FINAL_LONG_COLS)
            counters["n_source_events"] += self._count_rows(source_raw)
            counters["n_null_events"] += self._count_rows(null_raw)

            source_keys = source_raw.select(join_keys)
            null_keys = null_raw.select(join_keys)
            counters["n_source_only_positions"] += self._count_rows(
                source_keys.join(null_keys, on=join_keys, how="anti")
            )
            counters["n_null_only_positions"] += self._count_rows(
                null_keys.join(source_keys, on=join_keys, how="anti")
            )

            source = source_raw.select(
                *join_keys,
                *[
                    pl.col(col).alias(f"{col}__source")
                    for col in compare_cols
                ],
            )
            null = null_raw.select(
                *join_keys,
                *[
                    pl.col(col).alias(f"{col}__null")
                    for col in compare_cols
                ],
            )
            joined = source.join(null, on=join_keys, how="inner")

            source_structural = pl.col("code__source").str.contains(pattern)
            null_structural = pl.col("code__null").str.contains(pattern)
            expected_base = (
                pl.col("code__source")
                .str.replace(rf"^DX_ERA{gap}_START/", "")
                .str.replace(rf"^DX_ERA{gap}_END/", "")
                .str.replace(rf"^DX_ERA{gap}_COUNT_BIN/[^/]+/", "")
            )

            counters["n_time_mismatch"] += self._count_rows(
                joined.filter(pl.col("time__source") != pl.col("time__null"))
            )
            counters["n_order_mismatch"] += self._count_rows(
                joined.filter(
                    (pl.col("order_anchor_id__source") != pl.col("order_anchor_id__null"))
                    | (pl.col("role_order__source") != pl.col("role_order__null"))
                )
            )
            counters["n_base_code_mismatch"] += self._count_rows(
                joined.filter(pl.col("code__null") != expected_base)
            )
            counters["n_null_structural_codes"] += self._count_rows(
                joined.filter(null_structural)
            )
            counters["n_null_compression_flags"] += self._count_rows(
                joined.filter(pl.col("is_compression_token__null"))
            )
            counters["n_structural_features_not_removed"] += self._count_rows(
                joined.filter(
                    source_structural
                    & (
                        pl.col("numeric_value__null").is_not_null()
                        | pl.col("text_value__null").is_not_null()
                        | pl.col("is_compression_token__null")
                        | (pl.col("code__null") != expected_base)
                    )
                )
            )

            same_numeric = self._numeric_equal(
                pl.col("numeric_value__source"),
                pl.col("numeric_value__null"),
            )
            same_text = self._null_safe_equal(
                pl.col("text_value__source"),
                pl.col("text_value__null"),
            )
            counters["n_nonstructural_feature_mismatch"] += self._count_rows(
                joined.filter(
                    (~source_structural)
                    & (
                        (pl.col("code__source") != pl.col("code__null"))
                        | (~same_numeric)
                        | (~same_text)
                        | (
                            pl.col("is_compression_token__source")
                            != pl.col("is_compression_token__null")
                        )
                    )
                )
            )
            counters["n_invariant_field_mismatch"] += self._count_rows(
                joined.filter(
                    (pl.col("task__source") != pl.col("task__null"))
                    | (pl.col("subject_id__source") != pl.col("subject_id__null"))
                    | (pl.col("prediction_time__source") != pl.col("prediction_time__null"))
                    | (pl.col("label__source") != pl.col("label__null"))
                    | (pl.col("split__source") != pl.col("split__null"))
                    | (pl.col("days_before_prediction__source") != pl.col("days_before_prediction__null"))
                )
            )

        vocab_path = self.representation_dir(task, version) / "vocab.json"
        examples_path = self.representation_dir(task, version) / "examples.parquet"
        if not vocab_path.exists():
            raise FileNotFoundError(vocab_path)
        if not examples_path.exists():
            raise FileNotFoundError(examples_path)

        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        structural_vocab_codes = [
            code for code in vocab if re.match(pattern, str(code))
        ]
        structural_vocab_ids = {
            int(vocab[code]) for code in structural_vocab_codes
        }

        sequence_lf = pl.scan_parquet(str(examples_path))
        sequence_schema = sequence_lf.collect_schema().names()
        sequence_codes_missing = "codes" not in sequence_schema

        # -------------------------------------------------------------
        # Проверка специальных codes.
        #
        # Не используем list.eval(...) здесь: в Polars 1.31 сочетание
        # list.eval и streaming LazyFrame может ошибочно разрешаться
        # как обращение к колонке с пустым именем.
        # -------------------------------------------------------------
        if sequence_codes_missing:
            n_sequence_structural_codes = -1
        else:
            structural_code_rows = (
                sequence_lf
                .select(
                    pl.col("row_id"),
                    pl.col("codes"),
                )
                .explode("codes")
                .filter(
                    pl.col("codes")
                    .fill_null("")
                    .str.contains(pattern)
                )
                .select("row_id")
                .unique()
            )

            n_sequence_structural_codes = self._count_rows(
                structural_code_rows
            )

        # -------------------------------------------------------------
        # Проверка, что ID специальных tokens отсутствуют в sequence.
        # -------------------------------------------------------------
        if structural_vocab_ids:
            structural_token_rows = (
                sequence_lf
                .select(
                    pl.col("row_id"),
                    pl.col("token_ids"),
                )
                .explode("token_ids")
                .filter(
                    pl.col("token_ids").is_in(
                        sorted(structural_vocab_ids)
                    )
                )
                .select("row_id")
                .unique()
            )

            n_sequence_structural_token_ids = self._count_rows(
                structural_token_rows
            )
        else:
            n_sequence_structural_token_ids = 0

        # -------------------------------------------------------------
        # Проверка диапазона token IDs.
        # -------------------------------------------------------------
        invalid_token_id_rows = (
            sequence_lf
            .select(
                pl.col("row_id"),
                pl.col("token_ids"),
            )
            .explode("token_ids")
            .filter(
                (pl.col("token_ids") < 0)
                | (pl.col("token_ids") >= int(len(vocab)))
            )
            .select("row_id")
            .unique()
        )

        invalid_token_rows = self._count_rows(
            invalid_token_id_rows
        )

        source_examples = pd.read_parquet(
            self.representation_dir(task, source_version) / "examples.parquet"
        ).sort_values("row_id").reset_index(drop=True)
        null_examples = pd.read_parquet(examples_path).sort_values(
            "row_id"
        ).reset_index(drop=True)

        same_example_rows = np.array_equal(
            source_examples["row_id"].to_numpy(),
            null_examples["row_id"].to_numpy(),
        )
        same_seq_len = same_example_rows and np.array_equal(
            source_examples["seq_len"].to_numpy(),
            null_examples["seq_len"].to_numpy(),
        )
        same_time_axis = same_example_rows and all(
            np.allclose(
                np.asarray(as_python_list(a), dtype=float),
                np.asarray(as_python_list(b), dtype=float),
                equal_nan=True,
                atol=1e-6,
            )
            for a, b in zip(
                source_examples["days_before_prediction"],
                null_examples["days_before_prediction"],
            )
        )

        audit_cols = [
            "original_history_len",
            "transformed_history_len",
            "final_seq_len",
            "n_repeats_removed",
            "was_truncated",
            "earliest_retained_time",
            "covered_days",
            "n_backfill_events_added",
            "n_persistent_diagnoses",
        ]
        same_audit = same_example_rows
        if same_audit:
            for col in audit_cols:
                if col not in null_examples.columns or col not in source_examples.columns:
                    same_audit = False
                    break
                if pd.api.types.is_numeric_dtype(null_examples[col]):
                    a = pd.to_numeric(
                        null_examples[col], errors="coerce"
                    ).to_numpy(dtype=float)
                    b = pd.to_numeric(
                        source_examples[col], errors="coerce"
                    ).to_numpy(dtype=float)
                    if not np.allclose(a, b, equal_nan=True):
                        same_audit = False
                        break
                elif not null_examples[col].fillna("<NA>").equals(
                    source_examples[col].fillna("<NA>")
                ):
                    same_audit = False
                    break

        count_equal = counters["n_source_events"] == counters["n_null_events"]
        positions_equal = (
            counters["n_source_only_positions"] == 0
            and counters["n_null_only_positions"] == 0
        )
        only_expected_differences = (
            positions_equal
            and counters["n_time_mismatch"] == 0
            and counters["n_order_mismatch"] == 0
            and counters["n_base_code_mismatch"] == 0
            and counters["n_structural_features_not_removed"] == 0
            and counters["n_nonstructural_feature_mismatch"] == 0
            and counters["n_invariant_field_mismatch"] == 0
        )

        return [
            self._invariant_row(
                task,
                version,
                "structure_null_event_count_matches_backfill",
                count_equal and same_seq_len,
                (
                    f"source_events={counters['n_source_events']}; "
                    f"null_events={counters['n_null_events']}"
                ),
            ),
            self._invariant_row(
                task,
                version,
                "structure_null_event_times_match_backfill",
                counters["n_time_mismatch"] == 0 and same_time_axis,
                f"long_time_mismatches={counters['n_time_mismatch']}",
            ),
            self._invariant_row(
                task,
                version,
                "structure_null_event_order_matches_backfill",
                positions_equal and counters["n_order_mismatch"] == 0,
                (
                    f"source_only_positions={counters['n_source_only_positions']}; "
                    f"null_only_positions={counters['n_null_only_positions']}; "
                    f"order_mismatches={counters['n_order_mismatch']}"
                ),
            ),
            self._invariant_row(
                task,
                version,
                "structure_null_base_diagnoses_match_backfill",
                counters["n_base_code_mismatch"] == 0,
                f"mismatches={counters['n_base_code_mismatch']}",
            ),
            self._invariant_row(
                task,
                version,
                "structure_null_state_features_absent",
                (
                    counters["n_null_structural_codes"] == 0
                    and counters["n_null_compression_flags"] == 0
                    and counters["n_structural_features_not_removed"] == 0
                ),
                (
                    f"structural_codes={counters['n_null_structural_codes']}; "
                    f"compression_flags={counters['n_null_compression_flags']}; "
                    "structural_features_not_removed="
                    f"{counters['n_structural_features_not_removed']}"
                ),
            ),
            self._invariant_row(
                task,
                version,
                "structure_null_special_tokens_absent_from_vocabulary",
                len(structural_vocab_codes) == 0,
                f"structural_vocab_codes={structural_vocab_codes[:20]}",
            ),
            self._invariant_row(
                task,
                version,
                "structure_null_special_tokens_absent_from_sequence",
                (
                    not sequence_codes_missing
                    and n_sequence_structural_codes == 0
                    and n_sequence_structural_token_ids == 0
                    and invalid_token_rows == 0
                ),
                (
                    f"codes_column_missing={sequence_codes_missing}; "
                    f"rows_with_structural_codes={n_sequence_structural_codes}; "
                    "rows_with_structural_token_ids="
                    f"{n_sequence_structural_token_ids}; "
                    f"rows_with_invalid_token_ids={invalid_token_rows}"
                ),
            ),
            self._invariant_row(
                task,
                version,
                "structure_null_only_structural_features_differ",
                only_expected_differences and same_audit,
                (
                    "nonstructural_feature_mismatches="
                    f"{counters['n_nonstructural_feature_mismatch']}; "
                    "invariant_field_mismatches="
                    f"{counters['n_invariant_field_mismatch']}; "
                    f"same_episode_audit={same_audit}"
                ),
            ),
        ]

    def check_episode_audit(
        self,
        task: str,
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        path = self.representation_dir(task, spec["name"]) / "examples.parquet"
        df = pd.read_parquet(path)
        context = int(spec["context_length"])
        rows: list[dict[str, Any]] = []

        def add(check: str, passed: bool, details: str = "") -> None:
            rows.append(
                {
                    "task": task,
                    "version": spec["name"],
                    "check": check,
                    "passed": bool(passed),
                    "details": details,
                }
            )

        missing = [c for c in REQUIRED_EPISODE_AUDIT_COLUMNS if c not in df.columns]
        add("required_episode_audit_columns", not missing, f"missing={missing}")
        if missing:
            return rows

        add(
            "final_seq_len_equals_seq_len",
            bool((df["final_seq_len"] == df["seq_len"]).all()),
        )
        add("seq_len_within_context", bool((df["seq_len"] <= context).all()))
        add(
            "nonnegative_repeat_counts",
            bool((df["n_repeats_removed"] >= 0).all()),
        )
        add(
            "nonnegative_backfill_counts",
            bool((df["n_backfill_events_added"] >= 0).all()),
        )
        add(
            "backfill_not_above_final_len",
            bool((df["n_backfill_events_added"] <= df["seq_len"]).all()),
        )

        mode = str(spec["window_mode"])
        if spec["kind"] == "raw":
            add(
                "raw_transform_length_equals_original",
                bool(
                    (
                        df["transformed_history_len"]
                        == df["original_history_len"]
                    ).all()
                ),
            )
            add("raw_repeats_removed_zero", bool((df["n_repeats_removed"] == 0).all()))
        if mode == "backfill":
            add(
                "backfill_uses_full_history",
                bool(
                    (
                        df["transformation_input_len"]
                        == df["original_history_len"]
                    ).all()
                ),
            )
        if mode == "no_backfill":
            expected = np.minimum(df["original_history_len"].to_numpy(), context)
            add(
                "no_backfill_input_is_raw_window",
                bool(np.array_equal(df["transformation_input_len"].to_numpy(), expected)),
            )
            add(
                "no_backfill_added_zero",
                bool((df["n_backfill_events_added"] == 0).all()),
            )
        if mode == "raw":
            add(
                "raw_added_zero",
                bool((df["n_backfill_events_added"] == 0).all()),
            )

        return rows

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.build_base_cache()

        resolved_cfg_path = self.output_dir / "resolved_run_config.json"
        resolved_cfg_path.write_text(
            json.dumps(self.cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        matrix = expand_run_matrix(self.cfg)
        matrix.to_csv(self.output_dir / "resolved_run_matrix.csv", index=False)

        all_tasks = sorted(
            {task for spec in self.cfg["representations"] for task in spec["tasks"]}
        )
        labels_by_task = {task: self.load_labels(task) for task in all_tasks}
        history_parts_by_task = {
            task: self.build_task_history_parts(task, labels_by_task[task])
            for task in all_tasks
        }

        metadata_rows: list[dict[str, Any]] = []
        invariant_rows: list[dict[str, Any]] = []

        for spec in self.cfg["representations"]:
            for task in spec["tasks"]:
                metadata = self.build_representation(
                    task,
                    spec,
                    labels_by_task[task],
                    history_parts_by_task[task],
                )
                metadata_rows.append(metadata)
                new_invariants: list[dict[str, Any]] = []
                new_invariants.extend(self.check_episode_audit(task, spec))
                new_invariants.extend(
                    self.check_no_backfill(
                        task,
                        spec,
                        history_parts_by_task[task],
                    )
                )
                new_invariants.extend(self.check_structure_null(task, spec))
                invariant_rows.extend(new_invariants)
                self.persist_and_raise_invariant_failures(
                    invariant_rows,
                    only_new_rows=new_invariants,
                )

        summary_rows = []
        for metadata in metadata_rows:
            summary_rows.append(
                {
                    "task": metadata["task"],
                    "version": metadata["version"],
                    "n_examples": metadata["n_examples"],
                    "n_patients": metadata["n_patients"],
                    "n_positive": metadata["n_positive"],
                    "event_rate": metadata["event_rate"],
                    "vocab_size": metadata["vocab_size"],
                    "mean_original_history_len": metadata[
                        "mean_original_history_len"
                    ],
                    "mean_transformed_history_len": metadata[
                        "mean_transformed_history_len"
                    ],
                    "mean_seq_len": metadata["mean_seq_len"],
                    "median_seq_len": metadata["median_seq_len"],
                    "p90_seq_len": metadata["p90_seq_len"],
                    "max_seq_len": metadata["max_seq_len"],
                    "truncated_share": metadata["truncated_share"],
                    "mean_repeats_removed": metadata[
                        "mean_repeats_removed"
                    ],
                    "mean_backfill_events_added": metadata[
                        "mean_backfill_events_added"
                    ],
                    "min_days_before_prediction": metadata[
                        "min_days_before_prediction"
                    ],
                }
            )
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(
            self.output_dir / "all_compression_version_summary.csv",
            index=False,
        )
        invariants = pd.DataFrame(invariant_rows)
        self.persist_and_raise_invariant_failures(invariant_rows)

        # Once every structure-null dataset is finished, source long parts are
        # no longer required for training. They are retained only when requested.
        if not self.keep_long_parts:
            for spec in self.cfg["representations"]:
                for task in spec["tasks"]:
                    shutil.rmtree(
                        self.long_parts_dir(task, spec["name"]),
                        ignore_errors=True,
                    )
                    shutil.rmtree(
                        self.audit_parts_dir(task, spec["name"]),
                        ignore_errors=True,
                    )

        if not self.keep_history_cache:
            shutil.rmtree(self.history_cache_root, ignore_errors=True)

        print("=" * 100)
        print("DONE")
        print(f"Output: {self.output_dir}")
        print(summary.to_string(index=False))
        return summary, invariants


# -----------------------------------------------------------------------------
# Optional ClearML and upload
# -----------------------------------------------------------------------------

def is_clearml_agent_run() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def maybe_init_clearml(args: argparse.Namespace, cfg: dict[str, Any]):
    clearml_cfg = dict(cfg.get("clearml", {}))
    enabled = bool(clearml_cfg.get("enabled", False) or args.enable_clearml)
    remote = is_clearml_agent_run()
    if not enabled and not remote:
        return None

    from clearml import Task

    task = Task.current_task() if remote else None
    if task is None:
        task = Task.init(
            project_name=clearml_cfg.get(
                "project", "pershin-medailab/EHR_Risk_Profiling/EHRSHOT"
            ),
            task_name=clearml_cfg.get("task_name", cfg["run_set_id"]),
            output_uri=clearml_cfg.get("output_uri") or None,
            auto_connect_arg_parser=False,
        )
    task.connect(cfg)
    execute = bool(clearml_cfg.get("execute_remotely", False) or args.execute_remotely)
    if execute and not remote:
        task.execute_remotely(
            queue_name=clearml_cfg.get("queue", "cpu"), exit_process=True
        )
    return task


def upload_tree(local_root: Path, remote_prefix: str) -> pd.DataFrame:
    if not remote_prefix:
        return pd.DataFrame()
    from clearml import StorageManager

    rows = []
    for path in sorted(local_root.rglob("*")):
        if not path.is_file() or "_cache" in path.parts or "_long_parts" in path.parts:
            continue
        relative = path.relative_to(local_root).as_posix()
        remote = f"{remote_prefix.rstrip('/')}/{relative}"
        StorageManager.upload_file(
            local_file=str(path), remote_url=remote, wait_for_upload=True
        )
        rows.append({"local_path": str(path), "remote_url": remote})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    notebook_root = args.notebook_root.resolve()
    config_path = resolve_path(notebook_root, args.run_config).resolve()
    cfg = load_config(config_path)
    task = maybe_init_clearml(args, cfg)

    builder = CachedSequenceBuilder(
        cfg=cfg,
        notebook_root=notebook_root,
        run_config_path=config_path,
        rebuild=args.rebuild,
        rebuild_cache=args.rebuild_cache,
    )

    matrix = expand_run_matrix(cfg)
    print("=" * 100)
    print("CACHE-BASED CONFIG-DRIVEN SEQUENCE DATASET BUILD")
    print(f"run_set_id: {cfg['run_set_id']}")
    print(f"run_config: {config_path}")
    print(f"run_config_sha256: {sha256_file(config_path)}")
    print(f"ehrshot_root: {builder.ehrshot_root}")
    print(f"base_cache: {builder.base_cache_path}")
    print(f"persistent_code_list: {builder.persistent_code_list_path}")
    print(f"persistent concepts: {len(builder.persistent_codes)}")
    print(f"output_dir: {builder.output_dir}")
    print(f"include_prediction_time: {builder.include_prediction_time}")
    print(f"visit_anchor_max_days: {builder.visit_anchor_max_days}")
    print(f"subject_batch_size: {builder.subject_batch_size}")
    print(f"n_task_representation_pairs: {len(matrix)}")
    print(matrix.to_string(index=False))
    print("=" * 100)

    summary, invariants = builder.run()

    upload_enabled = bool(cfg["build"].get("upload", False)) and not args.skip_upload
    if upload_enabled:
        manifest = upload_tree(
            builder.output_dir, cfg["paths"].get("output_s3_prefix", "")
        )
        manifest.to_csv(builder.output_dir / "minio_upload_manifest.csv", index=False)
    if task is not None:
        task.upload_artifact("sequence_dataset_summary", summary)
        task.upload_artifact("representation_invariants", invariants)
        task.upload_artifact("resolved_run_matrix", matrix)


if __name__ == "__main__":
    main()