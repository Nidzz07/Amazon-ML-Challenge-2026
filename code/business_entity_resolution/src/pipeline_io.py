"""Shared plumbing for stages s1-s7: CLI flags, directory resolution, schema checks,
and the submission TSV writer. No stage logic lives here."""
import argparse
import subprocess
from pathlib import Path

import polars as pl

import config

NORM_SCHEMA = {
    "entity_id": pl.String,
    "name_norm": pl.String,
    "name_roman": pl.String,
    "name_tokens": pl.List(pl.String),
    "name_acronym": pl.String,
    "addr_norm": pl.String,
    "addr_roman": pl.String,
    "addr_tokens": pl.List(pl.String),
    "street_num": pl.String,
    "city_norm": pl.String,
    "state_canon": pl.String,
    "postcode": pl.String,
    "country": pl.String,
    "has_addr": pl.Boolean,
    "script": pl.UInt8,
    "name_suffix": pl.String,
}

CANDIDATES_SCHEMA = {
    "source1_entity_id": pl.String,
    "candidate_entity_id": pl.String,
    "channels": pl.UInt8,
    "n_channels": pl.UInt8,
    "best_rank": pl.UInt16,
    "prior_score": pl.Float32,
}

SCORED_SCHEMA = {
    "source1_entity_id": pl.String,
    "candidate_entity_id": pl.String,
    "prob": pl.Float32,
}

# Used only while features.py does not exist yet; Krrish's module replaces it.
STUB_FEATURE_NAMES = ("prior_score", "n_channels", "best_rank", "is_source3")
STUB_FEATURE_VERSION = 0


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--smoke", action="store_true", help="read and write artifacts/smoke/ instead of artifacts/")
    ap.add_argument("--input", type=Path, help="override the directory inputs are read from")
    ap.add_argument("--output", type=Path, help="override the directory outputs are written to")
    return ap


def artifacts_dir(smoke: bool) -> Path:
    return config.SMOKE_ARTIFACTS_DIR if smoke else config.ARTIFACTS_DIR


def dirs(args) -> tuple[Path, Path]:
    base = artifacts_dir(args.smoke)
    in_dir, out_dir = Path(args.input or base), Path(args.output or base)
    out_dir.mkdir(parents=True, exist_ok=True)
    return in_dir, out_dir


def check_schema(df: pl.DataFrame, schema: dict, name: str) -> None:
    got = dict(df.schema)
    assert list(got) == list(schema), f"{name}: columns {list(got)} != {list(schema)}"
    bad = {c: (got[c], t) for c, t in schema.items() if got[c] != t}
    assert not bad, f"{name}: dtype mismatch {bad}"


def feature_spec() -> tuple[tuple[str, ...], int]:
    try:
        from features import FEATURE_NAMES, FEATURE_VERSION
    except ImportError:
        return STUB_FEATURE_NAMES, STUB_FEATURE_VERSION
    return tuple(FEATURE_NAMES), int(FEATURE_VERSION)


def feature_columns(n: int) -> list[str]:
    return [f"f{i:03d}" for i in range(n)]


def val_ids(smoke: bool) -> pl.Series | None:
    """Held-out validation ids for full runs; None in smoke mode (no split there)."""
    if smoke or not config.VAL_ENTITY_IDS.exists():
        return None
    return pl.read_parquet(config.VAL_ENTITY_IDS)["entity_id"]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def write_id_lists(s1_ids: pl.Series, pairs: pl.DataFrame, list_col: str, path: Path) -> None:
    """One row per Source-1 entity, in s1_ids order; ids in `pairs` order, comma-joined;
    empty string (never null/NaN) when an entity has none. Tab-separated, UTF-8, no quoting."""
    lists = pairs.group_by("source1_entity_id", maintain_order=True).agg(
        pl.col("candidate_entity_id").str.join(",").alias(list_col)
    )
    out = (
        pl.DataFrame({"source1_entity_id": s1_ids})
        .join(lists, on="source1_entity_id", how="left", maintain_order="left")
        .with_columns(pl.col(list_col).fill_null(""))
    )
    assert out.height == len(s1_ids) and out["source1_entity_id"].n_unique() == out.height
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator=config.TSV_SEP, quote_style="never", line_terminator="\n")


def read_id_lists(path: Path, list_col: str) -> pl.DataFrame:
    """Inverse of write_id_lists: long (source1_entity_id, candidate_entity_id) pairs."""
    df = pl.read_csv(path, separator=config.TSV_SEP, infer_schema=False, quote_char=None, encoding="utf8")
    return (
        df.select("source1_entity_id", pl.col(list_col).fill_null("").str.split(",").alias("candidate_entity_id"))
        .explode("candidate_entity_id", empty_as_null=False)
        .filter(pl.col("candidate_entity_id") != "")
    )
