"""Shared plumbing for stages s1-s7: CLI flags, directory resolution, schema checks,
the submission TSV writer and the shared train-entity sampler. No stage logic lives here."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
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
    # Legal suffix stripped from name_roman ("private limited", "llc", "sarl"); '' if none.
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


def sample_train_entities(s1: pl.DataFrame, n: int, split: str, smoke: bool,
                          seed: int = config.SEED) -> pl.Series:
    """The ONE train-entity sampler; every stage that subsamples train imports this, so
    they can never sample different entity sets. s1 has entity_id and country.

    Returns the sorted entity_ids to use: every held-out validation entity (val_ids(smoke),
    looked up here so no caller can forget or substitute it; none on smoke) plus n of the
    others, stratified by country: each country gets n x its share of the others, floored,
    with the leftover handed out by largest remainder (ties by country name). Within a
    country the draw is over sorted ids, countries in sorted order, from one numpy
    default_rng(seed), so it depends only on the id set and seed. n <= 0 or n >= the
    number of others returns every id. Never call it on test."""
    assert split == "train", f"sample_train_entities called on split {split!r}: test must never be sampled"
    keep = val_ids(smoke)
    assert smoke or keep is not None, f"{config.VAL_ENTITY_IDS} missing: run validation_split.py first"
    ids = s1.select("entity_id", "country")
    kept = ids.filter(pl.col("entity_id").is_in(keep.implode())) if keep is not None else ids.head(0)
    rest = ids.join(kept, on="entity_id", how="anti") if keep is not None else ids
    if n <= 0 or n >= rest.height:
        return ids["entity_id"].sort()
    counts = rest.group_by("country").len().sort("country")
    exact = counts["len"].to_numpy().astype(np.int64) * n / rest.height  # u32 x 800k overflows
    quota = np.floor(exact).astype(np.int64)
    order = np.lexsort((np.arange(len(quota)), -(exact - quota)))  # largest remainder, ties by country order
    quota[order[: n - int(quota.sum())]] += 1
    rng = np.random.default_rng(seed)
    picked = []
    for country, q in zip(counts["country"], quota):
        pool = rest.filter(pl.col("country") == country)["entity_id"].sort()
        picked.append(pool.gather(np.sort(rng.choice(len(pool), size=int(q), replace=False))))
    return pl.concat([kept["entity_id"], *picked]).sort()


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def peak_rss_bytes() -> int | None:
    """Peak resident memory of this process so far (Windows: peak working set), stdlib
    only. It never goes down, so it covers everything the process has run until now."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (f, ctypes.c_size_t) for f in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        c = Counters(cb=ctypes.sizeof(Counters))
        return c.PeakWorkingSetSize if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb) else None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux reports KiB, macOS bytes


# Rows of `pairs` per list-building slice, and Source-1 entities per written block.
# Neither changes the output; together they bound write_id_lists' working set.
ID_LIST_PAIR_ROWS = 2_000_000
ID_LIST_WRITE_ROWS = 100_000


def write_id_lists(s1_ids: pl.Series, pairs: pl.DataFrame, list_col: str, path: Path) -> None:
    """One row per Source-1 entity, in s1_ids order; ids in `pairs` order, comma-joined;
    empty string (never null/NaN) when an entity has none. Tab-separated, UTF-8, no quoting.

    Bounded memory, same bytes as the one-shot version (one group_by + str.join over every
    pair, then one write_csv), which peaked at ~7 GB on the full test candidates (52M IDs):
      - lists are built over row slices of `pairs` cut at entity boundaries. That needs each
        entity's rows contiguous; if they are not, a stable sort by entity makes them so and
        keeps every entity's ID order;
      - the output is written in blocks of s1_ids through one file handle, header once."""
    assert s1_ids.n_unique() == len(s1_ids), "write_id_lists: duplicate Source-1 ids"
    p = pairs.select("source1_entity_id", "candidate_entity_id")
    # Contiguous iff no entity starts two runs; checked on the ~1.7M run values, not by
    # hashing all ~52M rows.
    runs = p["source1_entity_id"].rle().struct.field("value")
    if runs.n_unique() != runs.len():  # an entity's rows are split
        p = p.sort("source1_entity_id", maintain_order=True)
    del runs
    run = p["source1_entity_id"].rle_id()
    lists, start = [], 0
    while start < p.height:
        end = min(start + ID_LIST_PAIR_ROWS, p.height)
        if end < p.height:  # extend to the end of the entity straddling the cut
            last = run[end - 1]
            end = start + int(run.slice(start).search_sorted(last + 1))
        lists.append(
            p.slice(start, end - start).group_by("source1_entity_id", maintain_order=True)
            .agg(pl.col("candidate_entity_id").str.join(",").alias(list_col))
        )
        start = end
    del p, run
    lists = pl.concat(lists) if lists else pl.DataFrame(schema={"source1_entity_id": pl.String, list_col: pl.String})
    ids = pl.DataFrame({"source1_entity_id": s1_ids})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        for i, off in enumerate(range(0, max(ids.height, 1), ID_LIST_WRITE_ROWS)):
            block = (
                ids.slice(off, ID_LIST_WRITE_ROWS)
                .join(lists, on="source1_entity_id", how="left", maintain_order="left")
                .with_columns(pl.col(list_col).fill_null(""))
            )
            block.write_csv(fh, separator=config.TSV_SEP, quote_style="never", line_terminator="\n",
                            include_header=i == 0)


def read_id_lists(path: Path, list_col: str) -> pl.DataFrame:
    """Inverse of write_id_lists: long (source1_entity_id, candidate_entity_id) pairs."""
    df = pl.read_csv(path, separator=config.TSV_SEP, infer_schema=False, quote_char=None, encoding="utf8")
    return (
        df.select("source1_entity_id", pl.col(list_col).fill_null("").str.split(",").alias("candidate_entity_id"))
        .explode("candidate_entity_id", empty_as_null=False)
        .filter(pl.col("candidate_entity_id") != "")
    )
