"""Cut the smoke sample every track develops against.

Train smoke: config.SMOKE_TRAIN_ENTITIES random Source-1 entities, every one of
their true matches, plus a random distractor pool of config.SMOKE_DISTRACTOR_RATIO x
(true-match count) S2/S3 records that match none of the sampled entities. Distractors
are drawn from the rest of the pool, so they mix globally-unmatched records with records
belonging to non-sampled entities. That is the same mix a test entity sees.

Test smoke: config.SMOKE_TEST_ENTITIES Source-1 entities stratified by country (every
country gets >= config.SMOKE_TEST_MIN_PER_COUNTRY), plus a random S2/S3 pool sized
per country in proportion to the entities sampled. Test has no labels, so this pool
contains few true matches: use it for schema and runtime checks, not for recall.

Reads the S0 parquet (faithful to the raw TSVs) and writes
artifacts/smoke/smoke_{split}_{src}.tsv in the raw TSV format, so s0_ingest.py --smoke
ingests them exactly like the full files.

Usage:
    python make_smoke_sample.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import config
from s0_ingest import scan_tsv


def read(split: str, src: str) -> pl.DataFrame:
    pq = config.records_path(split, src)
    assert pq.exists(), f"{pq} missing - run s0_ingest.py first"
    return pl.read_parquet(pq)


def sample_ids(ids: pl.Series, n: int, rng: np.random.Generator) -> pl.Series:
    # Sort first so the draw depends only on SEED and the id set, not on row order.
    ids = ids.sort()
    return ids.gather(np.sort(rng.choice(len(ids), size=min(n, len(ids)), replace=False)))


def write_tsv(df: pl.DataFrame, path: Path) -> None:
    # Same format as the organisers' files: tab-separated, UTF-8, minimal quoting
    # (fields containing a quote get pandas-style "" escaping, which s0 reads back).
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator=config.TSV_SEP, quote_style="necessary", line_terminator="\n")


def verify_roundtrip(df: pl.DataFrame, path: Path) -> None:
    back = scan_tsv(path).select([pl.col(c).fill_null("") for c in df.columns]).collect()
    assert back.equals(df), f"{path.name}: round-trip through s0's reader changed the data"


def split_pool_ids(pool_ids: pl.Series) -> dict[str, pl.Series]:
    return {src: pool_ids.filter(pool_ids.str.starts_with(f"S{src[-1]}-")) for src in config.CANDIDATE_SRCS}


def make_train(rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    s1 = read("train", "source1")
    gt = read("train", "ground_truth")

    s1_ids = sample_ids(s1["entity_id"], config.SMOKE_TRAIN_ENTITIES, rng)
    gt_smoke = gt.filter(pl.col("source1_entity_id").is_in(s1_ids.implode()))
    true_ids = (
        gt_smoke.select(pl.col("matched_entity_ids").str.split(",").alias("id"))
        .explode("id", empty_as_null=False)
        .filter(pl.col("id") != "")["id"]
    )

    pools = {src: read("train", src) for src in config.CANDIDATE_SRCS}
    all_pool = pl.concat([p["entity_id"] for p in pools.values()])
    non_matching = all_pool.filter(~all_pool.is_in(true_ids.implode()))
    distractors = sample_ids(non_matching, round(config.SMOKE_DISTRACTOR_RATIO * len(true_ids)), rng)
    keep = pl.concat([true_ids, distractors]).implode()

    out = {
        "source1": s1.filter(pl.col("entity_id").is_in(s1_ids.implode())),
        "ground_truth": gt_smoke,
    }
    for src, df in pools.items():
        out[src] = df.filter(pl.col("entity_id").is_in(keep))

    n_pool = out["source2"].height + out["source3"].height
    assert n_pool == len(true_ids) + len(distractors), "true-match ids missing from S2/S3"
    print(f"train: {len(s1_ids):,} entities, {len(true_ids):,} true-match records, "
          f"{len(distractors):,} distractors ({len(distractors) / max(len(true_ids), 1):.2f}x)")
    return out


def allocate(counts: dict[str, int], total: int, floor: int) -> dict[str, int]:
    """Proportional allocation with a per-group floor, summing exactly to total."""
    n_all = sum(counts.values())
    alloc = {c: min(n, max(floor, round(total * n / n_all))) for c, n in counts.items()}
    # Absorb rounding / floor drift in the largest groups, never pushing below the floor.
    for c in sorted(counts, key=counts.get, reverse=True):
        diff = total - sum(alloc.values())
        if diff == 0:
            break
        alloc[c] = min(counts[c], max(floor, alloc[c] + diff))
    assert sum(alloc.values()) == total, alloc
    return alloc


def make_test(rng: np.random.Generator) -> dict[str, pl.DataFrame] | None:
    """Returns None (with a warning) when test_source2/3 parquets are missing.
    This lets the script generate a train-only smoke so pipeline stubs can still
    be validated end-to-end before the full test pool is downloaded.
    """
    missing = [
        src for src in config.CANDIDATE_SRCS
        if not config.records_path("test", src).exists()
    ]
    if missing:
        print(f"WARNING: test smoke skipped — missing parquets for {missing}.")
        print("         Download test_source2.tsv / test_source3.tsv and re-run to generate it.")
        return None

    s1 = read("test", "source1")
    pools = {src: read("test", src) for src in config.CANDIDATE_SRCS}

    s1_counts = dict(s1.group_by("country").len().sort("country").iter_rows())
    alloc = allocate(s1_counts, config.SMOKE_TEST_ENTITIES, config.SMOKE_TEST_MIN_PER_COUNTRY)

    s1_keep, pool_keep = [], {src: [] for src in pools}
    for country, n in alloc.items():
        s1_keep.append(sample_ids(s1.filter(pl.col("country") == country)["entity_id"], n, rng))
        frac = n / s1_counts[country]
        for src, df in pools.items():
            ids = df.filter(pl.col("country") == country)["entity_id"]
            pool_keep[src].append(sample_ids(ids, round(frac * len(ids)), rng))

    out = {"source1": s1.filter(pl.col("entity_id").is_in(pl.concat(s1_keep).implode()))}
    for src, df in pools.items():
        out[src] = df.filter(pl.col("entity_id").is_in(pl.concat(pool_keep[src]).implode()))
    print(f"test : allocation by country {alloc}")
    return out


def country_breakdown(split: str, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for src, df in frames.items():
        if src == "ground_truth":
            continue
        for country, n in df.group_by("country").len().iter_rows():
            rows.append({"split": split, "src": src, "country": country, "rows": n})
    return (
        pl.DataFrame(rows)
        .pivot(on="country", index=["split", "src"], values="rows", sort_columns=True)
        .fill_null(0)
        .sort("src")
    )


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(config.SEED)
    train_smoke = make_train(rng)
    test_smoke = make_test(rng)   # may return None if test pool not downloaded yet

    smoke = {"train": train_smoke}
    if test_smoke is not None:
        smoke["test"] = test_smoke

    print()
    print(f"{'file':<32} {'rows':>10}")
    for split, frames in smoke.items():
        for src in config.SPLITS[split]:
            if src not in frames:
                continue
            path = config.smoke_raw_path(split, src)
            write_tsv(frames[src], path)
            verify_roundtrip(frames[src], path)
            print(f"{path.name:<32} {frames[src].height:>10,}")

    gt = smoke["train"]["ground_truth"]
    singletons = (gt["matched_entity_ids"] == "").sum()
    print(f"\ntrain smoke singletons: {singletons:,} ({100 * singletons / gt.height:.2f}%)")
    print("\ncountry breakdown:")
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(country_breakdown("train", smoke["train"]))
        if test_smoke is not None:
            print(country_breakdown("test", smoke["test"]))
    print(f"\nwrote {config.SMOKE_DIR} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
