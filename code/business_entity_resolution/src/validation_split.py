"""Seeded validation split: hold out config.VAL_SIZE Source-1 train entities.

Writes only the held-out ids to artifacts/val_entity_ids.parquet (one column,
entity_id). Source-2/3 are deliberately NOT filtered or split: records that belong
to the remaining training entities stay in the pool as distractors, so validation
sees exactly the distractor density that test does.

Usage:
    python validation_split.py
"""
import sys
import time

import numpy as np
import polars as pl

import config


def load_source1_ids() -> pl.Series:
    pq = config.records_path("train", config.SOURCE1_SRC)
    assert pq.exists(), f"{pq} missing - run s0_ingest.py first"
    # Sort before sampling so the draw depends only on SEED and the id set,
    # never on file row order.
    return pl.read_parquet(pq, columns=["entity_id"])["entity_id"].sort()


def make_split(ids: pl.Series, n_val: int, seed: int) -> tuple[pl.Series, pl.Series]:
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(ids), dtype=bool)
    mask[rng.choice(len(ids), size=n_val, replace=False)] = True
    return ids.filter(pl.Series(mask)), ids.filter(pl.Series(~mask))


def main() -> None:
    t0 = time.perf_counter()
    ids = load_source1_ids()
    expected = config.EXPECTED_ROWS[("train", config.SOURCE1_SRC)]
    assert len(ids) == expected, f"train_source1 has {len(ids):,} ids, expected {expected:,}"
    assert ids.n_unique() == len(ids), "duplicate entity_id in train_source1"

    val, rest = make_split(ids, config.VAL_SIZE, config.SEED)
    overlap = len(set(val.to_list()) & set(rest.to_list()))
    assert overlap == 0, f"{overlap:,} ids in both held-out and train"
    assert len(val) + len(rest) == len(ids)

    config.VAL_ENTITY_IDS.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"entity_id": val}).write_parquet(config.VAL_ENTITY_IDS)

    # Sanity: the held-out slice should look like the whole in country mix and singleton rate.
    s1 = pl.scan_parquet(config.records_path("train", config.SOURCE1_SRC)).select("entity_id", "country")
    gt = pl.scan_parquet(config.records_path("train", "ground_truth")).select(
        pl.col("source1_entity_id").alias("entity_id"),
        (pl.col("matched_entity_ids") == "").alias("singleton"),
    )
    tagged = (
        s1.join(gt, on="entity_id", how="left")
        .with_columns(pl.col("entity_id").is_in(val.implode()).alias("held_out"))
        .group_by("held_out", "country")
        .agg(pl.len().alias("entities"), pl.col("singleton").mean().alias("singleton_rate"))
        .sort("held_out", "country")
        .collect()
    )

    print(f"train_source1 entities : {len(ids):,}")
    print(f"held out (validation)  : {len(val):,}")
    print(f"remaining train        : {len(rest):,}")
    print(f"overlap                : {overlap}")
    print(f"seed                   : {config.SEED}")
    print("S2/S3 pools            : untouched (full pool used for both train and validation)")
    print()
    print(tagged)
    print(f"\nwrote {config.VAL_ENTITY_IDS} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
