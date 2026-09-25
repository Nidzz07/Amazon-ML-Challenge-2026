"""S2 Block (STUB, owner: Nidhi): norm_{split}_{src} -> candidates_{split}.

Placeholder for the five-channel union under blocking/. It uses a single exact-key
block on (country, first name token), drops buckets over config.EXACT_KEY_MAX_BUCKET,
scores each pair with a crude name/street-number equality prior, and caps at
config.MAX_CANDIDATES_PER_ENTITY per entity. Its only job is to produce
schema-valid candidates, including some true positives, so S3-S7 can run.

Country is a hard partition: 0 of 7,638,365 ground-truth pairs cross US/India.
France is treated the same way but that is UNPROVEN, because France has no
training labels.

Usage:
    python s2_block.py [--smoke] [--input DIR] [--output DIR]
"""
import sys
import time

import polars as pl

import config
import pipeline_io as pio

STUB_CHANNEL_BIT = 1 << config.CHANNELS.index("exact_key")
KEY_COLS = ["country", "block_key"]


def keyed(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.select(
        "entity_id",
        "country",
        pl.col("name_tokens").list.first().alias("block_key"),
        "name_norm",
        "street_num",
    ).filter(pl.col("block_key").is_not_null() & (pl.col("block_key") != ""))


def block(split: str, in_dir, cap: int = config.MAX_CANDIDATES_PER_ENTITY) -> pl.DataFrame:
    s1 = keyed(pl.scan_parquet(config.norm_path(split, config.SOURCE1_SRC, in_dir)))
    pool = keyed(pl.concat([pl.scan_parquet(config.norm_path(split, s, in_dir)) for s in config.CANDIDATE_SRCS]))
    pool = pool.filter(pl.len().over(KEY_COLS) <= config.EXACT_KEY_MAX_BUCKET)

    pairs = s1.join(pool, on=KEY_COLS, how="inner", suffix="_c")
    name_eq = (pl.col("name_norm") == pl.col("name_norm_c")).cast(pl.Float32)
    street_eq = ((pl.col("street_num") != "") & (pl.col("street_num") == pl.col("street_num_c"))).cast(pl.Float32)
    return (
        pairs.select(
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_c").alias("candidate_entity_id"),
            (0.5 * name_eq + 0.5 * street_eq).alias("prior_score"),
        )
        .sort(["source1_entity_id", "prior_score", "candidate_entity_id"], descending=[False, True, False])
        .with_columns((pl.int_range(pl.len()).over("source1_entity_id") + 1).alias("rank"))
        .filter(pl.col("rank") <= cap)
        .select(
            "source1_entity_id",
            "candidate_entity_id",
            pl.lit(STUB_CHANNEL_BIT, dtype=pl.UInt8).alias("channels"),
            pl.lit(1, dtype=pl.UInt8).alias("n_channels"),
            pl.col("rank").cast(pl.UInt16).alias("best_rank"),
            pl.col("prior_score").cast(pl.Float32),
        )
        .collect()
    )


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()
    for split in config.SPLITS:
        cands = block(split, in_dir)
        pio.check_schema(cands, pio.CANDIDATES_SCHEMA, f"candidates_{split}")
        out = config.candidates_path(split, out_dir)
        cands.write_parquet(out)
        n_s1 = cands["source1_entity_id"].n_unique()
        print(f"{out.name:<28} {cands.height:>10,} pairs over {n_s1:,} entities")
    print(f"s2 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
