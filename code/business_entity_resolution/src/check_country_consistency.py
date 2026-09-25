"""Hour-0 task 1: do true matched pairs ever cross countries?

For every (source1_entity_id, matched_id) pair in train_ground_truth, compare the
country of the Source-1 entity with the country of the matched S2/S3 record.
Reads only entity_id + country columns, via polars lazy scans.
"""
import time

import polars as pl

import config
from s0_ingest import scan_tsv


def scan(split: str, src: str) -> pl.LazyFrame:
    pq = config.records_path(split, src)
    if pq.exists():
        return pl.scan_parquet(pq)
    return scan_tsv(config.raw_path(split, src))


def main() -> None:
    t0 = time.perf_counter()

    pairs = (
        scan("train", "ground_truth")
        .select(
            pl.col("source1_entity_id"),
            pl.col("matched_entity_ids").str.split(",").alias("matched_id"),
        )
        .explode("matched_id", empty_as_null=False)
        .with_columns(pl.col("matched_id").str.strip_chars())
        .filter(pl.col("matched_id").is_not_null() & (pl.col("matched_id") != ""))
    )
    s1 = scan("train", "source1").select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("country").alias("country_s1"),
    )
    s23 = pl.concat(
        [scan("train", s).select("entity_id", "country") for s in ("source2", "source3")]
    ).select(pl.col("entity_id").alias("matched_id"), pl.col("country").alias("country_m"))

    joined = (
        pairs.join(s1, on="source1_entity_id", how="left")
        .join(s23, on="matched_id", how="left")
        .with_columns(pl.col("matched_id").str.slice(0, 2).alias("m_src"))
        .collect()
    )

    total = joined.height
    missing_s1 = joined["country_s1"].is_null().sum()
    missing_m = joined["country_m"].is_null().sum()
    both = joined.filter(pl.col("country_s1").is_not_null() & pl.col("country_m").is_not_null())
    diff_mask = both["country_s1"] != both["country_m"]
    n_diff = int(diff_mask.sum())
    n_same = both.height - n_diff

    print("=== Country consistency of ground-truth matched pairs (train) ===")
    print(f"total matched pairs in ground truth : {total:,}")
    print(f"  S1 id not found in source1        : {missing_s1:,}")
    print(f"  matched id not found in source2/3 : {missing_m:,}")
    print(f"pairs checked (both countries known): {both.height:,}")
    print(f"same country                        : {n_same:,}")
    print(f"different country                   : {n_diff:,}  ({100 * n_diff / max(both.height, 1):.4f}%)")

    print("\nby matched source:")
    print(
        both.group_by("m_src")
        .agg(pl.len().alias("pairs"), (pl.col("country_s1") != pl.col("country_m")).sum().alias("diff"))
        .sort("m_src")
    )
    print("\nby Source-1 country:")
    print(
        both.group_by("country_s1")
        .agg(pl.len().alias("pairs"), (pl.col("country_s1") != pl.col("country_m")).sum().alias("diff"))
        .sort("pairs", descending=True)
    )

    if n_diff:
        print("\ntop 10 (country_A = S1, country_B = matched) mismatch combinations:")
        with pl.Config(tbl_rows=10):
            print(
                both.filter(diff_mask)
                .group_by("country_s1", "country_m")
                .len()
                .sort("len", descending=True)
                .head(10)
            )
        print("\nsample mismatched pairs:")
        print(both.filter(diff_mask).head(10))

    print("\ndistinct country labels per train file:")
    for src in ("source1", "source2", "source3"):
        vc = scan("train", src).group_by("country").len().sort("len", descending=True).collect()
        print(f"  {src}: " + ", ".join(f"{c!r}={n:,}" for c, n in vc.iter_rows()))

    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
