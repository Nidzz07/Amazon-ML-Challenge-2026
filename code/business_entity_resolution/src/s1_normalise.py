"""S1 Normalise (STUB, owner: Parth): records_{split}_{src} -> norm_{split}_{src}.

Pass-through placeholder. It emits the norm schema with no real normalisation:
lowercase and trimmed text, no transliteration (name_roman == name_norm), and
word-regex tokens. Replace the body of normalise_frame(). The output columns and
dtypes are fixed by PROJECT_ROADMAP.md.

Usage:
    python s1_normalise.py [--smoke] [--input DIR] [--output DIR]
"""
import sys
import time

import polars as pl

import config
import pipeline_io as pio

TOKEN_RE = r"\w+"  # Unicode-aware: keeps Devanagari/Tamil/etc. words intact


def normalise_frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    name = pl.col("business_name").str.to_lowercase().str.strip_chars()
    addr = pl.col("business_address").str.to_lowercase().str.strip_chars()
    return (
        lf.with_columns(name.alias("name_norm"), addr.alias("addr_norm"))
        .with_columns(
            pl.col("name_norm").str.extract_all(TOKEN_RE).alias("name_tokens"),
            pl.col("addr_norm").str.extract_all(TOKEN_RE).alias("addr_tokens"),
        )
        .select(
            "entity_id",
            "name_norm",
            pl.col("name_norm").alias("name_roman"),
            "name_tokens",
            pl.col("name_tokens").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("name_acronym"),
            "addr_norm",
            pl.col("addr_norm").alias("addr_roman"),
            "addr_tokens",
            pl.col("addr_norm").str.extract(r"\b(\d+)\b", 1).fill_null("").alias("street_num"),
            pl.lit("").alias("city_norm"),
            pl.lit("").alias("state_canon"),
            pl.col("addr_norm").str.extract(r"\b(\d{5,6})\b", 1).fill_null("").alias("postcode"),
            "country",
            (pl.col("addr_norm") != "").alias("has_addr"),
            pl.lit(0, dtype=pl.UInt8).alias("script"),  # 0 = unknown until script detection lands
            pl.lit("").alias("name_suffix"),
        )
    )


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()
    for split, srcs in config.SPLITS.items():
        for src in srcs:
            if src == "ground_truth":
                continue
            out = config.norm_path(split, src, out_dir)
            normalise_frame(pl.scan_parquet(config.records_path(split, src, in_dir))).sink_parquet(out)
            df = pl.read_parquet(out)
            pio.check_schema(df, pio.NORM_SCHEMA, out.name)
            print(f"{out.name:<32} {df.height:>10,}")
    print(f"s1 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
