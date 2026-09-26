"""S2 Block (owner: Nidhi): norm_{split}_{src} -> candidates_{split}.

Runs every channel in config.CHANNELS (modules under blocking/) independently on
each country shard, then takes the union:
  channels    uint8 bitmask, bit i = config.CHANNELS[i]
  n_channels  number of channels that proposed the pair
  best_rank   best (lowest) channel_rank across those channels
  prior_score sum over contributing channels of 1 / channel_rank
and keeps the top config.MAX_CANDIDATES_PER_ENTITY per entity by prior_score
(ties: n_channels desc, best_rank asc, candidate id asc). The capped set is what
S5 scores, and so it is what candidate_pairs.tsv contains.

Country is a hard partition: 0 of 7,638,365 ground-truth pairs cross US/India.
France (test only, no labels) is sharded the same way, but that is UNPROVEN.
Shards come from the data, not a hard-coded list, so any new label gets its own shard.

Usage:
    python s2_block.py [--smoke] [--input DIR] [--output DIR]
"""
import importlib
import sys
import time

import polars as pl

import config
import pipeline_io as pio
from blocking.common import CHANNEL_SCHEMA, pop_resolved

CHANNEL_MODULES = {name: importlib.import_module(f"blocking.{name}") for name in config.CHANNELS}
assert all(m.NAME == n for n, m in CHANNEL_MODULES.items())


def load_shard(split: str, in_dir, country: str, s1_ids: pl.Series | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One country's Source-1 rows (optionally only `s1_ids`, e.g. the held-out
    validation entities) and its full S2+S3 pool."""
    s1 = pl.scan_parquet(config.norm_path(split, config.SOURCE1_SRC, in_dir)).filter(pl.col("country") == country)
    if s1_ids is not None:
        s1 = s1.filter(pl.col("entity_id").is_in(s1_ids.implode()))
    pool = pl.concat(
        [pl.scan_parquet(config.norm_path(split, s, in_dir)) for s in config.CANDIDATE_SRCS]
    ).filter(pl.col("country") == country)
    return s1.collect(), pool.collect()


def channel_pairs(split: str, in_dir, verbose: bool = True, s1_ids: pl.Series | None = None,
                  countries: list[str] | None = None, smoke: bool = False) -> tuple[pl.DataFrame, list[dict]]:
    """Long (source1_entity_id, candidate_entity_id, channel_rank, channel_score, bit)
    from every channel on every shard, plus per shard x channel stats (including the
    document-frequency ceilings each channel resolved). `s1_ids` restricts the
    Source-1 side (the pool is always complete); `countries` restricts which shards run
    (shards are independent, so per-country runs combine exactly); `smoke` is passed to
    every channel (embed_ann uses it to pick its pairs file)."""
    if countries is None:
        countries = (
            pl.scan_parquet(config.norm_path(split, config.SOURCE1_SRC, in_dir))
            .select(pl.col("country").unique().sort())
            .collect()["country"]
            .to_list()
        )
    parts, stats = [], []
    for country in countries:
        s1, pool = load_shard(split, in_dir, country, s1_ids)
        if verbose:
            print(f"[{split}/{country}] {s1.height:,} source1 x {pool.height:,} pool")
        for bit, (name, module) in enumerate(CHANNEL_MODULES.items()):
            pop_resolved()
            t0 = time.perf_counter()
            out = module.run(s1, pool, smoke=smoke)
            sec = time.perf_counter() - t0
            pio.check_schema(out, CHANNEL_SCHEMA, f"{name} output")
            assert out.select(pl.struct("source1_entity_id", "candidate_entity_id").is_unique().all()).item(), name
            row = {
                "split": split, "country": country, "channel": name, "pairs": out.height,
                "entities": out["source1_entity_id"].n_unique(), "sec": sec, "resolved": pop_resolved(),
            }
            stats.append(row)
            if verbose:
                print(f"  {name:<11} {row['pairs']:>10,} pairs  {row['entities']:>8,} entities  {sec:6.1f}s")
            parts.append(out.with_columns(pl.lit(1 << bit, dtype=pl.UInt8).alias("bit")))
    long = pl.concat(parts) if parts else pl.DataFrame(schema={**CHANNEL_SCHEMA, "bit": pl.UInt8})
    return long, stats


def union(long: pl.DataFrame) -> pl.DataFrame:
    """One row per pair in the candidates schema, uncapped."""
    return long.lazy().group_by("source1_entity_id", "candidate_entity_id").agg(
        pl.col("bit").sum().cast(pl.UInt8).alias("channels"),  # each channel emits a pair at most once, so sum == OR
        pl.len().cast(pl.UInt8).alias("n_channels"),
        pl.col("channel_rank").min().alias("best_rank"),
        (1.0 / pl.col("channel_rank").cast(pl.Float64)).sum().cast(pl.Float32).alias("prior_score"),
    ).collect(engine="streaming")


def cap(cands: pl.DataFrame, n: int = config.MAX_CANDIDATES_PER_ENTITY) -> pl.DataFrame:
    return (
        cands.sort(
            ["source1_entity_id", "prior_score", "n_channels", "best_rank", "candidate_entity_id"],
            descending=[False, True, True, False, False],
        )
        .filter(pl.int_range(pl.len()).over("source1_entity_id") < n)
        .select(list(pio.CANDIDATES_SCHEMA))
    )


def block(split: str, in_dir, verbose: bool = True, smoke: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    """(capped candidates, uncapped union, stats)."""
    long, stats = channel_pairs(split, in_dir, verbose, smoke=smoke)
    uncapped = union(long)
    return cap(uncapped), uncapped, stats


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()
    for split in config.SPLITS:
        if not config.norm_path(split, config.SOURCE1_SRC, in_dir).exists():
            print(f"  skipping {split} (norm_{split}_source1.parquet not found)")
            continue
        cands, uncapped, _ = block(split, in_dir, smoke=args.smoke)
        pio.check_schema(cands, pio.CANDIDATES_SCHEMA, f"candidates_{split}")
        out = config.candidates_path(split, out_dir)
        cands.write_parquet(out)
        n_s1 = cands["source1_entity_id"].n_unique()
        print(f"{out.name}: union {uncapped.height:,} pairs -> capped {cands.height:,} pairs over {n_s1:,} entities "
              f"(median {cands.group_by('source1_entity_id').len()['len'].median() or 0:.0f}/entity)")
    print(f"s2 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
