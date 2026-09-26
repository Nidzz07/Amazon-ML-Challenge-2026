"""The contract every blocking channel satisfies.

A channel is a module with `NAME` (an entry of config.CHANNELS) and
`run(s1, pool) -> pl.DataFrame`. `s1` holds the Source-1 rows and `pool` the
S2+S3 rows of ONE country shard, both in the norm_{split}_{src} schema. The
return value has exactly CHANNEL_SCHEMA. Rows are unique on the pair, and
channel_rank is 1 for the channel's best candidate for that entity.
"""
import polars as pl

CHANNEL_SCHEMA = {
    "source1_entity_id": pl.String,
    "candidate_entity_id": pl.String,
    "channel_rank": pl.UInt16,
    "channel_score": pl.Float32,
}


def empty() -> pl.DataFrame:
    return pl.DataFrame(schema=CHANNEL_SCHEMA)


def rank_within_entity(pairs: pl.DataFrame, by: list[str], descending: list[bool], score: str) -> pl.DataFrame:
    """Assign channel_rank 1..n per entity by `by`, ties broken by candidate id."""
    return (
        pairs.sort(["source1_entity_id", *by, "candidate_entity_id"], descending=[False, *descending, False])
        .with_columns((pl.int_range(pl.len()).over("source1_entity_id") + 1).alias("channel_rank"))
        .select(
            "source1_entity_id",
            "candidate_entity_id",
            pl.col("channel_rank").clip(upper_bound=65_535).cast(pl.UInt16),
            pl.col(score).cast(pl.Float32).alias("channel_score"),
        )
    )
