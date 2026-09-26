"""The contract every blocking channel satisfies.

A channel is a module with `NAME` (an entry of config.CHANNELS) and
`run(s1, pool, smoke=False) -> pl.DataFrame`. `s1` holds the Source-1 rows and
`pool` the S2+S3 rows of ONE country shard, both in the norm_{split}_{src} schema.
`smoke` says the shard comes from the smoke sample; only channels that read a
precomputed artifact (embed_ann) use it, to pick the smoke or full-scale file. The
return value has exactly CHANNEL_SCHEMA. Rows are unique on the pair, and
channel_rank is 1 for the channel's best candidate for that entity.

Document-frequency ceilings (config.TFIDF_MAX_DF, config.RARE_TOKEN_DF_MAX) go
through resolve_df when the channel builds its index, so a fraction becomes a count
against the shard actually being indexed.
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


# Ceilings resolved since the last pop_resolved(); s2_block attaches them to its
# per shard x channel stats. Logging only, never read by a channel.
_resolved: dict[str, dict] = {}


def resolve_df(knob: str, value: int | float | None, n_docs: int) -> int | None:
    """A document-frequency ceiling as an absolute count over a corpus of n_docs.
    A float in (0, 1] is a fraction of n_docs (floored); an int is already a count;
    None means no ceiling. Prints and records the result under `knob`."""
    if value is None:
        resolved = None
    elif isinstance(value, float):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{knob}: fraction {value} outside (0, 1]")
        resolved = int(value * n_docs)
    else:
        resolved = int(value)
    _resolved[knob] = {"value": value, "n_docs": n_docs, "resolved": resolved}
    print(f"    {knob} = {value!r} over {n_docs:,} docs -> {'none' if resolved is None else f'{resolved:,}'}")
    return resolved


def pop_resolved() -> dict[str, dict]:
    out = dict(_resolved)
    _resolved.clear()
    return out


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
