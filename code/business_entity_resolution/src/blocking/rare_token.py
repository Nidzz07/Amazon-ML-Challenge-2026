"""Channel rare_token: inverted index over name_tokens + addr_tokens.

Tokens are namespaced by field ("n:" / "a:") so a name word only matches a name
word. Document frequency counts distinct records over the whole shard (S1 + pool),
and only tokens with config.RARE_TOKEN_DF_MIN <= df <= config.RARE_TOKEN_DF_MAX are
indexed (df 1 can never form a pair). RARE_TOKEN_DF_MAX may be a fraction; it is
resolved against the shard's S1 + pool record count (blocking.common.resolve_df). For each Source-1 entity, its
config.RARE_TOKENS_PER_ENTITY rarest tokens are taken (lowest df, ties broken by
token), and the pool postings of those tokens are unioned.

channel_score = number of the entity's rare tokens a candidate shares.
channel_rank orders by that, then by the rarest shared token's df, and only the top
config.RARE_TOKEN_TOP_K per entity are kept.
Source-1 rows are processed in chunks so the postings join stays bounded.

The ceiling is min(RARE_TOKEN_DF_MAX x N, RARE_TOKEN_DF_MAX_ABS). Memory: the pool's
tokens are never materialised as one frame. Token dedup is per record and df counts
distinct records, so both are computed per slice of config.RARE_TOKEN_SLICE_ROWS pool
records (pass 1: df, summed; pass 2: postings of in-range tokens only). Same output as
one pass, given unique pool entity_ids, which run() asserts.

run_to_file() is what s2's checkpointed path uses: each chunk's final rows go straight
into the part file instead of accumulating until one concat (full India climbed ~0.5 GB
a minute; France jumped +2.35 GB at the concat). Rows keep chunk order, which is entity
order, but the file is written in row groups; the union does not depend on row order.
"""
from pathlib import Path

import polars as pl

import config
from blocking.common import empty, rank_within_entity, resolve_df, stream_to_parquet

NAME = "rare_token"
# The only norm columns run() reads; s2_block loads just these. s1.height + pool.height
# (the df ceiling's corpus size) does not depend on which columns are loaded.
COLUMNS = ("entity_id", "name_tokens", "addr_tokens")


def _tokens(df: pl.DataFrame) -> pl.DataFrame:
    name = df.lazy().select("entity_id", pl.col("name_tokens").alias("tok"), pl.lit("n:").alias("ns"))
    addr = df.lazy().select("entity_id", pl.col("addr_tokens").alias("tok"), pl.lit("a:").alias("ns"))
    return (
        pl.concat([name, addr])
        .explode("tok", empty_as_null=False)
        .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
        .select("entity_id", (pl.col("ns") + pl.col("tok")).alias("tok"))
        .unique()
        .collect(engine="streaming")
    )


def _pool_slices(pool: pl.DataFrame):
    step = max(1, config.RARE_TOKEN_SLICE_ROWS)
    for off in range(0, pool.height, step):
        yield _tokens(pool.slice(off, step))


def _ranked_chunks(s1: pl.DataFrame, pool: pl.DataFrame):
    """Final rows, one chunk of RARE_TOKEN_CHUNK_ROWS Source-1 entities at a time."""
    df_max = resolve_df("RARE_TOKEN_DF_MAX", config.RARE_TOKEN_DF_MAX, s1.height + pool.height)
    if config.RARE_TOKEN_DF_MAX_ABS is not None and (df_max is None or df_max > config.RARE_TOKEN_DF_MAX_ABS):
        print(f"    RARE_TOKEN_DF_MAX_ABS caps it at {config.RARE_TOKEN_DF_MAX_ABS:,}")
        df_max = config.RARE_TOKEN_DF_MAX_ABS
    in_range = pl.col("df") >= config.RARE_TOKEN_DF_MIN
    if df_max is not None:
        in_range &= pl.col("df") <= df_max
    # Per-slice dedup and df counts only add up to the one-pass values if no record id repeats.
    assert pool["entity_id"].n_unique() == pool.height, "rare_token: duplicate pool entity_id"
    s1_tok = _tokens(s1)
    counts = s1_tok.group_by("tok").len(name="df")
    for t in _pool_slices(pool):  # pass 1: df = distinct records per token, summed over slices
        counts = pl.concat([counts, t.group_by("tok").len(name="df")]).group_by("tok").agg(pl.col("df").sum())
        del t
    df = counts.with_columns(pl.col("df").cast(pl.UInt32)).filter(in_range)
    del counts
    rarest = (
        s1_tok.join(df, on="tok", how="inner")
        .sort(["entity_id", "df", "tok"])
        .group_by("entity_id", maintain_order=True)
        .head(config.RARE_TOKENS_PER_ENTITY)
        .rename({"entity_id": "source1_entity_id"})
    )
    # pass 2: postings of in-range tokens only, slice by slice
    posts = [t.join(df.select("tok"), on="tok", how="semi") for t in _pool_slices(pool)]
    postings = (pl.concat(posts) if posts else _tokens(pool)).rename({"entity_id": "candidate_entity_id"})
    del posts

    ids = rarest["source1_entity_id"].unique(maintain_order=True)
    for start in range(0, len(ids), config.RARE_TOKEN_CHUNK_ROWS):
        chunk = rarest.filter(pl.col("source1_entity_id").is_in(ids.slice(start, config.RARE_TOKEN_CHUNK_ROWS).implode()))
        pairs = (
            chunk.join(postings, on="tok", how="inner")
            .group_by("source1_entity_id", "candidate_entity_id")
            .agg(pl.len().alias("shared"), pl.col("df").min().alias("min_df"))
        )
        # Entities never span chunks, so ranking and capping per chunk is exact.
        ranked = rank_within_entity(pairs, ["shared", "min_df"], [True, False], "shared")
        del pairs
        yield ranked.filter(pl.col("channel_rank") <= config.RARE_TOKEN_TOP_K)


def run(s1: pl.DataFrame, pool: pl.DataFrame, smoke: bool = False) -> pl.DataFrame:
    """In memory; the cap sweep and tests use this."""
    parts = list(_ranked_chunks(s1, pool))
    return pl.concat(parts) if parts else empty()


def run_to_file(s1: pl.DataFrame, pool: pl.DataFrame, path: Path, bit: int, smoke: bool = False) -> tuple[int, int]:
    """Streams run()'s rows, plus the channel's `bit` column, chunk by chunk into parquet at
    `path`. Returns (pairs, entities). Pairs are unique: each chunk's group_by makes them
    unique and entities never span chunks."""
    return stream_to_parquet(_ranked_chunks(s1, pool), path, bit)
