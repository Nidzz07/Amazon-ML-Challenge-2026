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
"""
import polars as pl

import config
from blocking.common import empty, rank_within_entity, resolve_df

NAME = "rare_token"


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


def run(s1: pl.DataFrame, pool: pl.DataFrame, smoke: bool = False) -> pl.DataFrame:
    df_max = resolve_df("RARE_TOKEN_DF_MAX", config.RARE_TOKEN_DF_MAX, s1.height + pool.height)
    in_range = pl.col("df") >= config.RARE_TOKEN_DF_MIN
    if df_max is not None:
        in_range &= pl.col("df") <= df_max
    s1_tok, pool_tok = _tokens(s1), _tokens(pool)
    df = (
        pl.concat([s1_tok.lazy(), pool_tok.lazy()])
        .group_by("tok")
        .len(name="df")
        .filter(in_range)
        .collect(engine="streaming")
    )
    rarest = (
        s1_tok.join(df, on="tok", how="inner")
        .sort(["entity_id", "df", "tok"])
        .group_by("entity_id", maintain_order=True)
        .head(config.RARE_TOKENS_PER_ENTITY)
        .rename({"entity_id": "source1_entity_id"})
    )
    postings = pool_tok.join(df.select("tok"), on="tok", how="semi").rename({"entity_id": "candidate_entity_id"})

    ids = rarest["source1_entity_id"].unique(maintain_order=True)
    parts = []
    for start in range(0, len(ids), config.RARE_TOKEN_CHUNK_ROWS):
        chunk = rarest.filter(pl.col("source1_entity_id").is_in(ids.slice(start, config.RARE_TOKEN_CHUNK_ROWS).implode()))
        pairs = (
            chunk.join(postings, on="tok", how="inner")
            .group_by("source1_entity_id", "candidate_entity_id")
            .agg(pl.len().alias("shared"), pl.col("df").min().alias("min_df"))
        )
        # Entities never span chunks, so ranking and capping per chunk is exact.
        ranked = rank_within_entity(pairs, ["shared", "min_df"], [True, False], "shared")
        parts.append(ranked.filter(pl.col("channel_rank") <= config.RARE_TOKEN_TOP_K))
    return pl.concat(parts) if parts else empty()
