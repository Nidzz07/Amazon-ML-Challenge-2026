"""Channel embed_ann: multilingual embedding ANN pairs precomputed by s2a_embed.py (Tanuj).

Reads config.embed_ann_path(smoke): artifacts/smoke/embed_ann_pairs.parquet on smoke
runs, config.EMBED_ANN_PATH (default artifacts/embed_ann_pairs.parquet) otherwise.
The smoke flag comes from the caller, never from the shard's size. While the file is
missing it logs that and returns zero candidates, so the pipeline runs without it.

The file holds every split and country shard, with channel_rank / channel_score
already set. A pair is kept only if its Source-1 id is in this shard's s1 AND its
candidate is in this shard's pool, and only ranks up to config.EMBED_TOP_K. The pool
filter means a pairs file built from a different artifact set can never emit
candidates the pool does not contain (s3 would turn those into all-null rows).
"""
import polars as pl

import config
from blocking.common import CHANNEL_SCHEMA, empty

NAME = "embed_ann"


def run(s1: pl.DataFrame, pool: pl.DataFrame, smoke: bool = False) -> pl.DataFrame:
    if s1.is_empty() or pool.is_empty():
        return empty()
    path = config.embed_ann_path(smoke)
    if not path.exists():
        print(f"    embed_ann: {path} not found, 0 candidates")
        return empty()
    pairs = (
        pl.scan_parquet(path)
        .filter(pl.col("channel_rank") <= config.EMBED_TOP_K)
        .join(s1.lazy().select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id", how="semi")
        .select(list(CHANNEL_SCHEMA))
        .collect()
        .cast(CHANNEL_SCHEMA)
    )
    out = pairs.join(pool.select(pl.col("entity_id").alias("candidate_entity_id")), on="candidate_entity_id", how="semi")
    if out.height < pairs.height:
        print(f"    embed_ann: dropped {pairs.height - out.height:,} of {pairs.height:,} pairs whose candidate "
              f"is not in this shard's pool (is {path} from a different artifact set?)")
    return out
