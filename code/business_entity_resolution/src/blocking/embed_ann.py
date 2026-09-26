"""Channel embed_ann: multilingual embedding ANN (Tanuj, Gate 3). PLACEHOLDER.

Same interface as the other channels. Until config.EMBED_ANN_PATH points at an
existing file it logs that embeddings are not yet available and returns zero
candidates. s2_block needs no special case for it.

When the embeddings land, implement _load() to return precomputed ANN pairs
(source1_entity_id, candidate_entity_id, score) and filter them to this shard.
"""
from pathlib import Path

import polars as pl

import config
from blocking.common import empty

NAME = "embed_ann"


def _available(in_dir) -> bool:
    return (in_dir / "embed_ann_pairs.parquet").exists()


def run(s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    # We infer the in_dir, split, and country from the input dataframes since this is
    # called per shard inside s2_block.py.
    # The true in_dir isn't passed directly to run(), but we can check ARTIFACTS_DIR/smoke.
    country = s1["country"][0]
    
    # We assume if s1 is small, it's a smoke run (or we can just check if smoke file exists)
    if (config.SMOKE_DIR / "embed_ann_pairs.parquet").exists() and s1.height <= 50000:
        in_dir = config.SMOKE_DIR
    else:
        in_dir = config.ARTIFACTS_DIR
        
    if not _available(in_dir):
        print(f"  embed_ann: embeddings not yet available (embed_ann_pairs.parquet), 0 candidates")
        return empty()
        
    # Read the precomputed file
    df = pl.scan_parquet(in_dir / "embed_ann_pairs.parquet")
    
    # Filter to this country (split is implicit since s1 only contains one split's entities)
    s1_ids = s1.select("entity_id").rename({"entity_id": "source1_entity_id"})
    
    # Join to keep only the pairs for the requested s1 entities
    # This automatically filters to the correct split and country shard
    res = df.join(s1_ids.lazy(), on="source1_entity_id", how="inner").select([
        "source1_entity_id", "candidate_entity_id", "channel_rank", "channel_score"
    ]).collect()
    
    return res
