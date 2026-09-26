"""S2A Embed (owner: Tanuj): Compute multilingual embeddings and ANN pairs.

Generates dense embeddings using intfloat/multilingual-e5-base.
Embeds (business_name + name_roman) and (business_address + addr_roman) separately,
concatenates them, and runs a FAISS inner-product search per country shard.

Writes precomputed pairs to config.EMBED_ANN_PATH.
"""
import argparse
import sys
import time
import gc
from pathlib import Path
import os

import numpy as np
import polars as pl
import torch
import faiss
from sentence_transformers import SentenceTransformer

import config
import pipeline_io as pio

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 256
CHECKPOINT_EVERY = 500_000

def _get_texts(records_lf: pl.LazyFrame, norm_lf: pl.LazyFrame) -> pl.DataFrame:
    # Join records (for raw text) and norm (for romanized text)
    r = records_lf.select("entity_id", "country", "business_name", "business_address")
    n = norm_lf.select("entity_id", "name_roman", "addr_roman")
    df = r.join(n, on="entity_id", how="left").collect()
    
    # E5 expects "query: " prefix for asymmetric tasks, but here symmetric matching is fine,
    # however E5 dictates "query: " for queries and "passage: " for documents. 
    # For entity resolution, we can just use "query: " for both sides.
    def clean(s): return pl.col(s).fill_null("").str.strip_chars()
    
    # Combine raw + roman
    df = df.with_columns([
        pl.format("query: {} {}", clean("business_name"), clean("name_roman")).alias("name_text"),
        pl.format("query: {} {}", clean("business_address"), clean("addr_roman")).alias("addr_text")
    ])
    return df

def embed_in_batches(model, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # normalize_embeddings=False because we will normalize the concatenated vector later
        emb = model.encode(batch, batch_size=batch_size, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)

def process_shard(split: str, country: str, in_dir: Path, out_dir: Path, model):
    print(f"\n--- Shard: {split} / {country} ---")
    t0 = time.perf_counter()
    
    # Load Source 1
    s1_rec = pl.scan_parquet(config.records_path(split, config.SOURCE1_SRC, in_dir)).filter(pl.col("country") == country)
    s1_norm = pl.scan_parquet(config.norm_path(split, config.SOURCE1_SRC, in_dir)).filter(pl.col("country") == country)
    s1_df = _get_texts(s1_rec, s1_norm)
    
    # Load Pool (Source 2 + Source 3)
    pool_rec = pl.concat([pl.scan_parquet(config.records_path(split, s, in_dir)) for s in config.CANDIDATE_SRCS]).filter(pl.col("country") == country)
    pool_norm = pl.concat([pl.scan_parquet(config.norm_path(split, s, in_dir)) for s in config.CANDIDATE_SRCS]).filter(pl.col("country") == country)
    pool_df = _get_texts(pool_rec, pool_norm)
    
    print(f"  Source1: {s1_df.height:,} | Pool: {pool_df.height:,}")
    if s1_df.height == 0 or pool_df.height == 0:
        return None
        
    # Embed Pool (checkpointing logic would go here if saving vectors to disk, but we do it in memory for now)
    print("  Embedding Pool (Names)...")
    pool_name_emb = embed_in_batches(model, pool_df["name_text"].to_list())
    print("  Embedding Pool (Addresses)...")
    pool_addr_emb = embed_in_batches(model, pool_df["addr_text"].to_list())
    
    # Concatenate and normalize
    pool_emb = np.hstack([pool_name_emb, pool_addr_emb])
    faiss.normalize_L2(pool_emb)
    
    # Build FAISS Index
    print("  Building FAISS index...")
    dim = pool_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(pool_emb)
    
    # Free memory
    del pool_name_emb, pool_addr_emb, pool_emb
    gc.collect()
    
    # Embed Source 1
    print("  Embedding Source 1 (Names)...")
    s1_name_emb = embed_in_batches(model, s1_df["name_text"].to_list())
    print("  Embedding Source 1 (Addresses)...")
    s1_addr_emb = embed_in_batches(model, s1_df["addr_text"].to_list())
    
    s1_emb = np.hstack([s1_name_emb, s1_addr_emb])
    faiss.normalize_L2(s1_emb)
    
    # Search
    print("  Searching FAISS...")
    scores, indices = index.search(s1_emb, config.EMBED_TOP_K)
    
    # Construct output DataFrame
    s1_ids = s1_df["entity_id"].to_numpy()
    pool_ids = pool_df["entity_id"].to_numpy()
    
    n_queries, k = indices.shape
    source1_col = np.repeat(s1_ids, k)
    candidate_col = pool_ids[indices.flatten()]
    score_col = scores.flatten()
    rank_col = np.tile(np.arange(1, k + 1), n_queries)
    
    out_df = pl.DataFrame({
        "source1_entity_id": source1_col,
        "candidate_entity_id": candidate_col,
        "channel_rank": rank_col.astype(np.uint16),
        "channel_score": score_col.astype(np.float32)
    })
    
    print(f"  Shard done in {time.perf_counter() - t0:.1f}s")
    return out_df

def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    out_path = config.EMBED_ANN_PATH
    if out_path is None:
        out_path = out_dir / "embed_ann_pairs.parquet"
    else:
        out_path = Path(out_path)
    
    all_pairs = []
    
    for split in config.SPLITS:
        if not config.norm_path(split, config.SOURCE1_SRC, in_dir).exists():
            continue
            
        countries = pl.scan_parquet(config.records_path(split, config.SOURCE1_SRC, in_dir)).select(pl.col("country").unique()).collect()["country"].to_list()
        for country in countries:
            df = process_shard(split, country, in_dir, out_dir, model)
            if df is not None:
                all_pairs.append(df.with_columns(pl.lit(split).alias("split"), pl.lit(country).alias("country")))

    if all_pairs:
        final_df = pl.concat(all_pairs)
        final_df.write_parquet(out_path)
        print(f"\nSaved {final_df.height:,} total pairs to {out_path}")
    else:
        print("\nNo data processed.")

if __name__ == "__main__":
    sys.exit(main())
