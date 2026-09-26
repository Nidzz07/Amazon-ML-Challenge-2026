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
from sentence_transformers import SentenceTransformer

import config
import pipeline_io as pio

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 512
MAX_SEQ_LEN = 64      # names/addresses are short; 512 (the default) wastes T4 time
CHUNK = 250_000       # texts per resumable embedding chunk (also the pool slice searched per step)
QUERY_BATCH = 2048
EMBED_DIM = 768

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

def embed_texts(model, texts: list[str], checkpoint_prefix: str) -> np.ndarray:
    """L2-normalised fp16 embeddings, resumable in CHUNK-sized .npy files; preallocated so peak RAM = result size."""
    out = np.empty((len(texts), EMBED_DIM), dtype=np.float16)
    for ci, start in enumerate(range(0, len(texts), CHUNK)):
        out[start:start + CHUNK] = embed_chunk(model, texts[start:start + CHUNK], f"{checkpoint_prefix}_{ci:04d}.npy")
    return out


def embed_chunk(model, texts: list[str], path: str) -> np.ndarray:
    if Path(path).exists():
        return np.load(path)
    emb = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    np.save(path, emb)
    print(f"      [saved] {Path(path).name}", flush=True)
    return emb


class TopK:
    """Exact top-k by (cos_name + cos_addr) / 2 -- the same ranking as cosine on the normalised concat.
    Queries live on the GPU; pool slices are fed in one at a time, so the pool never sits in RAM."""

    def __init__(self, q_name: np.ndarray, q_addr: np.ndarray, k: int, device: str):
        self.k, self.device = k, device
        self.qn, self.qa = torch.from_numpy(q_name).to(device), torch.from_numpy(q_addr).to(device)
        n_q = self.qn.shape[0]
        self.best_s = torch.full((n_q, k), -1e9, dtype=torch.float32, device=device)
        self.best_i = torch.zeros((n_q, k), dtype=torch.int64, device=device)

    def update(self, p_name: np.ndarray, p_addr: np.ndarray, offset: int) -> None:
        pn = torch.from_numpy(p_name).to(self.device)
        pa = torch.from_numpy(p_addr).to(self.device)
        kk = min(self.k, pn.shape[0])
        for qs in range(0, self.qn.shape[0], QUERY_BATCH):
            sim = (self.qn[qs:qs + QUERY_BATCH] @ pn.T + self.qa[qs:qs + QUERY_BATCH] @ pa.T).float() / 2
            s, i = sim.topk(kk, dim=1)
            cs = torch.cat([self.best_s[qs:qs + QUERY_BATCH], s], dim=1)
            ci = torch.cat([self.best_i[qs:qs + QUERY_BATCH], i + offset], dim=1)
            top_s, pos = cs.topk(self.k, dim=1)
            self.best_s[qs:qs + QUERY_BATCH] = top_s
            self.best_i[qs:qs + QUERY_BATCH] = ci.gather(1, pos)

    def result(self):
        return self.best_s.cpu().numpy(), self.best_i.cpu().numpy()


def process_shard(split: str, country: str, in_dir: Path, out_dir: Path, model, base_out_path: Path):
    print(f"\n--- Shard: {split} / {country} ---")
    t0 = time.perf_counter()

    s1_rec = pl.scan_parquet(config.records_path(split, config.SOURCE1_SRC, in_dir)).filter(pl.col("country") == country)
    s1_norm = pl.scan_parquet(config.norm_path(split, config.SOURCE1_SRC, in_dir)).filter(pl.col("country") == country)
    s1_df = _get_texts(s1_rec, s1_norm)

    pool_rec = pl.concat([pl.scan_parquet(config.records_path(split, s, in_dir)) for s in config.CANDIDATE_SRCS]).filter(pl.col("country") == country)
    pool_norm = pl.concat([pl.scan_parquet(config.norm_path(split, s, in_dir)) for s in config.CANDIDATE_SRCS]).filter(pl.col("country") == country)
    pool_df = _get_texts(pool_rec, pool_norm)

    print(f"  Source1: {s1_df.height:,} | Pool: {pool_df.height:,}")
    if s1_df.height == 0 or pool_df.height == 0:
        return None

    chkpt_base = str(base_out_path.parent / f"embed_{split}_{country.replace(' ', '_')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Source 1 first (kept on the GPU), then the pool streams through chunk by chunk.
    print("  Embedding Source 1 (Names)...", flush=True)
    q_name = embed_texts(model, s1_df["name_text"].to_list(), f"{chkpt_base}_s1_name")
    print("  Embedding Source 1 (Addresses)...", flush=True)
    q_addr = embed_texts(model, s1_df["addr_text"].to_list(), f"{chkpt_base}_s1_addr")
    topk = TopK(q_name, q_addr, config.EMBED_TOP_K, device)
    del q_name, q_addr
    gc.collect()

    print("  Embedding Pool + searching (chunked)...", flush=True)
    n_pool = pool_df.height
    for ci, start in enumerate(range(0, n_pool, CHUNK)):
        names = pool_df["name_text"].slice(start, CHUNK).to_list()
        addrs = pool_df["addr_text"].slice(start, CHUNK).to_list()
        pn = embed_chunk(model, names, f"{chkpt_base}_pool_name_{ci:04d}.npy")
        pa = embed_chunk(model, addrs, f"{chkpt_base}_pool_addr_{ci:04d}.npy")
        topk.update(pn, pa, start)
        print(f"      pool {min(start + CHUNK, n_pool):,}/{n_pool:,}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    scores, indices = topk.result()

    s1_ids = s1_df["entity_id"].to_numpy()
    pool_ids = pool_df["entity_id"].to_numpy()

    n_queries, k = indices.shape
    out_df = pl.DataFrame({
        "source1_entity_id": np.repeat(s1_ids, k),
        "candidate_entity_id": pool_ids[indices.flatten()],
        "channel_rank": np.tile(np.arange(1, k + 1), n_queries).astype(np.uint16),
        "channel_score": scores.flatten().astype(np.float32),
    })

    for f in Path(chkpt_base).parent.glob(f"{Path(chkpt_base).name}_*.npy"):
        f.unlink()  # shard finished; free the disk
    print(f"  Shard done in {time.perf_counter() - t0:.1f}s")
    return out_df

def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    if device == "cuda":
        model.half()

    out_path = config.EMBED_ANN_PATH
    if out_path is None or args.smoke:
        out_path = out_dir / "embed_ann_pairs.parquet"
    else:
        out_path = Path(out_path)

    # Per-shard parquets go in a subdirectory next to the final output.
    # If the final output already exists, we're done.
    if out_path.exists():
        print(f"\nFinal output already exists at {out_path}. Nothing to do.")
        return

    shard_dir = out_path.parent / "embed_ann_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = []

    for split in config.SPLITS:
        if not config.norm_path(split, config.SOURCE1_SRC, in_dir).exists():
            continue

        countries = (
            pl.scan_parquet(config.records_path(split, config.SOURCE1_SRC, in_dir))
            .select(pl.col("country").unique())
            .collect()["country"]
            .to_list()
        )
        for country in sorted(countries):
            shard_file = shard_dir / f"embed_{split}_{country.replace(' ', '_')}.parquet"
            shard_paths.append(shard_file)

            if shard_file.exists():
                rows = pl.scan_parquet(shard_file).select(pl.len()).collect().item()
                print(f"[{split}/{country}] Already done ({rows:,} pairs) — skipping.")
                continue

            df = process_shard(split, country, in_dir, out_dir, model, out_path)
            if df is not None:
                df = df.with_columns(pl.lit(split).alias("split"), pl.lit(country).alias("country"))
                df.write_parquet(shard_file)
                print(f"  -> Wrote {df.height:,} pairs to {shard_file.name}")

    # Merge all completed shards into the final parquet.
    completed = [p for p in shard_paths if p.exists()]
    if completed:
        print(f"\nMerging {len(completed)} shard(s) into {out_path} ...")
        final_df = pl.concat([pl.read_parquet(p) for p in completed])
        final_df.write_parquet(out_path)
        print(f"Saved {final_df.height:,} total pairs to {out_path}")
        # Clean up shard files to save disk space.
        for p in completed:
            p.unlink()
        try:
            shard_dir.rmdir()
        except OSError:
            pass
    else:
        print("\nNo data processed.")


if __name__ == "__main__":
    sys.exit(main())
