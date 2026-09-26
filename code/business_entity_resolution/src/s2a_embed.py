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
CHUNK = 250_000       # texts per resumable embedding chunk
POOL_CHUNK = 400_000  # pool rows streamed through the GPU per search step
QUERY_BATCH = 2048

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

def embed_in_batches(model, texts: list[str], checkpoint_prefix: str) -> np.ndarray:
    """L2-normalised fp16 embeddings, computed in resumable chunks (one .npy per chunk)."""
    parts = []
    for ci, start in enumerate(range(0, len(texts), CHUNK)):
        path = Path(f"{checkpoint_prefix}_{ci:04d}.npy")
        if path.exists():
            parts.append(np.load(path))
            continue
        emb = model.encode(texts[start:start + CHUNK], batch_size=BATCH_SIZE, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        np.save(path, emb)
        parts.append(emb)
        print(f"      [chunk {ci}] {min(start + CHUNK, len(texts)):,}/{len(texts):,}", flush=True)
    return np.concatenate(parts)


def gpu_topk(q_name, q_addr, p_name, p_addr, k: int, device: str):
    """Exact top-k by (cos_name + cos_addr) / 2, the same ranking as cosine on the normalised concat.
    Pool is streamed through the GPU in POOL_CHUNK slices; only (n_queries x k) state is kept."""
    qn, qa = torch.from_numpy(q_name).to(device), torch.from_numpy(q_addr).to(device)
    n_q = qn.shape[0]
    best_s = torch.full((n_q, k), -1e9, dtype=torch.float32, device=device)
    best_i = torch.zeros((n_q, k), dtype=torch.int64, device=device)
    for ps in range(0, p_name.shape[0], POOL_CHUNK):
        pn = torch.from_numpy(p_name[ps:ps + POOL_CHUNK]).to(device)
        pa = torch.from_numpy(p_addr[ps:ps + POOL_CHUNK]).to(device)
        kk = min(k, pn.shape[0])
        for qs in range(0, n_q, QUERY_BATCH):
            sim = (qn[qs:qs + QUERY_BATCH] @ pn.T + qa[qs:qs + QUERY_BATCH] @ pa.T).float() / 2
            s, i = sim.topk(kk, dim=1)
            cs = torch.cat([best_s[qs:qs + QUERY_BATCH], s], dim=1)
            ci = torch.cat([best_i[qs:qs + QUERY_BATCH], i + ps], dim=1)
            top_s, pos = cs.topk(k, dim=1)
            best_s[qs:qs + QUERY_BATCH] = top_s
            best_i[qs:qs + QUERY_BATCH] = ci.gather(1, pos)
        del pn, pa
    return best_s.cpu().numpy(), best_i.cpu().numpy()


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

    # Names and addresses stay separate (fp16, normalised); the search sums the two cosines.
    print("  Embedding Pool (Names)...", flush=True)
    p_name = embed_in_batches(model, pool_df["name_text"].to_list(), f"{chkpt_base}_pool_name")
    print("  Embedding Pool (Addresses)...", flush=True)
    p_addr = embed_in_batches(model, pool_df["addr_text"].to_list(), f"{chkpt_base}_pool_addr")
    print("  Embedding Source 1 (Names)...", flush=True)
    q_name = embed_in_batches(model, s1_df["name_text"].to_list(), f"{chkpt_base}_s1_name")
    print("  Embedding Source 1 (Addresses)...", flush=True)
    q_addr = embed_in_batches(model, s1_df["addr_text"].to_list(), f"{chkpt_base}_s1_addr")

    print("  Searching (GPU, chunked exact top-k)...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores, indices = gpu_topk(q_name, q_addr, p_name, p_addr, config.EMBED_TOP_K, device)

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
