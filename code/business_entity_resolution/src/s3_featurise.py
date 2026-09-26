"""S3 Featurise (owner: Krrish): norm + candidates_{split} → features_{split}.

Joins normalised Source-1 and Source-2/3 data onto candidate pairs, adds
context and competition columns, then calls features.featurise() to produce
the full feature matrix.

Output schema: source1_entity_id, candidate_entity_id, f000…fNNN float32,
[label uint8 on train splits only].

Sharding
--------
The test split is 1,732,544 Source-1 entities at config.MAX_CANDIDATES_PER_ENTITY
candidates each — about 52M pairs, which is far past what a materialised
(52M, 71) float32 matrix plus its Python-side working set would fit in. So this
stage never holds more than one chunk of features, following the same shape
blocking already uses (s2_block.load_shard, rare_token.run):

    for each country shard          # hard partition, as in s2_block
        load that country's norm rows and its slice of candidates
        compute the entity- and candidate-level aggregates ONCE per shard
        for each join block of config.S3_JOIN_BLOCK_ROWS pairs
            join the norm columns (and labels) onto the block
            for each chunk of config.S3_CHUNK_ROWS pairs
                featurise and append the chunk to the open ParquetWriter

Two nested sizes because they bound different things: the block bounds how
often the wide norm frames get hash-joined (once per block, not once per
chunk), the chunk bounds peak memory inside features.featurise().

Every group-level feature (entity_best_score, entity_n_cands, cand_best_score,
cand_n_claims) is aggregated over the whole country shard before any chunking,
so chunk boundaries cannot change a feature value. Sharding by country is exact
for the candidate-level aggregates because blocking only ever pairs records
within one country, so a candidate_entity_id appears in exactly one shard.

Usage:
    python s3_featurise.py [--smoke] [--input DIR] [--output DIR]
    python s3_featurise.py --smoke --chunk-rows 25000 --block-rows 500000
"""
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

import config
import pipeline_io as pio
from features import FEATURE_NAMES, FEATURE_VERSION, NUM_FEATURES, featurise

try:  # optional: not in requirements.txt, only used for the memory line
    import psutil

    _PROC = psutil.Process()
except Exception:  # pragma: no cover
    _PROC = None

# ═══════════════════════════════════════════════════════════════════════
# Data loading and joining
# ═══════════════════════════════════════════════════════════════════════

# Columns from the normalised schema that feed into features.
# entity_id is used for the join key, country is not needed in features.
_NORM_COLS = [
    "entity_id", "name_norm", "name_roman", "name_tokens", "name_acronym",
    "addr_norm", "addr_roman", "addr_tokens",
    "street_num", "city_norm", "state_canon", "postcode",
    "has_addr", "script", "name_suffix",
]


def _rss_mb() -> float:
    return _PROC.memory_info().rss / (1024 * 1024) if _PROC is not None else 0.0


def split_inputs(split: str, in_dir) -> dict[str, Path]:
    """Every file this stage reads for `split`, keyed by a human label."""
    need = {f"norm_{split}_{src}": config.norm_path(split, src, in_dir)
            for src in (config.SOURCE1_SRC, *config.CANDIDATE_SRCS)}
    need[f"candidates_{split}"] = config.candidates_path(split, in_dir)
    if split == "train":
        need["records_train_ground_truth"] = config.records_path("train", "ground_truth", in_dir)
    return need


def missing_inputs(split: str, in_dir) -> dict[str, Path]:
    return {k: p for k, p in split_inputs(split, in_dir).items() if not p.exists()}


def _load_norm(split: str, src: str, in_dir, country: str | None = None) -> pl.DataFrame:
    """Load normalised data for one country, selecting only the columns we need.

    Gracefully handles a missing name_suffix column (in case the normaliser
    hasn't been updated yet) by filling it with empty strings.
    """
    lf = pl.scan_parquet(config.norm_path(split, src, in_dir))
    have = set(lf.collect_schema().names())
    if "name_suffix" not in have:
        lf = lf.with_columns(pl.lit("").alias("name_suffix"))
        have.add("name_suffix")
    if country is not None and "country" in have:
        lf = lf.filter(pl.col("country") == country)
    return lf.select([c for c in _NORM_COLS if c in have]).collect()


def _load_pool(split: str, in_dir, country: str | None = None) -> pl.DataFrame:
    """Concatenate Source-2 and Source-3 normalised data for one country."""
    return pl.concat([_load_norm(split, s, in_dir, country) for s in config.CANDIDATE_SRCS])


def shard_countries(split: str, in_dir) -> list[str]:
    """Country shards, taken from the data like s2_block does (never hard-coded)."""
    lf = pl.scan_parquet(config.norm_path(split, config.SOURCE1_SRC, in_dir))
    if "country" not in lf.collect_schema().names():
        return [None]
    return lf.select(pl.col("country").unique().sort()).collect()["country"].to_list()


def _true_pairs(in_dir) -> pl.LazyFrame:
    """Ground-truth as long (source1_entity_id, candidate_entity_id, label=1)."""
    return (
        pl.scan_parquet(config.records_path("train", "ground_truth", in_dir))
        .select(
            "source1_entity_id",
            pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id"),
        )
        .explode("candidate_entity_id", empty_as_null=False)
        .filter(pl.col("candidate_entity_id") != "")
        .with_columns(pl.lit(1, dtype=pl.UInt8).alias("label"))
    )


def _prefix_cols(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Rename all columns (except entity_id) with a prefix."""
    return df.rename(
        {c: f"{prefix}_{c}" for c in df.columns if c != "entity_id"}
    )


def _add_context_and_competition(cands: pl.DataFrame) -> pl.DataFrame:
    """Add context features (entity-level) and competition features
    (candidate-level) as new columns on the candidates DataFrame.

    Context: entity_best_score, entity_n_cands
    Competition: cand_best_score, cand_n_claims
    Derived: is_source3

    Called once per country shard, before chunking, so every aggregate covers
    the entity's / candidate's complete set of rows.
    """
    # Entity-level context
    entity_ctx = cands.group_by("source1_entity_id").agg(
        pl.col("prior_score").max().alias("entity_best_score"),
        pl.len().cast(pl.Float32).alias("entity_n_cands"),
    )

    # Competition: for each candidate, its best score across all entities
    cand_comp = cands.group_by("candidate_entity_id").agg(
        pl.col("prior_score").max().alias("cand_best_score"),
        pl.len().cast(pl.Float32).alias("cand_n_claims"),
    )

    return (
        cands
        .join(entity_ctx, on="source1_entity_id", how="left")
        .join(cand_comp, on="candidate_entity_id", how="left")
        .with_columns(
            pl.col("candidate_entity_id").str.starts_with("S3-")
            .cast(pl.Float32).alias("is_source3"),
        )
    )


def _load_embed_scores(split: str, in_dir) -> pl.DataFrame | None:
    """Load precomputed embedding cosine scores if available."""
    embed_path = Path(in_dir) / "embed_ann_pairs.parquet"
    if not embed_path.exists():
        return None
    df = pl.read_parquet(embed_path)
    # Filter to the requested split if split column exists
    if "split" in df.columns:
        df = df.filter(pl.col("split") == split)
    return df.select(
        "source1_entity_id", "candidate_entity_id",
        pl.col("channel_score").alias("embed_cosine"),
        pl.col("channel_rank").cast(pl.Float32).alias("embed_rank"),
    )


def _join_block(
    block: pl.DataFrame,
    s1_norm: pl.DataFrame,
    pool_norm: pl.DataFrame,
    embed: pl.DataFrame | None,
    labels: pl.DataFrame | None,
) -> pl.DataFrame:
    """Attach both sides' normalised columns (+ embeddings, + labels) to a block
    of candidate pairs. One hash join per wide frame per block."""
    out = (
        block
        .join(s1_norm, left_on="source1_entity_id", right_on="entity_id", how="left")
        .join(pool_norm, left_on="candidate_entity_id", right_on="entity_id", how="left")
    )
    if embed is not None:
        out = out.join(embed, on=["source1_entity_id", "candidate_entity_id"], how="left")
    out = out.with_columns(
        pl.col("embed_cosine").fill_null(0.0).cast(pl.Float32) if "embed_cosine" in out.columns
        else pl.lit(0.0, dtype=pl.Float32).alias("embed_cosine"),
        pl.col("embed_rank").fill_null(0.0).cast(pl.Float32) if "embed_rank" in out.columns
        else pl.lit(0.0, dtype=pl.Float32).alias("embed_rank"),
    )
    if labels is not None:
        out = out.join(labels, on=["source1_entity_id", "candidate_entity_id"], how="left") \
                 .with_columns(pl.col("label").fill_null(0).cast(pl.UInt8))
    return out


def out_schema(split: str) -> dict:
    """The exact features_{split}.parquet schema, fixed before the first chunk so
    an empty leading shard cannot set a different one."""
    schema = {"source1_entity_id": pl.String, "candidate_entity_id": pl.String}
    schema |= {c: pl.Float32 for c in pio.feature_columns(NUM_FEATURES)}
    if split == "train":
        schema["label"] = pl.UInt8
    return schema


def featurise_chunk(pairs: pl.DataFrame, split: str, fcols: list[str]) -> pl.DataFrame:
    """One chunk of joined pairs → one chunk of the output schema."""
    arr = featurise(pairs)
    cols = {
        "source1_entity_id": pairs["source1_entity_id"],
        "candidate_entity_id": pairs["candidate_entity_id"],
    }
    cols |= {c: arr[:, i] for i, c in enumerate(fcols)}
    if split == "train":
        cols["label"] = pairs["label"]
    return pl.DataFrame(cols, schema=out_schema(split))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def featurise_split(
    split: str,
    in_dir,
    out_dir,
    chunk_rows: int = None,
    block_rows: int = None,
    verbose: bool = True,
) -> dict:
    """Shard `split` by country, chunk within country, stream to features_{split}.

    Returns {rows, positives, sec, peak_rss_mb}.
    """
    chunk_rows = chunk_rows or config.S3_CHUNK_ROWS
    block_rows = max(block_rows or config.S3_JOIN_BLOCK_ROWS, chunk_rows)
    fcols = pio.feature_columns(NUM_FEATURES)
    schema = out_schema(split)
    out_path = config.features_path(split, out_dir)

    embed_all = _load_embed_scores(split, in_dir)
    labels_all = _true_pairs(in_dir).collect() if split == "train" else None
    if verbose:
        print(f"  embed_ann: {'joined from embed_ann_pairs.parquet' if embed_all is not None else 'no precomputed scores found, embed_cosine/rank = 0'}")

    t0 = time.perf_counter()
    rows = positives = 0
    peak = _rss_mb()
    writer = None
    try:
        for country in shard_countries(split, in_dir):
            ts = time.perf_counter()
            s1_norm = _prefix_cols(_load_norm(split, config.SOURCE1_SRC, in_dir, country), "s1")
            pool_norm = _prefix_cols(_load_pool(split, in_dir, country), "cand")
            s1_ids = s1_norm["entity_id"]

            cands = (
                pl.scan_parquet(config.candidates_path(split, in_dir))
                .filter(pl.col("source1_entity_id").is_in(s1_ids.implode()))
                .collect(engine="streaming")
            )
            if cands.is_empty():
                if verbose:
                    print(f"  [{split}/{country}] 0 candidate pairs — skipped")
                continue
            cands = _add_context_and_competition(cands)

            # Restrict the two pair-keyed side tables to this shard, so the
            # per-block joins never probe the other countries' rows.
            embed = (
                embed_all.filter(pl.col("source1_entity_id").is_in(s1_ids.implode()))
                if embed_all is not None else None
            )
            labels = (
                labels_all.filter(pl.col("source1_entity_id").is_in(s1_ids.implode()))
                if labels_all is not None else None
            )

            shard_rows = shard_pos = 0
            for block in cands.iter_slices(block_rows):
                joined = _join_block(block, s1_norm, pool_norm, embed, labels)
                for chunk in joined.iter_slices(chunk_rows):
                    out = featurise_chunk(chunk, split, fcols)
                    pio.check_schema(out, schema, out_path.name)
                    table = out.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema)
                    writer.write_table(table)
                    shard_rows += out.height
                    if split == "train":
                        shard_pos += int(out["label"].sum())
                    peak = max(peak, _rss_mb())
                    del out, table
                del joined
            del cands, s1_norm, pool_norm, embed, labels

            rows += shard_rows
            positives += shard_pos
            if verbose:
                sec = time.perf_counter() - ts
                print(f"  [{split}/{country}] {shard_rows:>12,} pairs  {sec:7.1f}s  "
                      f"{shard_rows / max(sec, 1e-9):>9,.0f} rows/s"
                      + (f"  {shard_pos:>9,} positives" if split == "train" else ""))

        if writer is None:  # no shard produced a row: still emit a schema-correct file
            pl.DataFrame(schema=schema).write_parquet(out_path)
    finally:
        if writer is not None:
            writer.close()

    return {"rows": rows, "positives": positives, "sec": time.perf_counter() - t0, "peak_rss_mb": peak}


def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    ap.add_argument("--chunk-rows", type=int, default=config.S3_CHUNK_ROWS,
                    help=f"pairs per featurise chunk (default {config.S3_CHUNK_ROWS:,})")
    ap.add_argument("--block-rows", type=int, default=config.S3_JOIN_BLOCK_ROWS,
                    help=f"pairs per norm-join block (default {config.S3_JOIN_BLOCK_ROWS:,})")
    ap.add_argument("--splits", nargs="+", default=list(config.SPLITS),
                    help="only featurise these splits")
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    print(f"Feature spec: {NUM_FEATURES} features, version {FEATURE_VERSION}")
    print(f"  Names: {', '.join(FEATURE_NAMES[:5])} … {', '.join(FEATURE_NAMES[-3:])}")
    print(f"  Shards: by country; blocks of {args.block_rows:,} pairs, chunks of {args.chunk_rows:,}")

    ran = []
    for split in args.splits:
        missing = missing_inputs(split, in_dir)
        if missing:
            names = "\n".join(f"    {k:<28} {p}" for k, p in missing.items())
            if len(missing) == len(split_inputs(split, in_dir)):
                # Nothing for this split exists at all — the normal state for test
                # before s1/s2 have been run on it. Say so and move on.
                print(f"\n[{split}] SKIPPED — none of this split's inputs exist in {in_dir}:\n{names}")
                print(f"    Run: python s0_ingest.py && python s1_normalise.py && python s2_block.py"
                      f"{' --smoke' if args.smoke else ''}")
                continue
            raise SystemExit(
                f"\ns3_featurise: {split} is partially prepared — {len(missing)} of "
                f"{len(split_inputs(split, in_dir))} input files are missing:\n{names}\n"
                f"  Re-run the stages that produce them (s0_ingest -> s1_normalise -> s2_block"
                f"{' --smoke' if args.smoke else ''}) before s3.\n"
                f"  Refusing to write a partial features_{split}.parquet."
            )

        print(f"\n[{split}]")
        st = featurise_split(split, in_dir, out_dir, args.chunk_rows, args.block_rows)
        ran.append(split)
        out = config.features_path(split, out_dir)
        pos = f", {st['positives']:,} positives" if split == "train" else ""
        print(f"  {out.name:<24} {st['rows']:>12,} rows × {NUM_FEATURES} features{pos}")
        per_m = st["rows"] / 1e6
        print(f"  {st['sec']:.1f}s  {st['rows'] / max(st['sec'], 1e-9):,.0f} rows/s"
              + (f"  peak RSS {st['peak_rss_mb']:,.0f} MB"
                 f"  ({st['peak_rss_mb'] / per_m:,.0f} MB per 1M pairs)" if _PROC and per_m else ""))

    if not ran:
        raise SystemExit(
            f"s3_featurise: no split had usable inputs in {in_dir}. Nothing was written."
        )
    print(f"\ns3 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
