"""S3 Featurise (owner: Krrish): norm + candidates_{split} → features_{split}.

Joins normalised Source-1 and Source-2/3 data onto candidate pairs, adds
context and competition columns, then calls features.featurise() to produce
the full feature matrix.

Output schema: source1_entity_id, candidate_entity_id, f000…fNNN float32,
[label uint8 on train splits only].

Usage:
    python s3_featurise.py [--smoke] [--input DIR] [--output DIR]
"""
import sys
import time

import numpy as np
import polars as pl

import config
import pipeline_io as pio
from features import FEATURE_NAMES, FEATURE_VERSION, NUM_FEATURES, featurise

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


def _load_norm(split: str, src: str, in_dir) -> pl.DataFrame:
    """Load normalised data, selecting only the columns we need.

    Gracefully handles a missing name_suffix column (in case the normaliser
    hasn't been updated yet) by filling it with empty strings.
    """
    path = config.norm_path(split, src, in_dir)
    df = pl.read_parquet(path)
    if "name_suffix" not in df.columns:
        df = df.with_columns(pl.lit("").alias("name_suffix"))
    available = [c for c in _NORM_COLS if c in df.columns]
    return df.select(available)


def _load_pool(split: str, in_dir) -> pl.DataFrame:
    """Concatenate Source-2 and Source-3 normalised data."""
    parts = [_load_norm(split, s, in_dir) for s in config.CANDIDATE_SRCS]
    return pl.concat(parts)


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
    embed_path = in_dir / "embed_ann_pairs.parquet"
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


def build_pairs(split: str, in_dir) -> pl.DataFrame:
    """Join candidates with normalised data from both sides and add
    context/competition columns.  Returns the DataFrame ready for
    features.featurise().
    """
    # Load data
    s1_norm = _prefix_cols(_load_norm(split, config.SOURCE1_SRC, in_dir), "s1")
    pool_norm = _prefix_cols(_load_pool(split, in_dir), "cand")
    cands = pl.read_parquet(config.candidates_path(split, in_dir))

    # Add context and competition features
    cands = _add_context_and_competition(cands)

    # Join Source-1 normalised data
    pairs = cands.join(s1_norm, left_on="source1_entity_id", right_on="entity_id", how="left")

    # Join candidate normalised data
    pairs = pairs.join(pool_norm, left_on="candidate_entity_id", right_on="entity_id", how="left")

    # Join embedding cosine scores (Tanuj B5) — zero-filled if not yet available
    embed_scores = _load_embed_scores(split, in_dir)
    if embed_scores is not None:
        pairs = pairs.join(embed_scores, on=["source1_entity_id", "candidate_entity_id"], how="left")
        pairs = pairs.with_columns([
            pl.col("embed_cosine").fill_null(0.0),
            pl.col("embed_rank").fill_null(0.0),
        ])
        print(f"  embed_ann scores joined: {pairs['embed_cosine'].gt(0).sum():,}/{len(pairs):,} pairs have embeddings")
    else:
        pairs = pairs.with_columns([
            pl.lit(0.0, dtype=pl.Float32).alias("embed_cosine"),
            pl.lit(0.0, dtype=pl.Float32).alias("embed_rank"),
        ])
        print("  embed_ann: no precomputed scores found, embed_cosine/rank = 0")

    return pairs


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def compute_features(split: str, in_dir) -> pl.DataFrame:
    """Build the feature matrix for a split.

    Returns a DataFrame with: source1_entity_id, candidate_entity_id,
    f000…fNNN float32, [label uint8 on train].
    """
    pairs = build_pairs(split, in_dir)
    n = pairs.height

    print(f"  Computing {NUM_FEATURES} features for {n:,} pairs …")
    t = time.perf_counter()
    feat_arr = featurise(pairs)
    print(f"  Features computed in {time.perf_counter() - t:.1f}s")

    # Build output DataFrame
    fcols = pio.feature_columns(NUM_FEATURES)
    feat_df = pl.DataFrame(
        {c: feat_arr[:, i] for i, c in enumerate(fcols)},
        schema={c: pl.Float32 for c in fcols},
    )
    out = pl.concat(
        [pairs.select("source1_entity_id", "candidate_entity_id"), feat_df],
        how="horizontal",
    )

    if split == "train":
        labels = _true_pairs(in_dir).collect()
        out = out.join(
            labels, on=["source1_entity_id", "candidate_entity_id"], how="left"
        ).with_columns(pl.col("label").fill_null(0))

    return out


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    print(f"Feature spec: {NUM_FEATURES} features, version {FEATURE_VERSION}")
    print(f"  Names: {', '.join(FEATURE_NAMES[:5])} … {', '.join(FEATURE_NAMES[-3:])}")

    for split in config.SPLITS:
        print(f"\n[{split}]")
        feats = compute_features(split, in_dir)
        out = config.features_path(split, out_dir)
        feats.write_parquet(out)
        n_feat = feats.width - 2 - ("label" in feats.columns)
        pos = f", {int(feats['label'].sum()):,} positives" if "label" in feats.columns else ""
        print(f"  {out.name:<24} {feats.height:>10,} rows × {n_feat} features{pos}")

    print(f"\ns3 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
