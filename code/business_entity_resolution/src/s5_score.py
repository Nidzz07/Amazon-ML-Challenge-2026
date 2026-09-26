"""S5 Score (owner: Tanuj): model + features_{split} -> scored_{split}.

Loads the LightGBM model and isotonic calibrator. Refuses to run if the
model's FEATURE_VERSION doesn't match the current features.py. Scores in
~5M-pair shards so the full inference feature matrix is never materialised
on disk (it would be ~25 GB). Each scored shard is appended straight to
scored_{split}.parquet (written to a .tmp file, renamed when complete), so memory
is one shard however many pairs there are. Writes only (source1_entity_id,
candidate_entity_id, prob float32).

Usage:
    python s5_score.py [--smoke] [--input DIR] [--output DIR]
    python s5_score.py --splits test                 # only what is asked; default is every split with features
    python s5_score.py --splits train --val-only     # only the held-out validation entities (what s7 evaluates)
"""
import json
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import config
import pipeline_io as pio
from s4_train import ID, iter_entity_batches

SHARD_ROWS = 1_000_000   # pairs per in-memory shard: peak RAM ~2.7 GB flat (5M-row shards peaked at 9 GB)


def load_model_and_calibrator(in_dir):
    """Load model + calibrator and verify FEATURE_VERSION before anything else."""
    meta_path = config.model_path(in_dir).with_suffix(".meta")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {meta_path}. "
            "Run s4_train.py to regenerate both model.txt and model.meta."
        )

    meta = json.loads(meta_path.read_text(encoding=config.ENCODING))
    _, current_version = pio.feature_spec()

    if meta["feature_version"] != current_version:
        raise SystemExit(
            f"FEATURE_VERSION MISMATCH — model trained on v{meta['feature_version']}, "
            f"current features.py is v{current_version}. Re-run s4_train.py first."
        )
    if meta["feature_names"] != list(pio.feature_spec()[0]):
        raise SystemExit(
            "Feature name list has changed since training. Re-run s4_train.py first."
        )

    print(f"Feature version check passed (v{current_version}, {meta['kind']} model).")

    model_path = config.model_path(in_dir)
    if meta.get("kind") == "stub":
        # Stub model: no LightGBM file, return None and handle below
        return None, None, meta

    model = lgb.Booster(model_file=str(model_path))

    cal_path = config.calibrator_path(in_dir)
    with open(cal_path, "rb") as f:
        calibrator = pickle.load(f)

    return model, calibrator, meta


def score_shard(
    shard: pl.DataFrame,
    model,
    calibrator,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Score one shard, apply calibration, return (s1_id, cand_id, prob)."""
    X = shard.select(feature_cols).to_numpy(order="c")
    raw = model.predict(X)

    if hasattr(calibrator, "transform"):      # IsotonicRegression
        probs = calibrator.transform(raw)
    elif isinstance(calibrator, dict) and calibrator.get("kind") == "identity":
        probs = raw
    else:
        probs = raw

    return pl.DataFrame({
        "source1_entity_id":  shard["source1_entity_id"],
        "candidate_entity_id": shard["candidate_entity_id"],
        "prob": probs.astype(np.float32),
    })


def stub_score_shard(shard: pl.DataFrame) -> pl.DataFrame:
    """Fallback when only a stub model.txt exists (Gate 0/1 pipeline closure)."""
    score = pl.col("prior_score") if "prior_score" in shard.columns else pl.col("f000")
    lo = score.min().over("source1_entity_id")
    hi = score.max().over("source1_entity_id")
    prob_expr = pl.when(hi > lo).then((score - lo) / (hi - lo)).otherwise(0.5)
    return shard.select(
        "source1_entity_id",
        "candidate_entity_id",
        prob_expr.cast(pl.Float32).alias("prob"),
    )


def score_split(split: str, in_dir, out_dir, model, calibrator, is_stub: bool, feature_cols: list[str],
                shard_rows: int, val_only: bool, smoke: bool) -> int:
    """Stream features_{split} -> scored_{split}.parquet, one shard in memory at a time."""
    feat_path = config.features_path(split, in_dir)
    keep = None
    if val_only and split == "train":
        keep = pio.val_ids(smoke)
        if keep is None:
            print("  --val-only: no held-out validation ids found (smoke run?), scoring every entity")
    n_total = pq.ParquetFile(str(feat_path)).metadata.num_rows
    print(f"\n{split}: {n_total:,} pairs in {feat_path.name}"
          + (f", keeping only {keep.len():,} validation entities" if keep is not None else ""), flush=True)

    out = config.scored_path(split, out_dir)
    tmp = out.with_name(out.name + ".tmp")
    schema = pl.DataFrame(schema=pio.SCORED_SCHEMA).to_arrow().schema
    written, seen, t0 = 0, 0, time.perf_counter()
    with pq.ParquetWriter(str(tmp), schema) as writer:
        for i, shard in enumerate(iter_entity_batches(feat_path, [ID, "candidate_entity_id", *feature_cols], shard_rows)):
            seen += shard.height
            if keep is not None:
                shard = shard.filter(pl.col(ID).is_in(keep.implode()))
            if not shard.height:
                continue
            result = stub_score_shard(shard) if is_stub else score_shard(shard, model, calibrator, feature_cols)
            pio.check_schema(result, pio.SCORED_SCHEMA, f"scored_{split}")
            writer.write_table(result.to_arrow().cast(schema))
            written += result.height
            del shard, result   # release RAM immediately
            rate = seen / max(time.perf_counter() - t0, 1e-9)
            print(f"    shard {i:4d}: {seen:>12,} read  {written:>12,} scored  {rate:>9,.0f} rows/s", flush=True)
    os.replace(tmp, out)
    print(f"  -> {out.name}  {written:,} pairs")
    return written


def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    ap.add_argument("--splits", nargs="+", choices=list(config.SPLITS), default=None,
                    help="splits to score (default: every split that has a features file)")
    ap.add_argument("--val-only", action="store_true",
                    help="for the train split, score only the held-out validation entities")
    ap.add_argument("--shard-rows", type=int, default=SHARD_ROWS, help="pairs per in-memory shard")
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    model, calibrator, meta = load_model_and_calibrator(in_dir)
    feature_names, _ = pio.feature_spec()
    feature_cols = pio.feature_columns(len(feature_names))
    is_stub = meta.get("kind") == "stub"

    for split in (args.splits or list(config.SPLITS)):
        if not config.features_path(split, in_dir).exists():
            print(f"  {config.features_path(split, in_dir).name} not found — skipping {split}")
            continue
        score_split(split, in_dir, out_dir, model, calibrator, is_stub, feature_cols,
                    args.shard_rows, args.val_only, args.smoke)

    peak = pio.peak_rss_bytes()
    print(f"\ns5 done in {time.perf_counter() - t0:.1f}s" + (f", peak RSS {peak / 2**30:.2f} GB" if peak else ""))


if __name__ == "__main__":
    sys.exit(main())
