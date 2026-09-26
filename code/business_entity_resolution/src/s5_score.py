"""S5 Score (owner: Tanuj): model + features_{split} -> scored_{split}.

Loads the LightGBM model and isotonic calibrator. Refuses to run if the
model's FEATURE_VERSION doesn't match the current features.py. Scores in
~5M-pair shards so the full inference feature matrix is never materialised
on disk (it would be ~25 GB). Writes only (source1_entity_id,
candidate_entity_id, prob float32).

Usage:
    python s5_score.py [--smoke] [--input DIR] [--output DIR]
"""
import json
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
import pipeline_io as pio

SHARD_ROWS = 5_000_000   # pairs per in-memory shard; fits comfortably in 24 GB


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
    X = shard.select(feature_cols).to_numpy(allow_copy=True)
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


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    model, calibrator, meta = load_model_and_calibrator(in_dir)
    feature_names, _ = pio.feature_spec()
    feature_cols = list(feature_names)
    is_stub = meta.get("kind") == "stub"

    for split in config.SPLITS:
        feat_path = config.features_path(split, in_dir)
        if not feat_path.exists():
            print(f"  {feat_path.name} not found — skipping {split}")
            continue

        lf = pl.scan_parquet(feat_path)
        total = lf.select(pl.len()).collect().item()
        n_shards = max(1, (total + SHARD_ROWS - 1) // SHARD_ROWS)
        print(f"\n{split}: {total:,} pairs across {n_shards} shard(s)")

        results = []
        for i in range(n_shards):
            shard = lf.slice(i * SHARD_ROWS, SHARD_ROWS).collect()

            if is_stub:
                result = stub_score_shard(shard)
            else:
                result = score_shard(shard, model, calibrator, feature_cols)

            results.append(result)
            del shard   # release RAM immediately

        scored = pl.concat(results)
        pio.check_schema(scored, pio.SCORED_SCHEMA, f"scored_{split}")

        out = config.scored_path(split, out_dir)
        scored.write_parquet(out)
        print(f"  -> {out.name}  {scored.height:,} pairs")
        del results, scored

    print(f"\ns5 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
