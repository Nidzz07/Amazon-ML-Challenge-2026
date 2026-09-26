"""
s5_score.py  —  Tanuj
Scores candidate pairs in shards to avoid materialising the full 25GB feature matrix.
Reads: model.txt + calibrator.pkl + features_{split}.parquet (produced shard by shard)
Writes: scored_{split}.parquet  —  (source1_entity_id, candidate_entity_id, prob)
"""

import polars as pl
import lightgbm as lgb
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from features import FEATURE_NAMES, FEATURE_VERSION

SHARD_SIZE = 5_000_000   # ~5M pairs per shard — stays comfortably inside 24 GB RAM


def load_model_and_calibrator(model_path: str, calibrator_path: str):
    """Load model and verify FEATURE_VERSION before anything else."""
    meta_path = model_path + ".meta"
    if not Path(meta_path).exists():
        raise FileNotFoundError(
            f"Model metadata file not found at {meta_path}. "
            "Re-train the model to regenerate it."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    saved_version = meta.get("FEATURE_VERSION")
    if saved_version != FEATURE_VERSION:
        raise RuntimeError(
            f"FEATURE_VERSION MISMATCH — model was trained on v{saved_version} "
            f"but features.py is now at v{FEATURE_VERSION}. "
            "Re-train the model or revert features.py."
        )

    print(f"Feature version check passed (v{FEATURE_VERSION}).")

    model = lgb.Booster(model_file=model_path)

    with open(calibrator_path, "rb") as f:
        calibrator = pickle.load(f)

    return model, calibrator


def score_shard(shard: pl.DataFrame, model, calibrator, feature_cols: list[str]) -> pl.DataFrame:
    """Featurise a shard and return (source1_entity_id, candidate_entity_id, prob)."""
    X = shard.select(feature_cols).to_numpy(allow_copy=True)
    raw_probs = model.predict(X)
    cal_probs = calibrator.transform(raw_probs)

    return pl.DataFrame({
        "source1_entity_id": shard["source1_entity_id"],
        "candidate_entity_id": shard["candidate_entity_id"],
        "prob": cal_probs.astype(np.float32),
    })


def score(features_path: str, model_path: str, calibrator_path: str, output_path: str):
    """
    Main entry point.
    Streams the feature file in SHARD_SIZE chunks, scores each shard,
    discards the feature columns immediately, and appends to output parquet.
    """
    model, calibrator = load_model_and_calibrator(model_path, calibrator_path)
    feature_cols = [f[0] for f in FEATURE_NAMES]

    print(f"Scanning {features_path} ...")
    # Use lazy scan so we never pull the full file into RAM
    lf = pl.scan_parquet(features_path)
    total_rows = lf.select(pl.len()).collect().item()
    print(f"Total candidate pairs to score: {total_rows:,}")

    n_shards = (total_rows + SHARD_SIZE - 1) // SHARD_SIZE
    all_results = []

    for shard_idx in tqdm(range(n_shards), desc="Scoring shards"):
        offset = shard_idx * SHARD_SIZE

        # Slice just this shard into memory
        shard = (
            lf.slice(offset, SHARD_SIZE)
              .collect()
        )

        result = score_shard(shard, model, calibrator, feature_cols)
        all_results.append(result)

        # Explicitly release the shard so Python can GC it
        del shard

    print("Concatenating results and writing output parquet...")
    scored = pl.concat(all_results)
    scored.write_parquet(output_path)

    print(f"Scored {len(scored):,} pairs -> {output_path}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score candidate pairs with calibrated probabilities.")
    parser.add_argument("--features",    type=str, required=True, help="Path to features_{split}.parquet")
    parser.add_argument("--model",       type=str, required=True, help="Path to model.txt")
    parser.add_argument("--calibrator",  type=str, required=True, help="Path to calibrator.pkl")
    parser.add_argument("--out",         type=str, required=True, help="Output path for scored_{split}.parquet")
    args = parser.parse_args()

    score(args.features, args.model, args.calibrator, args.out)
