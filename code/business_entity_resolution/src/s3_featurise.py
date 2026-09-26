"""S3 Featurise (STUB, owner: Krrish): norm + candidates_{split} -> features_{split}.

Placeholder. It emits the four stub features in pipeline_io.STUB_FEATURE_NAMES,
taken straight from the candidate row, as f000..f003 float32. On train it adds a
uint8 label from the ground truth. Once features.py defines FEATURE_NAMES /
FEATURE_VERSION this stub refuses to run, because it cannot compute those; replace
compute_features() at that point.

Output: source1_entity_id, candidate_entity_id, f000..fNNN float32, [label uint8 on train].

Usage:
    python s3_featurise.py [--smoke] [--input DIR] [--output DIR]
"""
import sys
import time

import polars as pl

import config
import pipeline_io as pio

STUB_EXPRS = {
    "prior_score": pl.col("prior_score"),
    "n_channels": pl.col("n_channels"),
    "best_rank": pl.col("best_rank"),
    "is_source3": pl.col("candidate_entity_id").str.starts_with("S3-"),
}


def true_pairs(in_dir) -> pl.LazyFrame:
    return (
        pl.scan_parquet(config.records_path("train", "ground_truth", in_dir))
        .select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id"))
        .explode("candidate_entity_id", empty_as_null=False)
        .filter(pl.col("candidate_entity_id") != "")
        .with_columns(pl.lit(1, dtype=pl.UInt8).alias("label"))
    )


def compute_features(split: str, in_dir) -> pl.DataFrame:
    names, _ = pio.feature_spec()
    missing = [n for n in names if n not in STUB_EXPRS]
    assert not missing, f"stub s3 cannot compute {missing}; replace this stub with the real featuriser"

    lf = pl.scan_parquet(config.candidates_path(split, in_dir))
    cols = [STUB_EXPRS[n].cast(pl.Float32).alias(c) for n, c in zip(names, pio.feature_columns(len(names)))]
    out = lf.select("source1_entity_id", "candidate_entity_id", *cols)
    if split == "train":
        out = out.join(true_pairs(in_dir), on=["source1_entity_id", "candidate_entity_id"], how="left").with_columns(
            pl.col("label").fill_null(0)
        )
    return out.collect()


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()
    for split in config.SPLITS:
        feats = compute_features(split, in_dir)
        out = config.features_path(split, out_dir)
        feats.write_parquet(out)
        pos = f", {int(feats['label'].sum()):,} positives" if "label" in feats.columns else ""
        print(f"{out.name:<24} {feats.height:>10,} rows x {feats.width - 2 - ('label' in feats.columns)} features{pos}")
    print(f"s3 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
