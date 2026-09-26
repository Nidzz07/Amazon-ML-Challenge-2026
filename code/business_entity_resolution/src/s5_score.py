"""S5 Score (STUB, owner: Tanuj): model + features_{split} -> scored_{split}.

Placeholder. It refuses to run if model.txt's feature_version differs from the
current FEATURE_VERSION; keep that guard in the real version. The stub
"probability" is the blocking prior_score (feature f000) min-max scaled to [0, 1]
within each Source-1 entity (see stub_prob).

Output: source1_entity_id str, candidate_entity_id str, prob float32.

Usage:
    python s5_score.py [--smoke] [--input DIR] [--output DIR]
"""
import json
import sys
import time

import polars as pl

import config
import pipeline_io as pio


def load_model(in_dir) -> dict:
    model = json.loads(config.model_path(in_dir).read_text(encoding=config.ENCODING))
    names, version = pio.feature_spec()
    if model["feature_version"] != version or model["feature_names"] != list(names):
        raise SystemExit(
            f"model.txt was trained on FEATURE_VERSION {model['feature_version']}, "
            f"current is {version}: retrain before scoring"
        )
    return model


def stub_prob(score: pl.Expr) -> pl.Expr:
    """STAND-IN for Tanuj's calibrated model. REMOVE once model.py exists and
    FEATURE_VERSION is real.

    prior_score is a sum of 1/rank across channels (range 0..n_channels), not a
    probability. Here it is min-max scaled per Source-1 entity so S6 receives valid
    [0, 1] values. These are NOT calibrated. An entity's top candidate is always
    1.0 and its bottom one 0.0, so nothing downstream should read them as real
    match probabilities. When an entity's candidates all tie (including a single
    candidate), each gets 0.5, which is maximally uncertain.
    """
    lo, hi = score.min().over("source1_entity_id"), score.max().over("source1_entity_id")
    return pl.when(hi > lo).then((score - lo) / (hi - lo)).otherwise(0.5)


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()
    load_model(in_dir)
    for split in config.SPLITS:
        scored = (
            pl.scan_parquet(config.features_path(split, in_dir))
            .select("source1_entity_id", "candidate_entity_id", stub_prob(pl.col("f000")).cast(pl.Float32).alias("prob"))
            .collect()
        )
        pio.check_schema(scored, pio.SCORED_SCHEMA, f"scored_{split}")
        out = config.scored_path(split, out_dir)
        scored.write_parquet(out)
        print(f"{out.name:<22} {scored.height:>10,} pairs")
    print(f"s5 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
