"""S4 Train (STUB, owner: Tanuj): features_train -> model.txt, calibrator.pkl.

Placeholder. Nothing is fitted. model.txt is a JSON header recording the
FEATURE_VERSION and feature names it was "trained" against, plus the positive rate,
so S5's version guard is live from day one. calibrator.pkl is an identity mapping
stored as a plain dict, so it unpickles without importing this module. On full runs,
held-out validation entities are excluded from training.

Usage:
    python s4_train.py [--smoke] [--input DIR] [--output DIR]
"""
import json
import pickle
import sys
import time

import polars as pl

import config
import pipeline_io as pio


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()
    names, version = pio.feature_spec()

    lf = pl.scan_parquet(config.features_path("train", in_dir))
    held_out = pio.val_ids(args.smoke)
    if held_out is not None:
        lf = lf.filter(~pl.col("source1_entity_id").is_in(held_out.implode()))
    stats = lf.select(pl.len().alias("rows"), pl.col("label").mean().alias("pos_rate")).collect().row(0, named=True)

    model = {
        "kind": "stub",
        "feature_version": version,
        "feature_names": list(names),
        "train_rows": stats["rows"],
        "pos_rate": stats["pos_rate"],
        "seed": config.SEED,
    }
    config.model_path(out_dir).write_text(json.dumps(model, indent=2), encoding=config.ENCODING)
    with open(config.calibrator_path(out_dir), "wb") as f:
        pickle.dump({"kind": "identity"}, f)
    print(f"model.txt: stub, feature_version={version}, {stats['rows']:,} rows, pos_rate={stats['pos_rate']:.4f}")
    print(f"s4 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
