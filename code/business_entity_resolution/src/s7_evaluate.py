"""S7 Evaluate (owner: Krrish): S6 train output + ground truth → report_{tag}.json.

Uses metric.py for the F₀.₅ computation (no more inline formula).
Reports overall, by country, by true-match-count bucket, singleton accuracy,
and blocking recall ceiling. Full runs score only the held-out validation
entities; smoke runs score every smoke entity (the smoke sample has no split).

Usage:
    python s7_evaluate.py [--smoke] [--input DIR] [--output DIR] [--tag TAG]
"""
import json
import sys
import time

import polars as pl

import config
import pipeline_io as pio
from metric import macro_f05_arrays

SPLIT = "train"


def per_entity(
    truth: pl.DataFrame,
    pred: pl.DataFrame,
    cand: pl.DataFrame,
    entities: pl.DataFrame,
) -> pl.DataFrame:
    """One row per entity with c, k, m, k_in_cand, f05, country, k_bucket."""
    keys = ["source1_entity_id", "candidate_entity_id"]
    hit = pred.join(truth, on=keys, how="semi")
    in_cand = truth.join(cand, on=keys, how="semi")
    count = lambda df, name: df.group_by("source1_entity_id").len(name=name)  # noqa: E731

    out = entities
    for df, name in ((truth, "k"), (pred, "m"), (hit, "c"), (in_cand, "k_in_cand")):
        out = out.join(count(df, name), on="source1_entity_id", how="left")
    out = out.with_columns(pl.col("k", "m", "c", "k_in_cand").fill_null(0).cast(pl.Int64))

    # Use metric.py's formula directly for per-entity scores
    return out.with_columns(
        pl.when(pl.col("k") == 0)
        .then((pl.col("m") == 0).cast(pl.Float64))
        .otherwise(
            pl.when(pl.col("m") == 0)
            .then(0.0)
            .otherwise(1.25 * pl.col("c") / (0.25 * pl.col("k") + pl.col("m")))
        )
        .alias("f05"),
        pl.col("k")
        .cut([0, 1, 3, 5], labels=["0", "1", "2-3", "4-5", "6+"])
        .cast(pl.String)
        .alias("k_bucket"),
    )


def summarise(df: pl.DataFrame) -> dict:
    """Summary statistics for a group of entities."""
    c = df["c"].to_numpy()
    k = df["k"].to_numpy()
    m = df["m"].to_numpy()

    # Use the canonical macro_f05_arrays from metric.py
    f05 = macro_f05_arrays(c, k, m)

    k_in_cand = df["k_in_cand"].to_numpy()
    total_k = int(k.sum())
    total_m = int(m.sum())
    total_c = int(c.sum())
    total_k_in_cand = int(k_in_cand.sum())

    return {
        "entities": df.height,
        "macro_f05": f05,
        "precision": total_c / total_m if total_m > 0 else None,
        "recall": total_c / total_k if total_k > 0 else None,
        "blocking_recall": total_k_in_cand / total_k if total_k > 0 else None,
        "mean_predicted": total_m / df.height if df.height > 0 else 0.0,
    }


def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    ap.add_argument("--tag", help="report tag (default: git short sha, prefixed smoke_ in smoke mode)")
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    # Load ground truth
    truth = (
        pl.read_parquet(config.records_path(SPLIT, "ground_truth", in_dir))
        .select(
            "source1_entity_id",
            pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id"),
        )
        .explode("candidate_entity_id", empty_as_null=False)
        .filter(pl.col("candidate_entity_id") != "")
    )

    # Load entity list with country
    entities = pl.read_parquet(
        config.records_path(SPLIT, config.SOURCE1_SRC, in_dir),
        columns=["entity_id", "country"],
    ).rename({"entity_id": "source1_entity_id"})

    # Filter to validation entities on full runs
    held_out = pio.val_ids(args.smoke)
    if held_out is not None:
        entities = entities.filter(pl.col("source1_entity_id").is_in(held_out.implode()))

    # Load predictions and check coverage
    pred_file = config.matching_results_path(SPLIT, in_dir)
    pred_ids = pl.read_csv(
        pred_file, separator=config.TSV_SEP, infer_schema=False,
        quote_char=None, columns=[0],
    )
    missing = entities.join(pred_ids, on="source1_entity_id", how="anti").height
    if missing:
        raise SystemExit(f"{pred_file.name} is missing {missing:,} evaluated entities")

    pred = pio.read_id_lists(pred_file, "matched_entity_ids")
    cand = pio.read_id_lists(config.candidate_pairs_path(SPLIT, in_dir), "candidate_entity_ids")

    # Score
    scored = per_entity(truth, pred, cand, entities)
    report = {
        "tag": args.tag or (("smoke_" if args.smoke else "") + pio.git_sha()),
        "split": "smoke_train" if args.smoke else ("val" if held_out is not None else "train"),
        "metric_impl": "metric.py",
        "feature_version": None,
        "overall": summarise(scored),
        "by_country": {
            c: summarise(g) for (c,), g in scored.group_by("country", maintain_order=True)
        },
        "by_k_bucket": {
            b: summarise(g) for (b,), g in scored.sort("k").group_by("k_bucket", maintain_order=True)
        },
        "singleton_accuracy": scored.filter(pl.col("k") == 0).select(
            (pl.col("m") == 0).mean()
        ).item(),
    }

    # Try to get the feature version if features.py is available
    try:
        from features import FEATURE_VERSION
        report["feature_version"] = FEATURE_VERSION
    except ImportError:
        pass

    # Write report
    out = config.report_path(report["tag"], out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding=config.ENCODING)

    o = report["overall"]
    print(
        f"{o['entities']:,} entities: macro F0.5 {o['macro_f05']:.4f}, "
        f"P {o['precision'] or 0:.4f}, R {o['recall']:.4f}, "
        f"blocking recall {o['blocking_recall']:.4f}, "
        f"singleton acc {report['singleton_accuracy']:.4f}"
    )
    for country, cs in report["by_country"].items():
        print(f"  {country}: F0.5 {cs['macro_f05']:.4f}, P {cs['precision'] or 0:.4f}, R {cs['recall']:.4f}")
    print(f"wrote {out}")
    print(f"s7 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
