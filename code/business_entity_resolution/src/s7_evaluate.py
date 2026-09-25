"""S7 Evaluate (STUB, owner: Krrish): S6 train output + ground truth -> report_{tag}.json.

Placeholder harness. It uses an inline macro F0.5 (F = 1.25c / (0.25k + m);
k = 0 scores 1.0 for an empty prediction and 0.0 otherwise) until metric.py exists.
It reports overall, by country, by true-match-count bucket, singleton accuracy, and
the blocking recall ceiling. Full runs score only the held-out validation entities;
smoke runs score every smoke entity (the smoke sample has no held-out split).

Usage:
    python s7_evaluate.py [--smoke] [--input DIR] [--output DIR] [--tag TAG]
"""
import json
import sys
import time

import polars as pl

import config
import pipeline_io as pio

SPLIT = "train"


def per_entity(truth: pl.DataFrame, pred: pl.DataFrame, cand: pl.DataFrame, entities: pl.DataFrame) -> pl.DataFrame:
    keys = ["source1_entity_id", "candidate_entity_id"]
    hit = pred.join(truth, on=keys, how="semi")
    in_cand = truth.join(cand, on=keys, how="semi")
    count = lambda df, name: df.group_by("source1_entity_id").len(name=name)  # noqa: E731
    out = entities
    for df, name in ((truth, "k"), (pred, "m"), (hit, "c"), (in_cand, "k_in_cand")):
        out = out.join(count(df, name), on="source1_entity_id", how="left")
    out = out.with_columns(pl.col("k", "m", "c", "k_in_cand").fill_null(0).cast(pl.Int64))
    return out.with_columns(
        pl.when(pl.col("k") == 0)
        .then((pl.col("m") == 0).cast(pl.Float64))
        .otherwise(1.25 * pl.col("c") / (0.25 * pl.col("k") + pl.col("m")))
        .alias("f05"),
        pl.col("k").cut([0, 1, 3, 5], labels=["0", "1", "2-3", "4-5", "6+"]).cast(pl.String).alias("k_bucket"),
    )


def summarise(df: pl.DataFrame) -> dict:
    s = df.select(
        pl.len().alias("entities"),
        pl.col("f05").mean().alias("macro_f05"),
        (pl.col("c").sum() / pl.col("m").sum()).alias("precision"),
        (pl.col("c").sum() / pl.col("k").sum()).alias("recall"),
        (pl.col("k_in_cand").sum() / pl.col("k").sum()).alias("blocking_recall"),
        (pl.col("m").sum() / pl.len()).alias("mean_predicted"),
    ).row(0, named=True)
    return {k: (None if v is None or v != v else v) for k, v in s.items()}  # NaN -> null in JSON


def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    ap.add_argument("--tag", help="report tag (default: git short sha, prefixed smoke_ in smoke mode)")
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    truth = (
        pl.read_parquet(config.records_path(SPLIT, "ground_truth", in_dir))
        .select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id"))
        .explode("candidate_entity_id", empty_as_null=False)
        .filter(pl.col("candidate_entity_id") != "")
    )
    entities = pl.read_parquet(config.records_path(SPLIT, config.SOURCE1_SRC, in_dir), columns=["entity_id", "country"])
    entities = entities.rename({"entity_id": "source1_entity_id"})
    held_out = pio.val_ids(args.smoke)
    if held_out is not None:
        entities = entities.filter(pl.col("source1_entity_id").is_in(held_out.implode()))

    pred_file = config.matching_results_path(SPLIT, in_dir)
    pred_ids = pl.read_csv(pred_file, separator=config.TSV_SEP, infer_schema=False, quote_char=None, columns=[0])
    missing = entities.join(pred_ids, on="source1_entity_id", how="anti").height
    if missing:
        raise SystemExit(f"{pred_file.name} is missing {missing:,} evaluated entities")
    pred = pio.read_id_lists(pred_file, "matched_entity_ids")
    cand = pio.read_id_lists(config.candidate_pairs_path(SPLIT, in_dir), "candidate_entity_ids")

    scored = per_entity(truth, pred, cand, entities)
    report = {
        "tag": args.tag or (("smoke_" if args.smoke else "") + pio.git_sha()),
        "split": "smoke_train" if args.smoke else ("val" if held_out is not None else "train"),
        "metric_impl": "stub_inline",
        "overall": summarise(scored),
        "by_country": {c: summarise(g) for (c,), g in scored.group_by("country", maintain_order=True)},
        "by_k_bucket": {b: summarise(g) for (b,), g in scored.sort("k").group_by("k_bucket", maintain_order=True)},
        "singleton_accuracy": scored.filter(pl.col("k") == 0).select((pl.col("m") == 0).mean()).item(),
    }
    out = config.report_path(report["tag"], out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding=config.ENCODING)
    o = report["overall"]
    print(f"{o['entities']:,} entities: macro F0.5 {o['macro_f05']:.4f}, P {o['precision'] or 0:.4f}, "
          f"R {o['recall']:.4f}, blocking recall {o['blocking_recall']:.4f}, singleton acc {report['singleton_accuracy']:.4f}")
    print(f"wrote {out}")
    print(f"s7 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
