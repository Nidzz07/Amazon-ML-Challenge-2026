"""Candidate-cap sweep (owner: Nidhi): recall vs candidate-set size at several K.

Runs every blocking channel ONCE on the train split (via s2_block.channel_pairs),
builds the uncapped union, then applies s2_block.cap at each K in SWEEP_K. Nothing is
recomputed per K. Truth is the train ground truth; smoke runs score every smoke
entity (the smoke sample has no held-out split), full runs only the held-out ones.

Per K it reports:
  recall          true pairs inside the capped set / all true pairs (overall, by country)
  pairs           candidate pairs written at that cap
  reduction       1 - pairs / (entities x |S2+S3 pool|); the naive cross product
  reduction_shard 1 - pairs / sum over entities of their country shard's pool size
and per channel: of the true pairs that channel retrieved (uncapped), the share that
survive the cap, and the same for pairs ONLY that channel retrieved.

Writes artifacts[/smoke]/reports/cap_sweep.json. Does not write candidates.

Usage:
    python sweep_candidate_cap.py [--smoke] [--input DIR] [--output DIR]
"""
import json
import sys
import time

import polars as pl

import config
import pipeline_io as pio
import s2_block

SPLIT = "train"
SWEEP_K = (10, 15, 20, 25, 30, 40, 50, 60)
KEYS = ["source1_entity_id", "candidate_entity_id"]


def load_truth(in_dir, entities: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.read_parquet(config.records_path(SPLIT, "ground_truth", in_dir))
        .select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id"))
        .explode("candidate_entity_id", empty_as_null=False)
        .filter(pl.col("candidate_entity_id") != "")
        .join(entities, on="source1_entity_id", how="inner")
    )


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    entities = pl.read_parquet(config.norm_path(SPLIT, config.SOURCE1_SRC, in_dir), columns=["entity_id", "country"])
    entities = entities.rename({"entity_id": "source1_entity_id"})
    held_out = pio.val_ids(args.smoke)
    if held_out is not None:
        entities = entities.filter(pl.col("source1_entity_id").is_in(held_out.implode()))
    truth = load_truth(in_dir, entities)

    pool = pl.concat(
        [pl.read_parquet(config.norm_path(SPLIT, s, in_dir), columns=["country"]) for s in config.CANDIDATE_SRCS]
    )
    shard_pool = pool.group_by("country").len(name="pool")
    cross = entities.height * pool.height
    cross_shard = entities.join(shard_pool, on="country", how="left")["pool"].fill_null(0).sum()

    long, _ = s2_block.channel_pairs(SPLIT, in_dir)
    uncapped = s2_block.union(long).join(entities.select("source1_entity_id"), on="source1_entity_id", how="semi")
    t_block = time.perf_counter() - t0

    # True pairs with the channels that retrieved them (0 = never retrieved).
    truth_ch = truth.join(uncapped.select(*KEYS, "channels", "n_channels"), on=KEYS, how="left").with_columns(
        pl.col("channels").fill_null(0), pl.col("n_channels").fill_null(0)
    )
    countries = entities["country"].unique().sort().to_list()

    rows = []
    for k in [*SWEEP_K, None]:
        capped = uncapped if k is None else s2_block.cap(uncapped, k)
        hit = truth_ch.join(capped.select(KEYS), on=KEYS, how="semi")
        row = {
            "K": "uncapped" if k is None else k,
            "recall": hit.height / truth.height,
            "pairs": capped.height,
            "mean_per_entity": capped.height / entities.height,
            "reduction": 1 - capped.height / cross,
            "reduction_shard": 1 - capped.height / cross_shard,
            "recall_by_country": {
                c: hit.filter(pl.col("country") == c).height / max(1, truth.filter(pl.col("country") == c).height)
                for c in countries
            },
            "by_channel": {},
        }
        for bit, name in enumerate(config.CHANNELS):
            m = 1 << bit
            found = truth_ch.filter((pl.col("channels") & m) != 0)
            only = found.filter(pl.col("n_channels") == 1)
            hf = hit.filter((pl.col("channels") & m) != 0)
            ho = hf.filter(pl.col("n_channels") == 1)
            row["by_channel"][name] = {
                "true_found": found.height,
                "kept": hf.height / found.height if found.height else None,
                "true_only": only.height,
                "only_kept": ho.height / only.height if only.height else None,
            }
        rows.append(row)

    report = {
        "split": "smoke_train" if args.smoke else ("val" if held_out is not None else "train"),
        "entities": entities.height,
        "true_pairs": truth.height,
        "true_pairs_by_country": {c: truth.filter(pl.col("country") == c).height for c in countries},
        "pool": pool.height,
        "pool_by_country": dict(shard_pool.sort("country").iter_rows()),
        "block_sec": t_block,
        "sweep": rows,
    }
    out = out_dir / config.REPORTS_DIRNAME / "cap_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding=config.ENCODING)

    print(f"\n{report['split']}: {entities.height:,} entities, {truth.height:,} true pairs, pool {pool.height:,}")
    print(f"{'K':>9} {'recall':>7} " + " ".join(f"{c:>7}" for c in countries)
          + f" {'pairs':>10} {'/ent':>6} {'reduction':>10} {'red_shard':>10}")
    for r in rows:
        print(f"{r['K']!s:>9} {r['recall']:7.4f} " + " ".join(f"{r['recall_by_country'][c]:7.4f}" for c in countries)
              + f" {r['pairs']:>10,} {r['mean_per_entity']:6.1f} {r['reduction']:10.6f} {r['reduction_shard']:10.6f}")
    print("\nper channel: share of that channel's true pairs kept (share of its exclusive true pairs kept)")
    print(f"{'K':>9} " + " ".join(f"{n:>18}" for n in config.CHANNELS))
    fmt = lambda v: "   -" if v is None else f"{v:.4f}"  # noqa: E731
    for r in rows:
        print(f"{r['K']!s:>9} " + " ".join(
            f"{fmt(r['by_channel'][n]['kept']):>8} ({fmt(r['by_channel'][n]['only_kept']):>6})" + "  "
            for n in config.CHANNELS))
    print(" found:    " + " ".join(
        f"{r['by_channel'][n]['true_found']:>9,} ({r['by_channel'][n]['true_only']:>6,})" for n in config.CHANNELS
        for r in rows[-1:]))
    print(f"\nwrote {out}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
