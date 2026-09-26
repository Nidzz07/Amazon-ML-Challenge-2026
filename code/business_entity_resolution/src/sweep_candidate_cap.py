"""Candidate-cap sweep (owner: Nidhi): recall vs candidate-set size at several K.

Runs every blocking channel ONCE on the train split (via s2_block.channel_pairs),
builds the uncapped union, then applies s2_block.cap at each K in SWEEP_K. Nothing is
recomputed per K. Truth is the train ground truth; smoke runs score every smoke
entity (the smoke sample has no held-out split). Full runs block and score only the
held-out validation entities, against the complete S2+S3 pool. Channel statistics
computed on the Source-1 side (rare_token document frequency, exact_key Source-1
bucket sizes) therefore see 400k entities, not all 2.2M.

Per K it reports:
  recall          true pairs inside the capped set / all true pairs (overall, by country)
  pairs           candidate pairs written at that cap
  reduction       1 - pairs / (entities x |S2+S3 pool|); the naive cross product
  reduction_shard 1 - pairs / sum over entities of their country shard's pool size
and per channel: of the true pairs that channel retrieved (uncapped), the share that
survive the cap, and the same for pairs ONLY that channel retrieved.

Two exact_key diagnostics, because candidates only carry one exact_key bit:
  slots_at_k        at --slots-k, per country, the mean per-entity share of capped slots
                    each source occupies (name_tfidf, addr_tfidf, exact_key state
                    family, exact_key other families, rare_token), the share it
                    occupies alone, and how often those sole-source slots are true.
  ablate_state      the same K sweep with the state_canon families removed from
                    exact_key: if recall at a K goes UP, those families are crowding
                    true pairs out of the cap.

Writes artifacts[/smoke]/reports/<--report> (default cap_sweep.json). Does not write
candidates.

Usage:
    python sweep_candidate_cap.py [--smoke] [--input DIR] [--output DIR] [--report NAME] [--slots-k K] [--country C ...]
"""
import json
import sys
import time

import polars as pl

import config
import pipeline_io as pio
import s2_block
from blocking import exact_key

SPLIT = "train"
SWEEP_K = (10, 15, 20, 25, 30, 40, 50, 60)
KEYS = ["source1_entity_id", "candidate_entity_id"]
STATE_FAMILIES = tuple(f for f in config.EXACT_KEY_FAMILIES if "state_canon" in f)
OTHER_FAMILIES = tuple(f for f in config.EXACT_KEY_FAMILIES if "state_canon" not in f)
EK_BIT = 1 << config.CHANNELS.index("exact_key")
SOURCES = ("name_tfidf", "addr_tfidf", "ek_state", "ek_other", "rare_token")
SOURCE_BITS = {"name_tfidf": 1 << config.CHANNELS.index("name_tfidf"),
               "addr_tfidf": 1 << config.CHANNELS.index("addr_tfidf"),
               "rare_token": 1 << config.CHANNELS.index("rare_token")}


def load_truth(in_dir, entities: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.read_parquet(config.records_path(SPLIT, "ground_truth", in_dir))
        .select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id"))
        .explode("candidate_entity_id", empty_as_null=False)
        .filter(pl.col("candidate_entity_id") != "")
        .join(entities, on="source1_entity_id", how="inner")
    )


def exact_key_detail(in_dir, s1_ids: pl.Series | None, countries: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(pair -> ek_state / ek_other flags, exact_key output without the state families),
    both over the given country shards."""
    flags, ablated = [], []
    for country in countries:
        s1, pool = s2_block.load_shard(SPLIT, in_dir, country, s1_ids)
        for fams, col in ((STATE_FAMILIES, "ek_state"), (OTHER_FAMILIES, "ek_other")):
            for fam in fams:
                flags.append(exact_key.family_pairs(s1, pool, fam).select(KEYS).with_columns(pl.lit(col).alias("src")))
        ablated.append(exact_key.run(s1, pool, families=OTHER_FAMILIES))
    flag = pl.concat(flags).unique().group_by(KEYS).agg(
        (pl.col("src") == "ek_state").any().alias("ek_state"),
        (pl.col("src") == "ek_other").any().alias("ek_other"),
    )
    return flag, pl.concat(ablated)


def slot_breakdown(capped: pl.DataFrame, flags: pl.DataFrame, truth: pl.DataFrame, entities: pl.DataFrame) -> dict:
    """Per country: mean per-entity share of capped slots by source (any / sole source),
    and the true-match rate of the slots each source fills alone."""
    df = (
        capped.join(flags, on=KEYS, how="left")
        .join(truth.select(KEYS).with_columns(pl.lit(True).alias("true")), on=KEYS, how="left")
        .join(entities, on="source1_entity_id")
        .with_columns(
            *[((pl.col("channels") & b) != 0).alias(n) for n, b in SOURCE_BITS.items()],
            pl.col("ek_state", "ek_other", "true").fill_null(False),
        )
        .with_columns(pl.sum_horizontal(*SOURCES).alias("n_src"))
    )
    out = {}
    for (country,), g in df.group_by("country"):
        per_ent = g.group_by("source1_entity_id").agg(
            pl.len().alias("slots"),
            *[pl.col(s).mean().alias(f"{s}_any") for s in SOURCES],
            *[(pl.col(s) & (pl.col("n_src") == 1)).mean().alias(f"{s}_only") for s in SOURCES],
        )
        out[country] = {
            "entities": per_ent.height,
            "mean_slots": per_ent["slots"].mean(),
            "true_rate_all_slots": g["true"].mean(),
            **{s: {
                "share_any": per_ent[f"{s}_any"].mean(),
                "share_only": per_ent[f"{s}_only"].mean(),
                "only_slots": g.filter(pl.col(s) & (pl.col("n_src") == 1)).height,
                "only_true_rate": g.filter(pl.col(s) & (pl.col("n_src") == 1))["true"].mean(),
            } for s in SOURCES},
        }
    return out


def recall_rows(uncapped: pl.DataFrame, truth: pl.DataFrame, countries: list[str]) -> list[dict]:
    rows = []
    for k in [*SWEEP_K, None]:
        capped = uncapped if k is None else s2_block.cap(uncapped, k)
        hit = truth.join(capped.select(KEYS), on=KEYS, how="semi")
        rows.append({
            "K": "uncapped" if k is None else k,
            "recall": hit.height / truth.height,
            "pairs": capped.height,
            "recall_by_country": {
                c: hit.filter(pl.col("country") == c).height / max(1, truth.filter(pl.col("country") == c).height)
                for c in countries
            },
        })
    return rows


def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    ap.add_argument("--report", default="cap_sweep.json", help="report file name under reports/ (default: %(default)s)")
    ap.add_argument("--slots-k", type=int, default=config.MAX_CANDIDATES_PER_ENTITY,
                    help="cap at which to break slots down by source (default: %(default)s)")
    ap.add_argument("--country", action="append",
                    help="run only this country shard (repeatable). Shards are independent, so "
                         "per-country reports combine exactly; use it when all shards at once do not fit in RAM")
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    entities = pl.read_parquet(config.norm_path(SPLIT, config.SOURCE1_SRC, in_dir), columns=["entity_id", "country"])
    entities = entities.rename({"entity_id": "source1_entity_id"})
    held_out = pio.val_ids(args.smoke)
    if held_out is not None:
        entities = entities.filter(pl.col("source1_entity_id").is_in(held_out.implode()))
    if args.country:
        entities = entities.filter(pl.col("country").is_in(args.country))
    truth = load_truth(in_dir, entities)

    pool = pl.concat(
        [pl.read_parquet(config.norm_path(SPLIT, s, in_dir), columns=["country"]) for s in config.CANDIDATE_SRCS]
    )
    shard_pool = pool.group_by("country").len(name="pool")
    cross = entities.height * pool.height
    cross_shard = entities.join(shard_pool, on="country", how="left")["pool"].fill_null(0).sum()

    # Full runs block only the held-out Source-1 entities (against the complete pool).
    long, _ = s2_block.channel_pairs(SPLIT, in_dir, s1_ids=held_out,
                                     countries=entities["country"].unique().sort().to_list())
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

    flags, ablated_ek = exact_key_detail(in_dir, held_out, countries)
    slots = slot_breakdown(s2_block.cap(uncapped, args.slots_k), flags, truth, entities)
    ablated_long = pl.concat([
        long.filter(pl.col("bit") != EK_BIT),
        ablated_ek.with_columns(pl.lit(EK_BIT, dtype=pl.UInt8).alias("bit")),
    ])
    ablated = s2_block.union(ablated_long).join(entities.select("source1_entity_id"), on="source1_entity_id", how="semi")
    ablate_rows = recall_rows(ablated, truth, countries)

    report = {
        "split": "smoke_train" if args.smoke else ("val" if held_out is not None else "train"),
        "entities": entities.height,
        "true_pairs": truth.height,
        "true_pairs_by_country": {c: truth.filter(pl.col("country") == c).height for c in countries},
        "pool": pool.height,
        "pool_by_country": dict(shard_pool.sort("country").iter_rows()),
        "block_sec": t_block,
        "sweep": rows,
        "state_families": [list(f) for f in STATE_FAMILIES],
        "slots_k": args.slots_k,
        "slots_at_k": slots,
        "ablate_state": ablate_rows,
    }
    out = out_dir / config.REPORTS_DIRNAME / args.report
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
    print("\nrecall with vs without the state families (ablate_state)")
    print(f"{'K':>9} " + " ".join(f"{'with ' + c:>12} {'w/o ' + c:>12}" for c in countries)
          + f" {'pairs with':>11} {'pairs w/o':>11}")
    for r, a in zip(rows, ablate_rows):
        print(f"{r['K']!s:>9} " + " ".join(f"{r['recall_by_country'][c]:12.4f} {a['recall_by_country'][c]:12.4f}"
                                          for c in countries) + f" {r['pairs']:>11,} {a['pairs']:>11,}")
    print(f"\nslots at K={args.slots_k}: mean per-entity share, any source / sole source (sole-source true rate)")
    for c, d in sorted(slots.items()):
        print(f"  {c}: {d['entities']:,} entities, {d['mean_slots']:.1f} slots/entity, "
              f"true rate {d['true_rate_all_slots']:.3f}")
        for src in SOURCES:
            v = d[src]
            tr = "-" if v["only_true_rate"] is None else f"{v['only_true_rate']:.3f}"
            print(f"    {src:<11} any {v['share_any']:.3f}  only {v['share_only']:.3f}  "
                  f"({v['only_slots']:>7,} slots, true {tr})")
    print(f"\nwrote {out}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
