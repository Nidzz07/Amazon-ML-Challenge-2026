"""S2 Block (owner: Nidhi): norm_{split}_{src} -> candidates_{split}.

Runs every channel in config.CHANNELS (modules under blocking/) independently on
each country shard, then takes the union:
  channels    uint8 bitmask, bit i = config.CHANNELS[i]
  n_channels  number of channels that proposed the pair
  best_rank   best (lowest) channel_rank across those channels
  prior_score sum over contributing channels of 1 / channel_rank
and keeps the top config.MAX_CANDIDATES_PER_ENTITY per entity by prior_score
(ties: n_channels desc, best_rank asc, candidate id asc). The capped set is what
S5 scores, and so it is what candidate_pairs.tsv contains.

Country is a hard partition: 0 of 7,638,365 ground-truth pairs cross US/India.
France (test only, no labels) is sharded the same way, but that is UNPROVEN.
Shards come from the data, not a hard-coded list, so any new label gets its own shard.

Train subsampling is OFF by default (config.S2_TRAIN_ENTITIES = 0; see config for why).
With --train-entities N, Source-1 is subsampled (pipeline_io.sample_train_entities) to
every held-out validation entity plus N others, stratified by country. The S2/S3 pool
is never sampled, and the test split never is either.

Checkpoints: every (country, channel) output and every country's capped candidates is
written under config.s2_parts_dir(split) as soon as it is done (tmp file + rename, so a
kill mid-write never leaves a file that looks finished). --resume reuses them, but only
if manifest.json (knobs, sample, input file sizes/mtimes) matches this run; a run
without --resume starts clean. Union + cap run per country in config.S2_UNION_BUCKETS
hash buckets of Source-1 ids, so memory holds at most one shard's work, never the whole
split's. The candidates file is identical to the in-memory block() path, which
sweep_candidate_cap.py still uses.

Usage:
    python s2_block.py [--smoke] [--input DIR] [--output DIR] [--splits train test] [--resume]
                       [--train-entities N | --no-train-subsample]
"""
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import polars as pl

import config
import pipeline_io as pio
from blocking.common import CHANNEL_SCHEMA, pop_notes, pop_resolved

CHANNEL_MODULES = {name: importlib.import_module(f"blocking.{name}") for name in config.CHANNELS}
assert all(m.NAME == n for n, m in CHANNEL_MODULES.items())
# Each channel declares the norm columns it reads, and is handed only those: loading every
# column of the full US shard alone passed 13 GB. A channel reading an undeclared column
# fails with ColumnNotFoundError, never silently.
assert all(set(m.COLUMNS) <= set(pio.NORM_SCHEMA) and "entity_id" in m.COLUMNS for m in CHANNEL_MODULES.values())


def load_shard(split: str, in_dir, country: str, s1_ids: pl.Series | None = None,
               columns: tuple[str, ...] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One country's Source-1 rows (optionally only `s1_ids`, e.g. the held-out
    validation entities) and its full S2+S3 pool: every column, or only `columns`.
    Projection selects after the same filters, so rows and their order do not change."""
    s1 = pl.scan_parquet(config.norm_path(split, config.SOURCE1_SRC, in_dir)).filter(pl.col("country") == country)
    if s1_ids is not None:
        s1 = s1.filter(pl.col("entity_id").is_in(s1_ids.implode()))
    pool = pl.concat(
        [pl.scan_parquet(config.norm_path(split, s, in_dir)) for s in config.CANDIDATE_SRCS]
    ).filter(pl.col("country") == country)
    if columns is not None:
        s1, pool = s1.select(columns), pool.select(columns)
    return s1.collect(), pool.collect()


def channel_pairs(split: str, in_dir, verbose: bool = True, s1_ids: pl.Series | None = None,
                  countries: list[str] | None = None, smoke: bool = False) -> tuple[pl.DataFrame, list[dict]]:
    """Long (source1_entity_id, candidate_entity_id, channel_rank, channel_score, bit)
    from every channel on every shard, plus per shard x channel stats (including the
    document-frequency ceilings each channel resolved). `s1_ids` restricts the
    Source-1 side (the pool is always complete); `countries` restricts which shards run
    (shards are independent, so per-country runs combine exactly); `smoke` is passed to
    every channel (embed_ann uses it to pick its pairs file)."""
    if countries is None:
        countries = split_countries(split, in_dir)
    parts, stats = [], []
    for country in countries:
        if verbose:
            n_s1, n_pool = (f.height for f in load_shard(split, in_dir, country, s1_ids, ("entity_id",)))
            print(f"[{split}/{country}] {n_s1:,} source1 x {n_pool:,} pool")
        for bit, (name, module) in enumerate(CHANNEL_MODULES.items()):
            s1, pool = load_shard(split, in_dir, country, s1_ids, module.COLUMNS)
            out, row = run_channel(split, country, bit, name, module, s1, pool, smoke, verbose)
            del s1, pool
            stats.append(row)
            parts.append(out)
    long = pl.concat(parts) if parts else pl.DataFrame(schema={**CHANNEL_SCHEMA, "bit": pl.UInt8})
    return long, stats


def split_countries(split: str, in_dir) -> list[str]:
    return (
        pl.scan_parquet(config.norm_path(split, config.SOURCE1_SRC, in_dir))
        .select(pl.col("country").unique().sort())
        .collect()["country"]
        .to_list()
    )


def run_channel(split: str, country: str, bit: int, name: str, module, s1: pl.DataFrame, pool: pl.DataFrame,
                smoke: bool, verbose: bool) -> tuple[pl.DataFrame, dict]:
    """One channel on one shard: its output plus the channel's `bit` column, and stats."""
    pop_resolved()
    pop_notes()
    t0 = time.perf_counter()
    out = module.run(s1, pool, smoke=smoke)
    sec = time.perf_counter() - t0
    pio.check_schema(out, CHANNEL_SCHEMA, f"{name} output")
    assert out.select(pl.struct("source1_entity_id", "candidate_entity_id").is_unique().all()).item(), name
    row = {
        "split": split, "country": country, "channel": name, "pairs": out.height,
        "entities": out["source1_entity_id"].n_unique(), "sec": sec, "resolved": pop_resolved(),
        "notes": pop_notes(),
    }
    if verbose:
        print(f"  {name:<11} {row['pairs']:>10,} pairs  {row['entities']:>8,} entities  {sec:6.1f}s", flush=True)
    return out.with_columns(pl.lit(1 << bit, dtype=pl.UInt8).alias("bit")), row


def union(long: pl.DataFrame) -> pl.DataFrame:
    """One row per pair in the candidates schema, uncapped."""
    return long.lazy().group_by("source1_entity_id", "candidate_entity_id").agg(
        pl.col("bit").sum().cast(pl.UInt8).alias("channels"),  # each channel emits a pair at most once, so sum == OR
        pl.len().cast(pl.UInt8).alias("n_channels"),
        pl.col("channel_rank").min().alias("best_rank"),
        (1.0 / pl.col("channel_rank").cast(pl.Float64)).sum().cast(pl.Float32).alias("prior_score"),
    ).collect(engine="streaming")


CAP_KEYS = ["source1_entity_id", "prior_score", "n_channels", "best_rank", "candidate_entity_id"]
CAP_DESC = [False, True, True, False, False]


def cap(cands: pl.DataFrame, n: int = config.MAX_CANDIDATES_PER_ENTITY) -> pl.DataFrame:
    return (
        cands.sort(CAP_KEYS, descending=CAP_DESC)
        .filter(pl.int_range(pl.len()).over("source1_entity_id") < n)
        .select(list(pio.CANDIDATES_SCHEMA))
    )


def write_atomic(df: pl.DataFrame, path: Path) -> None:
    """A kill mid-write leaves only the .tmp file, never a truncated file at `path`."""
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def cap_from_parts(paths: list[Path], buckets: int) -> tuple[pl.DataFrame, int]:
    """(capped candidates, uncapped pair count) from one shard's channel parts. Equal to
    cap(union(concat(parts))): cap is per entity, so it runs one hash bucket of Source-1
    ids at a time and the shard's whole uncapped union never sits in memory."""
    capped, n_uncapped = [], 0
    for b in range(buckets):
        long_b = (
            pl.scan_parquet(paths)
            .filter(pl.col("source1_entity_id").hash(seed=0) % buckets == b)
            .collect(engine="streaming")
        )
        u = union(long_b)
        del long_b
        n_uncapped += u.height
        capped.append(cap(u))
        del u
    return pl.concat(capped), n_uncapped


def manifest(split: str, in_dir, s1_ids: pl.Series | None, smoke: bool) -> dict:
    """What a checkpoint depends on. --resume refuses parts made under anything else."""
    knobs = {k: getattr(config, k) for k in dir(config)
             if k.isupper() and k.startswith(("TFIDF_", "EXACT_KEY_", "RARE_TOKEN_", "EMBED_"))}
    knobs |= {"CHANNELS": config.CHANNELS, "MAX_CANDIDATES_PER_ENTITY": config.MAX_CANDIDATES_PER_ENTITY}
    inputs = [config.norm_path(split, s, in_dir) for s in (config.SOURCE1_SRC, *config.CANDIDATE_SRCS)]
    inputs.append(config.embed_ann_path(smoke))
    return json.loads(json.dumps({
        "split": split, "smoke": smoke, "knobs": knobs,
        "s1_ids": None if s1_ids is None else {"n": s1_ids.len(), "hash": int(s1_ids.sort().hash(seed=0).sum())},
        "inputs": {str(p): [p.stat().st_size, p.stat().st_mtime] if p.exists() else None for p in inputs},
    }, default=str))


def block_checkpointed(split: str, in_dir, out_dir, s1_ids: pl.Series | None, smoke: bool,
                       resume: bool) -> tuple[pl.DataFrame, int]:
    """(capped candidates, uncapped pair count) for `split`, checkpointed per
    (country, channel) and per country under config.s2_parts_dir. Same output as block()."""
    pdir = config.s2_parts_dir(split, out_dir)
    want = manifest(split, in_dir, s1_ids, smoke)
    mpath = pdir / "manifest.json"
    if resume and mpath.exists():
        have = json.loads(mpath.read_text(encoding=config.ENCODING))
        if have != want:
            bad = sorted(k for k in want if have.get(k) != want[k])
            raise SystemExit(f"--resume refused: {mpath} was written with different {bad}. "
                             f"Rerun without --resume to start clean.")
        print(f"[{split}] resuming from {pdir}", flush=True)
    else:
        if resume:
            print(f"[{split}] --resume: no manifest in {pdir}, starting clean", flush=True)
        shutil.rmtree(pdir, ignore_errors=True)
        pdir.mkdir(parents=True)
        mpath.write_text(json.dumps(want, indent=2), encoding=config.ENCODING)

    country_parts, n_uncapped = [], 0
    for country in split_countries(split, in_dir):
        cpath = pdir / f"candidates_{country}.parquet"
        if cpath.exists():
            print(f"[{split}/{country}] done in an earlier run, skipped", flush=True)
            country_parts.append(cpath)
            continue
        t0 = time.perf_counter()
        n_s1, n_pool = (f.height for f in load_shard(split, in_dir, country, s1_ids, ("entity_id",)))
        print(f"[{split}/{country}] {n_s1:,} source1 x {n_pool:,} pool", flush=True)
        ch_paths = []
        for bit, (name, module) in enumerate(CHANNEL_MODULES.items()):
            path = pdir / f"{country}_{name}.parquet"
            if path.exists():
                print(f"  {name:<11} done in an earlier run, skipped", flush=True)
            else:
                s1, pool = load_shard(split, in_dir, country, s1_ids, module.COLUMNS)
                out, _ = run_channel(split, country, bit, name, module, s1, pool, smoke, True)
                del s1, pool
                write_atomic(out, path)
                del out
            ch_paths.append(path)
        capped, n_u = cap_from_parts(ch_paths, config.S2_UNION_BUCKETS)
        write_atomic(capped, cpath)
        n_uncapped += n_u
        print(f"[{split}/{country}] union {n_u:,} -> capped {capped.height:,} pairs, checkpointed "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        del capped
        country_parts.append(cpath)
    cands = pl.concat([pl.read_parquet(p) for p in country_parts]).sort(CAP_KEYS, descending=CAP_DESC)
    return cands.select(list(pio.CANDIDATES_SCHEMA)).rechunk(), n_uncapped


def block(split: str, in_dir, verbose: bool = True, smoke: bool = False,
          s1_ids: pl.Series | None = None) -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    """(capped candidates, uncapped union, stats). `s1_ids` restricts Source-1 only."""
    long, stats = channel_pairs(split, in_dir, verbose, s1_ids=s1_ids, smoke=smoke)
    uncapped = union(long)
    return cap(uncapped), uncapped, stats


def train_entities(in_dir, n: int, smoke: bool) -> pl.Series | None:
    """The train Source-1 ids to block, or None for all (the no-op case, so output is
    exactly what an unsampled run gives). Prints per-country counts either way."""
    s1 = pl.read_parquet(config.norm_path("train", config.SOURCE1_SRC, in_dir), columns=["entity_id", "country"])
    ids = pio.sample_train_entities(s1, n, "train", smoke)
    held_out = pio.val_ids(smoke)  # printout only; the sampler looks it up itself
    before = s1.group_by("country").len(name="all")
    after = s1.filter(pl.col("entity_id").is_in(ids.implode())).group_by("country").len(name="blocked")
    print(f"train entities: {ids.len():,} of {s1.height:,} blocked (n={n:,}, "
          f"{0 if held_out is None else held_out.len():,} held-out always kept)")
    for country, n_all, n_blk in before.join(after, on="country", how="left").sort("country").iter_rows():
        print(f"  {country:<8} {n_blk or 0:>9,} of {n_all:>9,} ({(n_blk or 0) / n_all:.1%})")
    return None if ids.len() == s1.height else ids


def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    ap.add_argument("--train-entities", type=int, default=config.S2_TRAIN_ENTITIES, metavar="N",
                    help="train only: block the held-out validation entities plus N others, "
                         "stratified by country (0 = all; default: %(default)s)")
    ap.add_argument("--no-train-subsample", action="store_true", help="block every train entity (the default)")
    ap.add_argument("--splits", nargs="+", choices=list(config.SPLITS), default=list(config.SPLITS),
                    help="only block these splits (run train and test as separate processes)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse finished checkpoints under s2_parts/<split> if their manifest matches")
    args = ap.parse_args(argv)
    n_train = 0 if args.no_train_subsample else args.train_entities
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()
    for split in args.splits:
        if not config.norm_path(split, config.SOURCE1_SRC, in_dir).exists():
            print(f"  skipping {split} (norm_{split}_source1.parquet not found)")
            continue
        s1_ids = train_entities(in_dir, n_train, args.smoke) if split == "train" else None
        assert split == "train" or s1_ids is None, f"{split} must never be subsampled"
        cands, n_uncapped = block_checkpointed(split, in_dir, out_dir, s1_ids, args.smoke, args.resume)
        pio.check_schema(cands, pio.CANDIDATES_SCHEMA, f"candidates_{split}")
        out = config.candidates_path(split, out_dir)
        write_atomic(cands, out)
        n_s1 = cands["source1_entity_id"].n_unique()
        print(f"{out.name}: union {n_uncapped:,} pairs -> capped {cands.height:,} pairs over {n_s1:,} entities "
              f"(median {cands.group_by('source1_entity_id').len()['len'].median() or 0:.0f}/entity)")
    print(f"s2 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
