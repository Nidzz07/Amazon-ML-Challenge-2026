"""S3 Featurise (owner: Krrish): norm + candidates_{split} → features_{split}.

Joins normalised Source-1 and Source-2/3 data onto candidate pairs, adds
context and competition columns, then calls features.featurise() to produce
the full feature matrix.

Output schema: source1_entity_id, candidate_entity_id, f000…fNNN float32,
[label uint8 on train splits only].

Sharding
--------
The test split is 1,732,544 Source-1 entities at config.MAX_CANDIDATES_PER_ENTITY
candidates each — about 52M pairs, which is far past what a materialised
(52M, 71) float32 matrix plus its Python-side working set would fit in. So this
stage never holds more than one chunk of features, following the same shape
blocking already uses (s2_block.load_shard, rare_token.run):

    for each country shard          # hard partition, as in s2_block
        load that country's norm rows and its slice of candidates
        compute the entity- and candidate-level aggregates ONCE per shard
        for each join block of config.S3_JOIN_BLOCK_ROWS pairs
            join the norm columns (and labels) onto the block
            for each chunk of config.S3_CHUNK_ROWS pairs
                featurise and append the chunk to the open ParquetWriter

Two nested sizes because they bound different things: the block bounds how
often the wide norm frames get hash-joined (once per block, not once per
chunk), the chunk bounds peak memory inside features.featurise().

Every group-level feature (entity_best_score, entity_n_cands, cand_best_score,
cand_n_claims) is aggregated over the whole country shard before any chunking,
so chunk boundaries cannot change a feature value. Sharding by country is exact
for the candidate-level aggregates because blocking only ever pairs records
within one country, so a candidate_entity_id appears in exactly one shard.

Usage:
    python s3_featurise.py [--smoke] [--input DIR] [--output DIR]
    python s3_featurise.py --smoke --chunk-rows 25000 --block-rows 500000
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

import config
import pipeline_io as pio
from features import EMBED_RANK_MISSING, FEATURE_NAMES, FEATURE_VERSION, NUM_FEATURES, featurise

try:  # optional: not in requirements.txt, only used for the memory line
    import psutil

    _PROC = psutil.Process()
except Exception:  # pragma: no cover
    _PROC = None

# ═══════════════════════════════════════════════════════════════════════
# Data loading and joining
# ═══════════════════════════════════════════════════════════════════════

# Columns from the normalised schema that feed into features.
# entity_id is used for the join key, country is not needed in features.
_NORM_COLS = [
    "entity_id", "name_norm", "name_roman", "name_tokens", "name_acronym",
    "addr_norm", "addr_roman", "addr_tokens",
    "street_num", "city_norm", "state_canon", "postcode",
    "has_addr", "script", "name_suffix",
]


def _rss_mb() -> float:
    return _PROC.memory_info().rss / (1024 * 1024) if _PROC is not None else 0.0


def split_inputs(split: str, in_dir) -> dict[str, Path]:
    """Every file this stage reads for `split`, keyed by a human label."""
    need = {f"norm_{split}_{src}": config.norm_path(split, src, in_dir)
            for src in (config.SOURCE1_SRC, *config.CANDIDATE_SRCS)}
    need[f"candidates_{split}"] = config.candidates_path(split, in_dir)
    if split == "train":
        need["records_train_ground_truth"] = config.records_path("train", "ground_truth", in_dir)
    return need


def missing_inputs(split: str, in_dir) -> dict[str, Path]:
    return {k: p for k, p in split_inputs(split, in_dir).items() if not p.exists()}


def _load_norm(split: str, src: str, in_dir, country: str | None = None) -> pl.DataFrame:
    """Load normalised data for one country, selecting only the columns we need.

    Gracefully handles a missing name_suffix column (in case the normaliser
    hasn't been updated yet) by filling it with empty strings.
    """
    lf = pl.scan_parquet(config.norm_path(split, src, in_dir))
    have = set(lf.collect_schema().names())
    if "name_suffix" not in have:
        lf = lf.with_columns(pl.lit("").alias("name_suffix"))
        have.add("name_suffix")
    if country is not None and "country" in have:
        lf = lf.filter(pl.col("country") == country)
    # Streaming: the country filter runs batch by batch, so only this country's rows are ever
    # accumulated. The in-memory engine decoded every row of the file first: loading India's
    # 4.7M-record pool (1.4 GB of data) peaked at +4.6 GB and pushed s3's India setup past 7.5 GB.
    return lf.select([c for c in _NORM_COLS if c in have]).collect(engine="streaming")


def _load_pool(split: str, in_dir, country: str | None = None) -> pl.DataFrame:
    """Concatenate Source-2 and Source-3 normalised data for one country."""
    return pl.concat([_load_norm(split, s, in_dir, country) for s in config.CANDIDATE_SRCS])


def shard_countries(split: str, in_dir) -> list[str]:
    """Country shards, taken from the data like s2_block does (never hard-coded)."""
    lf = pl.scan_parquet(config.norm_path(split, config.SOURCE1_SRC, in_dir))
    if "country" not in lf.collect_schema().names():
        return [None]
    return lf.select(pl.col("country").unique().sort()).collect()["country"].to_list()


def _true_pairs(in_dir) -> pl.LazyFrame:
    """Ground-truth as long (source1_entity_id, candidate_entity_id, label=1)."""
    return (
        pl.scan_parquet(config.records_path("train", "ground_truth", in_dir))
        .select(
            "source1_entity_id",
            pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id"),
        )
        .explode("candidate_entity_id", empty_as_null=False)
        .filter(pl.col("candidate_entity_id") != "")
        .with_columns(pl.lit(1, dtype=pl.UInt8).alias("label"))
    )


def _prefix_cols(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Rename all columns (except entity_id) with a prefix."""
    return df.rename(
        {c: f"{prefix}_{c}" for c in df.columns if c != "entity_id"}
    )


def _cand_competition(cands: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Candidate-level competition aggregates (cand_best_score, cand_n_claims). Must be computed
    over EVERY entity in the country shard, never one bucket: at inference every entity
    competes, and a per-bucket count would shrink cand_n_claims and inflate cand_is_argmax."""
    return (
        cands.lazy().group_by("candidate_entity_id").agg(
            pl.col("prior_score").max().alias("cand_best_score"),
            pl.len().cast(pl.Float32).alias("cand_n_claims"),
        ).collect(engine="streaming")
    )


def _add_context_and_competition(cands: pl.DataFrame, cand_comp: pl.DataFrame | None = None) -> pl.DataFrame:
    """Add context features (entity-level) and competition features
    (candidate-level) as new columns on the candidates DataFrame.

    Context: entity_best_score, entity_n_cands
    Competition: cand_best_score, cand_n_claims
    Derived: is_source3

    Called once per country shard, before chunking, so every aggregate covers
    the entity's / candidate's complete set of rows.
    """
    # Entity-level context
    entity_ctx = cands.group_by("source1_entity_id").agg(
        pl.col("prior_score").max().alias("entity_best_score"),
        pl.len().cast(pl.Float32).alias("entity_n_cands"),
    )

    # Competition: for each candidate, its best score across all entities. Callers that only
    # hold one bucket of a shard pass the shard-wide table in.
    if cand_comp is None:
        cand_comp = _cand_competition(cands)

    return (
        cands
        .join(entity_ctx, on="source1_entity_id", how="left")
        .join(cand_comp, on="candidate_entity_id", how="left")
        .with_columns(
            pl.col("candidate_entity_id").str.starts_with("S3-")
            .cast(pl.Float32).alias("is_source3"),
        )
    )


def _load_embed_scores(split: str, where) -> pl.LazyFrame | None:
    """Lazy handle on the precomputed embedding scores for `split`; nothing is read until a shard collects
    it (see _embed_for_shard). The full-scale file has ~79M rows, and reading it whole cost several GB per split.

    `where` is the embeddings file itself or a directory holding config.EMBED_ANN_FILENAME. featurise_split
    passes config.embed_ann_path(smoke) — the same resolver the s2 embed_ann channel uses — so the two stages
    always read the same file. (Resolving it from s3's --input dir instead made s2 and s3 read different
    places whenever --input was not artifacts/, e.g. the D: layout.)"""
    embed_path = Path(where)
    if embed_path.is_dir():
        embed_path = embed_path / config.EMBED_ANN_FILENAME
    if not embed_path.exists():
        return None
    lf = pl.scan_parquet(embed_path)
    names = lf.collect_schema().names()
    if "split" in names:  # older files without the column hold a single split
        lf = lf.filter(pl.col("split") == split)
    keep = ["source1_entity_id", "candidate_entity_id",
            pl.col("channel_score").alias("embed_cosine"),
            pl.col("channel_rank").cast(pl.Float32).alias("embed_rank")]
    if "country" in names:
        keep.append("country")
    return lf.select(keep)


def _embed_for_shard(embed_all: pl.LazyFrame | None, country: str, s1_ids: pl.Series) -> pl.DataFrame | None:
    """This shard's embedding rows only. The file is written per (split, country), so the country filter lets
    parquet skip every other shard's row groups; the id filter is the same restriction the old code applied."""
    if embed_all is None:
        return None
    lf = embed_all
    if "country" in lf.collect_schema().names():
        lf = lf.filter(pl.col("country") == country).drop("country")
    return lf.filter(pl.col("source1_entity_id").is_in(s1_ids.implode())).collect(engine="streaming")



def _join_block(
    block: pl.DataFrame,
    s1_norm: pl.DataFrame,
    pool_norm: pl.DataFrame,
    embed: pl.DataFrame | None,
    labels: pl.DataFrame | None,
) -> pl.DataFrame:
    """Attach both sides' normalised columns (+ embeddings, + labels) to a block
    of candidate pairs. One hash join per wide frame per block."""
    out = (
        block
        .join(s1_norm, left_on="source1_entity_id", right_on="entity_id", how="left")
        .join(pool_norm, left_on="candidate_entity_id", right_on="entity_id", how="left")
    )
    if embed is not None:
        out = out.join(embed, on=["source1_entity_id", "candidate_entity_id"], how="left")
    out = out.with_columns(
        pl.col("embed_cosine").fill_null(0.0).cast(pl.Float32) if "embed_cosine" in out.columns
        else pl.lit(0.0, dtype=pl.Float32).alias("embed_cosine"),
        pl.col("embed_rank").fill_null(EMBED_RANK_MISSING).cast(pl.Float32) if "embed_rank" in out.columns
        else pl.lit(EMBED_RANK_MISSING, dtype=pl.Float32).alias("embed_rank"),
    )
    if labels is not None:
        out = out.join(labels, on=["source1_entity_id", "candidate_entity_id"], how="left") \
                 .with_columns(pl.col("label").fill_null(0).cast(pl.UInt8))
    return out


def out_schema(split: str) -> dict:
    """The exact features_{split}.parquet schema, fixed before the first chunk so
    an empty leading shard cannot set a different one."""
    schema = {"source1_entity_id": pl.String, "candidate_entity_id": pl.String}
    schema |= {c: pl.Float32 for c in pio.feature_columns(NUM_FEATURES)}
    if split == "train":
        schema["label"] = pl.UInt8
    return schema


def featurise_chunk(pairs: pl.DataFrame, split: str, fcols: list[str]) -> pl.DataFrame:
    """One chunk of joined pairs → one chunk of the output schema."""
    arr = featurise(pairs)
    cols = {
        "source1_entity_id": pairs["source1_entity_id"],
        "candidate_entity_id": pairs["candidate_entity_id"],
    }
    cols |= {c: arr[:, i] for i, c in enumerate(fcols)}
    if split == "train":
        cols["label"] = pairs["label"]
    return pl.DataFrame(cols, schema=out_schema(split))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _featurise_country(split: str, in_dir, country, embed_all, labels_all, write, *,
                       chunk_rows: int, block_rows: int, fcols: list[str], schema: dict,
                       name: str, verbose: bool, fraction: float = 1.0) -> dict:
    """Featurise one country shard, handing each finished chunk (a pyarrow Table) to `write`.

    Memory: the country's Source-1 and pool norm frames are loaded once (any candidate can be any
    pool record). Candidate-level aggregates are computed once over the whole shard with a
    streaming group_by. Source-1 entities are then walked in config.S3_S1_BUCKETS hash buckets,
    and only one bucket's candidates, entity aggregates, embedding rows and labels are held at a
    time. Feature values are exactly as when the whole shard was held: entity aggregates are per
    entity, and candidate aggregates still cover every entity of the shard."""
    ts = time.perf_counter()
    s1_norm = _prefix_cols(_load_norm(split, config.SOURCE1_SRC, in_dir, country), "s1")
    s1_ids = s1_norm["entity_id"]
    cand_lf = pl.scan_parquet(config.candidates_path(split, in_dir)).filter(
        pl.col("source1_entity_id").is_in(s1_ids.implode()))
    if cand_lf.select(pl.len()).collect(engine="streaming").item() == 0:
        if verbose:
            print(f"  [{split}/{country}] 0 candidate pairs, skipped", flush=True)
        return {"rows": 0, "positives": 0, "peak_rss_mb": _rss_mb()}
    # Candidate-level aggregates over EVERY entity of the country, before any sampling below and
    # before the pool is loaded (so the two largest transient allocations never overlap).
    cand_comp = _cand_competition(cand_lf)
    pool_norm = _prefix_cols(_load_pool(split, in_dir, country), "cand")

    n_all = s1_ids.len()
    if fraction < 1.0:
        # Train-only entity sample, AFTER the shard-wide aggregates: a sampled entity's features
        # are exactly what a full run gives it (cand_n_claims etc. still count every entity).
        # An independent hash (seed=1) so the sample is exact to 0.1% and is not tied to the
        # processing buckets below; it is a random sample, so the country mix is preserved.
        keep = (s1_ids.hash(seed=1) % 1000) < round(fraction * 1000)
        s1_ids = s1_ids.filter(keep)
    n_b = max(1, config.S3_S1_BUCKETS)
    bucket = (s1_ids.hash(seed=0) % n_b).to_numpy()
    rows = positives = 0
    peak = _rss_mb()
    for b in range(n_b):
        ids_b = s1_ids.filter(pl.Series(bucket == b))
        if ids_b.is_empty():
            continue
        cands = cand_lf.filter(pl.col("source1_entity_id").is_in(ids_b.implode())).collect(engine="streaming")
        if cands.is_empty():
            continue
        cands = _add_context_and_competition(cands, cand_comp)
        embed = _embed_for_shard(embed_all, country, ids_b)
        labels = (
            labels_all.filter(pl.col("source1_entity_id").is_in(ids_b.implode()))
            if labels_all is not None else None
        )
        for block in cands.iter_slices(block_rows):
            joined = _join_block(block, s1_norm, pool_norm, embed, labels)
            for chunk in joined.iter_slices(chunk_rows):
                out = featurise_chunk(chunk, split, fcols)
                pio.check_schema(out, schema, name)
                write(out.to_arrow())
                rows += out.height
                if split == "train":
                    positives += int(out["label"].sum())
                peak = max(peak, _rss_mb())
                del out
            del joined
        del cands, embed, labels
    if verbose:
        sec = time.perf_counter() - ts
        print(f"  [{split}/{country}] {rows:>12,} pairs  {sec:7.1f}s  {rows / max(sec, 1e-9):>9,.0f} rows/s"
              + (f"  {positives:>9,} positives" if split == "train" else "")
              + (f"  {s1_ids.len():,} of {n_all:,} entities (fraction {fraction:g})" if fraction < 1.0 else "")
              + f"  peak {peak:,.0f} MB", flush=True)
    return {"rows": rows, "positives": positives, "peak_rss_mb": peak, "entities": s1_ids.len()}


def _embed_guard(split: str, in_dir, smoke: bool):
    """(embeddings LazyFrame or None, its path). Refuses, before anything is written, when the
    split has embed_ann-channel candidates but there is no embeddings file."""
    embed_path = config.embed_ann_path(smoke)
    embed_all = _load_embed_scores(split, embed_path)
    if embed_all is None:
        # Checked for the whole split before anything is written: a per-shard check would leave a
        # valid-looking features file holding only the shards that ran before it fired.
        n_ann = (
            pl.scan_parquet(config.candidates_path(split, in_dir))
            .filter((pl.col("channels") & (1 << config.CHANNELS.index("embed_ann"))) > 0)
            .select(pl.len()).collect(engine="streaming").item()
        )
        if n_ann:
            raise SystemExit(
                f"s3_featurise: {n_ann:,} {split} candidates came from the embed_ann channel, but there is no "
                f"embeddings file at {embed_path}. Featurising now would write ch_embed_ann = 1 beside "
                f"embed_cosine = 0 / embed_rank = {EMBED_RANK_MISSING:g}. Put the file s2 used at that path "
                f"(config.EMBED_ANN_PATH) and re-run. Nothing was written."
            )
    return embed_all, embed_path


def _country_part(split: str, in_dir, country, part: Path, *, chunk_rows, block_rows, smoke, verbose,
                  fraction: float = 1.0) -> dict:
    """Child-process entry: one country into `part`. A plain write: the parent owns atomicity."""
    embed_all, _ = _embed_guard(split, in_dir, smoke)
    labels_all = _true_pairs(in_dir).collect() if split == "train" else None
    fcols, schema = pio.feature_columns(NUM_FEATURES), out_schema(split)
    holder = {}

    def write(table):
        if "w" not in holder:
            holder["w"] = pq.ParquetWriter(part, table.schema)
        holder["w"].write_table(table)

    try:
        return _featurise_country(split, in_dir, country, embed_all, labels_all, write,
                                  chunk_rows=chunk_rows, block_rows=block_rows, fcols=fcols,
                                  schema=schema, name=part.name, verbose=verbose, fraction=fraction)
    finally:
        if "w" in holder:
            holder["w"].close()


def _child_env() -> dict:
    """Environment for the per-country child processes. polars on Windows allocates through
    mimalloc, which by default keeps freed memory reserved instead of returning it to the OS, so a
    process's RSS tracks its high-water mark. Measured on test India's setup (16 GB box): loading a
    4.7M-record pool that is 1.4 GB of data held 4.9 GB by default and 2.8 GB with purge delay 0;
    the whole setup peaked 5.0 GB instead of passing 7.2 GB. The values are only defaults: anything
    already set in the environment wins. mimalloc reads them at process start, hence children."""
    env = {**os.environ, "PYTHONUTF8": "1"}
    env.setdefault("MIMALLOC_PURGE_DELAY", "0")   # mimalloc v2
    env.setdefault("MIMALLOC_RESET_DELAY", "0")   # the same knob's older name
    return env


def featurise_split(
    split: str,
    in_dir,
    out_dir,
    chunk_rows: int = None,
    block_rows: int = None,
    verbose: bool = True,
    smoke: bool = False,
    isolate: bool | None = None,
    fraction: float = 1.0,
) -> dict:
    """Shard `split` by country, bucket and chunk within country, stream to features_{split}.

    With isolate (default config.S3_COUNTRY_PROCESSES) each country runs in its own child
    process and writes a part, and the parts are then streamed into the output, so nothing one
    country allocated is still held when the next one loads (in one process, France's retained
    memory plus India's load passed 7.5 GB on the 16 GB box).

    Returns {rows, positives, sec, peak_rss_mb}.
    """
    if fraction < 1.0 and split != "train":
        raise SystemExit(f"s3_featurise: --bucket-fraction samples TRAIN entities only; {split} is always featurised in full.")
    if not 0.0 < fraction <= 1.0:
        raise SystemExit(f"s3_featurise: --bucket-fraction must be in (0, 1], got {fraction}")
    chunk_rows = chunk_rows or config.S3_CHUNK_ROWS
    block_rows = max(block_rows or config.S3_JOIN_BLOCK_ROWS, chunk_rows)
    isolate = config.S3_COUNTRY_PROCESSES if isolate is None else isolate
    fcols = pio.feature_columns(NUM_FEATURES)
    schema = out_schema(split)
    out_path = config.features_path(split, out_dir)

    embed_all, embed_path = _embed_guard(split, in_dir, smoke)
    if verbose:
        print(f"  embed_ann: {f'joined from {embed_path}' if embed_all is not None else f'no file at {embed_path}: embed_cosine = 0, embed_rank = {EMBED_RANK_MISSING:g} (missing)'}")
        print(f"  mode: {'one child process per country' if isolate else 'in-process'}, "
              f"{config.S3_S1_BUCKETS} Source-1 buckets per country"
              + (f", TRAIN SAMPLE: {fraction:g} of entities (aggregates over all)" if fraction < 1.0 else ""), flush=True)

    # Atomic output: everything streams into <name>.partial, which is renamed over the real name
    # only after the last row is written and the writer closed. A crash (or a guard kill) can
    # never leave a valid-looking features file holding only the shards that ran. The previous
    # run's file goes first, so a failed re-run leaves nothing for s5 to score rather than a
    # complete-looking file built from an older candidate set.
    tmp_path = out_path.with_name(out_path.name + ".partial")
    out_path.unlink(missing_ok=True)
    tmp_path.unlink(missing_ok=True)
    countries = shard_countries(split, in_dir)
    parts = [out_path.with_name(f"{out_path.name}.{c}.part") for c in countries]
    for pth in parts:
        pth.unlink(missing_ok=True)
        pth.with_suffix(".json").unlink(missing_ok=True)

    t0 = time.perf_counter()
    rows = positives = 0
    peak = _rss_mb()
    ok = False
    try:
        if isolate:
            for country, part in zip(countries, parts):
                cmd = [sys.executable, str(Path(__file__).resolve()), "--splits", split,
                       "--input", str(in_dir), "--output", str(out_dir),
                       "--chunk-rows", str(chunk_rows), "--block-rows", str(block_rows),
                       "--_country", str(country), "--_part", str(part), "--bucket-fraction", str(fraction)]
                if smoke:
                    cmd.append("--smoke")
                rc = subprocess.run(cmd, env=_child_env()).returncode
                if rc != 0:
                    raise SystemExit(f"s3_featurise: the {split}/{country} child process failed (exit {rc}). "
                                     f"Nothing was written.")
                st = json.loads(part.with_suffix(".json").read_text(encoding=config.ENCODING))
                rows += st["rows"]
                positives += st["positives"]
                peak = max(peak, st["peak_rss_mb"])
            written = [p for p in parts if p.exists()]
            if written:
                pl.scan_parquet(written).sink_parquet(tmp_path)
        else:
            holder = {}

            def write(table):
                if "w" not in holder:
                    holder["w"] = pq.ParquetWriter(tmp_path, table.schema)
                holder["w"].write_table(table)

            labels_all = _true_pairs(in_dir).collect() if split == "train" else None
            try:
                for country in countries:
                    st = _featurise_country(split, in_dir, country, embed_all, labels_all, write,
                                            chunk_rows=chunk_rows, block_rows=block_rows, fcols=fcols,
                                            schema=schema, name=out_path.name, verbose=verbose, fraction=fraction)
                    rows += st["rows"]
                    positives += st["positives"]
                    peak = max(peak, st["peak_rss_mb"])
            finally:
                if "w" in holder:
                    holder["w"].close()
        if not tmp_path.exists():  # no shard produced a row: still emit a schema-correct file
            pl.DataFrame(schema=schema).write_parquet(tmp_path)
        ok = True
    finally:
        for pth in parts:
            pth.unlink(missing_ok=True)
            pth.with_suffix(".json").unlink(missing_ok=True)
        if ok:
            tmp_path.replace(out_path)  # atomic on one volume: complete, or not there at all
        else:
            tmp_path.unlink(missing_ok=True)

    return {"rows": rows, "positives": positives, "sec": time.perf_counter() - t0, "peak_rss_mb": peak}


def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    ap.add_argument("--chunk-rows", type=int, default=config.S3_CHUNK_ROWS,
                    help=f"pairs per featurise chunk (default {config.S3_CHUNK_ROWS:,})")
    ap.add_argument("--block-rows", type=int, default=config.S3_JOIN_BLOCK_ROWS,
                    help=f"pairs per norm-join block (default {config.S3_JOIN_BLOCK_ROWS:,})")
    ap.add_argument("--splits", nargs="+", default=list(config.SPLITS),
                    help="only featurise these splits")
    ap.add_argument("--bucket-fraction", type=float, default=1.0,
                    help="TRAIN ONLY: featurise this fraction of Source-1 entities (hash sample, taken after "
                         "the per-candidate aggregates are computed over every entity). Default 1 = all.")
    ap.add_argument("--_country", help=argparse.SUPPRESS)  # child mode: one country ...
    ap.add_argument("--_part", type=Path, help=argparse.SUPPRESS)  # ... into this part file
    args = ap.parse_args(argv)
    if args._country is not None:
        in_dir, _ = pio.dirs(args)
        st = _country_part(args.splits[0], in_dir, args._country, args._part,
                           chunk_rows=args.chunk_rows, block_rows=args.block_rows,
                           smoke=args.smoke, verbose=True, fraction=args.bucket_fraction)
        args._part.with_suffix(".json").write_text(json.dumps(st), encoding=config.ENCODING)
        return 0
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    print(f"Feature spec: {NUM_FEATURES} features, version {FEATURE_VERSION}")
    print(f"  Names: {', '.join(FEATURE_NAMES[:5])} … {', '.join(FEATURE_NAMES[-3:])}")
    print(f"  Shards: by country; blocks of {args.block_rows:,} pairs, chunks of {args.chunk_rows:,}")

    ran = []
    for split in args.splits:
        missing = missing_inputs(split, in_dir)
        if missing:
            names = "\n".join(f"    {k:<28} {p}" for k, p in missing.items())
            if len(missing) == len(split_inputs(split, in_dir)):
                # Nothing for this split exists at all — the normal state for test
                # before s1/s2 have been run on it. Say so and move on.
                print(f"\n[{split}] SKIPPED — none of this split's inputs exist in {in_dir}:\n{names}")
                print(f"    Run: python s0_ingest.py && python s1_normalise.py && python s2_block.py"
                      f"{' --smoke' if args.smoke else ''}")
                continue
            raise SystemExit(
                f"\ns3_featurise: {split} is partially prepared — {len(missing)} of "
                f"{len(split_inputs(split, in_dir))} input files are missing:\n{names}\n"
                f"  Re-run the stages that produce them (s0_ingest -> s1_normalise -> s2_block"
                f"{' --smoke' if args.smoke else ''}) before s3.\n"
                f"  Refusing to write a partial features_{split}.parquet."
            )

        print(f"\n[{split}]")
        st = featurise_split(split, in_dir, out_dir, args.chunk_rows, args.block_rows, smoke=args.smoke,
                             fraction=args.bucket_fraction)
        ran.append(split)
        out = config.features_path(split, out_dir)
        pos = f", {st['positives']:,} positives" if split == "train" else ""
        print(f"  {out.name:<24} {st['rows']:>12,} rows × {NUM_FEATURES} features{pos}")
        per_m = st["rows"] / 1e6
        print(f"  {st['sec']:.1f}s  {st['rows'] / max(st['sec'], 1e-9):,.0f} rows/s"
              + (f"  peak RSS {st['peak_rss_mb']:,.0f} MB"
                 f"  ({st['peak_rss_mb'] / per_m:,.0f} MB per 1M pairs)" if _PROC and per_m else ""))

    if not ran:
        raise SystemExit(
            f"s3_featurise: no split had usable inputs in {in_dir}. Nothing was written."
        )
    print(f"\ns3 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
