"""S1 Normalise (owner: Parth; taken over by Nidhi): records_{split}_{src} -> norm_{split}_{src}.

All text logic lives in normalise.py and translit.py. This stage only does I/O: it
reads each records file in slices of config.S1_CHUNK_ROWS, normalises each slice, and
streams it into the output Parquet. The output schema is fixed by PROJECT_ROADMAP.md.
One translit.Romaniser (and so one token memo) is shared across every file in the run.

Usage:
    python s1_normalise.py [--smoke] [--input DIR] [--output DIR]
"""
import sys
import time

import polars as pl
import pyarrow.parquet as pq

import config
import normalise
import pipeline_io as pio
import translit


def normalise_file(src_path, out_path, romaniser: translit.Romaniser) -> int:
    records = pl.read_parquet(src_path)
    writer, rows = None, 0
    try:
        for chunk in records.iter_slices(config.S1_CHUNK_ROWS):
            out = normalise.normalise_frame(chunk, romaniser)
            pio.check_schema(out, pio.NORM_SCHEMA, out_path.name)
            table = out.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            rows += out.height
        if writer is None:  # empty input still gets a schema-correct file
            pl.DataFrame(schema=pio.NORM_SCHEMA).write_parquet(out_path)
    finally:
        if writer is not None:
            writer.close()
    assert rows == records.height, f"{out_path.name}: {rows} rows out, {records.height} in"
    return rows


def missing_inputs(in_dir) -> dict[str, list]:
    """{split: [missing records_{split}_{src}.parquet paths]} for every split with a gap."""
    gaps = {}
    for split, srcs in config.SPLITS.items():
        miss = [config.records_path(split, s, in_dir) for s in srcs
                if s != "ground_truth" and not config.records_path(split, s, in_dir).exists()]
        if miss:
            gaps[split] = miss
    return gaps


def main(argv=None) -> int:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)

    # Pre-flight, BEFORE anything is written. Without it s1 wrote all three norm_train_*
    # files and then died inside polars on records_test_source1.parquet, leaving a
    # directory where train looks finished and test is absent — and s2_block --splits
    # train runs happily on that. Same rule as s3_featurise: a split with SOME inputs
    # missing is an error; a split with ALL of them missing is skipped, loudly.
    gaps = missing_inputs(in_dir)
    n_srcs = {sp: sum(s != "ground_truth" for s in srcs) for sp, srcs in config.SPLITS.items()}
    partial = {sp: m for sp, m in gaps.items() if len(m) < n_srcs[sp]}
    if partial:
        lines = "\n".join(f"    {p}" for m in partial.values() for p in m)
        raise SystemExit(
            f"s1_normalise: split(s) {', '.join(partial)} partially ingested — these records files "
            f"are missing:\n{lines}\n  Re-run s0_ingest{' --smoke' if args.smoke else ''} first. "
            f"Nothing was written."
        )
    skip = set(gaps)
    if skip == set(config.SPLITS):
        raise SystemExit(f"s1_normalise: no records_*.parquet in {in_dir} — run s0_ingest first. Nothing was written.")
    for sp in sorted(skip):
        print(f"SKIPPING {sp}: none of its records files exist in {in_dir} — norm_{sp}_* will NOT be written.")

    romaniser = translit.Romaniser()
    t0 = time.perf_counter()
    total = 0
    for split, srcs in config.SPLITS.items():
        if split in skip:
            continue
        for src in srcs:
            if src == "ground_truth":
                continue
            out = config.norm_path(split, src, out_dir)
            t = time.perf_counter()
            fields_before = romaniser.fields
            rows = normalise_file(config.records_path(split, src, in_dir), out, romaniser)
            total += rows
            print(f"{out.name:<32} {rows:>10,} rows  {romaniser.fields - fields_before:>9,} Indic fields  "
                  f"{time.perf_counter() - t:6.1f}s")
    sec = time.perf_counter() - t0
    print(f"romaniser: {romaniser.fields:,} fields, {romaniser.hits + romaniser.misses:,} run lookups, "
          f"hit rate {romaniser.hit_rate:.4f}, {romaniser.cache_size:,} cached runs")
    print(f"s1 done in {sec:.1f}s ({total:,} records, {total / sec if sec else 0:,.0f} records/s)")
    if skip:
        print(f"INCOMPLETE — split(s) {', '.join(sorted(skip))} skipped (see above). Exiting 1 so a "
              f"chained `s1 && s2` stops here.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
