"""S0 Ingest: raw TSV -> records_{split}_{src}.parquet, every column kept as string.

Usage:
    python s0_ingest.py            # full dataset, asserts exact row counts
    python s0_ingest.py --smoke    # artifacts/smoke/smoke_*.tsv -> artifacts/smoke/, no row-count assertion
    python s0_ingest.py --smoke --input DIR --output DIR   # override raw-TSV and parquet dirs
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

import config


def scan_tsv(path: Path) -> pl.LazyFrame:
    # Files are pandas-style quoted ("""x" -> "x), so keep '"' as the quote char.
    # Empty fields stay "" rather than null: an empty address is data, not missing.
    return pl.scan_csv(
        path,
        separator=config.TSV_SEP,
        has_header=True,
        quote_char='"',
        infer_schema=False,
        encoding="utf8",
        empty_string_is_null=False,
    )


def expected_columns(src: str) -> tuple:
    return config.GROUND_TRUTH_COLUMNS if src == "ground_truth" else config.SOURCE_COLUMNS


def ingest_file(split: str, src: str, raw: Path, out: Path, expected_rows: int | None = None) -> dict:
    t0 = time.perf_counter()
    lf = scan_tsv(raw)

    cols = tuple(lf.collect_schema().names())
    want = expected_columns(src)
    assert cols == want, f"{raw.name}: columns {cols} != expected {want}"

    out.parent.mkdir(parents=True, exist_ok=True)
    lf.select([pl.col(c).fill_null("").cast(pl.String) for c in want]).sink_parquet(out)

    written = pl.scan_parquet(out)
    id_col = want[0]
    stats = written.select(
        pl.len().alias("rows"),
        pl.col(id_col).n_unique().alias("unique_ids"),
        (pl.col(want[-1]) == "").sum().alias("empty_last_col"),
    ).collect().row(0, named=True)

    rows = stats["rows"]
    if expected_rows is not None:
        assert rows == expected_rows, f"{raw.name}: {rows:,} rows != expected {expected_rows:,}"
    assert stats["unique_ids"] == rows, f"{raw.name}: {rows - stats['unique_ids']:,} duplicate {id_col}"

    if src != "ground_truth":
        prefix = f"S{src[-1]}-"
        bad = written.filter(~pl.col("entity_id").str.starts_with(prefix)).select(pl.len()).collect().item()
        assert bad == 0, f"{raw.name}: {bad:,} entity_ids without prefix {prefix}"
        empty_addr = written.filter(pl.col("business_address") == "").select(pl.len()).collect().item()
    else:
        empty_addr = None

    return {
        "split": split,
        "src": src,
        "rows": rows,
        "expected": expected_rows,
        "empty_addr": empty_addr,
        "singletons": stats["empty_last_col"] if src == "ground_truth" else None,
        "mb": out.stat().st_size / 1e6,
        "sec": time.perf_counter() - t0,
    }


def print_summary(results: list[dict]) -> None:
    hdr = f"{'split':<6} {'src':<13} {'rows':>11} {'expected':>11} {'ok':>3} {'empty_addr':>10} {'singletons':>10} {'MB':>7} {'sec':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        exp = f"{r['expected']:,}" if r["expected"] is not None else "-"
        ok = "-" if r["expected"] is None else ("yes" if r["rows"] == r["expected"] else "NO")
        ea = f"{r['empty_addr']:,}" if r["empty_addr"] is not None else "-"
        sg = f"{r['singletons']:,}" if r["singletons"] is not None else "-"
        print(f"{r['split']:<6} {r['src']:<13} {r['rows']:>11,} {exp:>11} {ok:>3} {ea:>10} {sg:>10} {r['mb']:>7.1f} {r['sec']:>6.1f}")


def run(dataset_dir: Path, artifacts_dir: Path, check_counts: bool, raw_path_fn=config.raw_path) -> list[dict]:
    results = []
    for split, srcs in config.SPLITS.items():
        for src in srcs:
            raw = raw_path_fn(split, src, dataset_dir)
            out = config.records_path(split, src, artifacts_dir)
            expected = config.EXPECTED_ROWS[(split, src)] if check_counts else None
            print(f"ingesting {raw.name} ...", flush=True)
            results.append(ingest_file(split, src, raw, out, expected))
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="read artifacts/smoke/smoke_*.tsv instead of data/dataset/")
    ap.add_argument("--input", type=Path, help="override the raw-TSV directory")
    ap.add_argument("--output", type=Path, help="override the parquet output directory")
    args = ap.parse_args(argv)
    dataset_dir = args.input or (config.SMOKE_DIR if args.smoke else config.DATASET_DIR)
    artifacts_dir = args.output or (config.SMOKE_ARTIFACTS_DIR if args.smoke else config.ARTIFACTS_DIR)
    raw_path_fn = config.smoke_raw_path if args.smoke else config.raw_path
    t0 = time.perf_counter()
    results = run(dataset_dir, artifacts_dir, check_counts=not args.smoke, raw_path_fn=raw_path_fn)
    print()
    print_summary(results)
    print(f"\ntotal {time.perf_counter() - t0:.1f}s -> {artifacts_dir}")


if __name__ == "__main__":
    sys.exit(main())
