"""S6 Assemble (owner: Nidhi): scored_{split} + candidates_{split}
-> matching_results.tsv, candidate_pairs.tsv.

Logic lives in assemble.py:
  1. hard uniqueness: each S2/S3 record keeps only its highest-probability claim;
  2. expected-F0.5 prefix search per entity, with m = 0 scored per
     config.ASSEMBLY_EMPTY_SCORE.
Every Source-1 entity gets exactly one row, in source1 file order. The row is an
empty string when it has nothing. Output is tab-separated, UTF-8, unquoted, with no
index. candidate_pairs.tsv is the capped S2 set that S5 scored.

Test output goes to output/ (artifacts/smoke/output/ with --smoke) under the exact
names the validator expects; other splits go to the artifacts dir with a _{split}
suffix. --output overrides both.

Usage:
    python s6_assemble.py [--smoke] [--input DIR] [--output DIR] [--splits train test]
    python s6_assemble.py --splits test     # the submission only (e.g. a laptop with no train files)

--splits defaults to every split, as before. Every split you ask for must have its
records, candidates and scored files: a missing one is an error before anything is
written (non-zero exit), never a silent skip that could hide a crashed s5.
"""
import sys
import time

import polars as pl

import assemble
import config
import pipeline_io as pio
import validate_submission_native

UNIQUENESS_PARTS = 8          # candidate-hash partitions for the uniqueness sort
CAND_SORT_ROWS = 4_000_000    # candidate rows per (entity, best_rank) sort slice


def load_candidates(path) -> pl.DataFrame:
    """The (entity, candidate) pairs of candidates_{split} in (source1_entity_id, best_rank) order,
    with only the columns s6 uses. s2 writes the file entity-contiguous (sorted by entity first),
    so it is sorted in slices cut at entity boundaries instead of one 52M-row multi-key sort;
    the sort is stable, so candidates tied on best_rank keep the file's order."""
    keys = ["source1_entity_id", "best_rank"]
    c = pl.read_parquet(path, columns=[*assemble.KEYS, "best_rank"])
    runs = c["source1_entity_id"].rle().struct.field("value")
    if runs.n_unique() != runs.len():  # not entity-contiguous: one sort (fine for small files)
        return c.sort(keys, maintain_order=True).select(assemble.KEYS)
    del runs
    run = c["source1_entity_id"].rle_id()
    out, start = [], 0
    while start < c.height:
        end = min(start + CAND_SORT_ROWS, c.height)
        if end < c.height:  # extend to the end of the entity straddling the cut
            end = start + int(run.slice(start).search_sorted(run[end - 1] + 1))
        out.append(c.slice(start, end - start).sort(keys, maintain_order=True).select(assemble.KEYS))
        start = end
    return pl.concat(out)


def main(argv=None) -> int:
    ap = pio.parser(__doc__)
    ap.add_argument("--splits", nargs="+", choices=list(config.SPLITS), default=list(config.SPLITS),
                    help="splits to assemble (default: all). A requested split with a missing input is an "
                         "error, never a skip.")
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)

    # Pre-flight for EVERY requested split before anything is written, so a missing scored file
    # (e.g. a crashed s5) can never end in exit 0 with no submission, or in a half-written run.
    missing = [
        (split, p) for split in args.splits
        for p in (config.records_path(split, config.SOURCE1_SRC, in_dir),
                  config.candidates_path(split, in_dir),
                  config.scored_path(split, in_dir))
        if not p.exists()
    ]
    if missing:
        lines = "\n".join(f"    {split:<6}{p}" for split, p in missing)
        raise SystemExit(
            f"s6_assemble: requested split(s) {', '.join(sorted({s for s, _ in missing}))} are missing inputs:\n"
            f"{lines}\n  Run the stage that produces them (s0 records / s2 candidates / s5 scored), or drop the "
            f"split from --splits. Nothing was written."
        )

    t0 = time.perf_counter()
    for split in args.splits:
        dest = args.output or (
            (config.SMOKE_OUTPUT_DIR if args.smoke else config.OUTPUT_DIR) if split == "test" else out_dir
        )
        s1_ids = pl.read_parquet(config.records_path(split, config.SOURCE1_SRC, in_dir), columns=["entity_id"])["entity_id"]
        # Scored first and freed before the candidates are loaded: at full test size (52M pairs)
        # holding both plus their sorts went past 7.5 GB.
        scored = pl.read_parquet(config.scored_path(split, in_dir))
        n_claims = scored.height
        unique = assemble.enforce_uniqueness_partitioned(scored, UNIQUENESS_PARTS)
        del scored
        n_unique = unique.height
        matches, decisions = assemble.prefix_search(unique)
        del unique

        cands = load_candidates(config.candidates_path(split, in_dir))
        # Orphans = matches with no candidate row. Hash the small side (matches), not 52M candidates.
        hit = cands.join(matches.select(assemble.KEYS), on=assemble.KEYS, how="semi")
        orphans = matches.join(hit, on=assemble.KEYS, how="anti").height
        del hit
        assert orphans == 0, f"{orphans} matched pairs are not in the candidate set"
        assert matches["candidate_entity_id"].is_unique().all(), "uniqueness constraint violated"

        cand_path, match_path = config.candidate_pairs_path(split, dest), config.matching_results_path(split, dest)
        pio.write_id_lists(s1_ids, cands, "candidate_entity_ids", cand_path)
        del cands
        pio.write_id_lists(s1_ids, matches, "matched_entity_ids", match_path)
        if split == "test":  # the only split that is ever a submission; a failure quarantines it
            validate_submission_native.gate(match_path, cand_path, smoke=args.smoke, quarantine=True)

        n = len(s1_ids)
        no_cands = n - decisions.height
        chose_empty = decisions.filter(pl.col("m") == 0).height
        m_pos = decisions.filter(pl.col("m") > 0)
        print(f"{split}: {n:,} entities | uniqueness stripped {n_claims - n_unique:,} of {n_claims:,} claims")
        print(f"  m=0: {no_cands + chose_empty:,} ({100 * (no_cands + chose_empty) / n:.2f}%) "
              f"= {no_cands:,} with no surviving candidates + {chose_empty:,} chose empty "
              f"[{config.ASSEMBLY_EMPTY_SCORE}]")
        print(f"  m>0: {m_pos.height:,} ({100 * m_pos.height / n:.2f}%), mean m {m_pos['m'].mean() or 0:.2f}, "
              f"{matches.height:,} matches -> {match_path.parent}")
    print(f"s6 done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
