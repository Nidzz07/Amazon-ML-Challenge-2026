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
    python s6_assemble.py [--smoke] [--input DIR] [--output DIR]
"""
import sys
import time

import polars as pl

import assemble
import config
import pipeline_io as pio


def main(argv=None) -> None:
    args = pio.parser(__doc__).parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()
    for split in config.SPLITS:
        dest = args.output or (
            (config.SMOKE_OUTPUT_DIR if args.smoke else config.OUTPUT_DIR) if split == "test" else out_dir
        )
        s1_ids = pl.read_parquet(config.records_path(split, config.SOURCE1_SRC, in_dir), columns=["entity_id"])["entity_id"]
        cands = pl.read_parquet(config.candidates_path(split, in_dir)).sort(["source1_entity_id", "best_rank"])
        scored = pl.read_parquet(config.scored_path(split, in_dir))

        unique = assemble.enforce_uniqueness(scored)
        matches, decisions = assemble.prefix_search(unique)

        orphans = matches.join(cands, on=assemble.KEYS, how="anti").height
        assert orphans == 0, f"{orphans} matched pairs are not in the candidate set"
        assert matches["candidate_entity_id"].is_unique().all(), "uniqueness constraint violated"

        cand_path, match_path = config.candidate_pairs_path(split, dest), config.matching_results_path(split, dest)
        pio.write_id_lists(s1_ids, cands, "candidate_entity_ids", cand_path)
        pio.write_id_lists(s1_ids, matches, "matched_entity_ids", match_path)

        n = len(s1_ids)
        no_cands = n - decisions.height
        chose_empty = decisions.filter(pl.col("m") == 0).height
        m_pos = decisions.filter(pl.col("m") > 0)
        print(f"{split}: {n:,} entities | uniqueness stripped {scored.height - unique.height:,} of {scored.height:,} claims")
        print(f"  m=0: {no_cands + chose_empty:,} ({100 * (no_cands + chose_empty) / n:.2f}%) "
              f"= {no_cands:,} with no surviving candidates + {chose_empty:,} chose empty "
              f"[{config.ASSEMBLY_EMPTY_SCORE}]")
        print(f"  m>0: {m_pos.height:,} ({100 * m_pos.height / n:.2f}%), mean m {m_pos['m'].mean() or 0:.2f}, "
              f"{matches.height:,} matches -> {match_path.parent}")
    print(f"s6 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
