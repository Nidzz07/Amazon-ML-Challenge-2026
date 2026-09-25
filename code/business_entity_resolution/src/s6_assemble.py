"""S6 Assemble (owner: Nidhi, naive version): scored_{split} + candidates_{split}
-> matching_results.tsv, candidate_pairs.tsv.

Current logic:
  1. Uniqueness: each S2/S3 record keeps only its highest-probability claim.
     Ties go to the lower source1_entity_id so the output is deterministic.
  2. Selection: prob >= config.STUB_MATCH_THRESHOLD (placeholder; to be replaced
     by the expected-F0.5 prefix search).
Every Source-1 entity gets exactly one row, in source1 file order, with an empty
string when it has nothing.

Test output goes to output/ (artifacts/smoke/output/ with --smoke) under the exact
names the validator expects; other splits go to the artifacts dir with a _{split}
suffix. --output overrides both.

Usage:
    python s6_assemble.py [--smoke] [--input DIR] [--output DIR]
"""
import sys
import time

import polars as pl

import config
import pipeline_io as pio


def select_matches(scored: pl.DataFrame) -> pl.DataFrame:
    return (
        scored.sort(["candidate_entity_id", "prob", "source1_entity_id"], descending=[False, True, False])
        .unique("candidate_entity_id", keep="first", maintain_order=True)
        .filter(pl.col("prob") >= config.STUB_MATCH_THRESHOLD)
        .sort(["source1_entity_id", "prob", "candidate_entity_id"], descending=[False, True, False])
    )


def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()
    for split in config.SPLITS:
        dest = args.output or (
            (config.SMOKE_OUTPUT_DIR if args.smoke else config.OUTPUT_DIR) if split == "test" else out_dir
        )
        s1_ids = pl.read_parquet(config.records_path(split, config.SOURCE1_SRC, in_dir), columns=["entity_id"])["entity_id"]
        cands = pl.read_parquet(config.candidates_path(split, in_dir)).sort(["source1_entity_id", "best_rank"])
        matches = select_matches(pl.read_parquet(config.scored_path(split, in_dir)))

        orphans = matches.join(cands, on=["source1_entity_id", "candidate_entity_id"], how="anti").height
        assert orphans == 0, f"{orphans} matched pairs are not in the candidate set"
        assert matches["candidate_entity_id"].is_unique().all(), "uniqueness constraint violated"

        cand_path, match_path = config.candidate_pairs_path(split, dest), config.matching_results_path(split, dest)
        pio.write_id_lists(s1_ids, cands, "candidate_entity_ids", cand_path)
        pio.write_id_lists(s1_ids, matches, "matched_entity_ids", match_path)
        n_matched = matches["source1_entity_id"].n_unique()
        print(f"{split}: {len(s1_ids):,} entities, {cands.height:,} candidate pairs, "
              f"{matches.height:,} matches over {n_matched:,} entities -> {match_path.parent}")
    print(f"s6 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
