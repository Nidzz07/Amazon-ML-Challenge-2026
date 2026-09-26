"""Throwaway probe (not a pipeline stage): all-empty submission.

Every test Source-1 entity gets one row with an empty match list and an empty
candidate list. Scored on the leaderboard, this returns exactly the singleton rate
of the public test subset (roadmap, Submission strategy).

Writes output/matching_results.tsv and output/candidate_pairs.tsv: tab-separated,
UTF-8, unquoted, no index, source1 file order.

Usage:
    python probe_empty_submission.py
"""
import polars as pl

import config


def main() -> None:
    s1 = pl.read_csv(
        config.TEST_SOURCE1,
        separator=config.TSV_SEP,
        encoding="utf8",
        quote_char=None,
        infer_schema=False,
        columns=["entity_id"],
    )
    ids = s1["entity_id"]
    expected = config.EXPECTED_ROWS[("test", config.SOURCE1_SRC)]
    assert len(ids) == expected, f"test_source1 has {len(ids)} rows, expected {expected}"
    assert ids.n_unique() == len(ids), "duplicate entity_id in test_source1"

    for path, list_col in (
        (config.matching_results_path("test", config.OUTPUT_DIR), "matched_entity_ids"),
        (config.candidate_pairs_path("test", config.OUTPUT_DIR), "candidate_entity_ids"),
    ):
        out = pl.DataFrame({"source1_entity_id": ids, list_col: [""] * len(ids)})
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_csv(path, separator=config.TSV_SEP, quote_style="never", line_terminator="\n")
        print(f"wrote {out.height:,} rows -> {path}")


if __name__ == "__main__":
    main()
