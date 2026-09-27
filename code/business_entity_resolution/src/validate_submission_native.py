"""Pre-upload submission gate: nothing is a submission until this passes.

Two layers, both must pass:

1. NATIVE CHECKS (this module). Stops at the FIRST violation and names it: the rule,
   the file, the line, the offending value, and how to fix it. Every rule is tagged
     [scorer]  the organisers' validator / scorer rejects a file that breaks it
     [ours]    stricter than the scorer; a break means a pipeline bug, not a rejection
2. OUTER GATE: the organisers' own student_resource/utils/validate_submission.py
   (config.VALIDATOR), run unmodified in a subprocess against a test dir whose files
   carry the names it hard-codes (test_source1/2/3.tsv). It checks matching_results.tsv
   only by default: given the ~52M-ID candidate file it passes 7.5 GB (its own docstring
   says to drop --candidate then), and layer 1 checks that file. --organiser-candidate
   adds it back. Optional: if the script is absent the gate says so loudly.

Measured at full size (1,732,544 rows, 51,975,919 candidate IDs): layer 1 47.5 s, peak
3.7 GB; layer 2 11.4 s, peak 1.4 GB.

Rules, applied to matching_results.tsv and (same rules, own header) candidate_pairs.tsv:
  file     [scorer] exists, non-empty, strict UTF-8
           [scorer] no byte-order mark (it would corrupt the header's first column)
           [ours]   LF line endings. The organisers' validator tolerates CRLF (Python's
                    universal newlines strip it), but pio.write_id_lists only writes LF, so
                    a CR means the file was re-saved by something else (e.g. a Windows
                    editor) and the scorer's own parser is unknown
  header   [scorer] exactly `source1_entity_id<TAB>matched_entity_ids`; comma-separated
                    files and a leading pandas index column get their own messages
  quoting  [scorer] no '"' anywhere (IDs never contain one, so any quote is a writer bug)
  row      [scorer] exactly one TAB per row
           [scorer] query ID has the S1- prefix, no surrounding whitespace
           [scorer] no S1 entity on two rows
  list     [scorer] no-match is the EMPTY string; literal None/nan/null/[] rejected by name
           [scorer] no empty element (",," or a trailing comma), no whitespace in an ID
           [scorer] every ID is S2-/S3- (an S1- ID is a self-match)
           [scorer] no ID twice in one list
           [ours]   an all-whitespace list must be truly empty
  set      [scorer] every query ID exists in test_source1
           [ours]   every S2/S3 ID exists in test_source2/3 (the organisers only warn:
                    a nonexistent ID just scores zero)
           [ours]   every matched ID is in that entity's candidate list (organisers: warn)
           [scorer] COMPLETENESS — every test_source1 entity has a row (empty = no match)

"First violation": format rules are checked in file order and the first one stops the
run. If the format is clean, the set rules report the lowest-numbered offending line,
then completeness reports the first missing entity in test_source1 order.

Usage:
    python validate_submission_native.py --smoke
    python validate_submission_native.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir data/dataset/test
    python validate_submission_native.py --smoke --no-organiser   # native layer only
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

import config

KEY = "source1_entity_id"
HEADERS = {"matching": [KEY, "matched_entity_ids"], "candidate": [KEY, "candidate_entity_ids"]}
NAN_LITERALS = {"none", "nan", "null", "na", "n/a", "[]", "nat", "<na>"}
TEST_SRCS = ("source1", "source2", "source3")


class SubmissionRejected(Exception):
    """One named rule violation. str() is the full human-readable message."""

    def __init__(self, tag: str, rule: str, path, line: int | None, detail: str, fix: str = ""):
        self.tag, self.rule, self.path, self.line, self.detail, self.fix = tag, rule, Path(path), line, detail, fix
        where = f"{self.path.name}" + (f", line {line}" if line else "")
        msg = f"[{tag}] {rule} — {where}: {detail}"
        super().__init__(msg + (f"\n    fix: {fix}" if fix else ""))


# ── test input files ──────────────────────────────────────────────────────────

def test_files(smoke: bool, test_dir: Path | None = None) -> dict[str, Path]:
    """{src: path} of the test input TSVs the submission is checked against.
    --test-dir uses the organisers' names (test_source1.tsv ...)."""
    if test_dir is not None:
        return {s: Path(test_dir) / f"test_{s}.tsv" for s in TEST_SRCS}
    fn = config.smoke_raw_path if smoke else config.raw_path
    return {s: fn("test", s) for s in TEST_SRCS}


def _ids(path: Path) -> pl.Series:
    """First column of a raw source TSV, parsed exactly as s0_ingest parses it."""
    from s0_ingest import scan_tsv  # the one canonical raw-TSV reader

    return scan_tsv(path).select("entity_id").collect()["entity_id"]


# ── layer 1: native checks ────────────────────────────────────────────────────

def _format_pass(path: Path, kind: str) -> int:
    """Stream the file once, raising on the first format violation. Returns the row count.
    Holds only the set of S1 IDs seen, never the lists, so it is cheap at full scale."""
    want = HEADERS[kind]
    if not path.is_file():
        raise SubmissionRejected("scorer", "file missing", path, None, "file not found")
    if path.stat().st_size == 0:
        raise SubmissionRejected("scorer", "empty file", path, None, "0 bytes")

    seen: set[str] = set()
    rows = 0
    with open(path, "rb") as f:
        for n, raw in enumerate(f, start=1):
            if b"\r" in raw:
                raise SubmissionRejected(
                    "ours", "CR line ending", path, n, "line ends in \\r\\n (Windows newline)",
                    "our writer only emits LF, so this file was re-saved by something else "
                    "(e.g. a Windows editor); regenerate it with s6_assemble")
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                raise SubmissionRejected(
                    "scorer", "not UTF-8", path, n, f"invalid byte at column {e.start}: {raw[e.start:e.start + 4]!r}",
                    "write with encoding='utf-8' (never cp1252 — PYTHONUTF8=1)") from None
            line = line[:-1] if line.endswith("\n") else line

            if n == 1:
                if line.startswith("﻿"):
                    raise SubmissionRejected("scorer", "byte-order mark", path, 1, "file starts with a UTF-8 BOM",
                                             "write with encoding='utf-8', not 'utf-8-sig'")
                if "\t" not in line and "," in line:
                    raise SubmissionRejected("scorer", "comma-separated", path, 1, f"header {line!r} has no TAB",
                                             "write with separator='\\t'")
                cols = line.split("\t")
                if cols == want:
                    continue
                if len(cols) == len(want) + 1 and cols[1:] == want:
                    raise SubmissionRejected("scorer", "index column", path, 1,
                                             f"header has a leading extra column {cols[0]!r}",
                                             "write without the index (pandas: index=False)")
                tag = "ours" if [c.strip().lower() for c in cols] == want else "scorer"
                raise SubmissionRejected(tag, "wrong header", path, 1, f"got {cols}, expected {want}")

            if '"' in line:
                raise SubmissionRejected("scorer", "quoted field", path, n, f"row contains a '\"': {line[:80]!r}",
                                         "write with quote_style='never'")
            tabs = line.count("\t")
            if tabs != 1:
                if tabs == 0 and not line.strip():
                    raise SubmissionRejected("scorer", "blank line", path, n, "empty line inside the file")
                raise SubmissionRejected("scorer", "wrong column count", path, n,
                                         f"{tabs + 1} tab-separated fields, expected 2: {line[:80]!r}")
            s1, rest = line.split("\t")
            rows += 1

            if s1 != s1.strip():
                raise SubmissionRejected("scorer", "whitespace in query ID", path, n, f"{s1!r}")
            if not s1.startswith("S1-"):
                raise SubmissionRejected("scorer", "query ID prefix", path, n,
                                         f"{s1!r} does not start with 'S1-'")
            if s1 in seen:
                raise SubmissionRejected("scorer", "duplicate row", path, n,
                                         f"{s1} already has a row; each entity gets exactly one")
            seen.add(s1)

            if rest == "":
                continue
            if not rest.strip():
                raise SubmissionRejected("ours", "whitespace-only list", path, n,
                                         f"{s1}: list is {rest!r}; a no-match row must be exactly empty")
            if rest.strip().lower() in NAN_LITERALS:
                raise SubmissionRejected(
                    "scorer", "NaN/None literal", path, n,
                    f"{s1}: list is the literal {rest!r}; a no-match row must be the EMPTY string",
                    "fill_null('') before writing; never str() a null")
            ids = rest.split(",")
            for mid in ids:
                if mid == "":
                    raise SubmissionRejected("scorer", "empty ID in list", path, n,
                                             f"{s1}: {rest[:80]!r} has an empty element (',,' or trailing ',')")
                if mid != mid.strip():
                    raise SubmissionRejected("scorer", "whitespace in ID", path, n, f"{s1}: {mid!r}",
                                             "join IDs with ',' not ', '")
                if mid.startswith("S1-"):
                    raise SubmissionRejected("scorer", "self-match", path, n,
                                             f"{s1}: list contains Source-1 ID {mid}; only S2-/S3- allowed")
                if not mid.startswith(("S2-", "S3-")):
                    raise SubmissionRejected("scorer", "ID prefix", path, n,
                                             f"{s1}: {mid!r} is not an S2-/S3- ID")
            if len(ids) != len(set(ids)):
                dup = next(i for i in ids if ids.count(i) > 1)
                raise SubmissionRejected("scorer", "repeated ID in list", path, n, f"{s1}: {dup} appears twice")
    if rows == 0:
        raise SubmissionRejected("scorer", "no rows", path, None, "header only")
    return rows


def _ids_lazy(path: Path, kind: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """(one row per line: line, s1), (one row per listed ID: line, s1, id), both LAZY.
    Only called after _format_pass, so plain splitting is safe. Nothing here materialises
    the exploded ID list: a full-size candidate_pairs.tsv is ~52M IDs, and holding it whole
    (plus the anti-join against it) took this gate past 7.5 GB."""
    col = HEADERS[kind][1]
    lf = (
        pl.scan_csv(path, separator="	", quote_char=None, infer_schema=False, encoding="utf8")
        .with_row_index("line", offset=2)
    )
    rows = lf.select("line", KEY)
    ids = (
        lf.select("line", KEY, pl.col(col).fill_null("").str.split(",").alias("id"))
        .explode("id", empty_as_null=False)
        .filter(pl.col("id").is_not_null() & (pl.col("id") != ""))
    )
    return rows, ids


def _first(lf: pl.LazyFrame) -> dict | None:
    """The lowest-line row of `lf`, found with a streaming min() so that even a file where every
    row offends never has to be materialised."""
    first = lf.select(pl.col("line").min()).collect(engine="streaming").item()
    if first is None:
        return None
    return lf.filter(pl.col("line") == first).collect(engine="streaming").row(0, named=True)


def native_check(matching: Path, candidate: Path | None, files: dict[str, Path]) -> dict:
    """Layer 1. Raises SubmissionRejected on the first violation, else returns counts.
    Memory stays bounded at full size: the per-ID checks stream over the exploded lists, and the
    matching-subset-of-candidates check probes the candidate stream against the (small) matched
    pairs instead of holding every candidate ID."""
    for s, p in files.items():
        if not p.is_file():
            raise SubmissionRejected("ours", "cannot verify", p, None,
                                     f"test {s} input not found, so the submission cannot be checked",
                                     "pass --test-dir with test_source1/2/3.tsv")
    test_s1 = _ids(files["source1"])
    pool = pl.concat([_ids(files["source2"]), _ids(files["source3"])])
    report = {"test_entities": test_s1.len()}

    parsed = {}
    for kind, path in (("matching", matching), ("candidate", candidate)):
        if path is None:
            continue
        report[f"{kind}_rows"] = _format_pass(path, kind)
        parsed[kind] = _ids_lazy(path, kind)

    for kind, path in (("matching", matching), ("candidate", candidate)):
        if kind not in parsed:
            continue
        rows_lf, ids_lf = parsed[kind]
        rows = rows_lf.collect()  # one row per entity: 1.73M at full size, cheap
        bad = _first(rows.lazy().filter(~pl.col(KEY).is_in(test_s1.implode())))
        if bad:
            raise SubmissionRejected("scorer", "query ID not in test set", path, bad["line"],
                                     f"{bad[KEY]} is not in {files['source1'].name}")
        bad = _first(ids_lf.filter(~pl.col("id").is_in(pool.implode())))
        if bad:
            raise SubmissionRejected("ours", "ID not in test pool", path, bad["line"],
                                     f"{bad[KEY]}: {bad['id']} is in neither {files['source2'].name} nor "
                                     f"{files['source3'].name} (the scorer would just score it 0)")
        if kind == "matching" and "candidate" in parsed:
            matched = ids_lf.collect(engine="streaming")  # ~5.7M pairs at full size
            found = (
                parsed["candidate"][1].select(KEY, "id")
                .join(matched.lazy().select(KEY, "id"), on=[KEY, "id"], how="semi")
                .collect(engine="streaming")
            )
            bad = _first(matched.lazy().join(found.lazy(), on=[KEY, "id"], how="anti"))
            if bad:
                raise SubmissionRejected("ours", "match not in candidate list", path, bad["line"],
                                         f"{bad[KEY]}: {bad['id']} is not in its candidate_pairs.tsv row")
            del matched, found
        missing = pl.DataFrame({KEY: test_s1}).join(rows.select(KEY), on=KEY, how="anti")
        if missing.height:
            raise SubmissionRejected("scorer", "missing entity (completeness)", path, None,
                                     f"{missing.height:,} of {test_s1.len():,} test entities have no row, "
                                     f"first: {missing[KEY][0]}",
                                     "every test_source1 entity needs a row; write '' for no match")
        n_ids, n_with = ids_lf.select(pl.len(), pl.col(KEY).n_unique()).collect(engine="streaming").row(0)
        report[f"{kind}_ids"] = n_ids
        report[f"{kind}_empty_rows"] = rows.height - n_with
    return report


# ── layer 2: organisers' validator ────────────────────────────────────────────

def organiser_check(matching: Path, candidate: Path | None, files: dict[str, Path],
                    check_ids: bool = False, script: Path = config.VALIDATOR) -> tuple[bool | None, str]:
    """(passed, output). passed is None when the script does not exist.

    `candidate=None` must really mean "no candidate check": the organisers' script otherwise falls
    back to output/candidate_pairs.tsv relative to the working directory, which on submission night
    is the real ~660 MB file, and with it the script passes 7.5 GB. So it is handed an explicit path
    that does not exist, which it skips with a warning."""
    if not Path(script).is_file():
        return None, f"organisers' validator not found at {script}"
    names_ok = all(p.name == f"test_{s}.tsv" for s, p in files.items())
    one_dir = len({p.parent for p in files.values()}) == 1
    with tempfile.TemporaryDirectory(prefix="val_testdir_", dir=files["source1"].parent) as tmp:
        d = Path(tmp)
        if names_ok and one_dir:
            test_dir = files["source1"].parent  # already the names it hard-codes: no copy at all
        else:
            test_dir = d  # same volume as the inputs, so these are hardlinks, not 1 GB of copies
            for s, p in files.items():
                try:
                    os.link(p, d / f"test_{s}.tsv")
                except OSError:
                    shutil.copy(p, d / f"test_{s}.tsv")
        cand = candidate if candidate is not None else d / "no_candidate_check.tsv"
        cmd = [sys.executable, str(script), "--matching", str(matching), "--test-dir", str(test_dir),
               "--candidate", str(cand)]
        if check_ids:
            cmd.append("--check-ids")
        env = {**os.environ, "PYTHONUTF8": "1"}
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    return res.returncode == 0, (res.stdout + res.stderr).strip()


# ── the gate ─────────────────────────────────────────────────────────────────

def gate(matching, candidate=None, *, smoke: bool, test_dir=None, organiser: bool = True,
         organiser_check_ids: bool = False, organiser_candidate: bool = False, quarantine: bool = False) -> dict:
    """Both layers. On failure prints the reason and raises SystemExit(1); with
    quarantine=True the submission files are first renamed to *.REJECTED so a failed
    file can never be uploaded by mistake."""
    matching = Path(matching)
    candidate = Path(candidate) if candidate else None
    files = test_files(smoke, Path(test_dir) if test_dir else None)
    bar = "=" * 78
    print(f"\n{bar}\nSUBMISSION GATE  {matching}" + (f"\n                 {candidate}" if candidate else ""))

    def reject(reason: str):
        moved = []
        if quarantine:
            for p in (matching, candidate):
                if p is not None and p.exists():
                    dst = p.with_name(p.name + ".REJECTED")
                    p.replace(dst)
                    moved.append(dst.name)
        print(f"GATE: FAIL\n  {reason}" + (f"\n  quarantined -> {', '.join(moved)}" if moved else "") + f"\n{bar}")
        raise SystemExit(1)

    try:
        report = native_check(matching, candidate, files)
    except SubmissionRejected as e:
        reject(f"native: {e}")
    print(f"  native     PASS  {report}")

    if organiser:
        # The organisers' script holds every candidate ID in Python sets when given --candidate:
        # at full size (~52M IDs) it passed 7.5 GB. Its own docstring says to drop --candidate
        # then ("the matching file is the only one scored"); layer 1 above already checks
        # candidate_pairs.tsv with the same or stricter rules.
        ok, out = organiser_check(matching, candidate if organiser_candidate else None, files, organiser_check_ids)
        if ok is None:
            print(f"  organisers SKIPPED — {out}. Native checks only; do NOT treat as final for upload.")
        elif not ok:
            reject("organisers' validate_submission.py failed:\n    " + out.replace("\n", "\n    "))
        else:
            print(f"  organisers PASS  ({out.splitlines()[-1]})")
    print(f"GATE: PASS\n{bar}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="default paths under artifacts/smoke/")
    ap.add_argument("--matching", type=Path, help="default: <output dir>/matching_results.tsv")
    ap.add_argument("--candidate", type=Path, help="default: <output dir>/candidate_pairs.tsv if it exists")
    ap.add_argument("--no-candidate", action="store_true", help="check matching_results.tsv alone")
    ap.add_argument("--test-dir", type=Path, help="dir with test_source1/2/3.tsv (organisers' names)")
    ap.add_argument("--no-organiser", action="store_true", help="skip the organisers' validator")
    ap.add_argument("--organiser-candidate", action="store_true",
                    help="also give candidate_pairs.tsv to the organisers' script (needs >7.5 GB at full size)")
    ap.add_argument("--organiser-check-ids", action="store_true",
                    help="also pass --check-ids to it (several GB at full scale; ours already checks IDs)")
    args = ap.parse_args(argv)
    out_dir = config.SMOKE_OUTPUT_DIR if args.smoke else config.OUTPUT_DIR
    matching = args.matching or config.matching_results_path("test", out_dir)
    candidate = None if args.no_candidate else (args.candidate or config.candidate_pairs_path("test", out_dir))
    if candidate is not None and not Path(candidate).exists() and args.candidate is None:
        candidate = None
    gate(matching, candidate, smoke=args.smoke, test_dir=args.test_dir,
         organiser=not args.no_organiser, organiser_check_ids=args.organiser_check_ids,
         organiser_candidate=args.organiser_candidate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
