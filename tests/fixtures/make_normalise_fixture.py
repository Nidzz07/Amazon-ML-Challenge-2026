"""Regenerate tests/fixtures/normalise_pairs.tsv from the full train data.

Not a test. Run it by hand (needs artifacts/records_train_*.parquet from s0):
    python tests/fixtures/make_normalise_fixture.py

Sampling is seeded (config.SEED) and category rules are mechanical, so the file is
reproducible. EXCLUDE lists matched ids dropped by hand during curation (for
example, true matches that share no name at all, which no normaliser can fix).

Categories:
  cross_script  S1 name Latin-only, matched name contains Indic script (field=name),
                stratified by script, plus mixed-script names
  typo          same-token-count Latin names with a character-level edit (field=name)
  accent        matched name carries Latin accents the S1 name lacks (field=name)
  abbrev        legal-suffix drift (Private Limited vs Pvt Ltd) (field=name) and
                street-type drift (Street vs St, Road vs Rd) (field=addr)
  reorder       same address tokens in a different component order (field=addr)
  quote         test-set names containing a literal '"' (the raw file CSV-escapes them
                as triple-quoted CSV; s0 unescapes it). Unlabelled; the test asserts the words
                inside survive.
  negative      random NON-matching pairs from the same country and script mix;
                must stay below the ceiling after normalisation
"""
import random
import sys
from pathlib import Path

import polars as pl
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "business_entity_resolution" / "src"))
import config  # noqa: E402

OUT = Path(__file__).resolve().parent / "normalise_pairs.tsv"
COLUMNS = ["case_id", "category", "field", "country", "left", "right", "ids"]
EXCLUDE: set[str] = set()

INDIC = r"[ऀ-෿]"
NON_ASCII_LATIN = r"[À-ɏ]"
SCRIPT_BLOCKS = {
    "devanagari": r"[ऀ-ॿ]", "bengali": r"[ঀ-৿]", "gurmukhi": r"[਀-੿]",
    "gujarati": r"[઀-૿]", "oriya": r"[଀-୿]", "tamil": r"[஀-௿]",
    "telugu": r"[ఀ-౿]", "kannada": r"[ಀ-೿]", "malayalam": r"[ഀ-ൿ]",
}
PER_SCRIPT = {"devanagari": 12}  # others get 3 each; 12 + 8*3 + 4 mixed = 40
LATIN_PER_CATEGORY = 5


def load_pairs() -> pl.DataFrame:
    s1 = pl.read_parquet(config.records_path("train", "source1"))
    pool = pl.concat([pl.read_parquet(config.records_path("train", s)) for s in config.CANDIDATE_SRCS])
    gt = pl.read_parquet(config.records_path("train", "ground_truth"))
    return (
        gt.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m", empty_as_null=False)
        .filter(pl.col("m") != "")
        .join(s1.select(pl.col("entity_id").alias("source1_entity_id"), "country",
                        pl.col("business_name").alias("n1"), pl.col("business_address").fill_null("").alias("a1")),
              on="source1_entity_id")
        .join(pool.select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"),
                          pl.col("business_address").fill_null("").alias("a2")), on="m")
        .filter(~pl.col("m").is_in(list(EXCLUDE)))
        .sort("source1_entity_id", "m")
    )


def take(df: pl.DataFrame, n: int, category: str, field: str, rows: list) -> None:
    left, right = ("n1", "n2") if field == "name" else ("a1", "a2")
    for r in df.sample(min(n, df.height), seed=config.SEED).iter_rows(named=True):
        rows.append({"category": category, "field": field, "country": r["country"],
                     "left": r[left], "right": r[right], "ids": f"{r['source1_entity_id']}|{r['m']}"})


def main() -> None:
    pairs = load_pairs()
    rows: list[dict] = []
    latin_name = ~pl.col("n1").str.contains(INDIC)

    cross = pairs.filter(latin_name & pl.col("n2").str.contains(INDIC))
    for script, pat in SCRIPT_BLOCKS.items():
        pure = cross.filter(pl.col("n2").str.contains(pat) & ~pl.col("n2").str.contains(r"[A-Za-z]"))
        take(pure, PER_SCRIPT.get(script, 3), "cross_script", "name", rows)
    take(cross.filter(pl.col("n2").str.contains(r"[A-Za-z]")), 4, "cross_script", "name", rows)

    latin = pairs.filter(latin_name & ~pl.col("n2").str.contains(INDIC))
    lo1, lo2 = pl.col("n1").str.to_lowercase(), pl.col("n2").str.to_lowercase()
    same_len = pl.col("n1").str.split(" ").list.len() == pl.col("n2").str.split(" ").list.len()
    plain = ~pl.col("n1").str.contains(NON_ASCII_LATIN) & ~pl.col("n2").str.contains(NON_ASCII_LATIN)
    words = lambda c: c.str.extract_all(r"[a-z0-9]+")  # noqa: E731
    diff = lambda a, b: words(a).list.set_difference(words(b))  # noqa: E731
    typo = latin.filter(plain & same_len & (lo1 != lo2) & (diff(lo1, lo2).list.len() == 1)
                        & (diff(lo2, lo1).list.len() == 1)).with_columns(
        diff(lo1, lo2).list.first().alias("d1"), diff(lo2, lo1).list.first().alias("d2"),
    )
    # A typo, not a word swap: the one differing token is a <=2-edit variant of a 5+ char word.
    typo = typo.filter(pl.col("d1").str.len_chars() >= 5).filter(
        pl.struct("d1", "d2").map_elements(lambda r: Levenshtein.distance(r["d1"], r["d2"]) <= 2, return_dtype=pl.Boolean)
    )
    take(typo, LATIN_PER_CATEGORY, "typo", "name", rows)

    accent = latin.filter(pl.col("n2").str.contains(NON_ASCII_LATIN) & ~pl.col("n1").str.contains(NON_ASCII_LATIN))
    take(accent, LATIN_PER_CATEGORY, "accent", "name", rows)

    # Legal-suffix drift: abbreviated vs spelled out (India), present vs absent (US).
    nodomain = ~lo2.str.contains(r"\.(com|in|net|org)\b")
    india_suffix = latin.filter(lo1.str.contains(r"\bprivate limited$") & lo2.str.contains(r"\bpvt\b") & nodomain & plain)
    take(india_suffix, 2, "abbrev", "name", rows)
    us_suffix = latin.filter((pl.col("country") == "US") & lo1.str.contains(r"\b(inc|llc)\.?$")
                             & ~lo2.str.contains(r"\b(inc|llc|corp|ltd|incorporated)\b") & nodomain & plain
                             & (words(lo1).list.len() - words(lo2).list.len() == 1))
    take(us_suffix, 1, "abbrev", "name", rows)
    street = latin.filter(pl.col("a1").str.contains(r"(?i)\b(street|road)\b") & pl.col("a2").str.contains(r"(?i)\b(st|rd)\b")
                          & ~pl.col("a2").str.contains(r"(?i)\b(street|road)\b"))
    take(street, 2, "abbrev", "addr", rows)

    tok = lambda c: pl.col(c).str.to_lowercase().str.extract_all(r"[a-z0-9]+").list.sort()  # noqa: E731
    reorder = latin.filter((pl.col("a1") != "") & (tok("a1") == tok("a2"))
                           & (pl.col("a1").str.to_lowercase() != pl.col("a2").str.to_lowercase()))
    take(reorder, LATIN_PER_CATEGORY, "reorder", "addr", rows)

    # Literal-quote names exist only in the (unlabelled) test split, all France.
    test_names = pl.concat([
        pl.read_parquet(config.records_path("test", s), columns=["entity_id", "business_name", "country"])
        for s in ("source1", *config.CANDIDATE_SRCS)
    ])
    quoted = test_names.filter(pl.col("business_name").str.contains('"'))
    for q in ('"ehpad Club SAS', 'SARL "ehpad Club'):
        hit = quoted.filter(pl.col("business_name") == q).row(0, named=True)
        rows.append({"category": "quote", "field": "name", "country": hit["country"],
                     "left": hit["business_name"], "right": "Lehpad Club", "ids": hit["entity_id"]})

    # Negatives: shuffle matched names within country + script class so pairs are not matches.
    rng = random.Random(config.SEED)
    for cls, df in (("latin", latin), ("cross", cross)):
        for country in ("India", "US") if cls == "latin" else ("India",):
            sub = df.filter(pl.col("country") == country).sample(40, seed=config.SEED)
            lefts, rights = sub["n1"].to_list(), sub["n2"].to_list()
            ids1, ids2 = sub["source1_entity_id"].to_list(), sub["m"].to_list()
            order = list(range(len(rights)))
            rng.shuffle(order)
            added = 0
            for i, j in enumerate(order):
                if ids1[i] == ids1[j] or added == (4 if cls == "cross" else 3):
                    continue
                rows.append({"category": "negative", "field": "name", "country": country,
                             "left": lefts[i], "right": rights[j], "ids": f"{ids1[i]}|{ids2[j]}"})
                added += 1

    out = pl.DataFrame(rows).with_row_index("case_id", offset=1).select(COLUMNS)
    out = out.with_columns(pl.col("case_id").cast(pl.String))
    for c in ("left", "right"):
        assert not out[c].str.contains("\t|\n").any(), f"tab/newline in {c}"
    out.write_csv(OUT, separator=config.TSV_SEP, quote_style="never", line_terminator="\n")
    print(out.group_by("category", maintain_order=True).len())
    print(f"wrote {out.height} cases -> {OUT}")


if __name__ == "__main__":
    main()
