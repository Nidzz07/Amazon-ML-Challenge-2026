"""Feature library for entity resolution (Track D, owner: Krrish).

Exports
-------
FEATURE_SPEC    – list of (feature_name, monotonic_direction) pairs
FEATURE_NAMES   – tuple of feature names in column order
FEATURE_MONO    – tuple of monotonic directions (for LightGBM constraints)
FEATURE_VERSION – integer, increment on ANY change to FEATURE_NAMES/order
featurise       – (pairs DataFrame) → (N, F) float32 ndarray

Monotonic directions
    +1  higher value → more likely a match  (similarity metrics)
    −1  higher value → less likely a match  (distance/rank metrics)
     0  non-monotonic or informational

The ``pairs`` DataFrame expected by ``featurise()`` must carry columns from
the normalised schema (prefixed ``s1_`` for source-1 and ``cand_`` for the
candidate) plus blocking context columns.  ``s3_featurise.py`` builds this
DataFrame by joining norm and candidate data.

Vectorisation
-------------
``featurise`` is called once per chunk of candidate pairs (s3_featurise shards
by country then chunks within country), so every block below works on whole
columns rather than row by row.  Three kinds of work remain elementwise:

* the four rapidfuzz scorers go through ``rapidfuzz.process.cpdist``, which
  runs the same C scorers over the pair of sequences on all cores;
* char 3/4-gram Jaccard is a Polars n-gram explode + join (see
  ``_char_jaccard``), the same 64-bit-hash trick ``blocking/tfidf_index.py``
  uses for its vocabulary;
* Jaro-Winkler and the domain-stem match stay Python loops.  Both are named
  and explained at their definitions.

Every rewritten block was checked to reproduce the previous per-row
implementation bit for bit on the smoke split, so FEATURE_VERSION is unchanged.
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl

# ═══════════════════════════════════════════════════════════════════════
# Feature metadata
# ═══════════════════════════════════════════════════════════════════════

FEATURE_SPEC: list[tuple[str, int]] = [
    # ── Name similarity on name_norm (11) ────────────────────────────
    ("name_norm_ratio",            1),
    ("name_norm_partial_ratio",    1),
    ("name_norm_token_sort",       1),
    ("name_norm_token_set",        1),
    ("name_norm_jaro_winkler",     1),
    ("name_norm_char3_jaccard",    1),
    ("name_norm_char4_jaccard",    1),
    ("name_norm_token_jaccard",    1),
    ("name_norm_token_containment",1),
    ("name_norm_prefix_ratio",     1),
    ("name_norm_length_ratio",     1),
    # ── Name similarity on name_roman (11) ───────────────────────────
    ("name_roman_ratio",            1),
    ("name_roman_partial_ratio",    1),
    ("name_roman_token_sort",       1),
    ("name_roman_token_set",        1),
    ("name_roman_jaro_winkler",     1),
    ("name_roman_char3_jaccard",    1),
    ("name_roman_char4_jaccard",    1),
    ("name_roman_token_jaccard",    1),
    ("name_roman_token_containment",1),
    ("name_roman_prefix_ratio",     1),
    ("name_roman_length_ratio",     1),
    # ── Name cross-features (6) ─────────────────────────────────────
    ("name_acronym_match",    1),
    ("name_first_token_match",1),
    ("name_suffix_agree",     1),
    ("name_digit_eq",         1),
    ("name_domain_match",     1),
    ("name_script_pair",      0),
    # ── Address similarity on addr_norm (9) ─────────────────────────
    ("addr_norm_ratio",            1),
    ("addr_norm_partial_ratio",    1),
    ("addr_norm_token_sort",       1),
    ("addr_norm_token_set",        1),
    ("addr_norm_jaro_winkler",     1),
    ("addr_norm_char3_jaccard",    1),
    ("addr_norm_char4_jaccard",    1),
    ("addr_norm_token_jaccard",    1),
    ("addr_norm_token_containment",1),
    # ── Address similarity on addr_roman (9) ────────────────────────
    ("addr_roman_ratio",            1),
    ("addr_roman_partial_ratio",    1),
    ("addr_roman_token_sort",       1),
    ("addr_roman_token_set",        1),
    ("addr_roman_jaro_winkler",     1),
    ("addr_roman_char3_jaccard",    1),
    ("addr_roman_char4_jaccard",    1),
    ("addr_roman_token_jaccard",    1),
    ("addr_roman_token_containment",1),
    # ── Address exact/structural (8) ────────────────────────────────
    ("addr_street_num_match",  1),
    ("addr_postcode_match",    1),
    ("addr_state_match",       1),
    ("addr_city_match",        1),
    ("addr_numeric_jaccard",   1),
    ("s1_has_addr",            0),
    ("cand_has_addr",          0),
    ("both_have_addr",         0),
    # ── Context features (12) ───────────────────────────────────────
    ("n_channels",         1),
    ("ch_name_tfidf",      0),
    ("ch_addr_tfidf",      0),
    ("ch_exact_key",       0),
    ("ch_rare_token",      0),
    ("ch_embed_ann",       0),
    ("best_rank",         -1),
    ("reciprocal_rank",    1),
    ("prior_score",        1),
    ("margin_to_best",     0),
    ("entity_n_cands",     0),
    ("is_source3",         0),
    # ── Competition features (3) ────────────────────────────────────
    ("cand_best_score",    0),
    ("cand_n_claims",      0),
    ("cand_is_argmax",     1),
    # ── Embedding ANN features (2) — Tanuj, B5 ──────────────────────
    ("embed_cosine",       1),   # cosine similarity from FAISS search (0 if not in ANN)
    ("embed_rank",        -1),   # rank within entity in ANN channel (0 if not in ANN)
]

FEATURE_NAMES: tuple[str, ...] = tuple(n for n, _ in FEATURE_SPEC)
FEATURE_MONO: tuple[int, ...] = tuple(d for _, d in FEATURE_SPEC)
FEATURE_VERSION: int = 2
NUM_FEATURES: int = len(FEATURE_NAMES)

# Channel bit positions (must match config.CHANNELS order)
_CHANNEL_BITS = ("name_tfidf", "addr_tfidf", "exact_key", "rare_token", "embed_ann")

# TLD pattern for domain-name matching
_TLD_RE = re.compile(r"\.(com|net|org|co\.in|co|io|in|fr|us|biz|info)$", re.IGNORECASE)

# Tokens that count as numeric for addr_numeric_jaccard: anything containing a digit.
_HAS_DIGIT_RE = r"[0-9]"

# ═══════════════════════════════════════════════════════════════════════
# Helpers — pure, no I/O
# ═══════════════════════════════════════════════════════════════════════

def _s(v) -> str:
    """None / null → empty string."""
    return v if v is not None else ""


def _tl(v) -> list:
    """None / null token list → empty list."""
    return v if v is not None else []


def _ngrams(s: str, n: int) -> set[str]:
    if len(s) < n:
        return set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    denom = min(len(a), len(b))
    return len(a & b) / denom if denom else 0.0


def _prefix_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    mx = max(len(a), len(b))
    common = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        common += 1
    return common / mx


def _length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _digit_tokens(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t.isdigit()}


def _numeric_tokens(tokens: list[str]) -> set[str]:
    return {t for t in tokens if any(c.isdigit() for c in t)}


def _domain_stem_match(a: str, b: str) -> float:
    """1.0 if one string is a domain whose stem matches the other name."""
    for s, other in [(a, b), (b, a)]:
        if "." not in s:
            continue
        stem = _TLD_RE.sub("", s)
        if stem == s:
            continue  # no TLD was stripped
        words = re.findall(r"[a-z]+", stem.lower())
        if not words:
            continue
        from rapidfuzz import fuzz as _fuzz
        if _fuzz.token_set_ratio(" ".join(words), other) > 70:
            return 1.0
    return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Column plumbing — every block below takes and returns whole columns
# ═══════════════════════════════════════════════════════════════════════

def _str_col(pairs: "pl.DataFrame", name: str) -> pl.Series:
    """String column with nulls as '' (the `_s` contract), or all-'' if absent."""
    if name not in pairs.columns:
        return pl.Series(name, [""] * pairs.height, dtype=pl.String)
    return pairs[name].cast(pl.String).fill_null("")


def _tok_col(pairs: "pl.DataFrame", name: str) -> pl.Series:
    """list[str] column with null lists as [] (the `_tl` contract)."""
    if name not in pairs.columns:
        return pl.Series(name, [[]] * pairs.height, dtype=pl.List(pl.String))
    return pairs[name].fill_null([])


def _np_col(pairs: "pl.DataFrame", name: str, dtype=np.float32) -> np.ndarray:
    if name not in pairs.columns:
        return np.zeros(pairs.height, dtype=dtype)
    return pairs[name].fill_null(0).to_numpy().astype(dtype)


def _nonempty_both(a: pl.Series, b: pl.Series) -> np.ndarray:
    """Boolean mask for the `if not a or not b: continue` guard the loops used."""
    return ((a.str.len_chars() > 0) & (b.str.len_chars() > 0)).to_numpy()


# ═══════════════════════════════════════════════════════════════════════
# Batch computation — string similarity
# ═══════════════════════════════════════════════════════════════════════

def _gram_hashes(vals: pl.Series, k: int) -> pl.DataFrame:
    """(s, h): the distinct char-k-gram hashes of each DISTINCT string in `vals`.

    n-grams are identified by their 64-bit polars hash rather than by the
    substring, exactly as blocking/tfidf_index.py builds its vocabulary. Doing
    the explode over distinct strings only is what makes this cheap: each
    Source-1 string repeats once per candidate (≈30x at the configured cap).
    Strings shorter than k drop out, which is `_ngrams` returning an empty set.
    """
    return (
        pl.DataFrame({"s": vals}).lazy()
        .unique()
        .with_columns(pl.col("s").str.len_chars().alias("L"))
        .filter(pl.col("L") >= k)
        .with_columns(pl.int_ranges(0, pl.col("L") - k + 1).alias("o"))
        .explode("o", empty_as_null=False)  # L >= k above, so no range is ever empty
        .select("s", pl.col("s").str.slice(pl.col("o"), k).hash(seed=0).alias("h"))
        .unique()
        .collect()
    )


def _char_jaccard(a: pl.Series, b: pl.Series, k: int) -> np.ndarray:
    """Char k-gram Jaccard per row, as |A∩B| / (|A| + |B| - |A∩B|).

    Replaces `_jaccard(_ngrams(a, k), _ngrams(b, k))`. Using |A|+|B|-|A∩B| for
    the union means one join instead of building two Python sets per row.
    """
    n = len(a)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    df = pl.DataFrame({"i": np.arange(n, dtype=np.uint32), "a": a.rename("a"), "b": b.rename("b")})
    ga = _gram_hashes(df["a"], k).rename({"s": "a"})
    gb = _gram_hashes(df["b"], k).rename({"s": "b"})
    inter = (
        df.lazy()
        .join(ga.lazy(), on="a", how="inner")
        .join(gb.lazy(), on=["b", "h"], how="inner")
        .group_by("i").len("ni")
        .collect()
    )
    out = (
        df.lazy()
        .join(ga.group_by("a").len("na").lazy(), on="a", how="left")
        .join(gb.group_by("b").len("nb").lazy(), on="b", how="left")
        .join(inter.lazy(), on="i", how="left")
        .with_columns(pl.col("^(ni|na|nb)$").fill_null(0))
        .sort("i")  # the joins do not promise input order; features are positional
        .select(
            pl.when((pl.col("na") > 0) & (pl.col("nb") > 0))
            .then(pl.col("ni") / (pl.col("na") + pl.col("nb") - pl.col("ni")))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("j")
        )
        .collect()
    )
    return out["j"].to_numpy()


def _string_sims_7(s1: pl.Series, s2: pl.Series) -> np.ndarray:
    """7 string-similarity features for paired string columns.

    Columns: ratio, partial_ratio, token_sort, token_set,
             jaro_winkler, char3_jaccard, char4_jaccard

    Returns (N, 7) float32.
    """
    from rapidfuzz import fuzz, process
    import jellyfish

    n = len(s1)
    out = np.zeros((n, 7), dtype=np.float32)
    if n == 0:
        return out

    both = _nonempty_both(s1, s2)
    a, b = s1.to_list(), s2.to_list()

    # cpdist runs the same C scorers elementwise across both sequences on all
    # cores. float64 then /100 reproduces the old `fuzz.ratio(a, b) / 100.0`.
    for j, scorer in enumerate((fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio)):
        out[:, j] = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float64) / 100.0

    # NOT VECTORISED: jellyfish's Jaro-Winkler. rapidfuzz's JaroWinkler is the
    # only vectorised (cpdist-compatible) implementation available and it does
    # not agree with jellyfish — 14,248 of 200,000 smoke pairs differ, worst
    # case 0.0 vs 0.381 on a Latin/Devanagari pair. Swapping it would silently
    # move a trained model's inputs, so the loop stays. It costs ~9% of this
    # block; the scorers above and the Jaccards below were the 88%.
    jw = np.zeros(n, dtype=np.float64)
    for i in np.flatnonzero(both):
        jw[i] = jellyfish.jaro_winkler_similarity(a[i], b[i])
    out[:, 4] = jw

    out[:, 5] = _char_jaccard(s1, s2, 3)
    out[:, 6] = _char_jaccard(s1, s2, 4)

    out[~both, :] = 0.0  # the loops' `if not a or not b: continue`
    return out


def _token_sims_2(t1: pl.Series, t2: pl.Series) -> np.ndarray:
    """Token jaccard + containment over the token SETS. Returns (N, 2) float32."""
    n = len(t1)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    got = (
        pl.DataFrame({"a": t1.rename("a"), "b": t2.rename("b")})
        .lazy()
        .with_columns(pl.col("a").list.unique(), pl.col("b").list.unique())
        .with_columns(
            pl.col("a").list.len().alias("na"),
            pl.col("b").list.len().alias("nb"),
            pl.col("a").list.set_intersection("b").list.len().alias("ni"),
        )
        .select(
            pl.when((pl.col("na") > 0) & (pl.col("nb") > 0))
            .then(pl.col("ni") / (pl.col("na") + pl.col("nb") - pl.col("ni")))
            .otherwise(0.0).cast(pl.Float32).alias("jac"),
            pl.when((pl.col("na") > 0) & (pl.col("nb") > 0))
            .then(pl.col("ni") / pl.min_horizontal("na", "nb"))
            .otherwise(0.0).cast(pl.Float32).alias("cont"),
        )
        .collect()
    )
    return np.column_stack([got["jac"].to_numpy(), got["cont"].to_numpy()]).astype(np.float32)


def _name_extras_2(s1: pl.Series, s2: pl.Series) -> np.ndarray:
    """prefix_ratio + length_ratio. Returns (N, 2) float32.

    The common-prefix length is found by binary search on the prefix length:
    `a[:m] == b[:m]` is monotone in m, so ceil(log2(min_len)) vectorised slice
    comparisons pin it exactly, with no per-row Python.
    """
    n = len(s1)
    out = np.zeros((n, 2), dtype=np.float32)
    if n == 0:
        return out

    la = s1.str.len_chars().fill_null(0).to_numpy().astype(np.int64)
    lb = s2.str.len_chars().fill_null(0).to_numpy().astype(np.int64)
    both = (la > 0) & (lb > 0)

    lo = np.zeros(n, dtype=np.int64)
    hi = np.minimum(la, lb)
    df = pl.DataFrame({"a": s1.rename("a"), "b": s2.rename("b")})
    while True:
        active = lo < hi
        if not active.any():
            break
        mid = (lo + hi + 1) // 2
        eq = (
            df.lazy()
            .with_columns(pl.Series("m", mid))
            .select(
                (pl.col("a").str.slice(0, pl.col("m")) == pl.col("b").str.slice(0, pl.col("m"))).alias("eq")
            )
            .collect()["eq"]
            .fill_null(False)
            .to_numpy()
        )
        take = active & eq
        lo = np.where(take, mid, lo)
        hi = np.where(active & ~eq, mid - 1, hi)

    mx = np.maximum(la, lb)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, 0] = np.where(both, lo / np.maximum(mx, 1), 0.0)
        out[:, 1] = np.where(both, np.minimum(la, lb) / np.maximum(mx, 1), 0.0)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Batch computation — cross-features and structural
# ═══════════════════════════════════════════════════════════════════════

def _set_equal(a: str, b: str) -> pl.Expr:
    """Expr: the two list columns hold the same set of values (both empty → true)."""
    ua, ub = pl.col(a).list.unique(), pl.col(b).list.unique()
    return (ua.list.len() == ub.list.len()) & (ua.list.set_difference(ub).list.len() == 0)


def _name_cross_6(
    s1_tokens: pl.Series,
    cand_tokens: pl.Series,
    s1_acronym: pl.Series,
    cand_acronym: pl.Series,
    s1_suffix: pl.Series,
    cand_suffix: pl.Series,
    s1_name_norm: pl.Series,
    cand_name_norm: pl.Series,
    s1_script: pl.Series,
    cand_script: pl.Series,
) -> np.ndarray:
    """6 name cross-features. Returns (N, 6) float32."""
    n = len(s1_tokens)
    out = np.zeros((n, 6), dtype=np.float32)
    if n == 0:
        return out

    df = pl.DataFrame({
        "t1": s1_tokens.rename("t1"), "t2": cand_tokens.rename("t2"),
        "a1": s1_acronym.rename("a1"), "a2": cand_acronym.rename("a2"),
        "x1": s1_suffix.rename("x1"), "x2": cand_suffix.rename("x2"),
        "n1": s1_name_norm.rename("n1"), "n2": cand_name_norm.rename("n2"),
        "p1": s1_script.rename("p1"), "p2": cand_script.rename("p2"),
    })
    # Digit-only tokens, kept as a set: both-empty counts as agreement, which is
    # what set equality already gives.
    digits = pl.element().filter(pl.element().str.contains(r"^[0-9]+$"))
    got = (
        df.lazy()
        .with_columns(
            pl.col("t1").list.eval(digits).alias("d1"),
            pl.col("t2").list.eval(digits).alias("d2"),
        )
        .select(
            # acronym match
            ((pl.col("a1") != "") & (pl.col("a2") != "") & (pl.col("a1") == pl.col("a2")))
            .cast(pl.Float32).alias("f0"),
            # first token match
            (
                (pl.col("t1").list.len() > 0) & (pl.col("t2").list.len() > 0)
                & (pl.col("t1").list.first() == pl.col("t2").list.first())
            ).fill_null(False).cast(pl.Float32).alias("f1"),
            # suffix agreement (both have same suffix, including both empty)
            (pl.col("x1") == pl.col("x2")).cast(pl.Float32).alias("f2"),
            # digit token equality
            _set_equal("d1", "d2").fill_null(False).cast(pl.Float32).alias("f3"),
            # script pair (encode as s1_script * 10 + cand_script for categorisation)
            (pl.col("p1").cast(pl.Float32) * 10 + pl.col("p2").cast(pl.Float32)).alias("f5"),
            # only rows with a dot on either side can be a domain
            (pl.col("n1").str.contains(".", literal=True) | pl.col("n2").str.contains(".", literal=True))
            .alias("maybe_domain"),
        )
        .collect()
    )
    for j, col in ((0, "f0"), (1, "f1"), (2, "f2"), (3, "f3"), (5, "f5")):
        out[:, j] = got[col].to_numpy()

    # NOT VECTORISED: domain-stem match. It strips a TLD, re-splits the stem into
    # words and runs a rapidfuzz threshold on the rebuilt string, so there is no
    # column form. Gating on "does either side contain a dot" keeps the loop off
    # ~99% of rows (0.4% of smoke pairs qualify).
    maybe = got["maybe_domain"].fill_null(False).to_numpy()
    if maybe.any():
        n1, n2 = s1_name_norm.to_list(), cand_name_norm.to_list()
        col4 = out[:, 4]
        for i in np.flatnonzero(maybe):
            col4[i] = _domain_stem_match(n1[i], n2[i])
    return out


def _addr_exact_8(
    s1_street: pl.Series,
    cand_street: pl.Series,
    s1_post: pl.Series,
    cand_post: pl.Series,
    s1_state: pl.Series,
    cand_state: pl.Series,
    s1_city: pl.Series,
    cand_city: pl.Series,
    s1_addr_tokens: pl.Series,
    cand_addr_tokens: pl.Series,
    s1_has_addr: pl.Series,
    cand_has_addr: pl.Series,
) -> np.ndarray:
    """8 address exact/structural features. Returns (N, 8) float32."""
    n = len(s1_street)
    out = np.zeros((n, 8), dtype=np.float32)
    if n == 0:
        return out

    df = pl.DataFrame({
        "st1": s1_street.rename("st1"), "st2": cand_street.rename("st2"),
        "pc1": s1_post.rename("pc1"),   "pc2": cand_post.rename("pc2"),
        "sa1": s1_state.rename("sa1"),  "sa2": cand_state.rename("sa2"),
        "ci1": s1_city.rename("ci1"),   "ci2": cand_city.rename("ci2"),
        "at1": s1_addr_tokens.rename("at1"), "at2": cand_addr_tokens.rename("at2"),
        "ha1": s1_has_addr.fill_null(False).rename("ha1"),
        "ha2": cand_has_addr.fill_null(False).rename("ha2"),
    })

    def both_eq(x: str, y: str) -> pl.Expr:
        # exact component matches (empty == empty → no useful signal → 0)
        return ((pl.col(x) != "") & (pl.col(y) != "") & (pl.col(x) == pl.col(y))).cast(pl.Float32)

    numeric = pl.element().filter(pl.element().str.contains(_HAS_DIGIT_RE))
    got = (
        df.lazy()
        .with_columns(
            pl.col("at1").list.eval(numeric).list.unique().alias("q1"),
            pl.col("at2").list.eval(numeric).list.unique().alias("q2"),
        )
        .with_columns(
            pl.col("q1").list.len().alias("nq1"),
            pl.col("q2").list.len().alias("nq2"),
            pl.col("q1").list.set_intersection("q2").list.len().alias("nqi"),
        )
        .select(
            both_eq("st1", "st2").alias("f0"),
            both_eq("pc1", "pc2").alias("f1"),
            both_eq("sa1", "sa2").alias("f2"),
            both_eq("ci1", "ci2").alias("f3"),
            # numeric token jaccard (all tokens containing digits)
            pl.when((pl.col("nq1") > 0) & (pl.col("nq2") > 0))
            .then(pl.col("nqi") / (pl.col("nq1") + pl.col("nq2") - pl.col("nqi")))
            .otherwise(0.0).cast(pl.Float32).alias("f4"),
            # address presence flags
            pl.col("ha1").cast(pl.Float32).alias("f5"),
            pl.col("ha2").cast(pl.Float32).alias("f6"),
            (pl.col("ha1") & pl.col("ha2")).cast(pl.Float32).alias("f7"),
        )
        .collect()
    )
    for j in range(8):
        out[:, j] = got[f"f{j}"].to_numpy()
    return out


# ═══════════════════════════════════════════════════════════════════════
# Context + competition features (from blocking metadata)
# ═══════════════════════════════════════════════════════════════════════

def _context_12(
    channels: np.ndarray,    # uint8 bitmask
    n_channels: np.ndarray,  # uint8
    best_rank: np.ndarray,   # uint16
    prior_score: np.ndarray, # float32
    entity_best: np.ndarray, # float32 — best prior_score for this entity
    entity_n: np.ndarray,    # int — candidates per entity
    is_s3: np.ndarray,       # bool
) -> np.ndarray:
    """12 context features. Returns (N, 12) float32."""
    n = len(channels)
    out = np.zeros((n, 12), dtype=np.float32)
    out[:, 0] = n_channels.astype(np.float32)
    for bit_idx in range(len(_CHANNEL_BITS)):
        out[:, 1 + bit_idx] = ((channels.astype(np.int32) >> bit_idx) & 1).astype(np.float32)
    rank_f = best_rank.astype(np.float32)
    out[:, 6] = rank_f
    out[:, 7] = np.where(rank_f > 0, 1.0 / rank_f, 0.0)
    out[:, 8] = prior_score.astype(np.float32)
    out[:, 9] = prior_score.astype(np.float32) - entity_best.astype(np.float32)
    out[:, 10] = entity_n.astype(np.float32)
    out[:, 11] = is_s3.astype(np.float32)
    return out


def _competition_3(
    prior_score: np.ndarray,
    cand_best: np.ndarray,   # best prior_score for this candidate across all entities
    cand_n_claims: np.ndarray,
) -> np.ndarray:
    """3 competition features. Returns (N, 3) float32."""
    n = len(prior_score)
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, 0] = cand_best.astype(np.float32)
    out[:, 1] = cand_n_claims.astype(np.float32)
    out[:, 2] = (prior_score >= cand_best - 1e-7).astype(np.float32)  # is_argmax (float tol)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def featurise(pairs: "pl.DataFrame") -> np.ndarray:
    """Compute all features for a pre-joined pairs DataFrame.

    Expected columns (see s3_featurise.py for the join logic):
        s1_name_norm, s1_name_roman, s1_name_tokens, s1_name_acronym,
        s1_addr_norm, s1_addr_roman, s1_addr_tokens,
        s1_street_num, s1_city_norm, s1_state_canon, s1_postcode,
        s1_has_addr, s1_script, s1_name_suffix,
        cand_name_norm, cand_name_roman, cand_name_tokens, cand_name_acronym,
        cand_addr_norm, cand_addr_roman, cand_addr_tokens,
        cand_street_num, cand_city_norm, cand_state_canon, cand_postcode,
        cand_has_addr, cand_script, cand_name_suffix,
        channels, n_channels, best_rank, prior_score,
        entity_best_score, entity_n_cands,
        cand_best_score, cand_n_claims,
        is_source3, embed_cosine, embed_rank

    Every entity- or candidate-level aggregate above is computed by
    s3_featurise over the whole country shard before chunking, so this function
    only ever reads within its own rows and chunk boundaries cannot change a
    feature value.

    Returns (N, NUM_FEATURES) float32 ndarray in FEATURE_NAMES order.
    """
    n = pairs.height
    blocks: list[np.ndarray] = []

    # ── Name similarity on name_norm (11) ────────────────────────────
    s1_nn = _str_col(pairs, "s1_name_norm")
    c_nn = _str_col(pairs, "cand_name_norm")
    blocks.append(_string_sims_7(s1_nn, c_nn))

    s1_nt = _tok_col(pairs, "s1_name_tokens")
    c_nt = _tok_col(pairs, "cand_name_tokens")
    blocks.append(_token_sims_2(s1_nt, c_nt))
    blocks.append(_name_extras_2(s1_nn, c_nn))

    # ── Name similarity on name_roman (11) ───────────────────────────
    s1_nr = _str_col(pairs, "s1_name_roman")
    c_nr = _str_col(pairs, "cand_name_roman")
    blocks.append(_string_sims_7(s1_nr, c_nr))

    # Reuse name_tokens for roman (tokens are from the same source in the stub;
    # when real transliteration lands, s3_featurise will supply roman-specific tokens)
    blocks.append(_token_sims_2(s1_nt, c_nt))
    blocks.append(_name_extras_2(s1_nr, c_nr))

    # ── Name cross-features (6) ─────────────────────────────────────
    blocks.append(_name_cross_6(
        s1_nt, c_nt,
        _str_col(pairs, "s1_name_acronym"), _str_col(pairs, "cand_name_acronym"),
        _str_col(pairs, "s1_name_suffix"), _str_col(pairs, "cand_name_suffix"),
        s1_nn, c_nn,
        pairs["s1_script"].fill_null(0) if "s1_script" in pairs.columns
        else pl.Series("s1_script", np.zeros(n, dtype=np.uint8)),
        pairs["cand_script"].fill_null(0) if "cand_script" in pairs.columns
        else pl.Series("cand_script", np.zeros(n, dtype=np.uint8)),
    ))

    # ── Address similarity on addr_norm (9) ──────────────────────────
    s1_an = _str_col(pairs, "s1_addr_norm")
    c_an = _str_col(pairs, "cand_addr_norm")
    blocks.append(_string_sims_7(s1_an, c_an))

    s1_at = _tok_col(pairs, "s1_addr_tokens")
    c_at = _tok_col(pairs, "cand_addr_tokens")
    blocks.append(_token_sims_2(s1_at, c_at))

    # ── Address similarity on addr_roman (9) ─────────────────────────
    s1_ar = _str_col(pairs, "s1_addr_roman")
    c_ar = _str_col(pairs, "cand_addr_roman")
    blocks.append(_string_sims_7(s1_ar, c_ar))
    blocks.append(_token_sims_2(s1_at, c_at))  # reuse addr_tokens

    # ── Address exact/structural (8) ─────────────────────────────────
    blocks.append(_addr_exact_8(
        _str_col(pairs, "s1_street_num"), _str_col(pairs, "cand_street_num"),
        _str_col(pairs, "s1_postcode"), _str_col(pairs, "cand_postcode"),
        _str_col(pairs, "s1_state_canon"), _str_col(pairs, "cand_state_canon"),
        _str_col(pairs, "s1_city_norm"), _str_col(pairs, "cand_city_norm"),
        s1_at, c_at,
        pairs["s1_has_addr"] if "s1_has_addr" in pairs.columns
        else pl.Series("s1_has_addr", np.zeros(n, dtype=bool)),
        pairs["cand_has_addr"] if "cand_has_addr" in pairs.columns
        else pl.Series("cand_has_addr", np.zeros(n, dtype=bool)),
    ))

    # ── Context features (12) ───────────────────────────────────────
    blocks.append(_context_12(
        _np_col(pairs, "channels", np.uint8),
        _np_col(pairs, "n_channels", np.uint8),
        _np_col(pairs, "best_rank", np.uint16),
        _np_col(pairs, "prior_score"),
        _np_col(pairs, "entity_best_score"),
        _np_col(pairs, "entity_n_cands"),
        _np_col(pairs, "is_source3"),
    ))

    # ── Competition features (3) ────────────────────────────────────
    blocks.append(_competition_3(
        _np_col(pairs, "prior_score"),
        _np_col(pairs, "cand_best_score"),
        _np_col(pairs, "cand_n_claims"),
    ))

    # ── Embedding ANN features (2) — Tanuj, B5 ──────────────────────
    embed = np.zeros((n, 2), dtype=np.float32)
    embed[:, 0] = _np_col(pairs, "embed_cosine")   # 0.0 if pair not in ANN output
    embed[:, 1] = _np_col(pairs, "embed_rank")     # 0.0 if pair not in ANN output
    blocks.append(embed)

    result = np.hstack(blocks)
    assert result.shape == (n, NUM_FEATURES), f"shape {result.shape} != ({n}, {NUM_FEATURES})"
    return result.astype(np.float32)
