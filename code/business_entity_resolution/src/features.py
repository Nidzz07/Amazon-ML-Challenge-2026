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
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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
]

FEATURE_NAMES: tuple[str, ...] = tuple(n for n, _ in FEATURE_SPEC)
FEATURE_MONO: tuple[int, ...] = tuple(d for _, d in FEATURE_SPEC)
FEATURE_VERSION: int = 1
NUM_FEATURES: int = len(FEATURE_NAMES)

# Channel bit positions (must match config.CHANNELS order)
_CHANNEL_BITS = ("name_tfidf", "addr_tfidf", "exact_key", "rare_token", "embed_ann")

# TLD pattern for domain-name matching
_TLD_RE = re.compile(r"\.(com|net|org|co\.in|co|io|in|fr|us|biz|info)$", re.IGNORECASE)

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
# Batch computation — string similarity
# ═══════════════════════════════════════════════════════════════════════

def _string_sims_7(s1: list[str], s2: list[str]) -> np.ndarray:
    """7 string-similarity features for paired string lists.

    Columns: ratio, partial_ratio, token_sort, token_set,
             jaro_winkler, char3_jaccard, char4_jaccard

    Returns (N, 7) float32.
    """
    from rapidfuzz import fuzz
    import jellyfish

    n = len(s1)
    out = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        a, b = s1[i], s2[i]
        if not a or not b:
            continue
        out[i, 0] = fuzz.ratio(a, b) / 100.0
        out[i, 1] = fuzz.partial_ratio(a, b) / 100.0
        out[i, 2] = fuzz.token_sort_ratio(a, b) / 100.0
        out[i, 3] = fuzz.token_set_ratio(a, b) / 100.0
        out[i, 4] = jellyfish.jaro_winkler_similarity(a, b)
        out[i, 5] = _jaccard(_ngrams(a, 3), _ngrams(b, 3))
        out[i, 6] = _jaccard(_ngrams(a, 4), _ngrams(b, 4))
    return out


def _token_sims_2(t1: list[list[str]], t2: list[list[str]]) -> np.ndarray:
    """Token jaccard + containment. Returns (N, 2) float32."""
    n = len(t1)
    out = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        a, b = set(t1[i]), set(t2[i])
        out[i, 0] = _jaccard(a, b)
        out[i, 1] = _containment(a, b)
    return out


def _name_extras_2(s1: list[str], s2: list[str]) -> np.ndarray:
    """prefix_ratio + length_ratio. Returns (N, 2) float32."""
    n = len(s1)
    out = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        a, b = s1[i], s2[i]
        out[i, 0] = _prefix_ratio(a, b)
        out[i, 1] = _length_ratio(a, b)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Batch computation — cross-features and structural
# ═══════════════════════════════════════════════════════════════════════

def _name_cross_6(
    s1_tokens: list[list[str]],
    cand_tokens: list[list[str]],
    s1_acronym: list[str],
    cand_acronym: list[str],
    s1_suffix: list[str],
    cand_suffix: list[str],
    s1_name_norm: list[str],
    cand_name_norm: list[str],
    s1_script: list[int],
    cand_script: list[int],
) -> np.ndarray:
    """6 name cross-features. Returns (N, 6) float32."""
    n = len(s1_tokens)
    out = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        # acronym match
        a_acr, c_acr = s1_acronym[i], cand_acronym[i]
        out[i, 0] = 1.0 if (a_acr and c_acr and a_acr == c_acr) else 0.0

        # first token match
        t1, t2 = s1_tokens[i], cand_tokens[i]
        out[i, 1] = 1.0 if (t1 and t2 and t1[0] == t2[0]) else 0.0

        # suffix agreement (both have same suffix, including both empty)
        out[i, 2] = 1.0 if s1_suffix[i] == cand_suffix[i] else 0.0

        # digit token equality
        d1, d2 = _digit_tokens(t1), _digit_tokens(t2)
        out[i, 3] = 1.0 if (d1 and d2 and d1 == d2) else (1.0 if (not d1 and not d2) else 0.0)

        # domain stem match
        out[i, 4] = _domain_stem_match(s1_name_norm[i], cand_name_norm[i])

        # script pair (encode as s1_script * 10 + cand_script for categorisation)
        out[i, 5] = float(s1_script[i] * 10 + cand_script[i])

    return out


def _addr_exact_8(
    s1_street: list[str],
    cand_street: list[str],
    s1_post: list[str],
    cand_post: list[str],
    s1_state: list[str],
    cand_state: list[str],
    s1_city: list[str],
    cand_city: list[str],
    s1_addr_tokens: list[list[str]],
    cand_addr_tokens: list[list[str]],
    s1_has_addr: list[bool],
    cand_has_addr: list[bool],
) -> np.ndarray:
    """8 address exact/structural features. Returns (N, 8) float32."""
    n = len(s1_street)
    out = np.zeros((n, 8), dtype=np.float32)
    for i in range(n):
        # exact component matches (empty == empty → no useful signal → 0)
        out[i, 0] = 1.0 if (s1_street[i] and cand_street[i] and s1_street[i] == cand_street[i]) else 0.0
        out[i, 1] = 1.0 if (s1_post[i] and cand_post[i] and s1_post[i] == cand_post[i]) else 0.0
        out[i, 2] = 1.0 if (s1_state[i] and cand_state[i] and s1_state[i] == cand_state[i]) else 0.0
        out[i, 3] = 1.0 if (s1_city[i] and cand_city[i] and s1_city[i] == cand_city[i]) else 0.0

        # numeric token jaccard (all tokens containing digits)
        num1 = _numeric_tokens(s1_addr_tokens[i])
        num2 = _numeric_tokens(cand_addr_tokens[i])
        out[i, 4] = _jaccard(num1, num2)

        # address presence flags
        out[i, 5] = float(bool(s1_has_addr[i]))
        out[i, 6] = float(bool(cand_has_addr[i]))
        out[i, 7] = float(bool(s1_has_addr[i]) and bool(cand_has_addr[i]))
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
        is_source3

    Returns (N, NUM_FEATURES) float32 ndarray in FEATURE_NAMES order.
    """
    n = pairs.height
    blocks: list[np.ndarray] = []

    # ── Extract columns to Python lists (fast; most work is in the C loops) ──
    def _col_str(name: str) -> list[str]:
        if name not in pairs.columns:
            return [""] * n
        return [_s(v) for v in pairs[name].to_list()]

    def _col_tokens(name: str) -> list[list[str]]:
        if name not in pairs.columns:
            return [[]] * n
        return [_tl(v) for v in pairs[name].to_list()]

    def _col_bool(name: str) -> list[bool]:
        if name not in pairs.columns:
            return [False] * n
        return [bool(v) if v is not None else False for v in pairs[name].to_list()]

    def _col_int(name: str, default: int = 0) -> list[int]:
        if name not in pairs.columns:
            return [default] * n
        return [int(v) if v is not None else default for v in pairs[name].to_list()]

    def _col_np(name: str, dtype=np.float32) -> np.ndarray:
        if name not in pairs.columns:
            return np.zeros(n, dtype=dtype)
        return pairs[name].to_numpy().astype(dtype)

    # ── Name similarity on name_norm (11) ────────────────────────────
    s1_nn = _col_str("s1_name_norm")
    c_nn = _col_str("cand_name_norm")
    blocks.append(_string_sims_7(s1_nn, c_nn))

    s1_nt = _col_tokens("s1_name_tokens")
    c_nt = _col_tokens("cand_name_tokens")
    blocks.append(_token_sims_2(s1_nt, c_nt))
    blocks.append(_name_extras_2(s1_nn, c_nn))

    # ── Name similarity on name_roman (11) ───────────────────────────
    s1_nr = _col_str("s1_name_roman")
    c_nr = _col_str("cand_name_roman")
    blocks.append(_string_sims_7(s1_nr, c_nr))

    # Reuse name_tokens for roman (tokens are from the same source in the stub;
    # when real transliteration lands, s3_featurise will supply roman-specific tokens)
    blocks.append(_token_sims_2(s1_nt, c_nt))
    blocks.append(_name_extras_2(s1_nr, c_nr))

    # ── Name cross-features (6) ─────────────────────────────────────
    blocks.append(_name_cross_6(
        s1_nt, c_nt,
        _col_str("s1_name_acronym"), _col_str("cand_name_acronym"),
        _col_str("s1_name_suffix"), _col_str("cand_name_suffix"),
        s1_nn, c_nn,
        _col_int("s1_script"), _col_int("cand_script"),
    ))

    # ── Address similarity on addr_norm (9) ──────────────────────────
    s1_an = _col_str("s1_addr_norm")
    c_an = _col_str("cand_addr_norm")
    blocks.append(_string_sims_7(s1_an, c_an))

    s1_at = _col_tokens("s1_addr_tokens")
    c_at = _col_tokens("cand_addr_tokens")
    blocks.append(_token_sims_2(s1_at, c_at))

    # ── Address similarity on addr_roman (9) ─────────────────────────
    s1_ar = _col_str("s1_addr_roman")
    c_ar = _col_str("cand_addr_roman")
    blocks.append(_string_sims_7(s1_ar, c_ar))
    blocks.append(_token_sims_2(s1_at, c_at))  # reuse addr_tokens

    # ── Address exact/structural (8) ─────────────────────────────────
    blocks.append(_addr_exact_8(
        _col_str("s1_street_num"), _col_str("cand_street_num"),
        _col_str("s1_postcode"), _col_str("cand_postcode"),
        _col_str("s1_state_canon"), _col_str("cand_state_canon"),
        _col_str("s1_city_norm"), _col_str("cand_city_norm"),
        s1_at, c_at,
        _col_bool("s1_has_addr"), _col_bool("cand_has_addr"),
    ))

    # ── Context features (12) ───────────────────────────────────────
    blocks.append(_context_12(
        _col_np("channels", np.uint8),
        _col_np("n_channels", np.uint8),
        _col_np("best_rank", np.uint16),
        _col_np("prior_score"),
        _col_np("entity_best_score"),
        _col_np("entity_n_cands"),
        _col_np("is_source3"),
    ))

    # ── Competition features (3) ────────────────────────────────────
    blocks.append(_competition_3(
        _col_np("prior_score"),
        _col_np("cand_best_score"),
        _col_np("cand_n_claims"),
    ))

    result = np.hstack(blocks)
    assert result.shape == (n, NUM_FEATURES), f"shape {result.shape} != ({n}, {NUM_FEATURES})"
    return result.astype(np.float32)
