"""Channel exact_key: hash join on the key families in config.EXACT_KEY_FAMILIES
((street_num, city_norm), (postcode, street_num), (name_acronym, city_norm),
(street_num, state_canon)). city_norm, state_canon and postcode come from
normalise.parse_address_components; street_num is the verbatim house-number token.

A key needs every component non-empty. Any key whose bucket holds more than
config.EXACT_KEY_MAX_BUCKET records on EITHER side is dropped as noise. No
similarity is computed. channel_rank only orders an entity's hits by specificity:
more families agreeing first, then smaller candidate bucket. channel_score is the
number of families that agreed.

Memory: run() keys every family ONCE on the whole shard (so bucket sizes, and therefore
which keys are dropped, are exactly as before), then does the join, the cross-family
group_by and the ranking for one hash bucket of Source-1 entities at a time
(config.EXACT_KEY_S1_BUCKETS). All three are per entity, so each bucket's ranks are
final; a stable sort by entity puts the rows back in the one-pass order. Done in one pass
over the full US shard, the group_by + rank alone passed 13 GB even at B = 200.
"""
import polars as pl

import config
from blocking.common import empty, rank_within_entity

NAME = "exact_key"
# The only norm columns run() reads (every family's key columns); s2_block loads just these.
COLUMNS = ("entity_id", *dict.fromkeys(c for fam in config.EXACT_KEY_FAMILIES for c in fam))


def _keyed(df: pl.DataFrame, cols: tuple[str, ...]) -> pl.DataFrame:
    """entity_id, the key columns and the key's bucket size, for keys with every component
    non-empty and at most EXACT_KEY_MAX_BUCKET records on this side of the shard."""
    nonempty = pl.all_horizontal([pl.col(c) != "" for c in cols])
    return (
        df.select("entity_id", *cols)
        .filter(nonempty)
        .filter(pl.len().over(cols) <= config.EXACT_KEY_MAX_BUCKET)
        .with_columns(pl.len().over(cols).alias("bucket"))
    )


def _join(k1: pl.DataFrame, k2: pl.DataFrame, cols: tuple[str, ...]) -> pl.DataFrame:
    return k1.join(k2, on=list(cols), how="inner", suffix="_c").select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_c").alias("candidate_entity_id"),
        pl.col("bucket_c").alias("bucket"),
    )


def family_pairs(s1: pl.DataFrame, pool: pl.DataFrame, cols: tuple[str, ...]) -> pl.DataFrame:
    """(source1_entity_id, candidate_entity_id, bucket) agreeing on every column of one family."""
    return _join(_keyed(s1, cols), _keyed(pool, cols), cols)


def run(s1: pl.DataFrame, pool: pl.DataFrame, smoke: bool = False, families=config.EXACT_KEY_FAMILIES) -> pl.DataFrame:
    """`families` defaults to config; the cap sweep passes a subset to ablate one family."""
    n_b = max(1, config.EXACT_KEY_S1_BUCKETS)
    keyed = [
        (cols, _keyed(s1, cols).with_columns((pl.col("entity_id").hash(seed=0) % n_b).alias("_b")), _keyed(pool, cols))
        for cols in families
    ]
    parts = []
    for b in range(n_b):
        pairs = pl.concat([_join(k1.filter(pl.col("_b") == b).drop("_b"), k2, cols) for cols, k1, k2 in keyed])
        if pairs.is_empty():
            continue
        pairs = pairs.group_by("source1_entity_id", "candidate_entity_id").agg(
            pl.len().alias("families"), pl.col("bucket").min()
        )
        parts.append(rank_within_entity(pairs, ["families", "bucket"], [True, False], "families"))
        del pairs
    if not parts:
        return empty()
    return pl.concat(parts).sort("source1_entity_id", maintain_order=True)
