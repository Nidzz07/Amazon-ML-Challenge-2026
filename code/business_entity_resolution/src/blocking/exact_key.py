"""Channel exact_key: hash join on the key families in config.EXACT_KEY_FAMILIES
((street_num, city_norm), (postcode, street_num), (name_acronym, city_norm)).

A key needs every component non-empty. Any key whose bucket holds more than
config.EXACT_KEY_MAX_BUCKET records on EITHER side is dropped as noise. No
similarity is computed. channel_rank only orders an entity's hits by specificity:
more families agreeing first, then smaller candidate bucket. channel_score is the
number of families that agreed.
"""
import polars as pl

import config
from blocking.common import empty, rank_within_entity

NAME = "exact_key"


def _family_pairs(s1: pl.DataFrame, pool: pl.DataFrame, cols: tuple[str, ...]) -> pl.DataFrame:
    def keyed(df: pl.DataFrame) -> pl.DataFrame:
        nonempty = pl.all_horizontal([pl.col(c) != "" for c in cols])
        return (
            df.select("entity_id", *cols)
            .filter(nonempty)
            .filter(pl.len().over(cols) <= config.EXACT_KEY_MAX_BUCKET)
            .with_columns(pl.len().over(cols).alias("bucket"))
        )

    return (
        keyed(s1)
        .join(keyed(pool), on=list(cols), how="inner", suffix="_c")
        .select(
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_c").alias("candidate_entity_id"),
            pl.col("bucket_c").alias("bucket"),
        )
    )


def run(s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    parts = [_family_pairs(s1, pool, cols) for cols in config.EXACT_KEY_FAMILIES]
    pairs = pl.concat(parts)
    if pairs.is_empty():
        return empty()
    pairs = pairs.group_by("source1_entity_id", "candidate_entity_id").agg(
        pl.len().alias("families"), pl.col("bucket").min()
    )
    return rank_within_entity(pairs, ["families", "bucket"], [True, False], "families")
