"""Channel addr_tfidf: char 3-4-gram TF-IDF cosine on addr_roman, top-k per entity.
Rows with has_addr false are skipped on both sides."""
import polars as pl

import config
from blocking.tfidf_index import tfidf_channel

NAME = "addr_tfidf"


def run(s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    return tfidf_channel(s1.filter(pl.col("has_addr")), pool.filter(pl.col("has_addr")), "addr_roman", config.TFIDF_TOP_K)
