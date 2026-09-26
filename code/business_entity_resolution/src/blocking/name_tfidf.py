"""Channel name_tfidf: char 3-4-gram TF-IDF cosine on name_roman, top-k per entity."""
import polars as pl

import config
from blocking.tfidf_index import tfidf_channel

NAME = "name_tfidf"


def run(s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    return tfidf_channel(s1, pool, "name_roman", config.TFIDF_TOP_K)
