"""Channel embed_ann: multilingual embedding ANN (Tanuj, Gate 3). PLACEHOLDER.

Same interface as the other channels. Until config.EMBED_ANN_PATH points at an
existing file it logs that embeddings are not yet available and returns zero
candidates. s2_block needs no special case for it.

When the embeddings land, implement _load() to return precomputed ANN pairs
(source1_entity_id, candidate_entity_id, score) and filter them to this shard.
"""
from pathlib import Path

import polars as pl

import config
from blocking.common import empty

NAME = "embed_ann"


def _available() -> bool:
    return config.EMBED_ANN_PATH is not None and Path(config.EMBED_ANN_PATH).exists()


def run(s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    if not _available():
        print(f"  embed_ann: embeddings not yet available (config.EMBED_ANN_PATH={config.EMBED_ANN_PATH}), 0 candidates")
        return empty()
    # Refuse rather than return nothing silently once a file exists but no loader does.
    raise NotImplementedError(f"embed_ann: {config.EMBED_ANN_PATH} exists but no loader is implemented yet")
