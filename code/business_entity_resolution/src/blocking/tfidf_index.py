"""Character n-gram TF-IDF top-k retrieval, shared by name_tfidf and addr_tfidf.

A backend is any class with
    fit(texts: pl.Series) -> self           # index the candidate pool
    search(texts: pl.Series, k) -> iterator of (query_idx, pool_idx, score) numpy chunks
make_index() picks one from config.TFIDF_BACKEND. The exact sparse backend lives
here. A TruncatedSVD(256) + faiss IVFFlat backend only needs to implement the same
two methods and register itself in BACKENDS; the channels do not change.
tfidf_matrix() gives such a backend the same weighted matrix to reduce.

N-grams are generated in polars, not with sklearn's pure-Python analyzer. They
follow char_wb semantics: each whitespace token is padded with one space on each
side and n-grams never cross tokens. Each n-gram is identified by its 64-bit polars
hash, which is stable for the pinned polars version, so the vocabulary never
materialises strings. Weights are sublinear TF x smoothed IDF (sklearn's formula),
L2-normalised per record, with the IDF fitted on the pool. Retrieval recall matches
sklearn's TfidfVectorizer on the smoke US shard (0.738 vs 0.736 recall@20).

config.TFIDF_MAX_DF may be a fraction; fit() resolves it against the number of pool
texts being indexed (blocking.common.resolve_df), so the ceiling scales with the shard.
"""
from typing import Iterator

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

import config
from blocking.common import empty, rank_within_entity, resolve_df


def char_ngrams(texts: pl.Series, ngram_range=config.TFIDF_NGRAM_RANGE) -> pl.DataFrame:
    """(doc u32, h u64, tf u32): one row per distinct n-gram hash per text."""
    chunk_size = 50_000
    all_chunks = []
    
    for offset in range(0, len(texts), chunk_size):
        chunk_texts = texts.slice(offset, chunk_size)
        words = (
            pl.DataFrame({"text": chunk_texts})
            .lazy()
            .with_row_index("doc", offset=offset)
            .select("doc", pl.col("text").str.extract_all(r"\S+").alias("w"))
            .explode("w", empty_as_null=False)
            .filter(pl.col("w").is_not_null() & (pl.col("w") != ""))
            .select("doc", (" " + pl.col("w") + " ").alias("w"))
            .with_columns(pl.col("w").str.len_chars().alias("L"))
        )
        grams = [
            words.with_columns(pl.int_ranges(0, pl.max_horizontal(pl.col("L") - n + 1, 1)).alias("off"))
            .explode("off", empty_as_null=False)
            .select("doc", pl.col("w").str.slice(pl.col("off"), n).hash(seed=0).alias("h"))
            for n in range(ngram_range[0], ngram_range[1] + 1)
        ]
        chunk_res = pl.concat(grams).group_by("doc", "h").len(name="tf").collect(engine="streaming")
        all_chunks.append(chunk_res)
        
    if not all_chunks:
        return pl.DataFrame(schema={"doc": pl.UInt32, "h": pl.UInt64, "tf": pl.UInt32})
    return pl.concat(all_chunks)


def _weights(grams: pl.DataFrame, vocab: pl.DataFrame) -> pl.DataFrame:
    """(doc, gid, w): sublinear TF x IDF, L2-normalised per doc. Sorted by (doc, gid) so
    the float sums, and therefore the scores, are identical run to run."""
    return (
        grams.join(vocab, on="h", how="inner")
        .sort("doc", "gid")
        .with_columns(((1.0 + pl.col("tf").cast(pl.Float64).log()) * pl.col("idf")).alias("w"))
        .with_columns((pl.col("w") / (pl.col("w") ** 2).sum().over("doc").sqrt()).cast(pl.Float32).alias("w"))
        .select("doc", "gid", "w")
    )


def _csr(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    """CSR from triplets already sorted by (row, col); skips scipy's COO sort."""
    indptr = np.zeros(shape[0] + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=shape[0]), out=indptr[1:])
    return sp.csr_matrix((data, cols.astype(np.int32), indptr), shape=shape)


def fit_vocab(grams: pl.DataFrame, n_docs: int, max_df: int | None = None) -> pl.DataFrame:
    """max_df is an absolute document count (resolve a fraction with resolve_df first)."""
    df = grams.group_by("h").len(name="df").filter(pl.col("df") >= config.TFIDF_MIN_DF)
    if max_df is not None:
        df = df.filter(pl.col("df") <= max_df)
    return (
        df.sort("h")
        .with_row_index("gid")
        .select("h", "gid", (((1 + n_docs) / (1 + pl.col("df"))).log() + 1.0).alias("idf"))
    )


def tfidf_matrix(grams: pl.DataFrame, vocab: pl.DataFrame, n_docs: int) -> sp.csr_matrix:
    """docs x vocab L2-normalised TF-IDF matrix."""
    w = _weights(grams, vocab)
    return _csr(w["doc"].to_numpy(), w["gid"].to_numpy(), w["w"].to_numpy(), (n_docs, vocab.height))


class SparseTopNIndex:
    """Exact sparse cosine top-k via sparse_dot_topn, queries processed in chunks.
    With max_df set, n-grams in more than max_df pool records are dropped first
    (see config.TFIDF_MAX_DF): cosine over the remaining n-grams. max_df is a fraction
    of the pool or a count; fit() stores the count it resolved to in max_df_resolved."""

    def __init__(self, max_df=config.TFIDF_MAX_DF, chunk_rows=config.TFIDF_CHUNK_ROWS, n_threads=config.BLOCKING_THREADS):
        self.max_df, self.chunk_rows, self.n_threads = max_df, chunk_rows, n_threads

    def fit(self, texts: pl.Series) -> "SparseTopNIndex":
        grams = char_ngrams(texts)
        self.max_df_resolved = resolve_df(f"TFIDF_MAX_DF[{texts.name}]", self.max_df, len(texts))
        self.vocab = fit_vocab(grams, len(texts), self.max_df_resolved)
        w = _weights(grams, self.vocab).sort("gid", "doc")
        # Stored transposed (vocab x pool): the right-hand operand of Q @ C.T.
        self.pool_t = _csr(w["gid"].to_numpy(), w["doc"].to_numpy(), w["w"].to_numpy(), (self.vocab.height, len(texts)))
        return self

    def search(self, texts: pl.Series, k: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        q = tfidf_matrix(char_ngrams(texts), self.vocab, len(texts))
        for start in range(0, q.shape[0], self.chunk_rows):
            res = sp_matmul_topn(
                q[start : start + self.chunk_rows], self.pool_t, top_n=k, sort=True, n_threads=self.n_threads
            ).tocoo()
            yield res.row.astype(np.int64) + start, res.col.astype(np.int64), res.data.astype(np.float32)


# Factories read config at call time, so a knob changed at runtime takes effect.
BACKENDS = {
    "sparse_topn": lambda: SparseTopNIndex(config.TFIDF_MAX_DF, config.TFIDF_CHUNK_ROWS, config.BLOCKING_THREADS),
}


def make_index():
    return BACKENDS[config.TFIDF_BACKEND]()


def tfidf_channel(s1: pl.DataFrame, pool: pl.DataFrame, text_col: str, k: int = config.TFIDF_TOP_K) -> pl.DataFrame:
    """Top-k pool records per Source-1 row by TF-IDF cosine on `text_col`."""
    s1 = s1.filter(pl.col(text_col) != "")
    pool = pool.filter(pl.col(text_col) != "")
    if s1.is_empty() or pool.is_empty():
        return empty()
    index = make_index().fit(pool[text_col])
    s1_ids, pool_ids = s1["entity_id"], pool["entity_id"]
    parts = [
        pl.DataFrame({
            "source1_entity_id": s1_ids.gather(qi),
            "candidate_entity_id": pool_ids.gather(ci),
            "score": score,
        })
        for qi, ci, score in index.search(s1[text_col], k)
        if len(qi)
    ]
    if not parts:
        return empty()
    return rank_within_entity(pl.concat(parts), ["score"], [True], "score")
