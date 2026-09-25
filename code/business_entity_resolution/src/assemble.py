"""Track A3 assembly logic (owner: Nidhi): pure functions, no I/O. s6_assemble.py
wires them to files.

1. enforce_uniqueness: every S2/S3 record belongs to at most one Source-1 entity
   (7,638,365 ground-truth matches, zero reuse). Hard version: each candidate keeps
   only its highest-probability claim, and losing claims are removed entirely.
2. prefix_search: per entity, sort surviving candidates by q descending and score
   every prefix length m = 0..n by expected F0.5, then keep the argmax prefix.
       m >= 1: E[F] ~= 1.25 * c_hat / (0.25 * k_hat + m)
               c_hat = sum of the top-m q, k_hat = sum of ALL the entity's surviving q
       m  = 0: see config.ASSEMBLY_EMPTY_SCORE ("p_none" = prod(1 - q_i))
   Ties go to the shorter prefix, since the metric weights precision.
"""
import polars as pl

import config

KEYS = ["source1_entity_id", "candidate_entity_id"]


def enforce_uniqueness(scored: pl.DataFrame) -> pl.DataFrame:
    """Keep each candidate's single best claim (ties: lower source1_entity_id)."""
    return (
        scored.sort(["candidate_entity_id", "prob", "source1_entity_id"], descending=[False, True, False])
        .unique("candidate_entity_id", keep="first", maintain_order=True)
    )


def empty_score(q: pl.Expr, k_hat: pl.Expr, mode: str) -> pl.Expr:
    if mode == "p_none":
        # prod(1 - q) computed in log space; q = 1 gives log(0) = -inf and so exp -> 0.
        return (-q).log1p().sum().exp()
    if mode == "plugin":
        return pl.when(k_hat == 0).then(1.0).otherwise(0.0)
    raise ValueError(f"unknown ASSEMBLY_EMPTY_SCORE {mode!r}")


def prefix_search(scored: pl.DataFrame, mode: str | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (selected pairs sorted by entity then q desc, one decision row per entity
    that has candidates: n, k_hat, e_empty, best_prefix_ef, m)."""
    mode = mode or config.ASSEMBLY_EMPTY_SCORE
    q = pl.col("prob").cast(pl.Float64)
    ranked = (
        scored.sort(["source1_entity_id", "prob", "candidate_entity_id"], descending=[False, True, False])
        .with_columns(
            (pl.int_range(pl.len()).over("source1_entity_id") + 1).alias("m"),
            q.cum_sum().over("source1_entity_id").alias("c_hat"),
            q.sum().over("source1_entity_id").alias("k_hat"),
        )
        .with_columns((1.25 * pl.col("c_hat") / (0.25 * pl.col("k_hat") + pl.col("m"))).alias("ef"))
    )
    decisions = (
        ranked.group_by("source1_entity_id", maintain_order=True)
        .agg(
            pl.len().alias("n"),
            pl.col("k_hat").first(),
            empty_score(q, pl.col("k_hat").first(), mode).alias("e_empty"),
            pl.col("ef").max().alias("best_prefix_ef"),
            pl.col("m").get(pl.col("ef").arg_max()).alias("best_m"),  # arg_max = first max = shortest prefix
        )
        .with_columns(
            pl.when(pl.col("e_empty") >= pl.col("best_prefix_ef")).then(0).otherwise(pl.col("best_m")).alias("m_star")
        )
        .drop("best_m")
        .rename({"m_star": "m"})
    )
    selected = (
        ranked.join(decisions.select("source1_entity_id", pl.col("m").alias("m_star")), on="source1_entity_id")
        .filter(pl.col("m") <= pl.col("m_star"))
        .select(*KEYS, "prob")
    )
    return selected, decisions


def assemble(scored: pl.DataFrame, mode: str | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    return prefix_search(enforce_uniqueness(scored), mode)
