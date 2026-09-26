"""Macro-averaged F₀.₅ for entity resolution (Track D, owner: Krrish).

Per-entity score: F = 1.25·c / (0.25·k + m), where
    c = |predicted ∩ truth|  (correct predictions)
    k = |truth|              (true matches)
    m = |predicted|          (number of predictions)

Special cases (easy to get wrong):
    k = 0, m = 0 → 1.0   singleton correctly predicted empty
    k = 0, m > 0 → 0.0   singleton incorrectly predicted non-empty
    k > 0, m = 0 → 0.0   matches exist but predicted empty

Macro average: over ALL entities including singletons.
Missing entity in predictions: raise KeyError, never silently score zero.

Verified against the organisers' worked example:
    pred = {S2-00047, S2-00193, S3-00812}, truth = {S2-00047, S3-00812}
    → 1.25·2 / (0.25·2 + 3) = 2.5 / 3.5 = 5/7 ≈ 0.714
"""
from __future__ import annotations

import numpy as np


def f05_per_entity(predicted: set[str], truth: set[str]) -> float:
    """F₀.₅ for a single entity."""
    k = len(truth)
    m = len(predicted)
    if k == 0:
        return 1.0 if m == 0 else 0.0
    if m == 0:
        return 0.0
    c = len(predicted & truth)
    return 1.25 * c / (0.25 * k + m)


def macro_f05(
    predictions: dict[str, set[str] | list[str]],
    ground_truth: dict[str, set[str] | list[str]],
) -> float:
    """Macro-averaged F₀.₅ over all entities in *ground_truth*.

    Every entity_id in ground_truth must also appear in predictions (the
    predicted set may be empty).  Raises KeyError if an entity is missing.
    """
    scores: list[float] = []
    for entity_id, truth_ids in ground_truth.items():
        if entity_id not in predictions:
            raise KeyError(
                f"Entity {entity_id!r} present in ground truth but missing from predictions"
            )
        pred = set(predictions[entity_id])
        truth = set(truth_ids)
        scores.append(f05_per_entity(pred, truth))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def macro_f05_arrays(
    c: np.ndarray,
    k: np.ndarray,
    m: np.ndarray,
) -> float:
    """Vectorised macro F₀.₅ from pre-computed per-entity counts.

    c[i] = correct predictions for entity i
    k[i] = true matches for entity i
    m[i] = number of predictions for entity i

    This is the fast path used by the evaluation harness when counts are
    already available from Polars joins.
    """
    c = np.asarray(c, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    denom = 0.25 * k + m
    # Guard against 0/0 in branches np.where will discard (avoids RuntimeWarning)
    safe_denom = np.where(denom > 0, denom, 1.0)
    scores = np.where(
        k == 0,
        np.where(m == 0, 1.0, 0.0),
        np.where(m == 0, 0.0, 1.25 * c / safe_denom),
    )
    return float(scores.mean())
