import math
import random

import polars as pl
import pytest

import assemble


def frame(rows):
    return pl.DataFrame(rows, schema={"source1_entity_id": pl.String, "candidate_entity_id": pl.String, "prob": pl.Float32}, orient="row")


def brute_force(probs, mode):
    """Reference: score every prefix of the q-descending list explicitly."""
    q = sorted(probs, reverse=True)
    k_hat = sum(q)
    if mode == "p_none":
        best_score, best_m = math.prod(1 - x for x in q), 0
    else:
        best_score, best_m = (1.0 if k_hat == 0 else 0.0), 0
    for m in range(1, len(q) + 1):
        ef = 1.25 * sum(q[:m]) / (0.25 * k_hat + m)
        if ef > best_score + 1e-12:
            best_score, best_m = ef, m
    return best_m


def decided_m(df, mode):
    _, dec = assemble.prefix_search(df, mode)
    return dict(dec.select("source1_entity_id", "m").iter_rows())


def test_uniqueness_keeps_best_claim_and_breaks_ties_by_s1_id():
    df = frame([
        ("S1-a", "S2-x", 0.9), ("S1-b", "S2-x", 0.6),  # b loses x
        ("S1-a", "S2-y", 0.5), ("S1-c", "S2-y", 0.5),  # tie -> lower s1 id (a)
        ("S1-b", "S2-z", 0.4),
    ])
    out = assemble.enforce_uniqueness(df)
    assert out["candidate_entity_id"].is_unique().all()
    owner = dict(out.select("candidate_entity_id", "source1_entity_id").iter_rows())
    assert owner == {"S2-x": "S1-a", "S2-y": "S1-a", "S2-z": "S1-b"}


@pytest.mark.parametrize("mode", ["p_none", "plugin"])
def test_all_zero_probabilities_predict_empty(mode):
    assert decided_m(frame([("S1-a", "S2-x", 0.0), ("S1-a", "S3-y", 0.0)]), mode) == {"S1-a": 0}


def test_empty_is_a_real_option_under_p_none():
    # One weak candidate: P(k=0) = 0.7 beats E[F|m=1] = 1.25*0.3/(0.075+1) = 0.349.
    df = frame([("S1-a", "S2-x", 0.3)])
    assert decided_m(df, "p_none") == {"S1-a": 0}
    assert decided_m(df, "plugin") == {"S1-a": 1}  # the plug-in empty score is 0 whenever k_hat > 0


def test_prefix_stops_where_marginal_candidate_no_longer_pays():
    # 0.95, 0.9 then 0.2: adding 0.2 lowers expected F0.5, so m = 2.
    df = frame([("S1-a", "S2-x", 0.95), ("S1-a", "S3-y", 0.9), ("S1-a", "S2-z", 0.2)])
    sel, dec = assemble.prefix_search(df, "p_none")
    assert dec["m"].to_list() == [2]
    assert sel["candidate_entity_id"].to_list() == ["S2-x", "S3-y"]


@pytest.mark.parametrize("mode", ["p_none", "plugin"])
def test_matches_brute_force_on_random_entities(mode):
    rng = random.Random(42)
    rows, truth = [], {}
    for e in range(300):
        n = rng.randint(1, 12)
        # Round to float32 so the reference sees the same values the frame stores.
        probs = [float(pl.Series([rng.random() ** rng.choice([1, 3, 8])], dtype=pl.Float32)[0]) for _ in range(n)]
        s1 = f"S1-{e:04d}"
        rows += [(s1, f"S2-{e:04d}-{i:02d}", p) for i, p in enumerate(probs)]
        truth[s1] = brute_force(probs, mode)
    assert decided_m(frame(rows), mode) == truth


def test_assemble_output_is_unique_and_subset_of_input():
    df = frame([("S1-a", "S2-x", 0.9), ("S1-b", "S2-x", 0.8), ("S1-b", "S3-y", 0.7)])
    sel, _ = assemble.assemble(df, "p_none")
    assert sel["candidate_entity_id"].is_unique().all()
    assert sel.join(df, on=assemble.KEYS, how="anti").is_empty()
    assert ("S1-b", "S2-x") not in set(sel.select(assemble.KEYS).iter_rows())
