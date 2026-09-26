"""exact_key channel: the city/state key families produce pairs once s1 fills them."""
import polars as pl

import config
import pipeline_io as pio
from blocking import exact_key


def norm(rows: list[dict]) -> pl.DataFrame:
    base = {c: "" for c, t in pio.NORM_SCHEMA.items() if t == pl.String}
    full = [{**base, "name_tokens": [], "addr_tokens": [], "has_addr": True, "script": 1, **r} for r in rows]
    return pl.DataFrame(full, schema=pio.NORM_SCHEMA)


def test_city_and_state_families_are_configured():
    fams = {tuple(f) for f in config.EXACT_KEY_FAMILIES}
    assert any("city_norm" in f for f in fams)
    assert any("state_canon" in f for f in fams)


def test_street_city_pair():
    s1 = norm([{"entity_id": "S1-1", "street_num": "8-2-293/82", "city_norm": "hyderabad", "state_canon": "tg"}])
    pool = norm([{"entity_id": "S2-1", "street_num": "8-2-293/82", "city_norm": "hyderabad", "state_canon": "tg"},
                 {"entity_id": "S2-2", "street_num": "8-2-293/82", "city_norm": "pune", "state_canon": "mh"},
                 {"entity_id": "S2-3", "street_num": "8-2-293/82", "city_norm": "", "state_canon": "tg"}])
    out = exact_key.run(s1, pool)
    got = dict(zip(out["candidate_entity_id"], out["channel_score"]))
    assert set(got) == {"S2-1", "S2-3"}, "S2-2 differs on city and state"
    assert got["S2-1"] > got["S2-3"], "more agreeing families rank first"
