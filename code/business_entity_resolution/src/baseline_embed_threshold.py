"""Standalone insurance baseline (owner: Tanuj, ad-hoc): threshold+top-k rule directly on
artifacts/embed_ann_pairs.parquet, no model, no other pipeline stage. Independent of s3/s4/s5/s6.

Does NOT touch any existing pipeline file or artifacts/output/. Writes only to output_baseline/ at the
repo root. Reads the embed file lazily throughout -- never materialises all 78.8M rows.

Rule: predict candidate c for entity e iff channel_score(e,c) >= t AND channel_rank(e,c) <= k
      [AND channel_score(e,c) >= top_score(e) - delta, if delta is not inf].
ONE global (t, k[, delta]) for every country (France has no training data to tune on).

Split: artifacts/val_entity_ids.parquet if present (tune on the rest of train, report on those
400k held out); a deterministic 80/20 hash split of train ids otherwise.

Sweep implementation: for each (k, delta) pair, every surviving row is binned once into the t-grid
by its score (grid step 0.0025), giving a dense (entity x 81-bin) count/true-count array via
np.add.at; a single reverse-cumsum along the bin axis then gives, for EVERY t in the grid at once,
each entity's (m, c) -- so F0.5(t) for all 81 t-values comes from one vectorised pass, not an 81-way
loop re-scanning the table.

Usage:
    python baseline_embed_threshold.py            # steps 1-3: sweep + report, no files written
    python baseline_embed_threshold.py --apply     # steps 1-5: also write output_baseline/*.tsv
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import metric

EMBED_PATH = config.ARTIFACTS_DIR / config.EMBED_ANN_FILENAME
GT_PATH = config.records_path("train", "ground_truth")
TEST_S1_TSV = config.DATASET_DIR / "test" / "test_source1.tsv"
OUT_DIR = config.ROOT / "output_baseline"

T_LO, T_HI, T_STEP = 0.80, 1.00, 0.0025
N_BINS = int(round((T_HI - T_LO) / T_STEP)) + 1  # 81
T_GRID = T_LO + T_STEP * np.arange(N_BINS)
K_GRID = [1, 2, 3, 5, 10, 20]
DELTA_GRID = [0.005, 0.01, 0.02, float("inf")]  # inf = no delta filter


def load_ground_truth() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(source1, candidate) truth pairs, and (source1_entity_id, k) true-match count per train entity."""
    gt = pl.read_parquet(GT_PATH).select(
        "source1_entity_id",
        pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id"),
    )
    pairs = gt.explode("candidate_entity_id").filter(pl.col("candidate_entity_id") != "")
    k_per_entity = pairs.group_by("source1_entity_id").agg(pl.len().alias("k"))
    return pairs, k_per_entity


def split_ids(all_train_ids: pl.Series) -> tuple[pl.Series, pl.Series, str]:
    val_path = config.VAL_ENTITY_IDS
    if val_path.exists():
        held = pl.read_parquet(val_path)["entity_id"]
        held = held.filter(held.is_in(all_train_ids.implode()))
        tune = all_train_ids.filter(~all_train_ids.is_in(held.implode()))
        return tune, held, f"config.VAL_ENTITY_IDS ({held.len():,} held out)"
    h = all_train_ids.hash(seed=0) % 100
    held = all_train_ids.filter(h < 20)
    tune = all_train_ids.filter(h >= 20)
    return tune, held, "deterministic 80/20 hash split (VAL_ENTITY_IDS not found)"


def macro_f05_grid(c: np.ndarray, k: np.ndarray, m: np.ndarray) -> np.ndarray:
    """c, k: (n_entities,); m: (n_entities, n_t). Returns mean F0.5 per t, shape (n_t,)."""
    k = k[:, None].astype(np.float64)
    c = c.astype(np.float64)
    m = m.astype(np.float64)
    denom = 0.25 * k + m
    safe = np.where(denom > 0, denom, 1.0)
    f = np.where(denom > 0, 1.25 * c / safe, 0.0)
    f = np.where((k[:, 0] == 0)[:, None] & (m == 0), 1.0, f)  # k=0,m=0 -> 1.0 (already 0 from c=0,denom=0 branch->fixed here)
    f = np.where((k[:, 0] == 0)[:, None] & (m > 0), 0.0, f)   # k=0,m>0 -> 0.0
    return f.mean(axis=0)


def entity_index(ids: pl.Series) -> tuple[dict, int]:
    uniq = ids.unique().to_list()
    return {v: i for i, v in enumerate(uniq)}, len(uniq)


def sweep_one_k(cand_k: pl.DataFrame, idx: dict, n_entities: int, k_arr: np.ndarray, deltas=DELTA_GRID
                ) -> list[dict]:
    """cand_k: rows already filtered to channel_rank<=k_param, columns entity_id,channel_score,is_true,
    plus (for delta) each row's own entity's top1 score. Returns one dict per delta with the full
    F0.5(t) array (len N_BINS) and its argmax."""
    ent_i = np.fromiter((idx[e] for e in cand_k["source1_entity_id"].to_list()), dtype=np.int64, count=cand_k.height)
    score = cand_k["channel_score"].to_numpy()
    is_true = cand_k["is_true"].to_numpy().astype(np.int64)
    top1 = cand_k["top1"].to_numpy() if "top1" in cand_k.columns else None

    out = []
    for delta in deltas:
        if delta == float("inf") or top1 is None:
            keep = np.ones(len(ent_i), dtype=bool)
        else:
            keep = score >= (top1 - delta)
        ei, sc, tr = ent_i[keep], score[keep], is_true[keep]
        # floor, not round: a row counts toward bin j (threshold T_GRID[j]) only while score >= T_GRID[j];
        # rounding would let e.g. score=0.8013 (< 0.8025) count toward the t=0.8025 bin, over-crediting it.
        bin_idx = np.clip(np.floor((sc - T_LO) / T_STEP + 1e-9).astype(np.int64), 0, N_BINS - 1)
        flat = ei * N_BINS + bin_idx
        local_m = np.bincount(flat, minlength=n_entities * N_BINS).reshape(n_entities, N_BINS)
        local_c = np.bincount(flat, weights=tr, minlength=n_entities * N_BINS).reshape(n_entities, N_BINS)
        m = np.cumsum(local_m[:, ::-1], axis=1)[:, ::-1]  # m[e, j] = count with score >= T_GRID[j]
        c = np.cumsum(local_c[:, ::-1], axis=1)[:, ::-1]
        out.append((delta, m, c))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="also write output_baseline/*.tsv (steps 4-5)")
    ap.add_argument("--skip-delta", action="store_true", help="sweep only (t, k), skip the delta grid (faster)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    deltas = [float("inf")] if args.skip_delta else DELTA_GRID

    truth_pairs, k_lookup = load_ground_truth()
    all_train_ids = pl.scan_parquet(EMBED_PATH).filter(pl.col("split") == "train").select(
        pl.col("source1_entity_id").unique()
    ).collect()["source1_entity_id"]
    tune_ids, held_ids, split_desc = split_ids(all_train_ids)
    print(f"[1] split: {split_desc}. tune={tune_ids.len():,}  held_out={held_ids.len():,}", flush=True)

    def load_slice(ids: pl.Series) -> pl.DataFrame:
        df = (
            pl.scan_parquet(EMBED_PATH)
            .filter(pl.col("split") == "train")
            .filter(pl.col("source1_entity_id").is_in(ids.implode()))
            .select("source1_entity_id", "candidate_entity_id", "channel_rank", "channel_score", "country")
            .collect(engine="streaming")
        )
        df = df.join(truth_pairs.with_columns(pl.lit(True).alias("is_true")),
                     on=["source1_entity_id", "candidate_entity_id"], how="left").with_columns(
            pl.col("is_true").fill_null(False))
        top1 = df.filter(pl.col("channel_rank") == 1).select("source1_entity_id", pl.col("channel_score").alias("top1"))
        return df.join(top1, on="source1_entity_id", how="left")

    tune_cand = load_slice(tune_ids)
    print(f"    tuning candidate rows: {tune_cand.height:,}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    idx, n = entity_index(tune_ids)
    k_arr = tune_ids.to_frame("source1_entity_id").join(k_lookup, on="source1_entity_id", how="left").with_columns(
        pl.col("k").fill_null(0))
    # order k_arr to match idx (idx built from tune_ids.unique(); tune_ids has no dupes so order matches unique().to_list())
    order = {v: i for i, v in enumerate(tune_ids.unique().to_list())}
    k_np = np.zeros(n, dtype=np.int64)
    for row in k_arr.iter_rows(named=True):
        k_np[order[row["source1_entity_id"]]] = row["k"]
    c_np = np.zeros(n, dtype=np.int64)  # placeholder unused

    best = None
    all_rows = []
    for k_param in K_GRID:
        cand_k = tune_cand.filter(pl.col("channel_rank") <= k_param)
        for delta, m, c in sweep_one_k(cand_k, idx, n, k_np, deltas):
            k_full = k_np.astype(np.float64)
            denom = 0.25 * k_full[:, None] + m
            safe = np.where(denom > 0, denom, 1.0)
            f = np.where(denom > 0, 1.25 * c / safe, 0.0)
            singleton = k_full == 0
            f = np.where(singleton[:, None] & (m == 0), 1.0, f)
            f = np.where(singleton[:, None] & (m > 0), 0.0, f)
            f05_t = f.mean(axis=0)
            empty_t = (m == 0).mean(axis=0)
            j = int(np.argmax(f05_t))
            row = {"k_param": k_param, "delta": delta, "t": float(T_GRID[j]), "f05": float(f05_t[j]),
                  "empty_pct": float(empty_t[j])}
            all_rows.append(row)
            if best is None or row["f05"] > best["f05"]:
                best = row
        del cand_k
    print(f"[2] swept {len(all_rows):,} (k,delta) combos x {N_BINS} t-values each  "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    b = best
    print(f"    BEST on tuning set: t={b['t']:.4f} k={b['k_param']} delta={b['delta']} "
          f"-> F0.5={b['f05']:.4f}  empty%={b['empty_pct']:.2%}")

    del tune_cand
    held_cand = load_slice(held_ids)
    t_b, k_b, d_b = b["t"], b["k_param"], b["delta"]
    capped = held_cand.filter(pl.col("channel_rank") <= k_b)
    if d_b != float("inf"):
        capped = capped.filter(pl.col("channel_score") >= pl.col("top1") - d_b)
    pred = capped.filter(pl.col("channel_score") >= t_b)

    def f05_on(ids: pl.Series) -> tuple[float, float]:
        base = ids.to_frame("source1_entity_id").join(k_lookup, on="source1_entity_id", how="left").with_columns(
            pl.col("k").fill_null(0))
        m = pred.filter(pl.col("source1_entity_id").is_in(ids.implode())).group_by("source1_entity_id").agg(
            pl.col("is_true").sum().alias("c"), pl.len().alias("m"))
        full = base.join(m, on="source1_entity_id", how="left").with_columns(pl.col("c").fill_null(0), pl.col("m").fill_null(0))
        f05 = metric.macro_f05_arrays(full["c"].to_numpy(), full["k"].to_numpy(), full["m"].to_numpy())
        return f05, float((full["m"] == 0).mean())

    overall_f05, overall_empty = f05_on(held_ids)
    singleton_rate = float(held_ids.to_frame("source1_entity_id").join(k_lookup, on="source1_entity_id", how="left")
                            .with_columns(pl.col("k").fill_null(0))["k"].eq(0).mean())
    print(f"\n[3] HELD-OUT ({held_ids.len():,} entities), chosen rule t={t_b:.4f} k={k_b} delta={d_b}:")
    print(f"    overall  F0.5={overall_f05:.4f}  empty%={overall_empty:.2%}  true_singleton_rate={singleton_rate:.2%}")
    country_of = held_cand.select("source1_entity_id", "country").unique()
    for c_name in country_of["country"].unique().sort().to_list():
        ids_c = country_of.filter(pl.col("country") == c_name)["source1_entity_id"]
        f05_c, empty_c = f05_on(ids_c)
        print(f"    {c_name:8s} F0.5={f05_c:.4f}  empty%={empty_c:.2%}  n={ids_c.len():,}")

    print(f"\nTotal time so far: {time.perf_counter() - t0:.0f}s")
    print("\n*** STOP HERE per instructions -- steps 4/5/6 need explicit go-ahead. ***")

    if not args.apply:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_ids = pl.read_csv(TEST_S1_TSV, separator="\t", infer_schema=False, columns=["entity_id"])["entity_id"].rename("source1_entity_id")
    print(f"[4] test entities from test_source1.tsv: {test_ids.len():,}")

    test_cand = (
        pl.scan_parquet(EMBED_PATH).filter(pl.col("split") == "test")
        .select("source1_entity_id", "candidate_entity_id", "channel_rank", "channel_score")
        .collect(engine="streaming")
    )
    capped_t = test_cand.filter(pl.col("channel_rank") <= k_b)
    if d_b != float("inf"):
        top1_t = test_cand.filter(pl.col("channel_rank") == 1).select("source1_entity_id", pl.col("channel_score").alias("top1"))
        capped_t = capped_t.join(top1_t, on="source1_entity_id", how="left").filter(pl.col("channel_score") >= pl.col("top1") - d_b)
    pred_t = capped_t.filter(pl.col("channel_score") >= t_b)

    import pipeline_io as pio
    match_path = OUT_DIR / "matching_results.tsv"
    cand_path = OUT_DIR / "candidate_pairs.tsv"
    pio.write_id_lists(test_ids, pred_t, "matched_entity_ids", match_path)
    pio.write_id_lists(test_ids, test_cand, "candidate_entity_ids", cand_path)
    print(f"[4] wrote {match_path} and {cand_path}")

    for p in (match_path, cand_path):
        raw = p.read_bytes()
        n_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
        assert b"\r\n" not in raw, f"{p}: CRLF found, expected LF"
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{p}: BOM found"
        assert b"nan" not in raw.lower() and b"none" not in raw.lower(), f"{p}: nan/None string found"
        print(f"    {p.name}: {n_lines:,} lines (incl. header)  sha256={hashlib.sha256(raw).hexdigest()}")
    m_df = pio.read_id_lists(match_path, "matched_entity_ids")  # already long: source1_entity_id, candidate_entity_id
    c_df = pio.read_id_lists(cand_path, "candidate_entity_ids")
    joined = m_df.join(c_df, on=["source1_entity_id", "candidate_entity_id"], how="anti")
    print(f"    matches not present in candidates for same entity: {joined.height} (must be 0)")
    print(f"\nTotal time: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
