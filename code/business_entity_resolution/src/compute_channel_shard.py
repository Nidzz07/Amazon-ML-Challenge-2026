"""One-off: compute a SINGLE (channel, country) shard and write it straight to the s2 checkpoint part
file, skipping every other country entirely. Use this to split blocking work across two machines without
wasting time redoing a country the other machine already has.

Reuses the exact production functions (s2_block.load_shard/run_channel/write_atomic) unchanged, so the
output is byte-identical to what a normal s2_block.py run would produce for that (channel, country) --
no risk of a subtly different format.

Usage (run from this directory, with your own norm_{split}_source*.parquet already in place):
    python compute_channel_shard.py addr_tfidf US --splits train
    python compute_channel_shard.py addr_tfidf US --splits train --input DIR --output DIR

Output: <output>/s2_parts/<split>/<country>_<channel>.parquet -- drop this file into the OTHER
machine's artifacts/s2_parts/<split>/ directory (or send it back to whoever is merging).

IMPORTANT -- the manifest also needs updating, or the receiving machine's --resume will delete this file
on first touch (its manifest has no record of this channel's knobs yet). This script also PRINTS the
exact "channels"."<channel>" JSON snippet to merge into that machine's artifacts/s2_parts/<split>/
manifest.json -- send that alongside the parquet file. It is pure config, so it will match exactly on
any machine running the same commit.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import s2_block

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("channel", choices=list(config.CHANNELS))
    ap.add_argument("country")
    ap.add_argument("--splits", default="train", choices=list(config.SPLITS))
    ap.add_argument("--input", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--output", type=Path, default=config.ARTIFACTS_DIR)
    args = ap.parse_args(argv)

    split = args.splits
    name = args.channel
    bit = list(config.CHANNELS).index(name)
    module = s2_block.CHANNEL_MODULES[name]

    pdir = config.s2_parts_dir(split, args.output)
    pdir.mkdir(parents=True, exist_ok=True)
    out_path = pdir / f"{args.country}_{name}.parquet"
    if out_path.exists():
        print(f"{out_path} already exists -- delete it first if you want to recompute.")
        return

    t0 = time.perf_counter()
    s1, pool = s2_block.load_shard(split, args.input, args.country, None, module.COLUMNS)
    print(f"[{split}/{args.country}] {s1.height:,} source1 x {pool.height:,} pool", flush=True)
    out, row = s2_block.run_channel(split, args.country, bit, name, module, s1, pool, False, True)
    del s1, pool
    s2_block.write_atomic(out, out_path)
    del out
    print(f"-> {out_path}  {row['pairs']:,} pairs  {row['entities']:,} entities  {time.perf_counter() - t0:.0f}s")

    snippet = {name: s2_block.manifest_channel(name, False)}
    print(f"\nMerge this into the RECEIVING machine's {pdir}/manifest.json under \"channels\":")
    print(json.dumps(snippet, indent=2))


if __name__ == "__main__":
    sys.exit(main())
