# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize an exp208 environment trace — issue #208.

The question these traces exist to answer is "did the rollout worker restart?",
so the summary is organised by BOOT ID: one interpreter, one boot id. Several
boot ids means the process is being recycled, which is what the nano runs
suggested but could not confirm.

    uv run python read_trace.py --path gs://.../exp208/trace/<run-name>
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import fsspec


def load(path: str, limit: int | None) -> list[dict]:
    fs, _ = fsspec.core.url_to_fs(path)
    files = sorted(fs.find(path))
    if limit:
        files = files[-limit:]

    def read(f):
        try:
            with fs.open(f, "r") as fh:
                return json.load(fh)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=32) as pool:
        return [e for e in pool.map(read, files) if e]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the last N events")
    ap.add_argument("--errors", action="store_true", help="print full tracebacks")
    a = ap.parse_args()

    events = load(a.path, a.limit)
    if not events:
        print(f"no events under {a.path}")
        return 1
    events.sort(key=lambda e: e["t"])
    print(f"{len(events)} events over {(events[-1]['t'] - events[0]['t']) / 60:.1f} min\n")

    boots: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        boots[e["boot"]].append(e)

    print(f"{'boot':14s} {'events':>7s} {'uptime_s':>9s} {'calls':>6s} {'kinds'}")
    for boot, evs in sorted(boots.items(), key=lambda kv: kv[1][0]["t"]):
        kinds = defaultdict(int)
        for e in evs:
            kinds[e["kind"]] += 1
        calls = max((e.get("call", 0) for e in evs), default=0)
        print(f"{boot:14s} {len(evs):7d} {evs[-1]['uptime_s']:9.1f} {calls:6d} "
              f"{dict(sorted(kinds.items()))}")

    hosts = {e["host"] for e in events}
    print(f"\nDISTINCT BOOTS: {len(boots)} across {len(hosts)} host(s)")
    if len(boots) > len(hosts):
        print("  -> more boots than hosts: some process RESTARTED")
    else:
        print("  -> one boot per host: no restart (N rollout workers give N boot ids, "
              "which is expected, not a restart)")

    done = [e for e in events if e["kind"] == "sample_done"]
    if done:
        gen = [e["gen_s"] for e in done]
        tot = [e["total_s"] for e in done]
        print(f"\nsample_done: {len(done)}  gen {min(gen):.1f}-{max(gen):.1f}s  "
              f"total {min(tot):.1f}-{max(tot):.1f}s")
        last = done[-1]
        for k in ("lesson", "n_rollouts", "n_groups", "n_empty", "n_dropped",
                  "p_bar_before", "p_bar_after", "best_f1", "n_pred", "max_output_tokens"):
            print(f"    {k:18s} {last.get(k)}")

    fails = [e for e in events if e["kind"] in ("sample_failed", "signal")]
    print(f"\nfailures/signals: {len(fails)}")
    for e in fails[-5:]:
        print(f"  [{e['kind']}] boot={e['boot'][:6]} {e.get('error') or e.get('name')}: "
              f"{str(e.get('message', ''))[:160]}")
        if a.errors and e.get("traceback"):
            print(e["traceback"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
