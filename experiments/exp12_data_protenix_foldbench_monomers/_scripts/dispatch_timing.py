"""Run predict_one with `force_run=True` for a curated stems-file subset
to capture per-(stem, mode) timing data.

The stems we re-run are spread log-uniformly across the 30-761 aa
sequence-length range used by exp20, so the resulting timings.csv
plot will overlay cleanly on exp20's
``data/timings.csv`` / ``plots/timing_vs_sequence_length.png``.

Reads ``/tmp/timing_stems.txt`` (one stem per line); both modes are
re-run for each. ``force_run=True`` bypasses the idempotency check, so
the existing outputs on the Modal Volume are overwritten in place —
the resulting structure/distogram/confidence files are equivalent (the
seeds + n_sample are unchanged) but they get a fresh ``timings.json``
sibling per (mode, stem).

Profile selection: stems that originally ran on ``timodonnell`` are
re-run on whichever profile is currently active. The output goes to
that profile's ``foldbench-protenix-runs`` Volume — for the 48
"originals" we'll be writing the timings to ``open-athena`` (since
that's the only workspace currently with an active token). The 52
"backfills" are already on ``open-athena``. All timings end up in the
same Volume.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modal_app  # noqa: E402


STEMS_FILE = Path("/tmp/timing_stems.txt")


def main() -> None:
    stems_to_time = [s.strip() for s in STEMS_FILE.read_text().splitlines() if s.strip()]
    if not stems_to_time:
        raise SystemExit(f"empty stems file: {STEMS_FILE}")
    manifest = list(csv.DictReader(open("inputs/manifest.csv")))
    by_stem = {r["stem"]: r for r in manifest}
    missing = [s for s in stems_to_time if s not in by_stem]
    if missing:
        raise SystemExit(f"stems not in manifest: {missing}")

    modes = ["single_seq", "msa"]
    call_args: list[dict] = []
    for stem in stems_to_time:
        row = by_stem[stem]
        job_json_str = (Path("inputs") / row["job_json"]).read_text()
        for mode in modes:
            call_args.append({
                "job_json_str": job_json_str,
                "stem": stem,
                "mode": mode,
                "seeds": [1, 2, 3, 4, 5],
                "n_sample": 8,
                "n_cycle": 10,
                "force_run": True,
            })
    print(
        f"dispatching {len(call_args)} timing jobs "
        f"({len(stems_to_time)} stems × {len(modes)} modes); "
        f"size range {min(int(by_stem[s]['n_residues']) for s in stems_to_time)}-"
        f"{max(int(by_stem[s]['n_residues']) for s in stems_to_time)} aa"
    )

    with modal_app.app.run():
        worker = modal_app.ProtenixWorker()
        # .spawn each so we can collect results as they finish.
        futures = [worker.predict_one.spawn(**kw) for kw in call_args]
        for fut, kw in zip(futures, call_args):
            try:
                result = fut.get()
                print(
                    f"[done] {kw['mode']}/{kw['stem']}: "
                    f"elapsed={result.get('elapsed_seconds'):.1f}s "
                    f"load={result.get('model_load_seconds'):.1f}s"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {kw['mode']}/{kw['stem']}: {e}")

    print("\nDone. Per-(mode, stem) timings written to "
          "/outputs/{mode}/{stem}/timings.json on the foldbench-protenix-runs Volume.")


if __name__ == "__main__":
    main()
