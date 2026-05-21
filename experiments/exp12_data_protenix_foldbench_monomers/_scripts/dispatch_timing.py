"""Run predict_one with ``force_run=True`` for a curated stems-file subset
to capture per-(stem, mode) timing data.

The stems we re-run are spread log-uniformly across the 30-761 aa
sequence-length range so the resulting ``data/timings.csv`` covers
short and long FoldBench monomers.

Reads ``/tmp/timing_stems.txt`` (one stem per line); both modes are
re-run for each. ``force_run=True`` bypasses the idempotency check, so
the existing outputs on the Modal Volume are overwritten in place —
the resulting structure/distogram/confidence files are equivalent (the
seeds + n_sample are unchanged) but they get a fresh ``timings.json``
sibling per (mode, stem).

MSA timings are only meaningful when the precomputed ColabFold cache is
already present. This script therefore audits the cache before
launching any MSA job and exits non-zero if any selected stem would
fall back to live MSA search.

Profile selection: stems that originally ran on ``timodonnell`` are
re-run on whichever profile is currently active. The output goes to
that profile's ``foldbench-protenix-runs`` Volume — for the 48
"originals" we'll be writing the timings to ``open-athena`` (since
that's the only workspace currently with an active token). The 52
"backfills" are already on ``open-athena``. All timings end up in the
same Volume.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modal_app  # noqa: E402


STEMS_FILE = Path("/tmp/timing_stems.txt")


def _require_cached_msas(audit_results: list[dict]) -> None:
    """Fail fast if any stem would have to compute an MSA during timing."""
    missing = sorted(
        result["stem"]
        for result in audit_results
        if not (result.get("paired_exists") and result.get("non_pairing_exists"))
    )
    if not missing:
        return
    stems = ", ".join(missing)
    raise SystemExit(
        "missing cached MSA for timing run; precompute these stems first: "
        f"{stems}"
    )


def _wait_for_results(futures: list, call_args: list[dict]) -> None:
    """Wait for Modal futures and fail the run if any job fails."""
    failures: list[str] = []
    for fut, kw in zip(futures, call_args):
        stem = kw["stem"]
        mode = kw["mode"]
        try:
            result = fut.get()
            elapsed = result.get("elapsed_seconds")
            model_load = result.get("model_load_seconds")
            if elapsed is None or model_load is None:
                raise ValueError(f"missing timing fields in result: {result}")
            print(
                f"[done] {mode}/{stem}: "
                f"elapsed={elapsed:.1f}s "
                f"load={model_load:.1f}s"
            )
        except Exception as e:  # noqa: BLE001
            failures.append(f"{mode}/{stem}: {e}")
            print(f"[FAIL] {mode}/{stem}: {e}")
    if failures:
        joined = "\n".join(f"  - {failure}" for failure in failures)
        raise SystemExit(f"timing jobs failed:\n{joined}")


def main() -> None:
    stems_to_time = [s.strip() for s in STEMS_FILE.read_text().splitlines() if s.strip()]
    if not stems_to_time:
        raise SystemExit(f"empty stems file: {STEMS_FILE}")
    with open("inputs/manifest.csv") as f:
        manifest = list(csv.DictReader(f))
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
        audit_results = list(modal_app.audit_msa.starmap((stem,) for stem in stems_to_time))
        _require_cached_msas(audit_results)
        print(f"verified cached MSA for {len(stems_to_time)} stems")

        worker = modal_app.ProtenixWorker()
        # .spawn each so we can collect results as they finish.
        futures = [worker.predict_one.spawn(**kw) for kw in call_args]
        _wait_for_results(futures, call_args)

    print("\nDone. Per-(mode, stem) timings written to "
          "/outputs/{mode}/{stem}/timings.json on the foldbench-protenix-runs Volume.")


if __name__ == "__main__":
    main()
