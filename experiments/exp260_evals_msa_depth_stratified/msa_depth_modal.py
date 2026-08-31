# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2 — measure ColabFold MSA depth for every protein in the universe.

Both depths come off Modal volumes that already hold the a3m files Protenix's
``+MSA`` arm ran with, so nothing re-searches a database and no alignment leaves
Modal (a deep a3m is tens of MB; 391 of them are not worth the egress). Two
numbers per protein, both from #74's pinned ``msa_depth.py`` — one definition,
one code path, both volumes:

``n_seqs``
    Raw sequence count including the query. This is "MSA depth" in the plain
    sense and the axis the tiers cut on.
``n_eff``
    Redundancy-weighted count (Meff) at 80 % and 62 % identity. A thousand
    near-duplicate homologs count once; this is closer to what a coevolution
    method can actually use, and it populates the shallow end that raw depth
    leaves nearly empty.

The 11 stems both volumes hold are measured twice, once per volume. They are the
only handle we have on how much the ~3-week gap between the two ColabFold runs
moved the numbers.

    uv run modal run msa_depth_modal.py                 # writes data/msa_depth.csv
    uv run modal run msa_depth_modal.py --limit 5       # smoke test
"""

import json
import time

import modal
import pandas as pd
import upstream as U

app = modal.App("exp260-msa-depth")

#: Three eval-denovo designs have no a3m on ``protenix-foldbench-msa`` (it holds
#: 330 of the 333 scorable monomers). They are designed proteins, so they are
#: outside the natural stratification this experiment is built on, and their
#: absence is recorded rather than fatal. Any *other* missing protein is a real
#: problem and stops the run.
KNOWN_MISSING = {"8ju8_A", "8k7o_A", "8oys_A"}

#: The depth computation is pure numpy over the a3m; pandas rides along only
#: because Modal imports this whole module inside the container.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=2,<3", "pandas>=2.2,<3")
    .add_local_file(U.EXP74 / "msa_depth.py", "/root/msa_depth.py", copy=True)
    # ``upstream`` lives beside this file and the remote functions read the
    # volume names and a3m layout from it, so it has to travel with them.
    .add_local_python_source("upstream")
)

VOLUMES = {
    f"/msa/{name}": modal.Volume.from_name(volume)
    for name, volume in U.MSA_VOLUMES.items()
}


@app.function(image=image, volumes=VOLUMES, cpu=1.0, timeout=600)
def list_stems() -> dict[str, list[str]]:
    """Return the stems each volume holds, so overlaps can be measured twice."""

    from pathlib import Path

    return {
        name: sorted(
            entry.name
            for entry in Path(f"/msa/{name}").iterdir()
            if (entry / "msa/0/0/non_pairing.a3m").exists()
        )
        for name in U.MSA_VOLUMES
    }


@app.function(image=image, volumes=VOLUMES, cpu=8.0, timeout=3600, max_containers=60)
def measure(spec: dict) -> dict:
    """Depth metrics for one ``(volume, stem)`` pair.

    Neff is O(N^2 L) and the deepest MSAs here run to ~19,000 sequences, which
    is why this asks for a whole container's worth of CPU per protein and
    records its own wall time.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    import msa_depth as md

    stem = spec["stem"]
    volume = spec["msa_volume"]
    path = Path(f"/msa/{volume}") / U.MSA_PATH.format(stem=stem)
    row = {"stem": stem, "msa_volume": volume, "found": path.exists()}
    if not row["found"]:
        return row
    started = time.monotonic()
    text = path.read_text()
    depth = md.msa_depth(text)
    row.update(
        {
            "a3m_bytes": len(text),
            "n_seqs": depth.n_seqs,
            "query_len": depth.query_len,
            **{f"n_eff_{threshold}": value for threshold, value in depth.n_eff.items()},
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return row


@app.local_entrypoint()
def main(limit: int = 0) -> None:
    """Measure every universe protein, plus both copies of every shared stem."""

    universe = pd.read_csv(U.DATA / "universe.csv")
    specs = universe[["stem", "msa_volume"]].to_dict("records")

    available = list_stems.remote()
    shared = sorted(set(available["foldbench"]) & set(available["exp74"]))
    scheduled = {(record["stem"], record["msa_volume"]) for record in specs}
    universe_stems = set(universe.stem)
    cross_check = [
        {"stem": stem, "msa_volume": volume}
        for stem in shared
        if stem in universe_stems
        for volume in U.MSA_VOLUMES
        if (stem, volume) not in scheduled
    ]
    specs.extend(cross_check)
    print(
        f"{len(universe)} universe proteins, {len(shared)} stems in both volumes, "
        f"{len(cross_check)} cross-check measurements"
    )
    if limit:
        specs = specs[:limit]

    rows = list(measure.map(specs))
    frame = pd.DataFrame(rows).sort_values(["msa_volume", "stem"], ignore_index=True)

    # Write before validating: these measurements cost real compute and a
    # validation failure should leave them on disk to look at.
    destination = U.DATA / ("msa_depth.csv" if not limit else "msa_depth.smoke.csv")
    frame.to_csv(destination, index=False)

    missing = set(frame.loc[~frame.found, "stem"])
    unexpected = missing - KNOWN_MISSING
    if unexpected and not limit:
        raise RuntimeError(f"no a3m for {sorted(unexpected)}")
    if missing and not limit:
        print(f"no a3m for the known-absent designs: {sorted(missing)}")
    print(
        json.dumps(
            {
                "rows": len(frame),
                "missing": sorted(set(frame.loc[~frame.found, "stem"])),
                "cpu_seconds": float(frame.elapsed_seconds.sum()),
                "slowest_seconds": float(frame.elapsed_seconds.max()),
                "median_n_seqs": float(frame.n_seqs.median()),
                "out": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
