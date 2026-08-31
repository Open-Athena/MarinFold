# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a Helico target set for the CAMEO-hard / CASP-FM low-depth proteins.

helico#14 ran only the FoldBench monomers, so the Helico arms cover 18 of this
set's 42 proteins and the structure comparison has to be taken on the subset
they share. This builds the missing 24 as a Helico target directory —
``targets.csv``, ``gt/<target_id>.cif.gz``, ``arms/<arm>.json`` — which
``modal/bench_byclass.py`` consumes directly.

**The index convention is the whole risk here.** Helico tokenizes the residues
of the ground-truth structure it is given; MarinFold indexes the evaluation
prompt. helico#14's ``build_index_map.py`` exists because those disagree on 281
of its 333 targets, and it warns that feeding prompt indices straight in "would
shift contacts and look exactly like real contacts do not help".

This side-steps the mapping rather than re-deriving it: the ground truth handed
to Helico is the structure this experiment already renumbered onto evaluation
indices, so a residue's token index is simply its rank among the residues
present. :func:`verify` then runs the same control helico#14 uses — the
evaluation's own ground-truth contacts, pushed through the map, have to land on
residue pairs that are actually in contact in the structure being supplied.

    uv run python helico/build_targets.py
"""

import gzip
import json
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import upstream as U  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = U.EXPERIMENT / "scratch" / "helico_cameo" / "data"
DASHBOARD = U.EXPERIMENT / "dashboard"

#: Cα distance under which a ground-truth contact should sit. pyconfind counts
#: all-atom contacts, so a true contact can be a little further apart at the Cα;
#: 14 Å is loose enough not to fire on correct maps and tight enough that a
#: shifted map (which scatters pairs at random) fails immediately.
CONTACT_CA_LIMIT = 14.0
MIN_SEPARATION = 6
MIN_DEGREE = 0.001


def ground_truth_records(units: set[tuple[str, str]]) -> dict[str, dict]:
    """#89's ground-truth records for the units we are building."""

    records = {}
    for url in (U.LEGACY_GROUND_TRUTH_URL, U.FOLDBENCH_GROUND_TRUTH_URL):
        import urllib.request

        with urllib.request.urlopen(url) as response:
            for line in response.read().decode().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if (record["dataset"], record["stem"]) in units:
                    records[record["stem"]] = record
    return records


def structure_residues(path: Path) -> tuple[gemmi.Structure, list[int], np.ndarray]:
    """Return the structure, its evaluation indices in order, and Cα coords."""

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    indices, coordinates = [], []
    for residue in structure[0][0]:
        atom = residue.find_atom("CA", "*")
        if atom is None:
            continue
        indices.append(residue.seqid.num - 1)
        coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return structure, indices, np.asarray(coordinates)


def verify(stem: str, indices: list[int], coordinates: np.ndarray, record: dict) -> dict:
    """helico#14's control: do the evaluation's own contacts land on contacts?"""

    position = {index: rank for rank, index in enumerate(indices)}
    checked = far = 0
    for i, j, degree in record["contacts"]:
        if degree < MIN_DEGREE or (j - i) < MIN_SEPARATION:
            continue
        if i not in position or j not in position:
            continue
        distance = float(
            np.linalg.norm(coordinates[position[i]] - coordinates[position[j]])
        )
        checked += 1
        far += distance > CONTACT_CA_LIMIT
    return {
        "stem": stem,
        "contacts_checked": checked,
        "contacts_beyond_limit": int(far),
        "fraction_beyond_limit": round(far / checked, 4) if checked else None,
    }


def main() -> None:
    low = pd.read_csv(U.DATA / "low_msa_depth_set.csv")
    missing = low[low.dataset != "foldbench_monomer"]
    units = set(zip(missing.dataset, missing.stem, strict=True))
    records = ground_truth_records(units)
    contacts = json.loads(
        (HERE.parent / "scratch" / "marinfold_low_depth_contacts.json").read_text()
    )["proteins"]

    (OUT / "gt").mkdir(parents=True, exist_ok=True)
    (OUT / "arms").mkdir(parents=True, exist_ok=True)
    rows, arm, checks = [], {}, []
    for record in missing.itertuples(index=False):
        key = f"{record.dataset}__{record.stem}"
        source = DASHBOARD / "structures" / key / "ground_truth.pdb"
        structure, indices, coordinates = structure_residues(source)

        # Helico's loader keys residues on `label_seq_id`. gemmi leaves that
        # field as "." unless it is assigned explicitly, and a file without it
        # silently collapses to one residue per residue *type* — 401 residues
        # became 19 tokens, every arm scored ~0.33, and the oracle arm derived
        # no contacts at all. Assign it, and check the round trip below.
        structure.setup_entities()
        structure.assign_label_seq_id(True)
        document = structure.make_mmcif_document()
        (OUT / "gt" / f"{record.stem}.cif.gz").write_bytes(
            gzip.compress(document.as_string().encode())
        )

        # Guard the exact field that broke, not a gemmi round trip: gemmi reads
        # its own output happily with `label_seq_id` unset, so a round-trip
        # check passes while Helico still folds a stub. Every ATOM row must
        # carry a numeric label_seq_id, and they must be as many as there are
        # residues.
        rows_written = [
            line.split() for line in document.as_string().splitlines()
            if line.startswith("ATOM")
        ]
        sequence_ids = {row[8] for row in rows_written}
        if "." in sequence_ids or len(sequence_ids) != len(indices):
            raise ValueError(
                f"{record.stem}: mmCIF has {len(sequence_ids)} distinct label_seq_id "
                f"for {len(indices)} residues — Helico would collapse them"
            )

        position = {index: rank for rank, index in enumerate(indices)}
        pairs = []
        for i, j, _votes in contacts[key]["top_pairs"][: int(record.L)]:
            if i in position and j in position:
                pairs.append([position[i], position[j]])
        arm[record.stem] = pairs

        sequence = gemmi.one_letter_code(
            [r.name for r in structure[0][0] if r.find_atom("CA", "*")]
        ).upper()
        rows.append(
            {
                "target_id": record.stem,
                "eval_set": record.dataset,
                "dataset": record.dataset,
                "stem": record.stem,
                "pdb_id": "",
                "gt_chain": structure[0][0].name,
                "L_helico": len(indices),
                "L_exp245": int(record.L),
                "n_resolved": len(indices),
                "is_viral": 0,
                "designed": int(bool(record.designed)),
                "kingdom": "",
                "exp199_stratum": "",
                "deposit_date": "",
                "initial_release_date": "",
                "msa_available": 0,
                "input_seq": sequence,
            }
        )
        checks.append(verify(record.stem, indices, coordinates, records[record.stem]))
        print(
            f"[targets] {record.stem}: {len(indices)} residues, {len(pairs)} contact pairs",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(OUT / "targets.csv", index=False)
    (OUT / "arms" / "mf_L_363k.json").write_text(json.dumps(arm))
    report = pd.DataFrame(checks)
    report.to_csv(U.DATA / "helico_index_control.csv", index=False)
    worst = report.sort_values("fraction_beyond_limit", ascending=False).head(3)
    print(
        json.dumps(
            {
                "targets": len(rows),
                "contact_pairs": sum(len(v) for v in arm.values()),
                "control_contacts_checked": int(report.contacts_checked.sum()),
                "control_fraction_beyond_limit": round(
                    float(report.contacts_beyond_limit.sum() / report.contacts_checked.sum()), 5
                ),
                "worst_targets": worst.to_dict("records"),
                "out": str(OUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
