# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Write one PDB per (protein, structure arm) for the dashboard to fetch.

Predicted structures are too big to inline — six arms across 29 proteins is
tens of megabytes — so they are served as separate files next to the page and
fetched on demand. That works because the dashboard is served from GitHub
Pages, where they are same-origin.

Arms, and where they come from:

``ground_truth``
    The deposited chain, already mapped to evaluation indices by
    :mod:`build_structures`.
``protenix_v2_msa`` / ``protenix_v2_single_seq``
    Protenix-v2's own prediction, best sample by ranking score — the same
    selection rule #74 used when it read contacts off these files.
``esmfold2``
    ESMFold2's prediction.
``helico_mf_L`` / ``helico_off`` / ``helico_oracle``
    Helico ``contacts-msafree-01`` step 6000 conditioned on, respectively,
    MarinFold's top-L contacts, nothing at all, and the ground-truth contacts —
    the three arms that isolate what the contacts are worth. Published by
    helico#14 over the FoldBench monomers only, so they exist for the 5
    FoldBench members of this set and not for the CAMEO/CASP ones.

Every arm is renumbered onto the evaluation sequence, so a contact drawn at
(i, j) means the same residues in every structure. Backbone plus Cβ only: the
viewer draws cartoons from these and the full atom set would quadruple the
bytes for no visible gain.

    uv run python dashboard/build_structure_files.py
"""

import gzip
import json
import sys
import tarfile
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import gemmi
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import upstream as U  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "structures"
ARCHIVES = U.EXPERIMENT / "scratch" / "helico"

#: Atoms kept per residue: enough for a cartoon and a Cβ direction.
KEEP_ATOMS = ("N", "CA", "C", "O", "CB")

#: #74's published Protenix and ESMFold predictions over the legacy 554, which
#: is where the 24 CAMEO-hard / CASP-FM members live.
LEGACY_SOURCES = {
    "protenix_v2_msa": f"{U.BUCKET}/data/protenix-contacts-eval-exp74/best_exp65/msa/{{stem}}/structure.cif",
    "protenix_v2_single_seq": (
        f"{U.BUCKET}/data/protenix-contacts-eval-exp74/best_exp65/single_seq/{{stem}}/structure.cif"
    ),
}

#: Arms packaged as one tarball each on the helico bucket.
ARCHIVE_SOURCES = {
    "helico_mf_L": ("helico_mf_L.tar.gz", "mf_L/{stem}.pdb.gz"),
    "helico_off": ("helico_off.tar.gz", "off/{stem}.pdb.gz"),
    "helico_oracle": ("helico_oracle.tar.gz", "oracle/{stem}.pdb.gz"),
    "esmfold2": ("esmfold2_esmfold2.tar.gz", "{stem}/structure.cif"),
}

#: Protenix ships several diffusion samples per target; the best one by
#: ranking score is the prediction #74 scored.
PROTENIX_ARCHIVES = {
    "protenix_v2_msa": ("protenix_v2_msa.tar.gz", "msa"),
    "protenix_v2_single_seq": ("protenix_v2_single_seq.tar.gz", "single_seq"),
}

#: helico#14's per-target structure accuracy, which covers every arm here over
#: the FoldBench monomers. ``mf_L`` there is Helico conditioned on the #232
#: m2-p06 *sweep* checkpoint's contacts (#245's scoring run), not on the
#: step-363000 training checkpoint this experiment scores — that variant has
#: published metrics but no published structures.
HELICO_SCORES = (
    "https://huggingface.co/buckets/timodonnell/helico-experiments/resolve/"
    "exp14_foldbench_held_out_monomers/scores/per_target.csv"
)
SCORE_ARMS = {
    "mf_L": "helico_mf_L",
    "off": "helico_off",
    "oracle": "helico_oracle",
    "esmfold2": "esmfold2",
    "protenix_v2_msa": "protenix_v2_msa",
    "protenix_v2_single_seq": "protenix_v2_single_seq",
}

LABELS = {
    "ground_truth": "Ground truth",
    "protenix_v2_msa": "Protenix-v2 + MSA",
    "protenix_v2_single_seq": "Protenix-v2 single-seq",
    "esmfold2": "ESMFold2",
    "helico_mf_L": "Helico + MarinFold contacts",
    "helico_off": "Helico, no contacts",
    "helico_oracle": "Helico + ground-truth contacts",
}


def read_structure(payload: bytes, name: str) -> gemmi.Structure:
    """Parse mmCIF or PDB bytes, transparently gunzipping.

    gemmi reads from a path rather than a buffer, so the payload goes through a
    scratch file whose suffix tells it which format to expect.
    """

    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
        name = name.removesuffix(".gz")
    suffix = ".cif" if name.endswith(".cif") else ".pdb"
    scratch = ARCHIVES / f"_read{suffix}"
    scratch.write_bytes(payload)
    structure = gemmi.read_structure(str(scratch))
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    return structure


def chain_records(structure: gemmi.Structure) -> list[dict]:
    """One record per polymer chain: sequence plus the atoms we keep."""

    traces = []
    for chain in structure[0]:
        letters, residues = [], []
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if residue.find_atom("CA", "*") is None or info is None:
                continue
            if not info.is_amino_acid():
                continue
            letters.append(info.one_letter_code.upper())
            residues.append(
                {
                    "name": residue.name,
                    "atoms": [
                        (a.name, a.element.name, a.pos.x, a.pos.y, a.pos.z)
                        for a in residue
                        if a.name in KEEP_ATOMS
                    ],
                }
            )
        if len(letters) >= 10:
            traces.append({"sequence": "".join(letters), "residues": residues})
    return traces


def to_eval_pdb(structure: gemmi.Structure, eval_sequence: str) -> tuple[str, float]:
    """Renumber the best-matching chain onto the evaluation sequence."""

    best, best_coverage = None, -1.0
    for trace in chain_records(structure):
        matcher = SequenceMatcher(None, trace["sequence"], eval_sequence, autojunk=False)
        residues: list = [None] * len(eval_sequence)
        covered = 0
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                residues[block.b + offset] = trace["residues"][block.a + offset]
                covered += 1
        coverage = covered / len(eval_sequence)
        if coverage > best_coverage:
            best, best_coverage = residues, coverage

    lines, serial = [], 1
    for index, residue in enumerate(best or []):
        if residue is None:
            continue
        for name, element, x, y, z in residue["atoms"]:
            atom_name = f" {name:<3}" if len(name) < 4 else name
            lines.append(
                f"ATOM  {serial:5d} {atom_name} {residue['name']:>3} A{index + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {(element or name[0]):>2}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines), best_coverage


def best_protenix_sample(archive: tarfile.TarFile, mode: str, stem: str) -> bytes | None:
    """Return the highest-ranked Protenix sample for one target."""

    prefix = f"{mode}/{stem}/predictions/{stem}/seed_42/predictions/"
    scores: dict[int, float] = {}
    for member in archive.getmembers():
        if not member.name.startswith(prefix):
            continue
        if "summary_confidence_sample_" in member.name:
            index = int(member.name.rsplit("_", 1)[-1].removesuffix(".json"))
            payload = json.loads(archive.extractfile(member).read())
            scores[index] = float(
                payload.get("ranking_score", payload.get("ptm", 0.0))
            )
    if not scores:
        return None
    best = max(scores, key=scores.get)
    member = archive.getmember(f"{prefix}{stem}_sample_{best}.cif")
    return archive.extractfile(member).read()


def fetch(url: str) -> bytes | None:
    """Download one published structure, or return None when absent."""

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError:
        return None


def structure_accuracy() -> dict[tuple[str, str], dict]:
    """lDDT / TM-score / GDT_TS per (stem, arm), where helico#14 published it."""

    frame = pd.read_csv(HELICO_SCORES)
    frame = frame[frame.arm.isin(SCORE_ARMS) & (frame.status == "ok")]
    return {
        (row.stem, SCORE_ARMS[row.arm]): {
            "lddt": round(float(row.lddt), 3),
            "tm_score": round(float(row.tm_score), 3),
            "gdt_ts": round(float(row.gdt_ts), 3),
        }
        for row in frame.itertuples(index=False)
    }


def main() -> None:
    sequences = pd.read_csv(U.DATA / "low_depth_sequences.csv").set_index("stem")
    accuracy = structure_accuracy()
    ground_truth = json.loads((U.DATA / "low_depth_structures.json").read_text())
    low = pd.read_csv(U.DATA / "low_msa_depth_set.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    archives = {
        name: tarfile.open(ARCHIVES / filename)
        for name, (filename, _) in ARCHIVE_SOURCES.items()
        if (ARCHIVES / filename).exists()
    }
    protenix = {
        name: tarfile.open(ARCHIVES / filename)
        for name, (filename, _) in PROTENIX_ARCHIVES.items()
        if (ARCHIVES / filename).exists()
    }

    index: dict[str, list[dict]] = {}
    for record in low.itertuples(index=False):
        key = f"{record.dataset}__{record.stem}"
        eval_sequence = sequences.loc[record.stem, "input_seq"]
        directory = OUT / key
        directory.mkdir(parents=True, exist_ok=True)
        arms = []

        truth = ground_truth[key]
        if truth["available"]:
            (directory / "ground_truth.pdb").write_text(truth["pdb"])
            arms.append(
                {
                    "arm": "ground_truth",
                    "label": LABELS["ground_truth"],
                    "file": f"structures/{key}/ground_truth.pdb",
                    "coverage": truth["coverage"],
                    "source": truth["pdb_id"],
                }
            )

        candidates: list[tuple[str, bytes | None, str]] = []
        for name, (_, pattern) in ARCHIVE_SOURCES.items():
            archive = archives.get(name)
            member_name = pattern.format(stem=record.stem)
            payload = None
            if archive is not None:
                try:
                    payload = archive.extractfile(archive.getmember(member_name)).read()
                except KeyError:
                    payload = None
            candidates.append((name, payload, member_name))
        for name, (_, mode) in PROTENIX_ARCHIVES.items():
            archive = protenix.get(name)
            payload = (
                best_protenix_sample(archive, mode, record.stem) if archive else None
            )
            if payload is None and name in LEGACY_SOURCES:
                payload = fetch(LEGACY_SOURCES[name].format(stem=record.stem))
            # Protenix ships mmCIF; the origin name carries the suffix the
            # parser dispatches on.
            candidates.append((name, payload, f"{record.stem}.cif"))

        for name, payload, origin in candidates:
            if payload is None:
                continue
            text, coverage = to_eval_pdb(read_structure(payload, origin), eval_sequence)
            if coverage < 0.5:
                print(f"[skip] {key}/{name}: coverage {coverage:.2f}", flush=True)
                continue
            (directory / f"{name}.pdb").write_text(text)
            arms.append(
                {
                    "arm": name,
                    "label": LABELS[name],
                    "file": f"structures/{key}/{name}.pdb",
                    "coverage": round(coverage, 4),
                    **(accuracy.get((record.stem, name)) or {}),
                }
            )
        index[key] = arms
        print(f"[structures] {key}: {', '.join(a['arm'] for a in arms)}", flush=True)

    (HERE / "structure_index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    total = sum(
        path.stat().st_size for path in OUT.rglob("*.pdb")
    )
    print(
        json.dumps(
            {
                "proteins": len(index),
                "files": sum(len(v) for v in index.values()),
                "megabytes": round(total / 1e6, 1),
                "arms_per_protein": {
                    key: len(value) for key, value in sorted(index.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
