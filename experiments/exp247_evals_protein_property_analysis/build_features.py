# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1 — one row per natural monomer, with everything that might explain it.

#245 left 314 natural proteins each carrying nine per-protein contact scores and
almost no explanation. This assembles the candidate explanations into one matrix,
grouped by where they come from and what they could plausibly drive:

``size / shape``
    length, resolved fraction, radius of gyration, contacts per residue.
``contact structure``
    mean contact order, fraction of long-range contacts, per-range counts. The
    classic difficulty axes, and the ones the metric's own denominator depends on
    — R-precision takes the top-N where N is the number of true contacts, so a
    protein's contact count is inside the metric as well as inside the biology.
``secondary structure``
    helix / sheet / coil fraction and SSE count, from ``annotate_sse`` (P-SEA) on
    the ground-truth chain. All-α proteins are mostly local contacts; β-rich ones
    carry the long-range pairings a single-sequence model has to infer.
``training support``
    best identity to the *pre*-decontamination corpora (pooled and per arm),
    number of significant homologs, and #94's KNN neighbour count and bitscore.
    This is the axis H1 is about, and #245 makes it well defined: the ≥30 %/50 %
    band was *removed* from the corpora the #232 checkpoints trained on, so what
    is left is a real gradient rather than a leak.
``MSA``
    depth and Neff of the colabfold MSA the Protenix +MSA arm ran with. A
    property of the protein, so every predictor is regressed on it — the point is
    which predictors it explains.
``domains / function / localisation``
    RCSB entity annotations (Pfam, CATH, SCOP, InterPro, GO, EC) and UniProt
    (domain features, subcellular location, transmembrane and signal segments).
``taxonomy / composition``
    kingdom and viral flag from #245; amino-acid class fractions, hydrophobicity
    and low-complexity fraction from the sequence.

Missing annotation is left missing. Recent PDB entries have no CATH assignment
and some have no UniProt cross-reference at all; imputing those would invent the
very signal the analysis is looking for, so `coverage.csv` reports what fraction
of proteins each feature actually has.

    uv run python build_features.py                 # all sources, ~4 min
    uv run python build_features.py --skip msa      # skip the Modal streaming
"""
import argparse
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import upstream as U

DATA = U.DATA
OUT = DATA / "protein_features.csv"
COVERAGE = DATA / "coverage.csv"
PROVENANCE = DATA / "features.provenance.json"

#: Kyte-Doolittle, for the mean-hydropathy feature.
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AA_CLASSES = {
    "hydrophobic": set("AVILMFWC"),
    "polar": set("STNQY"),
    "charged": set("DEKR"),
    "glycine": set("G"),
    "proline": set("P"),
}
#: A contact is "long-range" at this sequence separation — exp89's `long` range.
LONG_SEPARATION = 24
#: Window and identity ceiling for the low-complexity fraction.
LOW_COMPLEXITY_WINDOW = 20
LOW_COMPLEXITY_ENTROPY = 3.0


def base_table() -> pd.DataFrame:
    """The 314 natural, scorable monomers with what #245 already knows."""
    sets = pd.read_csv(U.EVAL_SETS)
    frame = sets[(sets.scorable == 1) & sets.eval_set.isin(U.NATURAL_SETS)].copy()
    keep = ["stem", "eval_set", "pdb_id", "chain_id", "entity_id", "entry_id",
            "seq_len", "sequence", "kingdom", "is_viral", "deposit_date",
            "n_uniprot_xrefs", "title", "source_organisms",
            "exp199_best_identity", "exp199_afdb_best_identity",
            "exp199_esm_atlas_best_identity", "exp199_stratum"]
    frame = frame[keep].rename(columns={"seq_len": "length"})
    return frame.reset_index(drop=True)


def contact_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Contact-map geometry from the frozen ground truth."""
    records = {}
    for line in U.GT_UNIVERSE.read_text().splitlines():
        record = json.loads(line)
        records[record["stem"]] = record
    rows = []
    for stem, length in zip(frame.stem, frame.length, strict=True):
        record = records[stem]
        pairs = [(i, j) for i, j, _ in record["contacts"]]
        separations = np.array([abs(j - i) for i, j in pairs], dtype=float)
        n = len(pairs)
        rows.append({
            "stem": stem,
            "n_contacts": n,
            "contacts_per_residue": n / length,
            "resolved_fraction": record["n_resolved"] / record["L"],
            "mean_contact_order": float(separations.mean()) if n else np.nan,
            "relative_contact_order": float(separations.mean() / length) if n else np.nan,
            "median_contact_separation": float(np.median(separations)) if n else np.nan,
            "frac_long_contacts": float((separations >= LONG_SEPARATION).mean()) if n else np.nan,
            "n_long_contacts": int((separations >= LONG_SEPARATION).sum()),
            "long_contacts_per_residue": float((separations >= LONG_SEPARATION).sum() / length),
        })
    return pd.DataFrame(rows)


def sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Composition, hydropathy and low-complexity content."""
    rows = []
    for stem, sequence in zip(frame.stem, frame.sequence, strict=True):
        residues = [r for r in sequence.upper() if r.isalpha()]
        counts = Counter(residues)
        total = max(1, len(residues))
        row = {"stem": stem}
        for name, members in AA_CLASSES.items():
            row[f"frac_{name}"] = sum(counts[r] for r in members) / total
        row["mean_hydropathy"] = float(
            np.mean([HYDROPATHY.get(r, 0.0) for r in residues]) if residues else np.nan)
        # Shannon entropy in a sliding window; a window below the threshold is
        # low-complexity (repeats, homopolymers, coiled-coil heptads).
        low = 0
        for start in range(0, max(1, len(residues) - LOW_COMPLEXITY_WINDOW + 1)):
            window = residues[start:start + LOW_COMPLEXITY_WINDOW]
            if len(window) < LOW_COMPLEXITY_WINDOW:
                break
            frequencies = np.array(list(Counter(window).values()), dtype=float)
            frequencies /= frequencies.sum()
            entropy = float(-(frequencies * np.log2(frequencies)).sum())
            low += entropy < LOW_COMPLEXITY_ENTROPY
        windows = max(1, len(residues) - LOW_COMPLEXITY_WINDOW + 1)
        row["frac_low_complexity"] = low / windows
        rows.append(row)
    return pd.DataFrame(rows)


def structure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Secondary-structure content and radius of gyration from the GT chain."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    manifest = pd.read_csv(U.GT_MANIFEST).set_index("stem")
    rows = []
    for stem in frame.stem:
        record = manifest.loc[stem]
        path = U.CIF_CACHE / record.gt_cif
        row = {"stem": stem}
        try:
            block = pdbx.CIFFile.read(str(path)).block
            array = pdbx.get_structure(block, model=1)
            chain = array[(array.chain_id == record.gt_chain)
                          & struc.filter_amino_acids(array)]
            if chain.array_length() == 0:  # auth vs label naming in the assembly
                chain = array[struc.filter_amino_acids(array)]
            calpha = chain[chain.atom_name == "CA"]
            sse = struc.annotate_sse(chain)
            total = max(1, len(sse))
            row["frac_helix"] = float((sse == "a").sum() / total)
            row["frac_sheet"] = float((sse == "b").sum() / total)
            row["frac_coil"] = float((sse == "c").sum() / total)
            # Number of contiguous secondary-structure elements.
            row["n_sse"] = int(sum(1 for index in range(1, len(sse))
                                   if sse[index] != sse[index - 1]
                                   and sse[index] in ("a", "b")))
            row["radius_of_gyration"] = float(struc.gyration_radius(calpha))
            row["rg_over_length"] = row["radius_of_gyration"] / len(calpha) ** 0.4
        except Exception as error:  # noqa: BLE001 — a parse failure is data, not a crash
            row["structure_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return pd.DataFrame(rows)


def training_support(frame: pd.DataFrame) -> pd.DataFrame:
    """How much similar sequence the pre-decontamination corpora held."""
    identity = pd.read_csv(U.IDENTITY_TABLE)
    identity = identity[identity.dataset.isin(("foldbench100", "foldbench_rest"))]
    identity = identity.drop_duplicates("stem").set_index("stem")
    residual = pd.read_csv(U.RESIDUAL_IDENTITY).drop_duplicates("stem").set_index("stem")
    knn = pd.read_csv(U.KNN_SUMMARY)
    knn["stem"] = knn["query"].str.split("__").str[-1]
    knn = knn.drop_duplicates("stem").set_index("stem")

    rows = []
    for stem in frame.stem:
        row = {"stem": stem}
        if stem in identity.index:
            source = identity.loc[stem]
            row.update({
                "train_n_hits_significant": float(source.n_hits_significant),
                "train_log_n_hits": math.log10(1 + float(source.n_hits_significant)),
                "train_best_evalue_log": (
                    -math.log10(max(float(source.best_evalue), 1e-300))
                    if pd.notna(source.best_evalue) else np.nan),
                "train_afdb_n_hits": float(source.afdb_n_hits_significant),
                "train_esm_n_hits": float(source.esm_atlas_n_hits_significant),
            })
        if stem in residual.index:
            source = residual.loc[stem]
            row.update({
                "residual_identity_cov40": source.best_surviving_identity_cov40,
                "residual_identity_cov30": source.best_surviving_identity_cov30,
                "residual_identity_anycov": source.best_surviving_identity_cov00,
                "n_surviving_alignments": source.n_alignments_surviving,
            })
        if stem in knn.index:
            source = knn.loc[stem]
            row.update({
                "knn_n_hits": float(source.n_hits),
                "knn_best_bits": float(source.best_bits),
                "knn_best_identity": float(source.best_fident),
                "knn_best_qcov": float(source.best_qcov),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def msa_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Depth and Neff of the colabfold MSA the +MSA arm actually used.

    Streamed from the Modal volume and counted in flight rather than downloaded:
    a deep MSA is tens of MB and nothing here needs the alignment itself.
    ``msa_depth`` is the sequence count; ``msa_n_near_identical`` counts rows at
    ≥80 % identity to the query (close relatives rather than remote ones), and
    ``msa_mean_coverage`` is how much of the query the average row spans.
    """
    import modal

    handle = modal.Volume.from_name(U.MSA_VOLUME)
    rows = []
    for index, stem in enumerate(frame.stem, 1):
        row = {"stem": stem}
        try:
            payload = b"".join(handle.read_file(U.MSA_PATH.format(stem=stem)))
        except Exception:  # noqa: BLE001 — absent MSA is reported, not fatal
            rows.append(row)
            continue
        # Counted in one streaming pass: a deep MSA is tens of MB and the
        # dense [n_sequences, length] matrix a vectorised version would build is
        # the only part of this that could not fit in memory.
        text = payload.decode("utf-8", errors="replace")
        query = None
        aligned = None
        depth = 0
        near_identical = 0
        coverage_sum = 0.0

        def consume(sequence: str) -> None:
            nonlocal query, depth, near_identical, coverage_sum
            # a3m lower-case columns are insertions relative to the query.
            columns = "".join(c for c in sequence if not c.islower())
            if query is None:
                query = columns
                return
            columns = columns[: len(query)].ljust(len(query), "-")
            depth += 1
            matches = sum(1 for a, b in zip(columns, query, strict=True) if a == b)
            near_identical += (matches / max(1, len(query))) >= 0.8
            coverage_sum += sum(1 for c in columns if c != "-") / max(1, len(query))

        for line in text.splitlines():
            if line.startswith(">"):
                if aligned is not None:
                    consume("".join(aligned))
                aligned = []
            elif aligned is not None:
                aligned.append(line.strip())
        if aligned:
            consume("".join(aligned))
        if query is None:
            rows.append(row)
            continue
        row["msa_depth"] = int(depth + 1)
        row["msa_log_depth"] = float(math.log10(depth + 1))
        row["msa_n_near_identical"] = int(near_identical)
        row["msa_mean_coverage"] = float(coverage_sum / depth) if depth else np.nan
        rows.append(row)
        if index % 50 == 0:
            print(f"  [msa] {index}/{len(frame)}", flush=True)
    return pd.DataFrame(rows)


def post(url: str, payload: dict, *, retries: int = 5) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        request = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


ENTITY_QUERY = """
query($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    rcsb_polymer_entity_annotation { type name annotation_id }
    rcsb_polymer_entity_container_identifiers {
      reference_sequence_identifiers { database_accession database_name }
    }
    rcsb_polymer_entity { pdbx_description rcsb_enzyme_class_combined { ec } }
  }
}
"""


def rcsb_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Domain and function annotation per polymer entity."""
    entity_ids = sorted(set(frame.entity_id))
    annotations = {}
    for start in range(0, len(entity_ids), 40):
        batch = entity_ids[start:start + 40]
        payload = post(U.RCSB_GRAPHQL, {"query": ENTITY_QUERY, "variables": {"ids": batch}})
        if "errors" in payload:
            raise RuntimeError(payload["errors"])
        for entity in payload["data"]["polymer_entities"] or []:
            annotations[entity["rcsb_id"].upper()] = entity
        print(f"  [rcsb] {min(start + 40, len(entity_ids))}/{len(entity_ids)}", flush=True)

    rows, accessions = [], {}
    for stem, entity_id in zip(frame.stem, frame.entity_id, strict=True):
        entity = annotations.get(str(entity_id).upper())
        row = {"stem": stem}
        if entity is None:
            rows.append(row)
            continue
        by_type = Counter()
        names = []
        for annotation in entity.get("rcsb_polymer_entity_annotation") or []:
            by_type[annotation.get("type", "?")] += 1
            if annotation.get("type") in ("Pfam", "InterPro", "CATH", "SCOP", "SCOP2"):
                names.append(f"{annotation.get('type')}:{annotation.get('name')}")
        row["n_pfam"] = by_type.get("Pfam", 0)
        row["n_interpro"] = by_type.get("InterPro", 0)
        row["n_cath"] = by_type.get("CATH", 0)
        row["n_scop"] = by_type.get("SCOP", 0) + by_type.get("SCOP2", 0)
        row["n_go"] = by_type.get("GO", 0)
        enzyme = ((entity.get("rcsb_polymer_entity") or {})
                  .get("rcsb_enzyme_class_combined") or [])
        row["has_ec"] = int(bool(enzyme))
        row["ec_class"] = (enzyme[0]["ec"].split(".")[0] if enzyme
                           and enzyme[0].get("ec") else "")
        row["domain_annotations"] = ";".join(sorted(set(names)))
        identifiers = (entity.get("rcsb_polymer_entity_container_identifiers") or {})
        refs = [r["database_accession"] for r in
                (identifiers.get("reference_sequence_identifiers") or [])
                if r.get("database_name") == "UniProt"]
        accessions[stem] = refs
        rows.append(row)
    return pd.DataFrame(rows), accessions


def uniprot_features(accessions: dict[str, list[str]]) -> pd.DataFrame:
    """Domain count, localisation, membrane and signal features per protein."""
    wanted = sorted({a for refs in accessions.values() for a in refs})
    entries = {}
    for start in range(0, len(wanted), 80):
        batch = wanted[start:start + 80]
        query = " OR ".join(f"accession:{a}" for a in batch)
        url = (f"{U.UNIPROT_REST}/search?query={urllib.parse.quote(query)}"
               "&format=json&size=500&fields=accession,ft_domain,ft_transmem,"
               "ft_signal,cc_subcellular_location,keyword,protein_existence,length")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    payload = json.loads(response.read().decode())
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        for entry in payload.get("results", []):
            entries[entry["primaryAccession"]] = entry
        print(f"  [uniprot] {min(start + 80, len(wanted))}/{len(wanted)}", flush=True)

    rows = []
    for stem, refs in accessions.items():
        row = {"stem": stem}
        entry = next((entries[a] for a in refs if a in entries), None)
        if entry is None:
            rows.append(row)
            continue
        features = entry.get("features", [])
        row["n_uniprot_domains"] = sum(1 for f in features if f.get("type") == "Domain")
        row["n_transmembrane"] = sum(1 for f in features if f.get("type") == "Transmembrane")
        row["has_signal_peptide"] = int(any(f.get("type") == "Signal" for f in features))
        locations = []
        for comment in entry.get("comments", []):
            if comment.get("commentType") != "SUBCELLULAR LOCATION":
                continue
            for item in comment.get("subcellularLocations", []):
                value = (item.get("location") or {}).get("value")
                if value:
                    locations.append(value)
        row["subcellular_location"] = ";".join(sorted(set(locations)))
        text = " ".join(locations).lower()
        row["is_membrane"] = int("membrane" in text or row.get("n_transmembrane", 0) > 0)
        row["is_secreted"] = int("secreted" in text or bool(row.get("has_signal_peptide")))
        row["is_cytoplasmic"] = int("cytoplasm" in text or "cytosol" in text)
        row["is_nuclear"] = int("nucleus" in text or "nucleoid" in text)
        keywords = [k.get("name", "") for k in entry.get("keywords", [])]
        row["uniprot_keywords"] = ";".join(keywords[:20])
        row["protein_existence"] = entry.get("proteinExistence", "")
        rows.append(row)
    return pd.DataFrame(rows)


def cached(name: str, builder, *, refresh: bool = False) -> pd.DataFrame:
    """Run `builder` once and keep the result on disk.

    The families cost very different amounts — contact geometry is instant, the
    MSA pass streams a few GB off a Modal volume — so a crash in a late family
    should not re-run the early ones. Delete the cache file (or pass --refresh)
    to force a rebuild.
    """
    path = U.WORK / f"features_{name}.csv"
    if path.exists() and not refresh:
        print(f"  [{name}] cached", flush=True)
        return pd.read_csv(path)
    frame = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["structure", "msa", "rcsb", "uniprot"],
                        help="feature families to leave out of this build")
    parser.add_argument("--refresh", nargs="*", default=[],
                        help="feature families to rebuild even if cached")
    args = parser.parse_args()

    frame = base_table()
    print(f"[features] {len(frame)} natural monomers", flush=True)
    parts = [
        cached("contacts", lambda: contact_features(frame), refresh="contacts" in args.refresh),
        cached("sequence", lambda: sequence_features(frame), refresh="sequence" in args.refresh),
        cached("training", lambda: training_support(frame), refresh="training" in args.refresh),
    ]
    sources = {"contacts": str(U.GT_UNIVERSE.name), "sequence": "eval_sets.csv",
               "training_support": "exp226 identity table + exp245 residual + exp94 KNN"}

    if "structure" not in args.skip:
        parts.append(cached("structure", lambda: structure_features(frame),
                            refresh="structure" in args.refresh))
        sources["structure"] = f"biotite annotate_sse over {U.CIF_CACHE}"
    if "msa" not in args.skip:
        parts.append(cached("msa", lambda: msa_features(frame),
                            refresh="msa" in args.refresh))
        sources["msa"] = f"modal volume {U.MSA_VOLUME}"
    accessions = {}
    if "rcsb" not in args.skip:
        rcsb, accessions = rcsb_features(frame)
        parts.append(rcsb)
        (U.WORK / "uniprot_accessions.json").write_text(json.dumps(accessions))
        sources["rcsb"] = U.RCSB_GRAPHQL
    if "uniprot" not in args.skip:
        if not accessions and (U.WORK / "uniprot_accessions.json").exists():
            accessions = json.loads((U.WORK / "uniprot_accessions.json").read_text())
        if accessions:
            parts.append(cached("uniprot", lambda: uniprot_features(accessions),
                                refresh="uniprot" in args.refresh))
            sources["uniprot"] = U.UNIPROT_REST

    for part in parts:
        frame = frame.merge(part, on="stem", how="left", validate="one_to_one")

    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    coverage = pd.DataFrame({
        "feature": frame.columns,
        "n_present": [int(frame[c].notna().sum()) for c in frame.columns],
        "coverage": [round(float(frame[c].notna().mean()), 4) for c in frame.columns],
        "dtype": [str(frame[c].dtype) for c in frame.columns],
    })
    coverage.to_csv(COVERAGE, index=False)
    PROVENANCE.write_text(json.dumps({
        "n_proteins": int(len(frame)), "n_columns": int(frame.shape[1]),
        "sources": sources, "skipped": args.skip,
    }, indent=2) + "\n")
    print(f"[features] {frame.shape[0]} x {frame.shape[1]} -> {OUT}", flush=True)
    print(coverage[coverage.coverage < 1.0].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
