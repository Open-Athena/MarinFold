"""Screen exact retained AFDB training entries against newer experimental chains.

This is a cheap candidate screen, not a contact-accuracy evaluation. It reports
how many serialized AFDB training contacts are separated by >12 A between CAs
in a sequence-matched PDB chain. Final cases still require pyconfind, chronology,
construct/context checks, and AFDB source-coordinate validation.
"""

import argparse
import difflib
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from marinfold.document_structures.contacts_v1.read import live_contacts, sequence_from_document

HERE = Path(__file__).resolve().parent


def screen(candidate: dict, training: dict, cache: Path, max_length: int = 800) -> dict:
    """Compare one exact training row to the specified experimental chain."""
    result = {key: candidate[key] for key in ("PDB", "CHAIN", "SP_PRIMARY", "entry_id", "shard")}
    sequence = sequence_from_document(training["document"], training["seq_len"], training["n_term_index"])
    result.update(L=len(sequence), training_global_plddt=training["global_plddt"],
                  training_document_sha256=hashlib.sha256(training["document"].encode()).hexdigest())
    if training["truncated"] or not 60 <= len(sequence) <= max_length:
        return result | {"status": "ineligible_training_length_or_truncation"}
    path = cache / f"{candidate['PDB']}_{candidate['CHAIN']}.cif"
    if not path.exists():
        response = requests.get(f"https://models.rcsb.org/v1/{candidate['PDB']}/atoms",
                                params={"auth_asym_id": candidate["CHAIN"], "model_nums": "1", "encoding": "cif"}, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    structure = gemmi.read_structure(str(path))
    if not len(structure) or candidate["CHAIN"] not in [c.name for c in structure[0]]:
        return result | {"status": "experimental_chain_unavailable"}
    residues = [r for r in structure[0][candidate["CHAIN"]] if r.het_flag == "A" and r.find_atom("CA", "*")]
    observed = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in residues)
    mapping = {}
    for block in difflib.SequenceMatcher(a=observed, b=sequence, autojunk=False).get_matching_blocks():
        mapping.update({block.b + k: block.a + k for k in range(block.size)})
    identity = len(mapping) / len(observed) if observed else 0
    coverage = len(mapping) / len(sequence)
    result.update(identity=identity, coverage=coverage, n_common=len(mapping),
                  pdb_chain_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if identity < .95 or coverage < .70:
        return result | {"status": "ineligible_mapping"}
    positions = {index: np.array(list(residues[obs].find_atom("CA", "*").pos)) for index, obs in mapping.items()}
    offset = training["n_term_index"]
    pairs = {tuple(sorted(((i - offset) % 2000, (j - offset) % 2000)))
             for i, j in live_contacts(training["document"])}
    distances = [np.linalg.norm(positions[i] - positions[j]) for i, j in pairs
                 if j - i >= 6 and i in positions and j in positions]
    if len(distances) < 20:
        return result | {"status": "too_few_common_contacts"}
    return result | {"status": "screened", "n_training_contacts_common": len(distances),
                     "fraction_over_12A": float(np.mean(np.array(distances) > 12)),
                     "fraction_over_16A": float(np.mean(np.array(distances) > 16)),
                     "median_contact_ca_distance": float(np.median(distances))}


def main() -> None:
    """Join SIFTS to exact retained training IDs, then resume a candidate screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/data/exp369"))
    parser.add_argument("--corpus", type=Path, default=Path("/data/exp225_decontam/contacts_v1_decontam"))
    parser.add_argument("--limit", type=int, default=700)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sifts = pd.read_csv(args.work / "pdb_chain_uniprot.csv.gz", comment="#")
    index = pd.read_parquet(args.corpus.parent / "index_afdb.parquet")
    index["SP_PRIMARY"] = index.entry_id.str.removeprefix("AF-").str.removesuffix("-F1")
    candidates = sifts.merge(index, on="SP_PRIMARY", validate="many_to_one")
    drop = pd.read_parquet(args.corpus.parent / "droplist_final.parquet").query("arm == 'afdb'")
    candidates = candidates.loc[~candidates.entry_id.isin(drop.entry_id)]
    # PDB identifier is only a triage heuristic. Actual release dates are checked
    # from the entry API before accepting any final case.
    candidates = candidates.loc[candidates.PDB.str.startswith(("8", "9")) & ((candidates.SP_END - candidates.SP_BEG) >= 60)]
    candidates = candidates.sort_values(["PDB", "CHAIN"], ascending=[False, True]).drop_duplicates("SP_PRIMARY").head(args.limit)
    cache = args.work / "pdb_chains"
    cache.mkdir(exist_ok=True)
    records = []
    failures = []
    for shard, group in candidates.groupby("shard"):
        path = args.corpus / f"contacts_v1-{shard:05d}-of-02067.parquet"
        table = pq.read_table(path, columns=["entry_id", "document", "seq_len", "n_term_index", "global_plddt", "truncated"])
        selected = {row["entry_id"]: row for row in table.to_pylist() if row["entry_id"] in set(group.entry_id)}
        for row in group.to_dict("records"):
            if row["entry_id"] not in selected:
                raise ValueError(f"Entry missing from retained training shard: {row}")
            records.append((row, selected[row["entry_id"]]))
    (args.work / "screen_inputs.json").write_text(json.dumps(records) + "\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(screen, row, training, cache): row for row, training in records}
        for future in as_completed(jobs):
            row = jobs[future]
            try:
                result = future.result()
            except (requests.RequestException, RuntimeError, ValueError) as error:
                # Expected candidate-level fetch/parse problems are retained in
                # an explicit failure ledger and make this command fail at end.
                result = row | {"status": "error", "error": f"{type(error).__name__}: {error}"}
                failures.append(result)
            results.append(result)
            if len(results) % 25 == 0 or len(results) == len(records):
                frame = pd.DataFrame(results)
                frame.to_csv(HERE / "data/afdb_candidate_screen.csv", index=False)
                print(f"Screened {len(results)}/{len(records)}; errors={len(failures)}", flush=True)
                if "fraction_over_12A" in frame:
                    print(frame.sort_values("fraction_over_12A", ascending=False)[["PDB", "CHAIN", "SP_PRIMARY", "fraction_over_12A"]].head(5).to_string(index=False), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} candidate failures are recorded in the screen CSV")


if __name__ == "__main__":
    main()
