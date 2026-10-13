"""Resolve the exact archived structures and ConFind tables used by each score.

The source-row lineage in pl5_figure_rows.csv chooses exp245 reruns, exp226
additions, or legacy predictions. Only missing legacy Protenix coordinates and
the two legacy contact tables are downloaded; no predictor is rerun.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
CACHE = HERE / "scratch/contact_confidence"
BUCKET = "https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/"
MODELS = {"esmfold": "esmfold", "esmfold2": "esmfold2",
          "protenix_msa": "protenix-v2_msa", "protenix_ss": "protenix-v2_single_seq"}


def digest(path: Path) -> str:
    """Hash a small immutable input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(task: tuple[str, Path]) -> None:
    """Cache an explicitly selected public artifact, with atomic replacement."""
    url, path = task
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        payload = response.read()
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    """Freeze exact prediction provenance and raw contact rows for rescoring."""
    CACHE.mkdir(parents=True, exist_ok=True)
    sources = pd.read_csv(HERE / "data/pl5_figure_rows.csv", float_precision="round_trip")
    sources["analysis_source_row"] = range(len(sources))
    sources = sources[(sources.figure == "04_contacts_pl5") & (sources.metric == "p_at_l5")
                      & ~sources.method.isin(["knn", "marinfold"])]
    old = pd.read_csv(REPO / "experiments/exp78_evals_esmfold_contacts/data/contact_precision_all.csv")
    tables = {
        "245_esm": Path("/data/exp245/baseline_scores/esm/contacts_raw.parquet"),
        "245_protenix": Path("/data/exp245/baseline_scores/protenix/contacts_raw.parquet"),
        "226_esm": Path("/data/exp226_gt/esm_scores/contacts_raw.parquet"),
        "226_protenix": Path("/data/exp226_gt/protenix_scores/contacts_raw.parquet"),
        "legacy_esm": CACHE / "inputs/esm_contacts_raw.parquet",
        "legacy_protenix": CACHE / "inputs/protenix_contacts_raw.parquet",
    }
    downloads = [(BUCKET + f"{prefix}/contacts_raw_all.parquet", tables[key]) for key, prefix in
                 [("legacy_esm", "esmfold-contacts-eval-exp78"), ("legacy_protenix", "protenix-contacts-eval-exp74")]]
    records = []
    for row in sources.itertuples():
        stem, method = row.stem, row.method
        raw_key, dataset, url = "", "", ""
        if method in ("af2", "af3", "boltz2"):
            family = "boltz2" if method == "boltz2" else "alphafold"
            path = HERE / f"scratch/{family}/results/{method}/{stem}" / ("selected.pdb" if method == "af2" else "selected.cif")
            raw_path = HERE / f"scratch/{family}/{method}_contacts.json"
        else:
            family = "esm" if method.startswith("esm") else "protenix"
            mode = "msa" if method == "protenix_msa" else "single_seq"
            if "exp245" in row.source:
                raw_key, dataset = f"245_{family}", "foldbench_monomer"
                path = Path("/data/exp245/pred") / MODELS[method] / stem / "structure.cif"
            elif "pl5_exp226" in row.source:
                raw_key, dataset = f"226_{family}", "foldbench_rest"
                path = (Path("/data/exp226_gt/pred/root") / method if family == "esm"
                        else Path("/data/exp226_gt/protenix_best") / mode) / stem / "structure.cif"
            else:
                original = old.iloc[int(row.source_row)]
                if original.stem != stem:
                    raise ValueError(f"Source-row mismatch: {stem}")
                raw_key, dataset = f"legacy_{family}", original.dataset
                if family == "esm":
                    path = Path("/data/exp226_gt/pred") / f"{method}_raw" / stem / "structure.cif"
                else:
                    prefix = "protenix-foldbench-monomers/best" if dataset == "foldbench100" else "protenix-contacts-eval-exp74/best_exp65"
                    url = BUCKET + f"{prefix}/{mode}/{stem}/structure.cif"
                    path = CACHE / "inputs/legacy" / method / stem / "structure.cif"
                    downloads.append((url, path))
            raw_path = tables[raw_key]
        records.append(dict(stem=stem, method=method, structure_file=str(path), structure_url=url,
                            raw_file=str(raw_path), raw_key=raw_key, dataset=dataset,
                            old_score_source=row.source, old_score_row=int(row.source_row),
                            analysis_source_row=int(row.analysis_source_row)))
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(download, downloads))
    raw_tables = {key: pd.read_parquet(path) for key, path in tables.items()}
    contacts = []
    for record in records:
        method, stem, key = record["method"], record["stem"], record["raw_key"]
        path = Path(record["structure_file"])
        record["structure_sha256"] = digest(path)
        if key:
            frame = raw_tables[key]
            subset = frame[(frame.stem == stem) & (frame.dataset == record["dataset"]) & (frame.role == "pred")]
            if "model" in subset:
                name = MODELS[method] if key.startswith("245") else method
                subset = subset[subset.model == name]
            else:
                subset = subset[subset["mode"] == ("msa" if method == "protenix_msa" else "single_seq")]
            pairs = subset[["i", "j", "degree"]].to_numpy().tolist()
        else:
            pairs = json.loads(Path(record["raw_file"]).read_text())[stem]
            run = json.loads((HERE / f"data/{method}_run.json").read_text())
            if record["structure_sha256"] != run["selected_sha256"][stem]:
                raise ValueError(f"Changed selected structure: {method}/{stem}")
        if not pairs:
            raise ValueError(f"Missing archived contacts: {method}/{stem}")
        contacts.extend(dict(stem=stem, method=method, i=int(i), j=int(j), degree=d) for i, j, d in pairs)
    pd.DataFrame(records).to_csv(CACHE / "sources.csv", index=False)
    pd.DataFrame(contacts).to_parquet(CACHE / "archived_contacts.parquet", index=False)
    pins = {str(path): digest(path) for path in tables.values()}
    pins.update({str(HERE / "data/pl5_figure_rows.csv"): digest(HERE / "data/pl5_figure_rows.csv")})
    (CACHE / "source_pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    print(f"Resolved {len(records)} predictions and {len(contacts)} archived contacts", flush=True)


if __name__ == "__main__":
    main()
