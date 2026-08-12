# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""ProteinGym v1.3 substitution benchmark: fetch, filter, and aggregate.

Everything in this module is about the *benchmark*, not about MarinFold — it
downloads the assay data, decides which assays we can score, and implements
ProteinGym's own aggregation so our number is directly comparable to the
published leaderboard.

Three artifacts, all fetched on demand into ``--data-dir`` (default
``data/proteingym``):

- ``DMS_substitutions.csv`` — per-assay metadata (target sequence, UniProt id,
  taxon, MSA depth, selection type). From the GitHub repo, ~1 MB.
- ``DMS_ProteinGym_substitutions.zip`` — the assays themselves, one CSV per
  assay with ``mutant`` / ``DMS_score`` columns. ~1 GB from the Marks lab host.
- ``DMS_substitutions_Spearman_DMS_level.csv`` — every published baseline's
  per-assay Spearman, so any subset can be re-aggregated on equal terms.

**Aggregation.** ProteinGym does not average assays uniformly. It averages
per-assay Spearman within a UniProt id first (several assays exist for some
proteins, and a uniform mean would over-weight them), then within each of the
five function categories, then takes the mean of those five. :func:`aggregate`
implements exactly that, and every reported number in this experiment goes
through it — including the baselines, re-aggregated on whatever subset we
actually scored.
"""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import spearmanr
from tqdm import tqdm

VERSION = "v1.3"
_RAW = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main"
REFERENCE_URL = f"{_RAW}/reference_files/DMS_substitutions.csv"
BASELINE_URL = (
    f"{_RAW}/benchmarks/DMS_zero_shot/substitutions/Spearman/"
    "DMS_substitutions_Spearman_DMS_level.csv"
)
ASSAYS_URL = (
    f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{VERSION}/"
    "DMS_ProteinGym_substitutions.zip"
)

# contacts-v1 indexes residues into 2000 wrap-around position tokens, so a
# chain longer than that cannot be uniquely numbered and is not serializable.
# This is a hard format limit, not a tuning choice.
MAX_CHAIN_LENGTH = 2000

# The five ProteinGym function categories, in the order the leaderboard lists
# them. The final score is the unweighted mean over these.
FUNCTION_CATEGORIES = (
    "Activity",
    "Binding",
    "Expression",
    "OrganismalFitness",
    "Stability",
)


@dataclass(frozen=True)
class Assay:
    """One DMS assay: its target sequence and its measured variants."""

    dms_id: str
    uniprot_id: str
    target_seq: str
    taxon: str
    selection_type: str
    msa_depth: str
    variants: pd.DataFrame  # columns: mutant, DMS_score

    @property
    def seq_len(self) -> int:
        return len(self.target_seq)


def data_dir(root: Path | None = None) -> Path:
    return (root or Path(__file__).resolve().parent / "data" / "proteingym")


def fetch(data_root: Path | None = None) -> Path:
    """Download the three benchmark artifacts if absent. Returns the directory."""
    out = data_dir(data_root)
    out.mkdir(parents=True, exist_ok=True)
    _download(REFERENCE_URL, out / "DMS_substitutions.csv")
    _download(BASELINE_URL, out / "baselines_Spearman_DMS_level.csv")
    _download(ASSAYS_URL, out / "DMS_ProteinGym_substitutions.zip")
    return out


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp.open("wb") as handle:
            bar = tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            )
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                bar.update(len(chunk))
            bar.close()
    tmp.rename(dest)


def reference(data_root: Path | None = None) -> pd.DataFrame:
    """The per-assay metadata table, with our scorability verdict attached.

    Adds two columns the rest of the experiment reads:

    - ``scorable``: whether contacts-v1 can serialize the target chain.
    - ``skip_reason``: why not, for the ones it cannot.
    """
    frame = pd.read_csv(data_dir(data_root) / "DMS_substitutions.csv")
    too_long = frame.seq_len > MAX_CHAIN_LENGTH
    frame["scorable"] = ~too_long
    frame["skip_reason"] = ""
    frame.loc[too_long, "skip_reason"] = (
        f"chain longer than the {MAX_CHAIN_LENGTH}-residue contacts-v1 limit"
    )
    return frame


def load_assay(row: pd.Series, data_root: Path | None = None) -> Assay:
    """Read one assay's variant table out of the zip (no extraction to disk)."""
    archive = data_dir(data_root) / "DMS_ProteinGym_substitutions.zip"
    with zipfile.ZipFile(archive) as zf:
        name = _member_for(zf, row.DMS_filename)
        with zf.open(name) as handle:
            variants = pd.read_csv(io.BytesIO(handle.read()))
    missing = {"mutant", "DMS_score"} - set(variants.columns)
    if missing:
        raise ValueError(f"{row.DMS_id}: assay CSV is missing columns {sorted(missing)}")
    return Assay(
        dms_id=row.DMS_id,
        uniprot_id=row.UniProt_ID,
        target_seq=row.target_seq,
        taxon=row.taxon,
        selection_type=row.coarse_selection_type,
        msa_depth=row.MSA_Neff_L_category,
        variants=variants[["mutant", "DMS_score"]],
    )


def _member_for(archive: zipfile.ZipFile, filename: str) -> str:
    """Resolve an assay filename inside the zip, which may nest a top folder."""
    names = archive.namelist()
    if filename in names:
        return filename
    suffix = "/" + filename
    matches = [n for n in names if n.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"{filename!r} matched {len(matches)} members of the substitutions "
            f"archive; expected exactly 1."
        )
    return matches[0]


def parse_mutants(variants: pd.DataFrame) -> list[list[tuple[str, int, str]]]:
    """``"A24G:D30N"`` → ``[("A", 23, "G"), ("D", 29, "N")]`` (0-based sites).

    ProteinGym writes positions 1-based against ``target_seq``; everything
    downstream of here is 0-based.
    """
    parsed = []
    for mutant in variants["mutant"]:
        sites = []
        for token in str(mutant).split(":"):
            token = token.strip()
            sites.append((token[0], int(token[1:-1]) - 1, token[-1]))
        parsed.append(sites)
    return parsed


def assay_spearman(scores: pd.Series, dms_scores: pd.Series) -> float:
    """Spearman ρ between predicted and measured effect for one assay."""
    return float(spearmanr(scores, dms_scores).statistic)


def aggregate(per_assay: pd.DataFrame, reference_frame: pd.DataFrame) -> dict:
    """ProteinGym's official aggregation of per-assay Spearman.

    ``per_assay`` needs columns ``DMS_id`` and ``spearman``. Returns the
    headline average plus the category / MSA-depth / taxon breakdowns the
    leaderboard reports.

    The headline is *not* a mean over assays: it is mean over UniProt ids
    within a function category, then mean over the five categories. Assays for
    a heavily-studied protein therefore do not get extra weight, and a category
    with few assays is not drowned out.
    """
    meta_columns = [
        "DMS_id",
        "UniProt_ID",
        "coarse_selection_type",
        "MSA_Neff_L_category",
        "taxon",
    ]
    merged = per_assay.merge(reference_frame[meta_columns], on="DMS_id", how="left")
    if merged.spearman.isna().any():
        raise ValueError("aggregate() received assays with a null Spearman.")
    if merged.UniProt_ID.isna().any():
        unknown = merged.loc[merged.UniProt_ID.isna(), "DMS_id"].tolist()
        raise ValueError(f"assays absent from the reference file: {unknown}")

    by_uniprot = (
        merged.groupby(["coarse_selection_type", "UniProt_ID"])["spearman"]
        .mean()
        .reset_index()
    )
    by_category = by_uniprot.groupby("coarse_selection_type")["spearman"].mean()
    present = [c for c in FUNCTION_CATEGORIES if c in by_category.index]
    return {
        "average_spearman": float(by_category[present].mean()),
        "n_assays": int(len(merged)),
        "n_uniprot": int(merged.UniProt_ID.nunique()),
        "by_function": {c: float(by_category[c]) for c in present},
        "by_msa_depth": _grouped(merged, "MSA_Neff_L_category"),
        "by_taxon": _grouped(merged, "taxon"),
    }


def _grouped(merged: pd.DataFrame, column: str) -> dict:
    """Mean over UniProt ids within each level of ``column``.

    The same de-duplication the headline applies, one stratum at a time. Note
    ProteinGym reports these breakdowns without the category averaging step,
    so they are UniProt-level means — matching the published columns.
    """
    by_uniprot = (
        merged.groupby([column, "UniProt_ID"])["spearman"].mean().reset_index()
    )
    return {
        str(level): float(value)
        for level, value in by_uniprot.groupby(column)["spearman"].mean().items()
    }


def baseline_table(
    dms_ids: list[str], data_root: Path | None = None
) -> pd.DataFrame:
    """Published per-assay Spearman for every baseline, restricted to ``dms_ids``.

    Used to re-aggregate the leaderboard on exactly the assays we scored — the
    only way a subset comparison is honest.
    """
    frame = pd.read_csv(data_dir(data_root) / "baselines_Spearman_DMS_level.csv")
    frame = frame.rename(columns={"DMS ID": "DMS_id"})
    missing = set(dms_ids) - set(frame.DMS_id)
    if missing:
        raise ValueError(
            f"{len(missing)} scored assays are absent from the baseline table, "
            f"e.g. {sorted(missing)[:3]}"
        )
    return frame[frame.DMS_id.isin(dms_ids)].reset_index(drop=True)


if __name__ == "__main__":
    directory = fetch()
    frame = reference()
    print(f"ProteinGym {VERSION} in {directory}")
    print(f"  {len(frame)} substitution assays, {frame.UniProt_ID.nunique()} UniProt ids")
    print(f"  scorable by contacts-v1: {int(frame.scorable.sum())}")
    for _, row in frame[~frame.scorable].iterrows():
        print(f"    skip {row.DMS_id} (L={row.seq_len}): {row.skip_reason}")
