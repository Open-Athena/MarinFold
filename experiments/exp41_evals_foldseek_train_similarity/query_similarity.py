# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure how close candidate structures are to MarinFold's training set.

MarinFold trains on ``afdb-24M``, which is already Foldseek-clustered:
every structure carries a ``struct_cluster_id`` (its structural cluster
representative) and a ``split`` (train/val/test) assigned by hashing that
cluster id, so a whole fold lands in one split. This tool answers the
question issue #41 asks for an external candidate (a FoldBench monomer, a
de novo design, a low-MSA natural protein): *how structurally close is it
to anything we trained on?*

The mechanism is ``foldseek easy-search`` of the candidate(s) against a
target database of the training-set cluster **representatives**
(``--alignment-type 1`` → TM-score), then joining each hit's representative
back to its split. We report, per candidate, the nearest training
representative, its TM-score, and a novel/same-fold/redundant verdict.

This module does NOT touch ``afdb-24M`` directly — it consumes a prebuilt
representative DB + a ``reps_manifest.csv`` mapping ``representative_id`` →
``split`` (built by ``build_db_modal.py`` and pulled local by
``fetch_db.py``). That keeps the repeatedly-run query path light and offline.

CLI::

    uv run python query_similarity.py \
        --candidate-dir candidates/foldbench/.../gt \
        --db db_full/db/targetDB \
        --reps-manifest db_full/reps_manifest.csv \
        --out data/foldbench_vs_full_reps_similarity.csv
"""

import argparse
import tempfile
import warnings
from pathlib import Path

import gemmi
import pandas as pd

from foldseek_env import foldseek_version as _foldseek_version
from foldseek_env import run_foldseek

# The Foldseek easy-search columns we request, in order. `alntmscore` is
# the TM-score over the alignment; `qtmscore`/`ttmscore` normalise it by
# query/target length; `fident` is sequence identity over the structural
# alignment (the free secondary sequence-similarity signal). Verify these
# field names against the installed Foldseek version — output codes have
# shifted across releases.
FORMAT_FIELDS: tuple[str, ...] = (
    "query",
    "target",
    "alntmscore",
    "qtmscore",
    "ttmscore",
    "lddt",
    "fident",
    "alnlen",
    "evalue",
)
_NUMERIC_FIELDS = ("alntmscore", "qtmscore", "ttmscore", "lddt", "fident", "alnlen", "evalue")

# 0.5 is the field-canonical same-fold TM boundary (Barrio-Hernandez 2023,
# the AFDB structural-cluster definition). 0.9 for "redundant" is a
# judgment call — tune via CLI. `qtmscore` (query-normalised) drives the
# verdict so short candidates aren't penalised against longer reps.
DEFAULT_FOLD_TM = 0.5
DEFAULT_REDUNDANT_TM = 0.9
DEFAULT_TM_FIELD = "qtmscore"

_STRUCT_EXTS = (".cif", ".mmcif", ".pdb", ".ent")
# Glob patterns for candidate structure files (plain + gzipped). Single
# source of truth shared by candidate_stems / run / collect_timings so the
# stem set, the residue-count map, and the timing set never disagree.
_STRUCT_GLOBS = ("*.cif", "*.cif.gz", "*.mmcif", "*.pdb", "*.pdb.gz", "*.ent")


def _strip_struct_ext(name: str) -> str:
    """Drop a trailing ``.gz`` then a structure-file extension."""
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    for ext in _STRUCT_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def iter_candidate_files(candidate_dir: Path) -> list[Path]:
    """All candidate structure files under ``candidate_dir``, sorted & deduped."""
    seen: dict[str, Path] = {}
    for pattern in _STRUCT_GLOBS:
        for p in candidate_dir.glob(pattern):
            seen.setdefault(p.name, p)
    return sorted(seen.values())


def normalize_name(fs_name: str, known: set[str]) -> str:
    """Map a Foldseek entry name back to a candidate stem / representative id.

    Foldseek names each DB entry ``<file-stem>`` and, for structures with
    named chains, appends ``_<chain>``. We resolve back to a known id by:
    strip the extension; exact-match the known set; else strip a trailing
    ``_<token>`` (the chain) and match; else return the stripped base
    unchanged (it simply won't join).
    """
    base = _strip_struct_ext(fs_name)
    if base in known:
        return base
    if "_" in base:
        head = base.rsplit("_", 1)[0]
        if head in known:
            return head
    return base


def candidate_stems(candidate_dir: Path) -> set[str]:
    """Stems (filename without structure extension) of the candidate cifs."""
    return {_strip_struct_ext(p.name) for p in iter_candidate_files(candidate_dir)}


def count_residues(cif_path: Path) -> int:
    """Count polymer residues in a structure (canonical 1..N length).

    Sums ``full_sequence`` over polypeptide entities (matching the N that
    exp20/exp12 use); falls back to counting Cα atoms in the first model if
    entity metadata is absent.
    """
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    n = 0
    for ent in st.entities:
        if ent.entity_type == gemmi.EntityType.Polymer and ent.polymer_type in (
            gemmi.PolymerType.PeptideL,
            gemmi.PolymerType.PeptideD,
        ):
            n += len(ent.full_sequence)
    if n:
        return n
    # Fallback: count Cα atoms in the first model.
    for model in st:
        return sum(
            1
            for chain in model
            for residue in chain
            if residue.find_atom("CA", "\0") is not None
        )
    return 0


def easy_search(
    candidate_dir: Path,
    db_prefix: Path,
    out_m8: Path,
    tmp_dir: Path,
    *,
    alignment_type: int = 1,
    max_seqs: int = 300,
    evalue: str = "inf",
    exhaustive: bool = False,
    gpu: bool = False,
) -> Path:
    """Run ``foldseek easy-search`` of ``candidate_dir`` against ``db_prefix``.

    Thresholds are loosened (``-c 0`` coverage, large e-value) so even a
    genuinely novel candidate emits its *best* (low-TM) hit rather than
    being filtered to no row. Returns ``out_m8``.
    """
    out_m8.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "easy-search",
        str(candidate_dir),
        str(db_prefix),
        str(out_m8),
        str(tmp_dir),
        "--alignment-type",
        str(alignment_type),
        "--format-output",
        ",".join(FORMAT_FIELDS),
        "-c",
        "0.0",
        "-e",
        evalue,
        "--max-seqs",
        str(max_seqs),
        "--exhaustive-search",
        "1" if exhaustive else "0",
    ]
    if gpu:
        args += ["--gpu", "1", "--prefilter-mode", "1"]
    run_foldseek(args)
    return out_m8


def parse_m8(m8_path: Path) -> pd.DataFrame:
    """Parse a Foldseek ``.m8`` written with ``FORMAT_FIELDS`` into a frame."""
    if m8_path.stat().st_size == 0:
        return pd.DataFrame(columns=list(FORMAT_FIELDS))
    df = pd.read_csv(m8_path, sep="\t", header=None, names=list(FORMAT_FIELDS))
    for col in _NUMERIC_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def verdict_for(
    best_train_tm: float | None, fold_tm: float, redundant_tm: float
) -> str:
    """Classify a candidate by the TM-score to its nearest *training* rep."""
    if best_train_tm is None or pd.isna(best_train_tm):
        return "novel_fold"
    if best_train_tm >= redundant_tm:
        return "redundant"
    if best_train_tm >= fold_tm:
        return "same_fold"
    return "novel_fold"


def summarize(
    hits: pd.DataFrame,
    manifest: pd.DataFrame,
    n_residues: dict[str, int],
    stems: set[str],
    *,
    tm_field: str = DEFAULT_TM_FIELD,
    fold_tm: float = DEFAULT_FOLD_TM,
    redundant_tm: float = DEFAULT_REDUNDANT_TM,
    foldseek_version: str = "unknown",
    db_snapshot_tag: str = "unknown",
) -> pd.DataFrame:
    """Reduce per-hit Foldseek output to one verdict row per candidate.

    ``hits`` columns are ``FORMAT_FIELDS``; ``manifest`` maps
    ``representative_id`` → ``split``. Candidates with no hits get an empty
    row with verdict ``novel_fold``.
    """
    manifest_ids = set(manifest["representative_id"].astype(str))
    split_by_id = dict(
        zip(manifest["representative_id"].astype(str), manifest["split"].astype(str))
    )

    grouped: dict[str, pd.DataFrame] = {}
    if not hits.empty:
        hits = hits.copy()
        hits["stem"] = hits["query"].map(lambda q: normalize_name(str(q), stems))
        hits["representative_id"] = hits["target"].map(
            lambda t: normalize_name(str(t), manifest_ids)
        )
        hits["target_split"] = hits["representative_id"].map(split_by_id)
        # A hit whose target rep is absent from the manifest gets split=NaN
        # and is silently dropped from every train-match verdict. The shipped
        # DB and manifest are co-derived so this is normally zero; warn loudly
        # if it ever isn't, rather than mislabel a candidate novel_fold.
        unmatched = hits["target_split"].isna()
        if unmatched.any():
            n_targets = hits.loc[unmatched, "representative_id"].nunique()
            warnings.warn(
                f"{int(unmatched.sum())} foldseek hits ({n_targets} distinct targets) "
                f"did not join to the reps manifest; their split is unknown and they "
                f"are excluded from train-match verdicts. Check that --db and "
                f"--reps-manifest come from the same build.",
                stacklevel=2,
            )
        grouped = {stem: g for stem, g in hits.groupby("stem")}

    rows: list[dict] = []
    for stem in sorted(stems):
        row: dict = {
            "stem": stem,
            "n_residues": n_residues.get(stem),
            "best_target_rep": None,
            "best_target_split": None,
            "best_alntmscore": None,
            "best_qtmscore": None,
            "best_train_target_rep": None,
            "best_train_alntmscore": None,
            "best_train_qtmscore": None,
            "best_train_fident": None,
            "tm_field": tm_field,
            "n_hits_tm_ge_fold": 0,
            "n_train_hits_tm_ge_fold": 0,
            "verdict": "novel_fold",
            "fold_tm": fold_tm,
            "redundant_tm": redundant_tm,
            "foldseek_version": foldseek_version,
            "db_snapshot_tag": db_snapshot_tag,
        }
        g = grouped.get(stem)
        if g is not None and not g.empty:
            g = g.sort_values(tm_field, ascending=False)
            best = g.iloc[0]
            row["best_target_rep"] = best["representative_id"]
            row["best_target_split"] = best["target_split"]
            row["best_alntmscore"] = float(best["alntmscore"])
            row["best_qtmscore"] = float(best["qtmscore"])
            row["n_hits_tm_ge_fold"] = int((g[tm_field] >= fold_tm).sum())

            train = g[g["target_split"] == "train"]
            row["n_train_hits_tm_ge_fold"] = int((train[tm_field] >= fold_tm).sum())
            best_train_tm = None
            if not train.empty:
                btrain = train.iloc[0]
                best_train_tm = float(btrain[tm_field])
                row["best_train_target_rep"] = btrain["representative_id"]
                row["best_train_alntmscore"] = float(btrain["alntmscore"])
                row["best_train_qtmscore"] = float(btrain["qtmscore"])
                row["best_train_fident"] = float(btrain["fident"])
            row["verdict"] = verdict_for(best_train_tm, fold_tm, redundant_tm)
        rows.append(row)

    return pd.DataFrame(rows)


def run(
    candidate_dir: Path,
    db_prefix: Path,
    reps_manifest: Path,
    out_csv: Path,
    *,
    tm_field: str = DEFAULT_TM_FIELD,
    fold_tm: float = DEFAULT_FOLD_TM,
    redundant_tm: float = DEFAULT_REDUNDANT_TM,
    db_snapshot_tag: str = "unknown",
    evalue: str = "inf",
    exhaustive: bool = False,
    gpu: bool = False,
) -> pd.DataFrame:
    """End-to-end: easy-search candidates, summarize, write ``out_csv``."""
    files = iter_candidate_files(candidate_dir)
    if not files:
        raise RuntimeError(f"no candidate structures found under {candidate_dir}")
    manifest = pd.read_csv(reps_manifest)

    # One directory walk: derive both the stem set and the residue map.
    n_residues = {_strip_struct_ext(p.name): count_residues(p) for p in files}
    stems = set(n_residues)

    with tempfile.TemporaryDirectory(prefix="foldseek_") as td:
        tmp = Path(td)
        out_m8 = tmp / "aln.m8"
        easy_search(
            candidate_dir,
            db_prefix,
            out_m8,
            tmp / "fs_tmp",
            evalue=evalue,
            exhaustive=exhaustive,
            gpu=gpu,
        )
        hits = parse_m8(out_m8)

    summary = summarize(
        hits,
        manifest,
        n_residues,
        stems,
        tm_field=tm_field,
        fold_tm=fold_tm,
        redundant_tm=redundant_tm,
        foldseek_version=_foldseek_version(),
        db_snapshot_tag=db_snapshot_tag,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-dir", type=Path, required=True, help="Dir of candidate .cif/.pdb files")
    ap.add_argument("--db", type=Path, required=True, help="Foldseek target DB prefix (e.g. .../targetDB)")
    ap.add_argument("--reps-manifest", type=Path, required=True, help="CSV: representative_id,split,...")
    ap.add_argument("--out", type=Path, required=True, help="Output similarity CSV path")
    ap.add_argument("--tm-field", default=DEFAULT_TM_FIELD, choices=["qtmscore", "ttmscore", "alntmscore"])
    ap.add_argument("--fold-tm", type=float, default=DEFAULT_FOLD_TM)
    ap.add_argument("--redundant-tm", type=float, default=DEFAULT_REDUNDANT_TM)
    ap.add_argument("--db-tag", default="unknown", help="Snapshot tag recorded in the output (which training DB)")
    ap.add_argument("--evalue", default="inf", help="Foldseek -e threshold (loose by default to keep weak hits)")
    ap.add_argument("--exhaustive", action="store_true", help="Skip prefilter (slow; only for tiny DBs)")
    ap.add_argument("--gpu", action="store_true", help="Use Foldseek GPU mode for the search")
    args = ap.parse_args()

    summary = run(
        args.candidate_dir,
        args.db,
        args.reps_manifest,
        args.out,
        tm_field=args.tm_field,
        fold_tm=args.fold_tm,
        redundant_tm=args.redundant_tm,
        db_snapshot_tag=args.db_tag,
        evalue=args.evalue,
        exhaustive=args.exhaustive,
        gpu=args.gpu,
    )
    counts = summary["verdict"].value_counts().to_dict()
    print(f"Wrote {args.out} ({len(summary)} candidates). Verdicts: {counts}")


if __name__ == "__main__":
    main()
