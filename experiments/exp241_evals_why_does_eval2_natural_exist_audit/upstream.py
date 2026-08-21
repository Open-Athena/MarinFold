# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The single seam onto the upstream experiments this audit reads.

exp241 asks *why* eval2's natural subset is non-empty. It adds no new search:
every identity number it uses is exp226's, every arm convention is exp213's, and
every provenance/date field is exp65's. So this module is the one place that
knows where those artifacts live, and it **verifies** them (row counts, header
grammar, the eval2 filter constants) rather than trusting them — a silent change
upstream should break this experiment loudly, not shift its conclusion.

Read-only. Nothing here writes.
"""
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent

EXP226_DIR = EXPERIMENTS / "exp226_evals_expand_foldbench_eval_set"
EXP213_DIR = EXPERIMENTS / "exp213_evals_train_sequence_overlap_audit"
EXP65_DIR = EXPERIMENTS / "exp65_evals_low_msa_depth_proteins"

for _d in (EXP226_DIR, EXP213_DIR, EXP65_DIR):
    if not _d.is_dir():  # pragma: no cover - branch-layout guard
        raise SystemExit(f"upstream experiment directory not found: {_d}")

# exp226's link module already puts exp213's overlap code on sys.path and
# re-exports the identity conventions. Import through it so exp241, exp226 and
# exp213 cannot drift on what "identity" means.
sys.path.insert(0, str(EXP226_DIR))
from exp213_link import (  # noqa: E402
    ARM_AFDB,
    ARM_ESM,
    ARMS,
    FORMAT,
    HOMOLOGY_EVALUE,
    MANIFEST_STRATA,
    MIN_QCOV,
    arm_of,
    ensure_mmseqs,
    reduce_alignments,
    run,
)

# --- eval2's definition, as constants we assert rather than restate ----------

#: The identity threshold eval2 cuts at. exp226 keeps a protein iff its
#: coverage-gated best identity against the *union* of both training arms is
#: strictly below this. 0.40 — not 0.30; 0.30 is carried as the retrospective
#: ``passes_30`` column, not the definition.
EVAL2_THRESHOLD = 0.40

#: Sequence counts of the two training arms of ``contacts-v1-exp199-1.5B``,
#: asserted against exp213's own FASTA headers in :func:`check_arm_sizes`.
N_AFDB = 4_129_682
N_ESM_ATLAS = 66_759_922
N_TRAIN = N_AFDB + N_ESM_ATLAS

#: Sizes of the *parent* databases each arm was sampled from, for the sampling
#: fraction in A0. ESM Atlas v0 is ~617 M predicted structures, of which the #91
#: curation kept the 40 %-linclust representatives.
N_AFDB_V4 = 214_683_829
N_AFDB_24M = 24_009_002  # timodonnell/afdb-24M, per its dataset card
N_ESM_ATLAS_V0 = 617_051_007

#: The AFDB arm's full provenance chain, read off ``timodonnell/afdb-24M``'s
#: dataset card and ``experiments/exp53_.../selection.py``. Both stages contain
#: a filter against *structurally singular* proteins, which is why this chain is
#: a finding rather than background:
#:
#: AFDB v4 --[full-length only; mean pLDDT >= 70; length <= 2048]--> ~30 M
#:        --[must be in BOTH the AFDB50 sequence-cluster file AND the structural
#:           cluster file with cluFlag=2; "fragments, singletons and
#:           sequence-only entries are excluded"]--> 24,009,002  (afdb-24M)
#:        --[exp53: keep the top EXP53_NUM_ROUNDS members per struct_cluster_id
#:           by pLDDT, and DROP any cluster with fewer than
#:           EXP53_MIN_CLUSTER_SIZE usable members]--> 4,129,682  (the arm)
AFDB24M_MIN_PLDDT = 70.0
AFDB24M_MAX_LEN = 2048
EXP53_NUM_ROUNDS = 5
EXP53_MIN_CLUSTER_SIZE = 3
EXP53_SEQ_LEN_RANGE = (2, 2000)

#: exp139/#91's ESM-Atlas curation clustered at this identity — the same number
#: eval2 cuts at, which is §A4's subject.
ESM_ATLAS_LINCLUST_ID = 0.40

# --- committed upstream artifacts -------------------------------------------

EVAL2_MANIFEST = EXP226_DIR / "data" / "eval2_manifest.csv"
IDENTITY_TABLE = EXP226_DIR / "data" / "eval_train_identity_expanded.csv"
#: The 776 MMseqs2 queries exp226 searched, ``>{dataset}__{stem}``. This is the
#: authoritative sequence for every eval protein — including the 469 outside
#: eval2, whose sequences the eval2 manifest does not carry.
QUERY_FASTA = EXP226_DIR / "data" / "eval_queries_expanded.fasta"
FOLDBENCH_TARGETS = EXP226_DIR / "data" / "foldbench_targets.csv"
EXP65_CAMEO_MANIFEST = EXP65_DIR / "data" / "cameo_hard_manifest.csv"
EXP65_CASP_MANIFEST = EXP65_DIR / "data" / "casp_fm_manifest.csv"
EXP65_CASP_FALLBACK = EXP65_DIR / "data" / "casp_fm_pdb_fallback.csv"
EXP65_DENOVO_MANIFEST = EXP65_DIR / "data" / "denovo_pdb_manifest.csv"

#: exp213's per-arm training FASTAs, built on the workstation's /data volume and
#: too large to commit. The AFDB one is what §A3's exact-accession membership
#: test reads; it is 1.2 GB and its headers carry the UniProt accession.
ARM_FASTA_DIR = Path("/data/exp213_overlap")
AFDB_ARM_FASTA = ARM_FASTA_DIR / "train_afdb.fasta"

#: ``>afdb|<shard>_<row>_AF-<ACCESSION>-F1`` — the grammar §A3 depends on.
AFDB_HEADER_RE = re.compile(r"^>afdb\|\d+_\d+_AF-([A-Z0-9]+)-F1\s*$")

#: Expected counts, so a rebuild that disagrees fails instead of drifting.
EXPECTED_EVAL2_N = 307
EXPECTED_EVAL2_NATURAL_N = 78
EXPECTED_IDENTITY_TABLE_N = 776


@dataclass(frozen=True)
class Protein:
    """One eval protein, as the upstream tables describe it."""

    dataset: str
    stem: str
    sequence: str
    length: int
    #: Coverage-gated best identity vs the union of both arms; ``None`` when no
    #: hit passed both the E-value and the coverage gate.
    best_identity: float | None
    best_identity_ungated: float | None
    afdb_best_identity: float | None
    esm_atlas_best_identity: float | None
    n_hits: int              # all reported alignments, E <= 10
    n_hits_significant: int  # E <= 1e-3, either arm
    stratum: str
    designed_any: bool
    source_organism: str
    msa_neff: float | None
    fold_verdict: str
    in_eval2: bool

    @property
    def pdb_id(self) -> str | None:
        """The 4-character RCSB entry id, when the stem names one.

        FoldBench and CAMEO stems are ``<pdbid>_<chain>``. CASP stems are
        ``T1027-D1`` and name no entry — those resolve through exp65's
        fallback map instead (see :mod:`annotate_rcsb`).
        """
        head = self.stem.split("_")[0]
        return head.lower() if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", head) else None

    @property
    def chain(self) -> str | None:
        parts = self.stem.split("_")
        return parts[1] if len(parts) == 2 else None


def _f(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def _i(value: str) -> int:
    return int(value) if value not in ("", None) else 0


def read_identity_table(path: Path = IDENTITY_TABLE) -> dict[tuple[str, str], dict]:
    """exp226's 776-row per-protein identity table, keyed ``(dataset, stem)``."""
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != EXPECTED_IDENTITY_TABLE_N:
        raise SystemExit(
            f"{path} has {len(rows)} rows, expected {EXPECTED_IDENTITY_TABLE_N}"
        )
    return {(r["dataset"], r["stem"]): r for r in rows}


def read_eval2(path: Path = EVAL2_MANIFEST) -> list[dict]:
    """exp226's eval2 manifest — the 307 proteins under 40 % identity."""
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != EXPECTED_EVAL2_N:
        raise SystemExit(f"{path} has {len(rows)} rows, expected {EXPECTED_EVAL2_N}")
    return rows


def read_query_sequences(path: Path = QUERY_FASTA) -> dict[tuple[str, str], str]:
    """``(dataset, stem) -> sequence`` from exp226's 776-record query FASTA."""
    seqs: dict[tuple[str, str], str] = {}
    key: tuple[str, str] | None = None
    parts: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if key is not None:
                    seqs[key] = "".join(parts)
                dataset, _, stem = line[1:].partition("__")
                key, parts = (dataset, stem), []
            else:
                parts.append(line.strip())
    if key is not None:
        seqs[key] = "".join(parts)
    if len(seqs) != EXPECTED_IDENTITY_TABLE_N:
        raise SystemExit(
            f"{path} has {len(seqs)} records, expected {EXPECTED_IDENTITY_TABLE_N}"
        )
    return seqs


def read_proteins() -> list[Protein]:
    """All 776 eval-universe proteins, joined across exp226's two tables.

    ``in_eval2`` is recomputed from ``best_identity`` and
    :data:`EVAL2_THRESHOLD` and then checked against eval2's own membership, so
    the threshold this experiment reports is demonstrably the one that built the
    set rather than a number copied out of a README.
    """
    identity = read_identity_table()
    eval2_rows = read_eval2()
    eval2_keys = {(r["dataset"], r["stem"]) for r in eval2_rows}
    eval2_by_key = {(r["dataset"], r["stem"]): r for r in eval2_rows}
    sequences = read_query_sequences()
    # eval2's manifest carries `input_seq` for its 307; the query FASTA carries
    # all 776. They must agree where they overlap, or the sequence this audit
    # annotates is not the one exp226 searched.
    for key, row in eval2_by_key.items():
        if sequences[key] != row["input_seq"]:
            raise SystemExit(
                f"sequence mismatch for {key} between {QUERY_FASTA.name} and "
                f"{EVAL2_MANIFEST.name}"
            )

    proteins: list[Protein] = []
    for key, row in identity.items():
        gated = _f(row["best_identity_covered"])
        derived = gated is None or gated < EVAL2_THRESHOLD
        if derived != (key in eval2_keys):
            raise SystemExit(
                f"eval2 membership for {key} disagrees with "
                f"best_identity_covered={gated} < {EVAL2_THRESHOLD}; the "
                "threshold assumed here is not the one that built the set"
            )
        e2 = eval2_by_key.get(key)
        proteins.append(Protein(
            dataset=key[0],
            stem=key[1],
            sequence=sequences[key],
            length=_i(row["length"] or row["query_len"]),
            best_identity=gated,
            best_identity_ungated=_f(row["best_identity_any"]),
            afdb_best_identity=_f(row["afdb_best_identity_covered"]),
            esm_atlas_best_identity=_f(row["esm_atlas_best_identity_covered"]),
            n_hits=_i(row["n_hits"]),
            n_hits_significant=_i(row["n_hits_significant"]),
            stratum=row["stratum"],
            # exp213's dataset-label rule for the 554; exp226's organism proxy
            # additionally flags synthetic constructs among the FoldBench rows.
            # Deliberately taken from eval2's manifest where available, because
            # that is the flag the published 78 was computed with — §A2 audits it.
            designed_any=bool(_i((e2 or {}).get("designed_any", row["designed"]))),
            source_organism=(e2 or {}).get("source_organism", ""),
            msa_neff=_f(row.get("msa_neff", "")),
            fold_verdict=row.get("fold_verdict", ""),
            in_eval2=key in eval2_keys,
        ))
    return proteins


def eval2_natural(proteins: list[Protein]) -> list[Protein]:
    """The 78 — eval2 minus everything the published designed flag catches."""
    out = [p for p in proteins if p.in_eval2 and not p.designed_any]
    if len(out) != EXPECTED_EVAL2_NATURAL_N:
        raise SystemExit(
            f"eval2-natural has {len(out)} proteins, expected "
            f"{EXPECTED_EVAL2_NATURAL_N}"
        )
    return out


def read_exp65_dates() -> dict[str, str]:
    """``stem -> deposit_date`` from exp65's three source manifests."""
    dates: dict[str, str] = {}
    for path in (EXP65_CAMEO_MANIFEST, EXP65_CASP_MANIFEST, EXP65_DENOVO_MANIFEST):
        if not path.exists():
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                if row.get("deposit_date"):
                    dates[row["stem"]] = row["deposit_date"]
    return dates


def read_casp_pdb_map() -> dict[str, tuple[str, str]]:
    """``CASP domain stem -> (pdb_id, chain)`` from exp65's fallback map.

    exp65 built this to clip FM domains out of deposited entries when the
    predictioncenter tarballs lacked them. Here it is the only route from a CASP
    target id to an RCSB entry, and therefore to a source organism — the field
    exp226 left blank for all 19 CASP rows in the 78.
    """
    out: dict[str, tuple[str, str]] = {}
    if not EXP65_CASP_FALLBACK.exists():
        return out
    with EXP65_CASP_FALLBACK.open() as fh:
        for row in csv.DictReader(fh):
            if (row.get("status") or "").strip() == "unavailable":
                continue
            pdb = (row.get("pdb_id") or "").strip().lower()
            if not pdb:
                continue
            # exp65 keys this file on ``domain`` (``T1027-D1``), which is the
            # same string the eval manifests use as ``stem``.
            out[row["domain"]] = (pdb, (row.get("chain") or "").strip())
    return out


def afdb_arm_accessions(path: Path = AFDB_ARM_FASTA) -> set[str]:
    """Every UniProt accession present in the AFDB training arm.

    One pass over a 1.2 GB FASTA reading only header lines; ~10 s. Each header
    must match :data:`AFDB_HEADER_RE`, so a change in the header grammar (which
    would silently make every membership test return "absent") raises instead.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found. exp213's per-arm training FASTAs live on this "
            "workstation's /data volume and are not committed; rebuild with "
            "exp213's fetch_train_sequences.py before running §A3."
        )
    accessions: set[str] = set()
    bad = 0
    with path.open() as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m = AFDB_HEADER_RE.match(line)
            if m is None:
                bad += 1
                if bad <= 5:
                    print(f"[warn] unparsed AFDB header: {line.rstrip()!r}")
                continue
            accessions.add(m.group(1))
    if bad:
        raise SystemExit(
            f"{bad} AFDB headers did not match {AFDB_HEADER_RE.pattern!r}; the "
            "accession membership test would silently under-report"
        )
    if len(accessions) != N_AFDB:
        raise SystemExit(
            f"{path} yielded {len(accessions)} unique accessions, expected "
            f"{N_AFDB} (the arm size) — accessions are not 1:1 with rows"
        )
    return accessions


__all__ = [
    "ARM_AFDB", "ARM_ESM", "ARMS", "FORMAT", "HOMOLOGY_EVALUE",
    "MANIFEST_STRATA", "MIN_QCOV",
    "arm_of", "ensure_mmseqs", "reduce_alignments", "run",
    "EVAL2_THRESHOLD", "N_AFDB", "N_ESM_ATLAS", "N_TRAIN", "N_AFDB_V4",
    "N_AFDB_24M", "N_ESM_ATLAS_V0", "AFDB24M_MIN_PLDDT", "AFDB24M_MAX_LEN",
    "EXP53_NUM_ROUNDS",
    "EXP53_MIN_CLUSTER_SIZE", "EXP53_SEQ_LEN_RANGE", "ESM_ATLAS_LINCLUST_ID",
    "Protein", "read_proteins", "eval2_natural", "read_identity_table", "read_query_sequences",
    "read_eval2", "read_exp65_dates", "read_casp_pdb_map",
    "afdb_arm_accessions", "AFDB_ARM_FASTA", "AFDB_HEADER_RE",
    "HERE", "EXP226_DIR", "EXP213_DIR", "EXP65_DIR",
]
