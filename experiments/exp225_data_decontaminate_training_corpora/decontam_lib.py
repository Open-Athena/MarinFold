# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for the exp225 eval-decontamination pass (issue #225).

Everything here is a *definition*, not a computation: the tier ladder, the
thresholds each tier applies, the id grammar that lets a hit be inverted back
into a corpus row, and the registry of corpora that can be filtered. The four
pipeline scripts import from here so they cannot disagree about what "Tier B"
means — and so a future corpus (#222's experimental-PDB set, anything after
it) can be decontaminated by adding one :class:`Corpus` entry rather than
re-deriving an ad-hoc reference.

**The reference is versioned.** :data:`REFERENCE_VERSION` is stamped into
every drop list and is meant to appear in the ``decontam_tier`` /
``decontam_reference`` fields of any dataset built from one. Changing which
proteins are in the eval set means bumping it; changing a *threshold* means a
new tier, not a new version, because the tiers are what the survival table is
reported against.

Thresholds and where they come from
-----------------------------------

* :data:`SEQ_MIN_IDENTITY` = 0.30 — exp65's ``seq_leakage.py`` ``REDUNDANT_ID``,
  the twilight-zone novel-family boundary. #91's ESM-Atlas funnel dropped at
  0.40 instead and so kept the 30–40 % band by design; this is gap 2 of the
  issue's audit.
* :data:`SEQ_MIN_QCOV` = 0.50 — exp213's ``MIN_QCOV``. A 95 %-identical
  12-residue local match is not homology.
* :data:`SEQ_MAX_EVALUE` = 1e-3 — exp213's ``HOMOLOGY_EVALUE``, the same
  significance bar that defined #213's 323-homologous / 231-novel split. It is
  a *catch-all* alongside the identity rule, not a conjunct: a remote homolog
  that aligns over 20 % of the query at 22 % identity still counts as
  contamination even though it clears neither identity nor coverage.
* :data:`STRUCT_REDUNDANT_TM` = 0.90 / :data:`STRUCT_FOLD_TM` = 0.50 — exp41's
  ``DEFAULT_REDUNDANT_TM`` / ``DEFAULT_FOLD_TM``. 0.5 is the field-canonical
  same-fold boundary (Barrio-Hernandez 2023, the AFDB cluster definition).

Identity convention matches exp213 exactly, because the sequence axis reuses
its 70.9 M-sequence MMseqs2 target database verbatim: the **eval protein is the
query**, so ``qcov`` is the fraction of the *eval* protein an alignment covers.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# --- the pinned reference ---------------------------------------------------

#: Bumped when the *membership* of the eval reference changes (a protein added
#: or removed). Thresholds moving is a new tier, not a new version.
REFERENCE_VERSION = "v1"

#: What the reference contains, asserted at build time so a silently-truncated
#: manifest can't produce an under-filtered corpus.
N_EVAL_PROTEINS = 554


# --- training arms / filterable corpora -------------------------------------

ARM_AFDB = "afdb"
ARM_ESM = "esm_atlas"
ARMS = (ARM_AFDB, ARM_ESM)

ARM_LABELS = {
    ARM_AFDB: "AFDB (AlphaFold2 labels)",
    ARM_ESM: "ESM Atlas (ESMFold2 labels)",
}


@dataclass(frozen=True)
class Corpus:
    """One filterable document corpus on the ``open-athena/MarinFold`` bucket.

    ``n_documents`` is the published train-split document count and is checked
    against the drop list's denominator — a survival percentage computed
    against the wrong total is worse than no percentage.

    ``has_struct_cluster_id`` says whether the corpus rows carry the AFDB
    structural cluster id. Only the AFDB arm does, which is why the fold-level
    purge (Tier C) is exact there and needs a Foldseek database of its own for
    ESM-Atlas.
    """

    arm: str
    prefix: str
    shard_name: str
    n_shards: int
    n_documents: int
    has_struct_cluster_id: bool
    label: str
    #: Where a decontaminated rebuild is published. New prefix, never in place,
    #: so every existing checkpoint stays reproducible against the corpus it saw.
    decontam_prefix: str

    def remote(self, shard: int) -> str:
        return f"{self.prefix}/{self.shard_name.format(shard)}"


CORPORA: dict[str, Corpus] = {
    ARM_AFDB: Corpus(
        arm=ARM_AFDB,
        prefix="data/document_structures/contacts_v1/train",
        shard_name="contacts_v1-{:05d}-of-02067.parquet",
        n_shards=2067,
        n_documents=4_129_682,
        has_struct_cluster_id=True,
        label="AFDB / AlphaFold2 labels (issue #53)",
        decontam_prefix="data/document_structures/contacts_v1_decontam/train",
    ),
    ARM_ESM: Corpus(
        arm=ARM_ESM,
        prefix="data/document_structures/contacts_v1_esm_atlas/train",
        shard_name="shard-{:05d}-of-03338.parquet",
        n_shards=3338,
        n_documents=66_759_922,
        has_struct_cluster_id=False,
        label="ESM Atlas / ESMFold2 labels (issue #139)",
        decontam_prefix="data/document_structures/contacts_v1_esm_atlas_decontam/train",
    ),
}


# --- the training-row id grammar --------------------------------------------
#
# exp213 wrote every training sequence into its FASTA as
# ``{arm}|{shard:05d}_{row}_{entry_id}``. That header is the whole reason
# decontamination is a row filter and not a regeneration: an MMseqs2 hit names
# the exact (shard, row) it came from, so inverting hits into a drop list needs
# no join against the corpus at all.


@dataclass(frozen=True)
class TrainingRow:
    """One row of one corpus shard, as named by an MMseqs2 target header."""

    arm: str
    shard: int
    row: int
    entry_id: str

    @property
    def key(self) -> str:
        """``{arm}|{entry_id}`` — the corpus-level identity of this row.

        The drop list is applied on ``entry_id`` (that is the column the
        published parquet carries), so this, not ``(shard, row)``, is what a
        filter matches on. ``(shard, row)`` is retained for provenance and to
        make the filter checkable shard-by-shard.
        """
        return f"{self.arm}|{self.entry_id}"


def parse_target(target: str) -> TrainingRow:
    """Invert exp213's FASTA header grammar.

    ``"esm_atlas|00123_45_0000052aa00ab212061f7c6987fd87ae"`` becomes
    ``TrainingRow(arm="esm_atlas", shard=123, row=45, entry_id="0000…")``.

    ``entry_id`` is taken as everything after the second underscore, so an
    entry id containing underscores round-trips; ``arm`` and the two integers
    are validated because a malformed header here would silently under-filter
    the corpus.
    """
    arm, _, local = target.partition("|")
    if arm not in ARMS:
        raise ValueError(f"target {target!r} has no recognised arm prefix")
    shard_text, _, rest = local.partition("_")
    row_text, _, entry_id = rest.partition("_")
    if not (shard_text.isdigit() and row_text.isdigit() and entry_id):
        raise ValueError(f"target {target!r} does not match {{arm}}|{{shard}}_{{row}}_{{entry_id}}")
    return TrainingRow(arm=arm, shard=int(shard_text), row=int(row_text), entry_id=entry_id)


# --- thresholds -------------------------------------------------------------

SEQ_MIN_IDENTITY = 0.30
SEQ_MIN_QCOV = 0.50
SEQ_MAX_EVALUE = 1e-3

STRUCT_REDUNDANT_TM = 0.90
STRUCT_FOLD_TM = 0.50


def is_sequence_contaminant(identity: float, qcov: float, evalue: float) -> bool:
    """Tier A's rule for one alignment: identity-and-coverage **or** significance.

    The disjunction is the point. The identity arm catches the obvious
    near-duplicates that #91's 40 % funnel let through in the 30–40 % band; the
    E-value arm catches the remote homologs that align over too little of the
    query to clear the coverage gate but are still evidence of a relative.
    """
    if evalue <= SEQ_MAX_EVALUE:
        return True
    return identity >= SEQ_MIN_IDENTITY and qcov >= SEQ_MIN_QCOV


# --- the tier ladder --------------------------------------------------------

TIER_A = "A"
TIER_B = "B"
TIER_C = "C"
TIERS = (TIER_A, TIER_B, TIER_C)

#: Each tier is cumulative: B is A plus a structural-redundancy rule, C is B
#: plus the fold-level cluster purge. The survival table reports all three so
#: the cost of each additional axis is visible before anything is retrained.
TIER_RULES = {
    TIER_A: (
        f"sequence: identity >= {SEQ_MIN_IDENTITY:.0%} over >= {SEQ_MIN_QCOV:.0%} "
        f"query coverage, or E <= {SEQ_MAX_EVALUE:g}"
    ),
    TIER_B: f"Tier A, plus structural TM >= {STRUCT_REDUNDANT_TM:.2f} to any eval structure",
    TIER_C: f"Tier B, plus every cluster with TM >= {STRUCT_FOLD_TM:.2f} to any eval structure",
}

TIER_LABELS = {
    TIER_A: "A — sequence",
    TIER_B: "B — + structurally redundant",
    TIER_C: "C — + fold-level purge",
}


def tiers_up_to(tier: str) -> tuple[str, ...]:
    """The tiers a given tier subsumes, itself included (tiers are cumulative)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    return TIERS[: TIERS.index(tier) + 1]


# --- external binaries ------------------------------------------------------

_CACHE = Path.home() / ".cache" / "marinfold"

MMSEQS_DOWNLOAD = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
FOLDSEEK_DOWNLOAD = "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"


def _ensure_binary(name: str, url: str, env_var: str) -> str:
    """Path to ``name``, installing the static build under the shared cache.

    Mirrors exp65/exp94/exp213's mmseqs installer and exp41's foldseek one so
    all of them share one cached copy per binary.
    """
    override = os.environ.get(env_var)
    if override and Path(override).exists():
        return str(Path(override).resolve())
    on_path = shutil.which(name)
    if on_path:
        return on_path
    root = _CACHE / name
    binary = root / name / "bin" / name
    if not binary.exists():
        root.mkdir(parents=True, exist_ok=True)
        tar = root / f"{name}.tar.gz"
        print(f"[decontam] downloading {name} from {url}", flush=True)
        urllib.request.urlretrieve(url, tar)
        with tarfile.open(tar) as tf:
            tf.extractall(root, filter="data")
        tar.unlink(missing_ok=True)
    return str(binary.resolve())


def ensure_mmseqs() -> str:
    return _ensure_binary("mmseqs", MMSEQS_DOWNLOAD, "MMSEQS_BIN")


def ensure_foldseek() -> str:
    return _ensure_binary("foldseek", FOLDSEEK_DOWNLOAD, "FOLDSEEK_BIN")


def run(cmd: list[str], *, quiet: bool = False) -> None:
    """Run a subprocess, echoing an abbreviated command line.

    Output is passed through by default. The searches here run for minutes to
    hours over databases of 1.3 M structures and 70.9 M sequences, and a
    swallowed progress bar means the only way to tell a slow run from a stuck
    one is to go poking at the tool's scratch directory.
    """
    print("  $", " ".join(str(c) for c in cmd[:8]), "...", flush=True)
    subprocess.run(
        [str(c) for c in cmd],
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
