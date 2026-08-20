# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The single seam onto everything exp245 inherits from other experiments.

exp245 is an assembly job: the FoldBench monomer universe comes from #226, the
contact definition from #89, the decontamination evidence from #225, the rollout
harness and the pinned checkpoint identities from #232's evaluation (PR #244).
Re-implementing any of those would produce numbers that agree today and drift
tomorrow, so this module -- and only this module -- resolves them, and everything
else in exp245 imports from here.

Three kinds of input, resolved three ways:

**Committed sibling experiments** (``exp226``, ``exp89``, ``exp232``) are read
from the worktree. exp245 is stacked on ``exp/232-evals-v1`` (PR #244) for the
rollout harness, the same way exp226 was stacked on exp213's branch.

**Local working artifacts** (#225's drop list and its all-FoldBench alignment
file) live under ``/data/exp225_decontam`` and are not in git -- they are 30 MB
and 1.3 GB. They are pinned by size so a rebuild on another machine fails loudly
rather than verifying against the wrong file.

**Branch-only artifacts** (#241's RCSB annotation, PR #242) are extracted from
their branch through ``git show`` into a cache dir. exp245 does not *depend* on
that file: it runs its own RCSB annotation pass and uses #241's only as a
cross-check, so an unmerged PR cannot block this one.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent

#: Where the big non-git inputs live. Both were produced by #225 on this
#: workstation; see its README for the commands that build them.
DECONTAM_WORK = Path("/data/exp225_decontam")

#: exp245's own scratch: mmCIF downloads, rollout parts, baseline outputs.
WORK = Path("/data/exp245")

# --- sibling experiment directories -----------------------------------------

EXP226_DIR = EXPERIMENTS / "exp226_evals_expand_foldbench_eval_set"
EXP89_DIR = EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set"
EXP78_DIR = EXPERIMENTS / "exp78_evals_esmfold_contacts"
EXP74_DIR = EXPERIMENTS / "exp74_evals_protenix_pyconfind_contacts"
EXP232_ROLLOUT_DIR = EXPERIMENTS / "exp232_sweep_cv1_decontam" / "evals" / "rollout_v2"

for _required in (EXP226_DIR, EXP89_DIR, EXP78_DIR, EXP74_DIR, EXP232_ROLLOUT_DIR):
    if not _required.is_dir():  # pragma: no cover - branch-layout guard
        raise SystemExit(
            f"required sibling experiment not found: {_required}\n"
            "exp245 is stacked on exp/232-evals-v1 (PR #244); check out a "
            "commit that contains both it and main."
        )

#: #226's committed FoldBench artifacts.
FOLDBENCH_TARGETS = EXP226_DIR / "data" / "foldbench_targets.csv"
EXPANDED_IDENTITY = EXP226_DIR / "data" / "eval_train_identity_expanded.csv"
EVAL2_MANIFEST = EXP226_DIR / "data" / "eval2_manifest.csv"
EVAL2_PER_PROTEIN = EXP226_DIR / "data" / "eval2_per_protein.csv.gz"

#: The published 554-unit eval targets parquet -- the only place the *input
#: sequences* the frozen ground truth was built from are recorded. The control
#: in ``build_ground_truth.py`` needs them to tell a genuine mismatch from a
#: frozen unit that was simply built from a different sequence for the same PDB
#: stem. Same file, digest and size PR #244 pins as ``LEGACY_TARGETS_URL``.
LEGACY_TARGETS = Path("/data/exp169_eval/eval_targets.parquet")
LEGACY_TARGETS_SIZE = 43_077
LEGACY_TARGETS_SHA256 = (
    "9de9bc1b99b7e7ab6d2b17a985f9e22bc7decd2b25e1b16be30dea921431c111"
)

#: The frozen 577-unit ground-truth universe, published by #226. Used only as a
#: control: exp245 rebuilds ground truth for all 334 monomers through one path
#: and checks that the units which overlap reproduce these records.
FROZEN_GT_UNIVERSE = Path("/data/exp226_gt/gt_universe_eval2_577.jsonl")

#: #225's applied drop list, and the alignment file its FoldBench half was
#: reduced from. Pinned so `confirm_decontamination.py` cannot silently verify
#: against a different search.
DROPLIST_FINAL = DECONTAM_WORK / "droplist_final.parquet"
FOLDBENCH_ALIGNMENTS = DECONTAM_WORK / "foldbench_all" / "aln_all_hits.m8"
DROPLIST_FINAL_SIZE = 31_510_179
FOLDBENCH_ALIGNMENTS_SIZE = 1_323_924_580

#: The reference FASTA #225 searched with, committed on its branch.
EXP225_BRANCH = "origin/claude/github-issue-225-fead58"
EXP225_REFERENCE_PATH = (
    "experiments/exp225_data_decontaminate_training_corpora/"
    "data/reference/foldbench_all_queries.fasta"
)

#: mmseqs2 ``--format-output`` field order used by #225's searches.
M8_FIELDS = ("query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits")

#: #225's published rule, as applied to both corpora: identity over coverage of
#: the *shorter* of the two sequences, with no E-value arm.
DECONTAM_MIN_IDENTITY = 0.30
DECONTAM_MIN_COVERAGE = 0.50

#: Published corpus sizes after that rule (#225 ``verify_published.py``), which
#: exp232's tokenizer independently re-asserts before training.
PUBLISHED_CORPUS_ROWS = {"afdb": 3_963_003, "esm_atlas": 65_553_178}
PUBLISHED_CORPUS_REMOVED = {"afdb": 166_679, "esm_atlas": 1_206_744}

# --- #241, which is still on a branch ---------------------------------------

EXP241_BRANCH = "origin/claude/eval2-natural-analysis-932188"
EXP241_ANNOTATION_PATH = (
    "experiments/exp241_evals_why_does_eval2_natural_exist_audit/"
    "data/rcsb_annotation.csv"
)


def sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file, for artifacts too large to read into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def require_pinned(path: Path, expected_size: int) -> Path:
    """Return ``path``, or fail with the reason it cannot be trusted."""
    if not path.exists():
        raise SystemExit(
            f"missing pinned input {path}. It is produced by #225 "
            "(experiments/exp225_data_decontaminate_training_corpora) and is "
            "too large for git; rebuild it there or copy it in."
        )
    size = path.stat().st_size
    if size != expected_size:
        raise SystemExit(
            f"{path} is {size} bytes, expected {expected_size}. This is not the "
            "artifact exp245's numbers were verified against."
        )
    return path


def from_branch(branch: str, path: str, out: Path) -> Path | None:
    """Extract one committed file from an unmerged branch into a cache dir."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        blob = subprocess.check_output(
            ["git", "show", f"{branch}:{path}"], cwd=REPO, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print(f"[upstream] {branch} unavailable; {path} not extracted", file=sys.stderr)
        return None
    out.write_bytes(blob)
    return out


def exp241_annotation(cache_dir: Path = WORK / "upstream") -> Path | None:
    """#241's RCSB annotation of all 776 eval proteins. Cross-check only.

    exp245 derives its own annotation, so a missing branch (a fresh clone, or
    #242 merged under a different ref) degrades this to a warning rather than
    failing a run that does not need it.
    """
    return from_branch(
        EXP241_BRANCH, EXP241_ANNOTATION_PATH, cache_dir / "exp241_rcsb_annotation.csv",
    )


def exp225_reference(cache_dir: Path = WORK / "upstream") -> Path | None:
    """The 1,940-chain FoldBench FASTA #225 decontaminated the corpora against."""
    return from_branch(
        EXP225_BRANCH, EXP225_REFERENCE_PATH, cache_dir / "foldbench_all_queries.fasta",
    )


def exp89_contacts():
    """#89's ``compute_contacts`` and its pyconfind geometry, imported not copied.

    Same seam pattern as exp226's ``build_gt_contacts.py``: the contact
    definition has one owner, and three experiments already carry copies of it.
    """
    if str(EXP89_DIR) not in sys.path:
        sys.path.insert(0, str(EXP89_DIR))
    from pyconfind_contacts import PYCONFIND_KWARGS, compute_contacts

    return compute_contacts, PYCONFIND_KWARGS


def exp89_metrics():
    """#89's metric implementation -- the one every published number uses."""
    if str(EXP89_DIR) not in sys.path:
        sys.path.insert(0, str(EXP89_DIR))
    import compute_metrics

    return compute_metrics
