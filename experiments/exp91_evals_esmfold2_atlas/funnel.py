# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality/novelty filtering funnel for distilling an ESM Atlas training subset.

The Atlas is a distillation source to *augment* our existing AFDB-based training
set for a new structure model. So the funnel keeps (sequence → structure) pairs
that are (a) reliable enough to train on and (b) not already covered by AFDB,
and drops anything that leaks into the held-out eval set.

**The spine is sequence + metadata, by design.** pLDDT and pTM are structure-quality
signals that ship as *metadata columns* of ``folds_1B.lance``; novelty, eval-leakage,
and clustering are all *sequence* operations (MMseqs2). So the whole filter never
reads the multi-TB ``structure_blob`` column — structures are decoded only for the
final selected representatives (materialization). The two *structural* gates are
**off by default**:

- ``afdb_dedup`` / ``max_afdb_tm`` (structural similarity to AFDB) — dropped on merit,
  not just cost: once a sequence is <40% identical to AFDB, a known-fold match is
  "new sequence on a known fold", which is the most valuable distillation signal
  (AF2's same-fold bucket), not redundancy. A foldseek all-vs-AFDB over 1.1B would
  also be the single most expensive stage.
- ``globularity`` (Cα–Cα contact ratio, ESMFold2's proxy) and ``plddt_uniform``
  (per-residue pLDDT std) — deferred to a cheap **post-selection QC** on the ~10–50M
  selected reps rather than a decode of all 1.1B structures.

This matches the cross-model consensus (AF3, Protenix, Boltz-1/2, Chai): gate per
*structure* (not per residue), keep *full chains* (do NOT trim disordered regions —
AF3/Protenix show training on extended-loop disorder predictions reduces
hallucination), and dedup at 40% sequence identity. Per-residue confidence loss
*masking* is an AF2-only technique and a training-time concern; this funnel preserves
``per_residue_plddt`` downstream so the model pipeline can optionally apply a soft
per-residue loss weight. The structural gates remain available (set the thresholds)
for sample-level analysis and the post-selection QC.

The funnel is a **pure function over a per-structure frame** (`apply_funnel`) with
these columns:

    protein_hash, seq_len, mean_plddt, ptm,
    contact_ratio, # Cα–Cα contact ratio: globularity proxy   (0..1, ESMFold2)
    plddt_std,     # per-residue pLDDT std-dev (rep tiebreaker) (0..1)
    afdb_seq_id,   # sequence identity to nearest AFDB entry  (0..1)
    afdb_tm,       # structural TM to nearest AFDB rep         (0..1)
    eval_tm,       # structural TM to nearest eval structure   (0..1)
    eval_seq_id    # sequence identity to nearest eval         (0..1)

``contact_ratio``/``plddt_std`` are optional: a gate whose column is absent is
skipped (like the eval stage), so the funnel still runs on the bare sample.

`load_funnel_frame` assembles that frame for the *sample* from the CSVs already in
``data/`` (so we can estimate survival rates per stage and tune thresholds). The
**production** run scans ``folds_1B.lance`` AWS-side and supplies the same columns
(mean_plddt/ptm straight from the Lance row; afdb_seq_id from an MMseqs2 search vs
AFDB sequences — stronger than the foldseek structural-alignment identity used as
the sample proxy here; afdb_tm/eval_tm from foldseek), then calls the *same*
`apply_funnel`. Keeping the gate logic I/O-free is what lets the sample-tuned
thresholds transfer verbatim to the full 1.1B-row run.

Stages (each is a keep-condition; a structure must pass all enabled stages). The
**default** pipeline enables only the sequence + metadata stages (1, 2, 3, 6, and
the sequence half of 9); the structural stages (4, 5, 7, 8) are off unless their
thresholds are set:

1. length          — ``min_len <= seq_len <= max_len``         (60–1000 upstream)
2. plddt           — ``mean_plddt >= min_plddt``               (label quality; meta)
3. ptm             — ``ptm >= min_ptm``                        (global coherence; meta)
4. globularity     — ``contact_ratio >= min_contact_ratio``   (structural; OFF, QC-only)
5. plddt_uniform   — ``plddt_std <= max_plddt_std``           (structural; OFF)
6. afdb_seq_novel  — ``afdb_seq_id < max_afdb_seq_id``         (MMseqs; adds coverage)
7. afdb_struct_novel — ``afdb_tm < max_afdb_tm``               (foldseek; OFF)
8. afdb_dedup      — ``afdb_tm < drop_afdb_redundant_tm``      (foldseek; OFF, see above)
9. eval_leakage    — ``eval_seq_id < eval_max_seq_id`` (MMseqs) AND, only if
                     ``eval_max_tm`` is set, ``eval_tm < eval_max_tm`` (foldseek)

NOTE on stage 7: requiring structural novelty vs AFDB (``max_afdb_tm = 0.5``) is
**anti-correlated with quality** in this dataset — novel folds have median pLDDT
~0.47 and only ~5% clear pLDDT>0.7, so stacking it on the quality gate collapses
the set to <1% and enriches for likely ESMFold2 artifacts. OFF by default.

Outputs: ``data/funnel_decisions.csv`` (per-structure stage flags),
``data/funnel_survival.csv`` (the survival table), ``plots/funnel.png``.
"""

import argparse
import dataclasses
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import gemmi
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
EXP41 = HERE.parent / "exp41_evals_foldseek_train_similarity"
EXP65 = HERE.parent / "exp65_evals_low_msa_depth_proteins"
sys.path.insert(0, str(EXP41))  # reuse the Foldseek binary wrapper + m8 parser
import query_similarity as qs  # noqa: E402

# Eval structures to dedup against (held-out, must not leak into training).
EVAL_STRUCT_DIRS = [
    EXP65 / "structures" / "cameo_hard",
    EXP65 / "structures" / "casp_fm_flat",
    EXP65 / "structures" / "denovo",
]

N_FOLDS_1B = 1_095_530_880  # for scaling sample fractions to the full Atlas


@dataclass(frozen=True)
class FunnelParams:
    """Tunable thresholds for the distillation funnel. Defaults are provisional."""

    min_len: int = 60
    max_len: int = 1000
    min_plddt: float = 0.70
    min_ptm: float = 0.50
    min_contact_ratio: float | None = None  # globularity: deferred post-selection QC
    max_plddt_std: float | None = None  # OFF: uniform-confidence gate (rep tiebreak)
    max_afdb_seq_id: float = 0.40  # 40% is the field dedup norm; permissive now
    max_afdb_tm: float | None = None  # OFF: structural-novelty gate (see module doc)
    drop_afdb_redundant_tm: float | None = None  # OFF: counterproductive for augment
    eval_max_tm: float | None = None  # OFF: leakage is sequence-based (eval_max_seq_id)
    eval_max_seq_id: float = 0.40


def apply_funnel(
    frame: pd.DataFrame, p: FunnelParams
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the funnel to a per-structure frame.

    Returns ``(decisions, survival)`` where ``decisions`` is ``frame`` plus a
    boolean column per stage and a final ``kept`` column, and ``survival`` is the
    cumulative stage table (one row per stage, with counts + fractions).
    """
    df = frame.copy()
    n0 = len(df)

    # Each stage: (name, keep-mask, enabled). Missing inputs (NaN) for an enabled
    # stage are treated as failing that stage, except eval columns when the eval
    # comparison was never computed (handled by enabling=False below).
    plddt = df["mean_plddt"]
    ptm = df["ptm"]
    length = df["seq_len"]
    aseq = df["afdb_seq_id"]
    atm = df["afdb_tm"]
    cratio = df.get("contact_ratio")
    pstd = df.get("plddt_std")
    etm = df.get("eval_tm")
    eseq = df.get("eval_seq_id")
    has_cratio = cratio is not None and cratio.notna().any()
    has_pstd = pstd is not None and pstd.notna().any()
    has_etm = etm is not None and etm.notna().any()
    has_eseq = eseq is not None and eseq.notna().any()
    eval_available = has_etm or has_eseq

    # Eval-leakage keep-mask: drop anything too similar to the held-out eval set.
    # Sequence identity is the primary axis (eval_max_seq_id); the structural TM
    # bound is layered on only if eval_tm was computed and eval_max_tm is set.
    eval_mask = pd.Series(True, index=df.index)
    if has_eseq:
        eval_mask &= eseq.fillna(0.0) < p.eval_max_seq_id
    if has_etm and p.eval_max_tm is not None:
        eval_mask &= etm.fillna(0.0) < p.eval_max_tm

    stages: list[tuple[str, pd.Series, bool]] = [
        ("length", (length >= p.min_len) & (length <= p.max_len), True),
        ("plddt", plddt >= p.min_plddt, True),
        ("ptm", ptm >= p.min_ptm, True),
        ("globularity",
         cratio >= p.min_contact_ratio if has_cratio
         else pd.Series(True, index=df.index),
         has_cratio and p.min_contact_ratio is not None),
        ("plddt_uniform",
         pstd <= p.max_plddt_std if has_pstd
         else pd.Series(True, index=df.index),
         has_pstd and p.max_plddt_std is not None),
        ("afdb_seq_novel", aseq.fillna(0.0) < p.max_afdb_seq_id, True),
        ("afdb_struct_novel", atm.fillna(0.0) < (p.max_afdb_tm or 1.0),
         p.max_afdb_tm is not None),
        ("afdb_dedup", atm.fillna(0.0) < (p.drop_afdb_redundant_tm or 1.1),
         p.drop_afdb_redundant_tm is not None),
        ("eval_leakage", eval_mask, eval_available),
    ]

    cum = pd.Series(True, index=df.index)
    rows = []
    for name, mask, enabled in stages:
        df[f"pass_{name}"] = mask
        if enabled:
            cum = cum & mask
        rows.append({
            "stage": name,
            "enabled": enabled,
            "passing_alone": int(mask.sum()),
            "cumulative_kept": int(cum.sum()),
            "cumulative_frac": float(cum.sum()) / n0 if n0 else 0.0,
            "est_full_atlas_millions": float(cum.sum()) / n0 * N_FOLDS_1B / 1e6
            if n0 else 0.0,
        })
    df["kept"] = cum
    survival = pd.DataFrame(rows)
    return df, survival


def load_funnel_frame(data_dir: Path = DATA) -> pd.DataFrame:
    """Assemble the per-structure funnel frame for the *sample* from data/ CSVs."""
    st = pd.read_csv(data_dir / "atlas_struct_sample.csv")
    afdb = pd.read_csv(data_dir / "atlas_vs_afdb_reps_similarity.csv")
    frame = st[["protein_hash", "seq_len", "mean_plddt", "ptm"]].merge(
        afdb[["stem", "best_train_qtmscore", "best_train_fident"]],
        left_on="protein_hash", right_on="stem", how="left",
    )
    frame = frame.rename(columns={
        "best_train_qtmscore": "afdb_tm",
        "best_train_fident": "afdb_seq_id",
    }).drop(columns=["stem"])
    metrics_csv = data_dir / "struct_metrics.csv"
    if metrics_csv.exists():
        m = pd.read_csv(metrics_csv)[["protein_hash", "contact_ratio", "plddt_std"]]
        frame = frame.merge(m, on="protein_hash", how="left")
    else:
        frame["contact_ratio"] = pd.NA
        frame["plddt_std"] = pd.NA
        print(f"[funnel] no {metrics_csv.name}; globularity/plddt_std gates "
              "disabled. Run with --compute-metrics to build it.")
    eval_csv = data_dir / "atlas_vs_eval.csv"
    if eval_csv.exists():
        ev = pd.read_csv(eval_csv)[["stem", "eval_tm", "eval_seq_id"]]
        frame = frame.merge(ev, left_on="protein_hash", right_on="stem",
                            how="left").drop(columns=["stem"])
    else:
        frame["eval_tm"] = pd.NA
        frame["eval_seq_id"] = pd.NA
        print(f"[funnel] no {eval_csv.name}; eval-leakage stage disabled. "
              "Run with --compute-eval to build it.")
    return frame


def compute_structure_metrics(
    candidate_dir: Path, contact_dist: float = 8.0, min_sep: int = 6
) -> Path:
    """Compute per-structure globularity + confidence-uniformity from the CIFs.

    ``contact_ratio`` = fraction of residues making at least one *tertiary* Cα–Cα
    contact (within ``contact_dist`` Å of a residue ≥ ``min_sep`` apart in
    sequence) — ESMFold2's globularity proxy. ``plddt_std`` = std-dev of the
    per-residue pLDDT (read from the CIF B-factor column, which stores
    per-residue pLDDT ×100). Writes ``data/struct_metrics.csv``.
    """
    rows = []
    for cif in sorted(candidate_dir.glob("*.cif")):
        st = gemmi.read_structure(str(cif))
        st.setup_entities()
        cas, bfs = [], []
        for res in st[0][0]:
            ca = res.find_atom("CA", "*")
            if ca is not None:
                cas.append([ca.pos.x, ca.pos.y, ca.pos.z])
                bfs.append(ca.b_iso)
        if len(cas) < 2:
            continue
        xyz = np.asarray(cas)
        bf = np.asarray(bfs)
        n = len(xyz)
        dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        seq_sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        tertiary = (dist <= contact_dist) & (seq_sep >= min_sep)
        plddt = bf / 100.0 if bf.max() > 1.5 else bf  # B-factor stores pLDDT×100
        rows.append({
            "protein_hash": cif.stem,
            "n_res_metric": n,
            "contact_ratio": float(tertiary.any(axis=1).mean()),
            "plddt_std": float(plddt.std()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "struct_metrics.csv", index=False)
    print(f"[metrics] wrote struct_metrics.csv ({len(df)} structures); "
          f"median contact_ratio {df['contact_ratio'].median():.2f}, "
          f"median plddt_std {df['plddt_std'].median():.2f}")
    return DATA / "struct_metrics.csv"


def compute_eval_similarity(candidate_dir: Path, eval_dirs: list[Path]) -> Path:
    """Foldseek the Atlas structure sample vs the held-out eval structures.

    Builds one combined eval directory of all eval ``.pdb``/``.cif`` files, runs
    ``easy-search`` (TM-align mode), keeps the best hit per Atlas structure, and
    writes ``data/atlas_vs_eval.csv`` with ``stem, eval_tm, eval_seq_id``.
    """
    combined = DATA / "_eval_structs"
    if combined.exists():
        shutil.rmtree(combined)
    combined.mkdir(parents=True)
    n = 0
    for d in eval_dirs:
        for f in sorted(d.glob("*.pdb")) + sorted(d.glob("*.cif")):
            (combined / f.name).symlink_to(f.resolve())
            n += 1
    print(f"[eval] {n} eval structures from {len(eval_dirs)} dirs")
    out_m8 = DATA / "atlas_vs_eval.m8"
    qs.easy_search(candidate_dir, combined, out_m8, DATA / "_tmp_eval",
                   alignment_type=1, max_seqs=50)
    hits = qs.parse_m8(out_m8)
    hits["stem"] = hits["query"].map(lambda s: str(s).rsplit(".", 1)[0])
    best = (hits.sort_values("qtmscore", ascending=False)
                .groupby("stem", as_index=False).first())
    best = best.rename(columns={"qtmscore": "eval_tm", "fident": "eval_seq_id"})
    best[["stem", "eval_tm", "eval_seq_id"]].to_csv(
        DATA / "atlas_vs_eval.csv", index=False)
    print(f"[eval] wrote atlas_vs_eval.csv ({len(best)} structures with a hit)")
    return DATA / "atlas_vs_eval.csv"


def plot_funnel(survival: pd.DataFrame, p: FunnelParams) -> None:
    """Horizontal funnel bar chart of cumulative survivors per enabled stage."""
    s = survival[survival["enabled"]].copy()
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(s) + 1.5))
    y = range(len(s))
    ax.barh(list(y), s["cumulative_frac"], color="#4477aa")
    ax.set_yticks(list(y))
    ax.set_yticklabels(s["stage"])
    ax.invert_yaxis()
    ax.set(xlabel="fraction of sampled Atlas structures surviving (cumulative)",
           xlim=(0, 1),
           title=f"ESM Atlas distillation funnel "
                 f"(pLDDT≥{p.min_plddt}, pTM≥{p.min_ptm}, "
                 f"seq<{p.max_afdb_seq_id:.0%} vs AFDB)")
    for i, (_, r) in enumerate(s.iterrows()):
        ax.text(r["cumulative_frac"] + 0.01, i,
                f"{r['cumulative_frac']:.1%}  (~{r['est_full_atlas_millions']:.0f}M)",
                va="center")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "funnel.png", dpi=120)
    plt.close(fig)


def build_params(args: argparse.Namespace) -> FunnelParams:
    return FunnelParams(
        min_len=args.min_len, max_len=args.max_len,
        min_plddt=args.min_plddt, min_ptm=args.min_ptm,
        min_contact_ratio=args.min_contact_ratio, max_plddt_std=args.max_plddt_std,
        max_afdb_seq_id=args.max_afdb_seq_id, max_afdb_tm=args.max_afdb_tm,
        drop_afdb_redundant_tm=args.drop_afdb_redundant_tm,
        eval_max_tm=args.eval_max_tm, eval_max_seq_id=args.eval_max_seq_id,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--min-plddt", type=float, default=0.70,
                    help="mean-pLDDT quality floor (provisional default)")
    ap.add_argument("--min-ptm", type=float, default=0.50,
                    help="set 0.70 to recover the classic ESM 'well-packed' bar")
    ap.add_argument("--min-contact-ratio", type=float, default=None,
                    help="ESMFold2 globularity proxy (structural); OFF by default — "
                         "deferred to post-selection QC. Skipped if column absent")
    ap.add_argument("--max-plddt-std", type=float, default=None,
                    help="optional uniform-confidence gate; mainly a cluster-rep "
                         "tiebreaker (ESMFold2). Off by default")
    ap.add_argument("--max-afdb-seq-id", type=float, default=0.40,
                    help="keep if seq identity to AFDB is below this (novelty); "
                         "40% is the field dedup norm, permissive (filter down later)")
    ap.add_argument("--max-afdb-tm", type=float, default=None,
                    help="OPTIONAL structural-novelty gate (foldseek); off by default "
                         "(anti-correlated with quality, see module docstring)")
    ap.add_argument("--drop-afdb-redundant-tm", type=float, default=None,
                    help="OPTIONAL structural dedup vs AFDB (foldseek); OFF by default "
                         "— counterproductive for augmentation (see module docstring)")
    ap.add_argument("--eval-max-tm", type=float, default=None,
                    help="OPTIONAL structural eval-leakage bound (foldseek); OFF — "
                         "leakage is sequence-based via --eval-max-seq-id")
    ap.add_argument("--eval-max-seq-id", type=float, default=0.40)
    ap.add_argument("--candidate-dir", type=Path, default=DATA / "struct_sample")
    ap.add_argument("--compute-metrics", action="store_true",
                    help="(re)build struct_metrics.csv (contact_ratio, plddt_std) "
                         "from the candidate CIFs before filtering")
    ap.add_argument("--contact-dist", type=float, default=8.0,
                    help="Cα–Cα tertiary-contact distance (Å) for contact_ratio")
    ap.add_argument("--contact-min-sep", type=int, default=6,
                    help="min sequence separation for a contact to count tertiary")
    ap.add_argument("--compute-eval", action="store_true",
                    help="(re)build atlas_vs_eval.csv via foldseek before filtering")
    args = ap.parse_args(argv)

    DATA.mkdir(parents=True, exist_ok=True)
    if args.compute_metrics:
        compute_structure_metrics(args.candidate_dir, args.contact_dist,
                                  args.contact_min_sep)
    if args.compute_eval:
        compute_eval_similarity(args.candidate_dir, EVAL_STRUCT_DIRS)

    params = build_params(args)
    frame = load_funnel_frame(DATA)
    decisions, survival = apply_funnel(frame, params)

    decisions.to_csv(DATA / "funnel_decisions.csv", index=False)
    survival.to_csv(DATA / "funnel_survival.csv", index=False)
    plot_funnel(survival, params)

    print(f"\nFunnel params: {dataclasses.asdict(params)}")
    print(survival.to_string(index=False))
    kept = int(decisions["kept"].sum())
    print(f"\nKept {kept}/{len(decisions)} = {kept / len(decisions):.1%} "
          f"of the sample  →  ~{kept / len(decisions) * N_FOLDS_1B / 1e6:.0f}M "
          f"of {N_FOLDS_1B / 1e6:.0f}M full Atlas (before cluster-dedup).")


if __name__ == "__main__":
    main()
