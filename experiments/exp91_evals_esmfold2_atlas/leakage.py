# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe whether our eval set leaks into the ESM Atlas (issue #91 Q4).

If we fold the Atlas into training, any Atlas protein that is (near-)identical
to a held-out eval protein would leak the eval answer into training. Our eval
set is the low-MSA-depth set built in exp65 (de-novo PDB designs, CAMEO-hard,
CASP-FM); its sequences live in
``../exp65_evals_low_msa_depth_proteins/data/candidate_sequences.csv``.

The Atlas deduplicates on the MD5 of the amino-acid sequence, and that MD5 *is*
the ``protein_hash``. So for each eval sequence we run two checks against the
Atlas REST API (``https://biohub.ai/esm/protein/api/v1alpha1``):

1. **Exact** — ``GET /proteins/{md5(seq)}?fold_on_miss=false``. A 200 means an
   identical sequence is in the Atlas (definitive exact-duplicate leakage).
2. **Near** — ``GET /similarity-search?sequence=...`` returns the Atlas proteins
   nearest in ESMC/SAE feature space. For the top hit we parse its sequence out
   of the returned PDB and compute global-alignment **sequence identity** +
   coverage to the eval query, applying the eval-design leakage criterion
   (id ≥ 0.30 over ≥ 0.50 coverage).

Output: ``data/atlas_leakage.csv`` (one row per eval protein) and
``data/atlas_leakage_summary.csv`` (counts overall and per source). This is an
upper-bound probe via a single sensitive search; the authoritative dedup remains
exp65's MMseqs2/Foldseek pass, which — if the Atlas is adopted — must be re-run
with the Atlas subset added to the training reference (see README §Leakage).
"""

import argparse
import hashlib
import io
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import gemmi
import pandas as pd
from Bio.Align import PairwiseAligner

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP65_SEQS = (HERE.parent / "exp65_evals_low_msa_depth_proteins"
              / "data" / "candidate_sequences.csv")
API = "https://biohub.ai/esm/protein/api/v1alpha1"

_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "global"
_ALIGNER.match_score = 1
_ALIGNER.mismatch_score = 0
_ALIGNER.open_gap_score = -1
_ALIGNER.extend_gap_score = -0.5


def _get(url: str, timeout: int = 60, retries: int = 3):
    """GET returning (status, bytes); retries on transient 429/5xx."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return e.code, e.read()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return None, b""


def _seq_from_pdb(pdb_text: str) -> str:
    """One-letter sequence of the first chain of a PDB string (via gemmi)."""
    st = gemmi.read_pdb_string(pdb_text)
    if not st or len(st) == 0 or len(st[0]) == 0:
        return ""
    chain = st[0][0]
    return gemmi.one_letter_code([r.name for r in chain]).upper().replace("X", "")


def identity_coverage(query: str, hit: str) -> tuple[float, float]:
    """Global-alignment fractional identity and coverage (over query)."""
    if not hit:
        return 0.0, 0.0
    aln = _ALIGNER.align(query, hit)[0]
    matches = sum(a == b for a, b in zip(aln[0], aln[1]) if a != "-" and b != "-")
    ident = matches / max(1, min(len(query), len(hit)))
    cov = sum(c != "-" for c in aln[1]) / max(1, len(query))
    return ident, cov


def probe_one(seq: str, topk: int) -> dict:
    md5 = hashlib.md5(seq.encode()).hexdigest()
    status, _ = _get(f"{API}/proteins/{md5}?fold_on_miss=false")
    exact = status == 200

    q = urllib.parse.urlencode({"sequence": seq, "topk_results": topk})
    s2, body = _get(f"{API}/similarity-search?{q}")
    rec = {"md5": md5, "exact_dup": exact, "top_sae_sim": None,
           "top_hit_accession": None, "top_seq_identity": 0.0,
           "top_coverage": 0.0, "search_status": s2}
    if s2 == 200 and body:
        import json
        hits = json.loads(body).get("similar_proteins", [])
        if hits:
            h = hits[0]
            rec["top_sae_sim"] = h.get("similarity_score")
            rec["top_hit_accession"] = h.get("protein_accession")
            ident, cov = identity_coverage(seq, _seq_from_pdb(h.get("pdb", "")))
            rec["top_seq_identity"] = ident
            rec["top_coverage"] = cov
    return rec


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqs", type=Path, default=EXP65_SEQS)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--id-threshold", type=float, default=0.30)
    ap.add_argument("--cov-threshold", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=0, help="0 = all eval proteins")
    args = ap.parse_args(argv)

    DATA.mkdir(parents=True, exist_ok=True)
    evals = pd.read_csv(args.seqs)
    if args.limit:
        evals = evals.head(args.limit)
    rows = []
    for i, r in evals.reset_index(drop=True).iterrows():
        rec = {"stem": r["stem"], "source": r["source"], "length": r["length"]}
        rec.update(probe_one(r["sequence"], args.topk))
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[leakage] {i + 1}/{len(evals)} probed", flush=True)
    df = pd.DataFrame(rows)
    df["near_dup"] = ((df["top_seq_identity"] >= args.id_threshold)
                      & (df["top_coverage"] >= args.cov_threshold))
    df.to_csv(DATA / "atlas_leakage.csv", index=False)

    def _summary(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            "n_exact_dup": int(g["exact_dup"].sum()),
            "n_near_dup": int(g["near_dup"].sum()),
            "max_seq_identity": float(g["top_seq_identity"].max()),
            "median_top_sae_sim": float(g["top_sae_sim"].dropna().median()
                                        if g["top_sae_sim"].notna().any() else 0.0),
        })

    overall = _summary(df).to_frame("ALL").T
    by_src = df.groupby("source").apply(_summary, include_groups=False)
    summary = pd.concat([overall, by_src])
    summary.to_csv(DATA / "atlas_leakage_summary.csv")
    print("[leakage] summary:\n", summary.to_string())


if __name__ == "__main__":
    main()
