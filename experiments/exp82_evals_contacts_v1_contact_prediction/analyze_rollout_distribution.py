# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-rollout R-precision distribution + R-precision vs rollout NLL.

For a few example proteins, draw N=100 resampled rollouts (the recipe) and, for
each individual rollout, record:
  * its contacts in generation order -> R-precision = (# of the first R generated
    contacts that are true) / R, with R = number of GT contacts (sep>=6, resolved);
  * the model's mean per-token negative log-likelihood of the generated section
    (raw, teacher-forced).
We also compute the **aggregate** R-precision the recipe actually uses (rank all
candidate pairs by occurrence count across the 100 rollouts, take top-R).

Per protein it writes a 2-panel figure: (1) histogram of per-rollout R-precision
with the aggregate / best / mean marked; (2) per-rollout R-precision vs NLL.

    PYTHONPATH=<repo>/marinfold uv run python analyze_rollout_distribution.py \
        --model /home/bizon/exp89_export/hf_step35679 --out-dir _scratch/rollout_dist
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from eval_contact_prediction import BEGIN, CONTACT_RE, MIN_SEP, NUM_POS, Scorer, candidate_pairs
from eval_full_curated_set import load_eval_records
from eval_rollout_resampled_dev import realization
from marinfold.document_structures.contacts_v1 import residues_from_sequence

EXAMPLES = [("foldbench100", "7y54_A"), ("foldbench100", "8axj_A"),
            ("foldbench100", "8bau_A"), ("foldbench100", "7qp5_A")]


@torch.no_grad()
def generate_with_nll(scorer, prefixes, temperature, top_p, top_k, max_new, batch):
    """One sampled completion per (equal-length) prefix; returns (gen_token_lists,
    mean_per_token_nll) aligned with prefixes. NLL is the raw teacher-forced model
    log-prob of the generated section (incl. <end>), length-normalised."""
    P = len(prefixes[0])
    eos = scorer.end_id
    gens, nlls = [], []
    for s in range(0, len(prefixes), batch):
        X = torch.tensor(prefixes[s:s + batch], device=scorer.device)        # [b, P]
        out = scorer.model.generate(X, do_sample=True, temperature=temperature, top_p=top_p,
                                    top_k=top_k, max_new_tokens=max_new,
                                    eos_token_id=eos, pad_token_id=eos)         # [b, P+G]
        logp = torch.log_softmax(scorer.model(out).logits.float(), -1)
        tok_lp = logp[:, :-1].gather(2, out[:, 1:, None]).squeeze(-1)           # [b, T-1]; col t = logP(out[t+1])
        T = out.shape[1]
        gen = out[:, P:]
        is_eos = gen == eos
        has = is_eos.any(1)
        first = torch.where(has, is_eos.float().argmax(1),
                            torch.full((gen.shape[0],), gen.shape[1] - 1, device=scorer.device))
        E = (P + first).long()                                                 # token idx of first <end> (incl.)
        cols = torch.arange(T - 1, device=scorer.device)[None, :]
        mask = (cols >= P - 1) & (cols <= (E - 1)[:, None])                     # generated-token columns
        cnt = mask.sum(1).clamp(min=1)
        mean_nll = (-(tok_lp * mask).sum(1) / cnt).cpu().numpy()
        Ei = E.tolist()
        for i in range(out.shape[0]):
            gens.append(out[i, P:Ei[i] + 1].tolist())
        nlls.extend(mean_nll.tolist())
    return gens, nlls


def ordered_contacts(scorer, gen_ids, seqidx):
    """Contacts in generation order (sep>=6, deduped), mapped to seq indices."""
    out, seen = [], set()
    for a, b in CONTACT_RE.findall(scorer.tok.decode(gen_ids, skip_special_tokens=False)):
        ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
        if ia is None or ib is None or ia == ib:
            continue
        key = (min(ia, ib), max(ia, ib))
        if abs(ia - ib) >= MIN_SEP and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recs = {(r["dataset"], r["stem"]): r for r in load_eval_records()}
    scorer = Scorer(args.model)
    print(f"{'stem':<9} {'L':>4} {'R':>4} {'gen/roll':>9} | {'rollout R-prec: mean':>20} {'med':>5} "
          f"{'best':>5} | {'vote(final)':>11} {'spearman(R,NLL)':>16}", flush=True)

    for ds, stem in EXAMPLES:
        rec = recs[(ds, stem)]
        L = rec["L"]
        resolved = frozenset(rec["resolved"])
        gt = {(i, j) for (i, j) in rec["contacts"] if i in resolved and j in resolved}
        R = len(gt)
        residues = residues_from_sequence(rec["input_seq"])
        prefixes, maps = [], []
        for r in range(args.n_rollouts):
            prefix, sp, _ = realization(stem, residues, f"r{r}")
            prefixes.append(scorer.tok(prefix, add_special_tokens=False).input_ids)
            maps.append({pos: i for i, pos in enumerate(sp)})
        max_new = min(8192 - len(prefixes[0]), 4 * L + 64)
        torch.manual_seed(args.seed)
        gens, nlls = generate_with_nll(scorer, prefixes, args.temperature, args.top_p,
                                       args.top_k, max_new, args.batch)

        rprec, ngen, counts = [], [], Counter()
        for g, m in zip(gens, maps):
            oc = ordered_contacts(scorer, g, m)
            ngen.append(len(oc))
            rprec.append(sum(1 for k in oc[:R] if k in gt) / R if R else float("nan"))
            for k in set(oc):
                counts[k] += 1
        rprec = np.array(rprec)
        nlls = np.array(nlls)
        ranked = sorted(candidate_pairs(L, resolved), key=lambda pr: -counts.get(pr, 0))
        vote_r = sum(1 for pr in ranked[:R] if pr in gt) / R if R else float("nan")
        rho = spearmanr(nlls, rprec).statistic

        # --- figure: distribution + R-vs-NLL ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
        ax1.hist(rprec, bins=24, range=(0, max(0.05, rprec.max() * 1.1)), color="#c6663b", alpha=0.85)
        for val, c, lab in [(vote_r, "#1f77b4", f"vote (final) = {vote_r:.3f}"),
                            (rprec.max(), "#2a9d63", f"best rollout = {rprec.max():.3f}"),
                            (rprec.mean(), "0.35", f"mean rollout = {rprec.mean():.3f}")]:
            ax1.axvline(val, color=c, lw=2, ls="--", label=lab)
        ax1.set_xlabel("per-rollout R-precision (first R generated, R=%d)" % R)
        ax1.set_ylabel("# rollouts")
        ax1.set_title(f"{stem}  (L={L}, R={R})  —  R-precision across {args.n_rollouts} rollouts")
        ax1.legend(fontsize=8)
        sc = ax2.scatter(nlls, rprec, c=ngen, cmap="viridis", s=18, alpha=0.8)
        ax2.axhline(vote_r, color="#1f77b4", lw=1.5, ls="--", label=f"vote (final) = {vote_r:.3f}")
        ax2.set_xlabel("rollout mean per-token NLL (lower = more likely)")
        ax2.set_ylabel("per-rollout R-precision")
        ax2.set_title(f"R-precision vs rollout NLL   (Spearman ρ = {rho:+.2f})")
        ax2.legend(fontsize=8)
        fig.colorbar(sc, ax=ax2, label="# contacts generated")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"rollout_dist_{stem}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        print(f"{stem:<9} {L:>4} {R:>4} {np.mean(ngen):>9.0f} | {rprec.mean():>20.3f} "
              f"{np.median(rprec):>5.3f} {rprec.max():>5.3f} | {vote_r:>11.3f} {rho:>16.2f}", flush=True)
        torch.cuda.empty_cache()
    print(f"\nfigures -> {args.out_dir}/rollout_dist_<stem>.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
