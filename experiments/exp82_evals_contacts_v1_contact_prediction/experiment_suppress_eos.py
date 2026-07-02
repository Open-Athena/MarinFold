# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does forcing more contacts per rollout (suppressing <end>) help?

The model emits <end> on its own after ~1/3-2/3 of R contacts, capping single
rollouts. Here we suppress <end> several ways and measure contacts/rollout,
per-rollout R-precision, and — the bottom line — the **vote** R-precision:

  * baseline   : natural <end>.
  * ban-R      : ban <end> until >= R contacts (R = #GT; ORACLE upper bound).
  * ban-1.5L   : ban <end> until >= ceil(1.5*L) contacts (deployable, GT-free; ~R).
  * penalty-4  : subtract 4 from the <end> logit every step (soft push).

    PYTHONPATH=<repo>/marinfold uv run python experiment_suppress_eos.py \
        --model /home/bizon/exp89_export/hf_step35679 --out-dir _scratch/suppress_eos
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
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

from eval_contact_prediction import CONTACT_RE, MIN_SEP, NUM_POS, Scorer, candidate_pairs
from eval_full_curated_set import load_eval_records
from eval_rollout_resampled_dev import realization
from marinfold.document_structures.contacts_v1 import residues_from_sequence

EXAMPLES = [("foldbench100", "7y54_A"), ("foldbench100", "8axj_A"),
            ("foldbench100", "8bau_A"), ("foldbench100", "7qp5_A")]


class BanEosUntilN(LogitsProcessor):
    """Force <end> logit to -inf until >= N <contact> tokens have been generated."""

    def __init__(self, contact_id, eos_id, prefix_len, n):
        self.cid, self.eos, self.P, self.n = contact_id, eos_id, prefix_len, n

    def __call__(self, input_ids, scores):
        n_contacts = (input_ids[:, self.P:] == self.cid).sum(1)
        scores[n_contacts < self.n, self.eos] = float("-inf")
        return scores


class EosPenalty(LogitsProcessor):
    def __init__(self, eos_id, penalty):
        self.eos, self.pen = eos_id, penalty

    def __call__(self, input_ids, scores):
        scores[:, self.eos] = scores[:, self.eos] - self.pen
        return scores


@torch.no_grad()
def generate(scorer, prefixes, max_new, batch, proc_factory):
    """proc_factory(prefix_len) -> LogitsProcessorList | None. Returns gen-token lists (trimmed at <end>)."""
    P = len(prefixes[0])
    eos = scorer.end_id
    gens = []
    for s in range(0, len(prefixes), batch):
        X = torch.tensor(prefixes[s:s + batch], device=scorer.device)
        kw = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=50, max_new_tokens=max_new,
                  eos_token_id=eos, pad_token_id=eos)
        lp = proc_factory(P)
        if lp is not None:
            kw["logits_processor"] = lp
        out = scorer.model.generate(X, **kw)
        for row in out[:, P:].tolist():
            gens.append(row[:row.index(eos) + 1] if eos in row else row)
    return gens


def ordered_contacts(scorer, gen_ids, seqidx):
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
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recs = {(r["dataset"], r["stem"]): r for r in load_eval_records()}
    scorer = Scorer(args.model)
    cid, eos = scorer.contact_id, scorer.end_id
    rows = []
    print(f"{'stem':<8} {'L':>4} {'R':>4}  {'variant':<10} {'contacts/roll':>13} "
          f"{'per-roll R mean':>15} {'best':>6} {'VOTE R':>8}", flush=True)
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
        P = len(prefixes[0])
        max_new = min(8192 - P, 8 * L + 128)
        variants = {
            "baseline": lambda P: None,
            "ban-R": lambda P, n=R: LogitsProcessorList([BanEosUntilN(cid, eos, P, n)]),
            "ban-1.5L": lambda P, n=int(np.ceil(1.5 * L)): LogitsProcessorList([BanEosUntilN(cid, eos, P, n)]),
            "penalty-4": lambda P: LogitsProcessorList([EosPenalty(eos, 4.0)]),
        }
        for vname, fac in variants.items():
            torch.manual_seed(args.seed)
            gens = generate(scorer, prefixes, max_new, args.batch, fac)
            ngen, rprec, counts = [], [], Counter()
            for g, m in zip(gens, maps):
                oc = ordered_contacts(scorer, g, m)
                ngen.append(len(oc))
                rprec.append(sum(1 for k in oc[:R] if k in gt) / R if R else float("nan"))
                for k in set(oc):
                    counts[k] += 1
            ranked = sorted(candidate_pairs(L, resolved), key=lambda pr: -counts.get(pr, 0))
            vote_r = sum(1 for pr in ranked[:R] if pr in gt) / R if R else float("nan")
            rows.append(dict(stem=stem, L=L, R=R, variant=vname, ngen=float(np.mean(ngen)),
                             rmean=float(np.mean(rprec)), rbest=float(np.max(rprec)), vote_r=vote_r))
            print(f"{stem:<8} {L:>4} {R:>4}  {vname:<10} {np.mean(ngen):>13.1f} "
                  f"{np.mean(rprec):>15.3f} {np.max(rprec):>6.3f} {vote_r:>8.3f}", flush=True)
        torch.cuda.empty_cache()

    # bar plot: vote R-precision by variant, grouped by protein
    stems = [s for _, s in EXAMPLES]
    vnames = ["baseline", "ban-R", "ban-1.5L", "penalty-4"]
    colors = {"baseline": "#9aa0a6", "ban-R": "#7f2704", "ban-1.5L": "#d94801", "penalty-4": "#fd8d3c"}
    by = {(r["stem"], r["variant"]): r["vote_r"] for r in rows}
    x = np.arange(len(stems))
    w = 0.8 / len(vnames)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, v in enumerate(vnames):
        ax.bar(x + (i - (len(vnames) - 1) / 2) * w, [by.get((s, v), np.nan) for s in stems],
               w, label=v, color=colors[v])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(R={[r['R'] for r in rows if r['stem'] == s][0]})" for s in stems])
    ax.set_ylabel("vote R-precision (top-R by occurrence)")
    ax.set_title("Forcing more contacts/rollout by suppressing <end> — effect on the vote")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "suppress_eos_vote_rprecision.png", dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {args.out_dir}/suppress_eos_vote_rprecision.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
