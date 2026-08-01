# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the contacts-v1 progress dataset: one row per trained checkpoint.

Two tables come out of here:

* ``data/val_loss_runs.csv`` — every finished contacts-v1 training run we can
  find a held-out `contacts-v1-val` loss for, with the date the run finished.
  Pulled live from W&B (`open-athena/MarinFold` + `eric-czech/marin` sweeps
  exp75/exp117/exp146/exp153).
* ``data/rprecision_checkpoints.csv`` — the (much smaller) set of checkpoints
  that have actually been scored on the fixed 554-protein contact benchmark.
  Hand-curated with a citation per row, because these numbers live in issue
  comments and per-experiment CSVs, not in W&B.

R-precision here is always **R-precision, all ranges (sep ≥ 6), mean over the
554-protein eval set**, computed by exp89's ``compute_metrics``. Two inference
recipes appear and they are *not* interchangeable — see INFERENCE below.

    uv run python build_dataset.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Inference recipes. The same checkpoint scores ~0.086 higher under `rollout`
# than under `pairwise` (measured on the two checkpoints run through both:
# #75 E8 0.3389 -> 0.4245, #120 0.3495 -> 0.4357), so a plot that mixes them
# has to say which is which.
# ---------------------------------------------------------------------------
PAIRWISE = "pairwise"           # exp89 original: autoregressive P(contact), symmetrised
ROLLOUT = "rollout"             # exp82 settled recipe: n=100 rollouts + document
                                # resampling + pairwise tie-break, top-k OFF (#142)
ORACLE_BEST100 = "oracle_best_of_100"  # NOT a deployable recipe: for each protein, the
                                # single BEST of the same 100 rollouts (first-R precision
                                # of that one rollout's own emission order), instead of the
                                # votes-aggregated ranking. An upper-bound/headroom diagnostic
                                # -- you cannot know which rollout is best without GT -- kept
                                # as a separate series from PAIRWISE/ROLLOUT, never blended in.

# ---------------------------------------------------------------------------
# R-precision (all ranges), 554-protein eval set, exp89 compute_metrics.
# (label, date the checkpoint finished training, params, val loss or None,
#  R-precision, inference recipe, source)
# ---------------------------------------------------------------------------
RPRECISION_ROWS = [
    dict(
        label="#67 quick 1.5B",
        model="protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked / step-11999",
        date="2026-06-14", params="1.5B", issue=67,
        val_loss=2.979953, val_loss_key="eval/contacts-v1-val/loss",
        r_precision=0.0294637, inference=PAIRWISE,
        source="exp89 data/marinfold_precision.csv (model=marinfold-cv1-exp67)",
    ),
    dict(
        label="#75 E1",
        model="prot-exp75-cv1-1_5b-e1-lr7e-4-wd0p05-v1-69de44",
        date="2026-06-14", params="1.5B", issue=75,
        val_loss=3.0457733, val_loss_key="eval/contacts-v1-val/loss",
        r_precision=0.0280270, inference=PAIRWISE,
        source="exp89 data/marinfold_precision.csv (model=marinfold-cv1-e1); PR #93",
    ),
    dict(
        label="#75 E2",
        model="prot-exp75-cv1-1_5b-e2-lr7e-4-wd0p8-v1-d4dfdb",
        date="2026-06-20", params="1.5B", issue=75,
        val_loss=2.9420996, val_loss_key="eval/contacts-v1-val/loss",
        r_precision=0.0289797, inference=PAIRWISE,
        source="exp89 data/marinfold_precision.csv (model=marinfold-cv1-e2); PR #93",
    ),
    dict(
        label="#75 E4",
        model="prot-exp75-cv1-1_5b-e4-lr1e-3-wd0p05-v1-dcd938",
        date="2026-06-21", params="1.5B", issue=75,
        val_loss=2.9238107, val_loss_key="eval/contacts-v1-val/loss",
        r_precision=0.0306772, inference=PAIRWISE,
        source="exp89 data/marinfold_precision.csv (model=marinfold-cv1-e4); PR #93",
    ),
    dict(
        label="#61/#75 E8",
        model="prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084 / step-35679",
        date="2026-06-21", params="1.5B", issue=75,
        val_loss=2.7566020, val_loss_key="eval/contacts-v1-val/loss",
        r_precision=0.3389373, inference=PAIRWISE,
        source="exp89 data/marinfold_precision.csv (model=marinfold-contacts-v1)",
    ),
    dict(
        label="#61/#75 E8",
        model="prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084 / step-35679",
        date="2026-06-21", params="1.5B", issue=75,
        val_loss=2.7566020, val_loss_key="eval/contacts-v1-val/loss",
        r_precision=0.4245291, inference=ROLLOUT,
        source="exp82 data/where_we_stand_summary.csv (marinfold-cv1-exp75-rollout)",
    ),
    dict(
        label="#120 re-epoch",
        model="exp120-cv1-1_5b-orig-lr3e-4-e1-cos / step-1005",
        date="2026-07-16", params="1.5B", issue=120,
        val_loss=2.7213, val_loss_key="eval/contacts-v1-val-orig/loss",
        r_precision=0.3495, inference=PAIRWISE,
        source="exp120 README headline table (all R-prec, lr3e-4 cosine 1ep)",
    ),
    dict(
        label="#120 re-epoch",
        model="exp120-cv1-1_5b-orig-lr3e-4-e1-cos / step-1005",
        date="2026-07-16", params="1.5B", issue=120,
        val_loss=2.7213, val_loss_key="eval/contacts-v1-val-orig/loss",
        r_precision=0.4357296, inference=ROLLOUT,
        source="exp160 data/exp160_summary.csv (model=exp120-base)",
    ),
    dict(
        label="#117 E8 bs64",
        model="prot-exp117-cv1-s02-1_5b-e8-lr3p162e-3-wd0p2-bs64-europe-west4 / step-71359",
        date="2026-07-19", params="1.5B", issue=117,
        val_loss=2.7130589, val_loss_key="eval/tokenized/contacts-v1-val/loss",
        r_precision=0.4192084, inference=PAIRWISE,
        source="eric-czech gist bfa78571 marinfold-checkpoint-metrics.csv (exp117-e8-s02)",
    ),
    dict(
        label="#117 E16 early stop",
        model="prot-exp117-...-e16-lr3p162e-3-wd0p2-bs256-europe-west4 / step-33450",
        date="2026-07-22", params="1.5B", issue=117,
        val_loss=2.696074, val_loss_key="eval/tokenized/contacts-v1-val/loss",
        r_precision=0.5318058, inference=ROLLOUT,
        source="exp169 data/exp169_summary.csv (exp117_e16_early_step33450)",
    ),
    dict(
        label="#117 E16 final",
        model="prot-exp117-...-e16-lr3p162e-3-wd0p2-bs256-europe-west4 / step-35679",
        date="2026-07-22", params="1.5B", issue=117,
        val_loss=2.7037086, val_loss_key="eval/tokenized/contacts-v1-val/loss",
        r_precision=0.5344182, inference=ROLLOUT,
        source="exp169 data/exp169_summary.csv (exp117_e16_final_step35679); "
               "exp82 where_we_stand_summary.csv has 0.53497 for the same ckpt",
    ),
    dict(
        label="#146 3B E8",
        model="prot-exp146-cv1-s01-3b-e8-lr3p162e-3-wd0p4-bs256-us-east1 / step-17839",
        date="2026-07-27", params="3B", issue=146,
        val_loss=2.7024784, val_loss_key="eval/tokenized/contacts-v1-val/loss",
        r_precision=0.5118632, inference=ROLLOUT,
        source="exp169 data/exp169_summary.csv (exp146_3b_e8_step17839)",
    ),
    dict(
        label="#160 backtracking",
        model="exp160-cv1-1_5b-bt50-lr3e-4-e1-cos-v3 / step-2058",
        date="2026-07-28", params="1.5B", issue=160,
        val_loss=None, val_loss_key="",   # superset vocab (3849) -> not comparable
        r_precision=0.4158492, inference=ROLLOUT,
        source="exp160 data/exp160_summary.csv (model=exp160-bt50, retraction enabled)",
    ),
    dict(
        label="#155 3-way restart 60k",
        model="exp137-3way-restart30k-lr2p5e-3-9e7568 / step-60000",
        # Checkpoint date, not run-finish date: this run is still training
        # (target step 74800) -- see the README caveat on in-flight checkpoints.
        date="2026-07-31", params="1.5B", issue=155,
        val_loss=None, val_loss_key="",   # superset crops tokenizer (3848) -> not comparable
        r_precision=0.5531834, inference=ROLLOUT,
        source="this session's eval: score_rollout_worker.py x8 v5p-8 TPU shards (us-east5) "
               "+ fetch_cw_scores.py + build_rollout_rows.py, 554/554 proteins; "
               "data/exp155_3way_restart_step60000_rollout_summary.csv",
    ),
    dict(
        label="#155 3-way restart 70k",
        model="exp137-3way-restart30k-lr2p5e-3-9e7568 / step-70000",
        date="2026-08-01", params="1.5B", issue=155,
        val_loss=None, val_loss_key="",   # superset crops tokenizer (3848) -> not comparable
        r_precision=0.5558576, inference=ROLLOUT,
        source="this session's eval: score_rollout_worker.py x8 v5p-8 TPU shards (us-east5) "
               "+ fetch_cw_scores.py + build_rollout_rows.py, 554/554 proteins; "
               "data/exp155_3way_restart_step70000_rollout_summary.csv",
    ),
    dict(
        # The run finished at step 74793 (target 74800) -- unlike the 60k/70k
        # rows above, this is a settled result, not an in-flight snapshot.
        label="#155 3-way restart final",
        model="exp137-3way-restart30k-lr2p5e-3-9e7568 / step-74793",
        date="2026-08-01", params="1.5B", issue=155,
        val_loss=None, val_loss_key="",   # superset crops tokenizer (3848) -> not comparable
        r_precision=0.5545, inference=ROLLOUT,
        source="this session's eval: score_rollout_worker_oracle.py x8 v5p-8 TPU shards "
               "(us-east5) + fetch_cw_scores.py + build_rollout_rows.py, 554/554 proteins; "
               "data/exp155_3way_restart_step74793_rollout_summary.csv",
    ),
    dict(
        # Oracle best-of-100: same checkpoint, same 100 rollouts per protein,
        # but scored by each rollout's OWN first-R precision (emission order)
        # and taking the max over the 100 -- not a deployable recipe, an
        # upper-bound/headroom diagnostic. See build_oracle_best_rollout.py.
        label="#155 3-way restart final",
        model="exp137-3way-restart30k-lr2p5e-3-9e7568 / step-74793",
        date="2026-08-01", params="1.5B", issue=155,
        val_loss=None, val_loss_key="",
        r_precision=0.594827, inference=ORACLE_BEST100,
        source="this session's eval: score_rollout_worker_oracle.py (per-rollout detail) "
               "+ build_oracle_best_rollout.py, 554/554 proteins, n=100 rollouts/protein; "
               "data/exp155_3way_restart_step74793_oracle_best100_summary.csv",
    ),
]

# ---------------------------------------------------------------------------
# Structure-predictor reference lines, same metric / same 554 proteins.
# Recomputed from exp89's per-protein table rather than transcribed, so the
# baselines and the LM points come out of one code path.
# ---------------------------------------------------------------------------
EXP89_ROWS = (HERE / ".." / "exp89_evals_contacts_v1_model_on_eval_set"
              / "data" / "contact_precision_all.csv").resolve()
BASELINES = [
    # (label, model, mode, predictor)
    ("Protenix-v2 single-seq", "protenix-v2", "single_seq", "structure"),
    ("ESMFold", "esmfold", "single_seq", "structure"),
    ("ESMFold2", "esmfold2", "single_seq", "structure"),
    ("Protenix-v2 + MSA", "protenix-v2", "msa", "structure"),
]


def build_baselines() -> list[dict]:
    import pandas as pd

    df = pd.read_csv(EXP89_ROWS)
    df = df[(df["cut"] == "R") & (df["range"] == "all")]
    out = []
    for label, model, mode, predictor in BASELINES:
        sel = df[(df["model"] == model) & (df["mode"] == mode)
                 & (df["predictor"] == predictor)]["precision"].dropna()
        out.append(dict(label=label, model=model, mode=mode, predictor=predictor,
                        r_precision=float(sel.mean()), n=int(sel.size),
                        source="exp89 data/contact_precision_all.csv"))
    return out


# Extra R-precision realisations that are NOT separate checkpoints — recorded
# so the numbers in the repo can be traced, but excluded from the plots.
FOOTNOTE_ROWS = [
    ("#61/#75 E8, K=10 document-resampling ensemble of pairwise", 0.3755950,
     "exp89 data/marinfold_precision.csv (marinfold-cv1-ens10)"),
    ("#61/#75 E8, rollout with the old top_k=50 (pre-#142)", 0.4130525,
     "exp82 data/where_we_stand_summary.csv (marinfold-cv1-exp75-rollout-topk50)"),
    ("#117 E16 final, rollout with the old top_k=50 (pre-#142)", 0.5279264,
     "exp82 data/where_we_stand_summary.csv (marinfold-cv1-exp117-rollout-topk50)"),
]

# W&B sources for the val-loss cloud.
WANDB_SOURCES = [
    ("open-athena/MarinFold", None),
    ("eric-czech/marin", "exp75"),
    ("eric-czech/marin", "exp117"),
    ("eric-czech/marin", "exp146"),
    ("eric-czech/marin", "exp153"),
]
LOSS_KEYS = [
    "eval/tokenized/contacts-v1-val/loss",
    "eval/contacts-v1-val/loss",
    "eval/contacts-v1-val-orig/loss",
]
# Runs that report a contacts-v1 val loss but are not contacts-v1 *models*
# (smoke tests, throughput probes, and mixture runs whose loss is logged
# through a superset tokenizer). Kept out of the frontier; see README.md.
EXCLUDE_SUBSTRINGS = ("smoke", "probe", "profile", "-prof", "vet-", "nemo")


def fetch_val_loss_runs() -> list[dict]:
    import wandb

    api = wandb.Api(timeout=180)
    rows = []
    for project, tag in WANDB_SOURCES:
        runs = (api.runs(project, filters={"tags": tag}, per_page=200) if tag
                else api.runs(project, per_page=200))
        for r in runs:
            s = r.summary._json_dict
            loss = key = None
            for k in LOSS_KEYS:
                if k in s and isinstance(s[k], (int, float)):
                    loss, key = float(s[k]), k
                    break
            if loss is None:
                continue
            name = r.name or ""
            rows.append(dict(
                project=project, tag=tag or "", name=name, run_id=r.id,
                state=r.state,
                started=str(r.created_at)[:19].replace("T", " "),
                finished=str(getattr(r, "heartbeatAt", ""))[:19].replace("T", " "),
                step=s.get("_step"), val_loss=loss, val_loss_key=key,
                excluded=any(x in name.lower() for x in EXCLUDE_SUBSTRINGS),
            ))
    rows.sort(key=lambda r: r["finished"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wandb", action="store_true",
                    help="reuse the committed val_loss_runs.csv")
    a = ap.parse_args()

    out = HERE / "data"
    out.mkdir(parents=True, exist_ok=True)

    with (out / "rprecision_checkpoints.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(RPRECISION_ROWS[0]))
        w.writeheader()
        w.writerows(RPRECISION_ROWS)
    print(f"wrote {out / 'rprecision_checkpoints.csv'} ({len(RPRECISION_ROWS)} rows)")

    baselines = build_baselines()
    with (out / "structure_baselines.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(baselines[0]))
        w.writeheader()
        w.writerows(baselines)
    print(f"wrote {out / 'structure_baselines.csv'} ({len(baselines)} rows)")

    with (out / "rprecision_footnotes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["description", "r_precision_all", "source"])
        w.writerows(FOOTNOTE_ROWS)

    if not a.skip_wandb:
        runs = fetch_val_loss_runs()
        with (out / "val_loss_runs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(runs[0]))
            w.writeheader()
            w.writerows(runs)
        print(f"wrote {out / 'val_loss_runs.csv'} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
