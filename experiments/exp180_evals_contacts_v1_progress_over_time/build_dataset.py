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

Validation loss has the same problem one axis over: marin changed the objective
partway through this tracker's history, so a raw loss is only meaningful once
you know which implementation produced it — see LOSS SCALE below.

    uv run python build_dataset.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# LOSS SCALE. marin #7209 (merged 2026-07-16) changed the default packed-LM
# objective to MASK positions whose next token is padding. Padding targets are
# nearly free to predict, so including them pulls the mean *down*: the same
# checkpoint reads ~0.38 nats LOWER under the old implementation than under the
# current one. Everything in this tracker up to and including #166 was recorded
# under the old one; exp199 is the first sweep recorded under the new one.
#
# Runs do not carry the objective in their W&B config, so the scale is declared
# per source in WANDB_SOURCES / per row in RPRECISION_ROWS. Get it wrong and a
# current-scale run looks 0.38 nats worse than a model it in fact beats.
#
# The conversion is empirical: Eric re-evaluated four exp166 checkpoints under
# the current implementation and fitted both an offset and a line.
#   https://gist.github.com/eric-czech/9c40252457790a513eeb62a6a965c049
# The offset is the more stable of the two over the observed range and is what
# exp199 and PR #204 quote, so it is what this tracker uses.
# ---------------------------------------------------------------------------
HISTORICAL = "historical"   # padding-target positions INCLUDED in the mean
CURRENT = "current"         # padding-target positions MASKED (marin >= #7209)
SCALE_OFFSET = 0.38171      # current ≈ historical + 0.38171
# Spread across the four measured checkpoints (+0.37934 … +0.38511); the
# alternative line fit `current = 0.86358*historical + 0.75716` disagrees by
# ~0.025 nats once you extrapolate a third of a nat past the four points it was
# fitted on. Both numbers are carried so figure footnotes can quote them.
SCALE_OFFSET_RANGE = (0.37934, 0.38511)
SCALE_LINE = (0.86358, 0.75716)   # current = a*historical + b


def to_historical(loss: float | None, scale: str) -> float | None:
    """Put a loss on the tracker's plotting axis (the historical scale).

    The historical scale is the axis rather than the current one purely because
    that is where the overwhelming majority of the runs natively are — the
    approximate conversion is then applied to the few, not the many.
    """
    if loss is None or loss == "":
        return None
    return float(loss) if scale == HISTORICAL else float(loss) - SCALE_OFFSET


def historical_via_line(loss: float) -> float:
    """The same conversion under the gist's fitted line instead of its offset.

    Only used to quote the disagreement between the two; never plotted.
    """
    a, b = SCALE_LINE
    return (float(loss) - b) / a

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
        # The run finished at step 74793 (target 74800), a settled result.
        # Earlier in-flight checkpoints from this run (step 60000: R 0.5532,
        # step 70000: R 0.5559) were scored during training but are not
        # plotted -- only the final checkpoint represents this run here.
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
    dict(
        # Continues from #117 rather than training from scratch, so it inherits
        # that run's tokenizer and its val loss IS comparable -- unlike #155,
        # which shares the accuracy axis but not the loss axis. #190 scored it
        # and re-scored its own #117 init in the same run (0.5336 against
        # #169's 0.5344), so the +0.0282 delta is a within-run paired result,
        # not a cross-harness subtraction.
        label="#166 AA aug",
        model="prot-exp166-cv1-aaaug-1_5b-e8-lr3p162e-3-wd0p1-bs128-exp117-init-us-east1 "
              "/ step-35679",
        date="2026-07-31", params="1.5B", issue=166,
        val_loss=2.6641791, val_loss_key="eval/tokenized/contacts-v1-val/loss",
        r_precision=0.5617739, inference=ROLLOUT,
        source="exp166 data/exp166_summary.csv (exp166_aaaug_step35679); PR #190",
    ),
    # ---- #199: AFDB + ESM-Atlas mixture sweep (PR #205), scored in #204 -----
    # Four final checkpoints, all n=100 rollout under the same harness as #190
    # and validated against it: the analyzer first rescored #190's archived
    # votes for the #117 control and recovered 0.5335961 exactly, then three
    # fresh control evaluations landed at 0.5348 / 0.5352 / 0.5329. So these
    # numbers sit on the same axis as #166's and #117's without a cross-harness
    # subtraction -- see FOOTNOTE_ROWS for the control replicates themselves.
    #
    # Their losses are the FIRST on the CURRENT scale (marin >= #7209); the
    # conversion above is what puts them on this tracker's axis.
    dict(
        label="#199 TRC p06-aug",
        model="prot-exp199-cv1-s01-m1-p06-aug-us-east1 / step-72599",
        date="2026-08-08", params="1.5B", issue=199,
        val_loss=3.054504156112671, val_loss_scale=CURRENT,
        val_loss_key="eval/tokenized/contacts-v1-val/loss",
        r_precision=0.5244069975064393, inference=ROLLOUT,
        source="exp199 data/contact_eval_final_checkpoint_summary.csv "
               "(rerun02-20260809); issue #204",
    ),
    dict(
        label="#199 TRC p03-aug",
        model="prot-exp199-cv1-s01-m1-p03-aug-us-east1 / step-72599",
        date="2026-08-09", params="1.5B", issue=199,
        val_loss=3.011530637741089, val_loss_scale=CURRENT,
        val_loss_key="eval/tokenized/contacts-v1-val/loss",
        r_precision=0.5743326909766765, inference=ROLLOUT,
        source="exp199 data/contact_eval_final_checkpoint_summary.csv "
               "(rerun02-20260809); issue #204",
    ),
    dict(
        label="#199 TRC p03-base",
        model="prot-exp199-cv1-s01-m1-p03-base-us-east5 / step-72599",
        date="2026-08-09", params="1.5B", issue=199,
        val_loss=3.00742244720459, val_loss_scale=CURRENT,
        val_loss_key="eval/tokenized/contacts-v1-val/loss",
        r_precision=0.5779648259578161, inference=ROLLOUT,
        source="exp199 data/contact_eval_final_checkpoint_summary.csv "
               "(finals03-20260810); issue #204",
    ),
    dict(
        # Not a continue-train: trained from scratch on CoreWeave H100s with a
        # WSD schedule for 2x the TRC step count (145,199 vs 72,599), so its
        # gap to the TRC p06-aug row is training history, not hardware.
        label="#199 CW p06-aug",
        model="prot-exp199-cw-cv1-s02-m1-p06-aug / step-145199",
        date="2026-08-10", params="1.5B", issue=199,
        val_loss=2.971200942993164, val_loss_scale=CURRENT,
        val_loss_key="eval/tokenized/contacts-v1-val/loss",
        r_precision=0.5873483777949621, inference=ROLLOUT,
        source="exp199 data/contact_eval_final_checkpoint_summary.csv "
               "(finals03-20260810); issue #204",
    ),
]

# Column order for rprecision_checkpoints.csv. Fixed here rather than read off
# the first row, because the scale columns are filled in by normalise_rows().
RPRECISION_FIELDS = [
    "label", "model", "date", "params", "issue",
    "val_loss", "val_loss_raw", "val_loss_scale", "val_loss_key",
    "r_precision", "inference", "source",
]


def normalise_rows(rows: list[dict]) -> list[dict]:
    """Fill in the loss-scale columns and put every loss on the plotting axis.

    ``val_loss`` is always the historical-scale value the figures use;
    ``val_loss_raw`` preserves what the source actually reported, so a reader
    who wants the current scale can recover it without re-deriving anything.
    """
    out = []
    for r in rows:
        r = dict(r)
        scale = r.get("val_loss_scale", HISTORICAL)
        raw = r.get("val_loss")
        r["val_loss_scale"] = scale if raw is not None else ""
        r["val_loss_raw"] = raw
        r["val_loss"] = to_historical(raw, scale)
        out.append({k: r.get(k, "") for k in RPRECISION_FIELDS})
    return out

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
    # #204's three fresh evaluations of the #117 E16 final checkpoint. Same
    # weights, same recipe, three independent generation runs -- so together
    # with #190's 0.5335961 they are the first direct estimate of how much of a
    # gap between two checkpoints is just sampling noise: the four span 0.0023.
    # That is the yardstick for the exp199 rows above (p03-base beats p03-aug
    # by 0.0036, i.e. barely more than this span).
    ("#117 E16 final, fresh rollout replicate r1 (#204)", 0.5347972614575084,
     "exp199 data/contact_eval_pr_comparison_summary.csv (control-r1)"),
    ("#117 E16 final, fresh rollout replicate r2 (#204)", 0.535215598085612,
     "exp199 data/contact_eval_pr_comparison_summary.csv (control-r2, rerun02-20260809)"),
    ("#117 E16 final, fresh rollout replicate r3 (#204)", 0.5328883690891095,
     "exp199 data/contact_eval_pr_comparison_summary.csv (control-r3, finals03-20260810)"),
]

# W&B sources for the val-loss cloud, each with the loss scale its runs were
# recorded under. A new sweep added here without the right scale will be off by
# 0.38 nats — the single easiest way to corrupt the loss frontier.
#
# The scale is a property of the pinned marin version, NOT of the run date, and
# it is not in the run config. Read it off the run's own `requirements.txt`
# artifact in W&B (`api.runs(...)[0].file("requirements.txt")`) and compare the
# marin-core version against #7209. As checked 2026-08-10:
#   open-athena/MarinFold  marin-core 0.2.19.dev202606171019  (Jun 17) -> historical
#     ...including runs *launched* in August: the pin, not the calendar, decides.
#   eric-czech/marin exp166 marin-core 0.2.0                          -> historical
#   eric-czech/marin exp199 marin-core 0.2.73.dev30987879744          -> current
WANDB_SOURCES = [
    ("open-athena/MarinFold", None, HISTORICAL),
    ("eric-czech/marin", "exp75", HISTORICAL),
    ("eric-czech/marin", "exp117", HISTORICAL),
    ("eric-czech/marin", "exp146", HISTORICAL),
    ("eric-czech/marin", "exp153", HISTORICAL),
    ("eric-czech/marin", "exp166", HISTORICAL),
    ("eric-czech/marin", "exp199", CURRENT),
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
    for project, tag, scale in WANDB_SOURCES:
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
                step=s.get("_step"),
                # val_loss is always on the historical plotting axis; val_loss_raw
                # is what the run itself logged. See LOSS SCALE at the top.
                val_loss=to_historical(loss, scale), val_loss_raw=loss,
                val_loss_scale=scale, val_loss_key=key,
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

    rows = normalise_rows(RPRECISION_ROWS)
    with (out / "rprecision_checkpoints.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RPRECISION_FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_conv = sum(1 for r in rows if r["val_loss_scale"] == CURRENT)
    print(f"wrote {out / 'rprecision_checkpoints.csv'} ({len(rows)} rows, "
          f"{n_conv} loss values converted from the current scale)")

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
        n_conv = sum(1 for r in runs if r["val_loss_scale"] == CURRENT)
        print(f"wrote {out / 'val_loss_runs.csv'} ({len(runs)} runs, "
              f"{n_conv} on the current scale and converted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
