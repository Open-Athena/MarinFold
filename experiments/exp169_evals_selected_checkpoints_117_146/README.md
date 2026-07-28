---
marinfold_experiment:
  issue: 169
  title: 'exp: evaluate selected checkpoints from #117 and #146'
  kind: evals
  branch: claude/marinfold-issue-169-evals-be31d1
---

# exp: evaluate selected checkpoints from #117 and #146

**Issue:** [#169](https://github.com/Open-Athena/MarinFold/issues/169) · **Kind:** `evals` · **Branch:** `claude/marinfold-issue-169-evals-be31d1`

## Question

Issue #169 lists three checkpoints — the final and early-stop winners of
[#117](https://github.com/Open-Athena/MarinFold/issues/117) (1.5B) and the
[#146](https://github.com/Open-Athena/MarinFold/issues/146) 3B — all selected by
`eval/tokenized/contacts-v1-val/loss`. **What R-precision does each get on the
554-protein contact eval set?**

The three sit within **0.008 nats** of each other, so the real question underneath
is: *at that separation, is val loss still a useful selection signal for contact
accuracy, or has the loss → accuracy relationship gone flat?* The prior evidence
([exp82](../exp82_evals_contacts_v1_contact_prediction/)) is a steep slope —
2.7566 → 2.7037 (0.053 nats) bought **+0.11** R-precision — but that was measured
across a much wider loss gap, and a 3B model was not in the picture.

## Hypothesis

If the exp82 slope (~2.1 R-precision per nat) held locally, the 0.0076-nat gap
between the #117 final and early-stop checkpoints would be worth ~0.016
R-precision — small but resolvable with a paired test over 554 proteins. The
#146 3B, at 2.7025, should land between the two #117 points if loss alone
predicts accuracy; if extra capacity buys accuracy that the val loss does not
see, it should beat both.

## Background

- **The eval set** ([#74](https://github.com/Open-Athena/MarinFold/issues/74) /
  [#78](https://github.com/Open-Athena/MarinFold/issues/78)) — 554 `(dataset, stem)`
  units over 552 unique proteins (FoldBench-100 + exp65 low-MSA / novel-fold
  candidates). Ground truth is pyconfind side-chain contacts on the experimental
  structure, defined identically to the contacts-v1 training documents.
- **The measurement spec** —
  [exp89](../exp89_evals_contacts_v1_model_on_eval_set/)'s ground-truth universe,
  candidate-pair universe and `compute_metrics.py`. Nothing here re-derives them.
- **The inference recipe** —
  [exp82](../exp82_evals_contacts_v1_contact_prediction/)'s settled
  **rollout + per-rollout document resampling, n = 100, top-k off**. exp82 also
  owns the dispatcher, the worker and the metric code; this experiment supplies
  only the checkpoints, an S3 prefix and a job name, so the numbers land on
  exactly the scale as the published #75 / #117 rows.

## Approach

**1. Get all three checkpoints into one place, in one format.** #169 notes the
checkpoints are region-split (`europe-west4` / `us-east1`) and that the #117
early-stop checkpoint has **no HF export at all**. Rather than run compute in two
regions, everything is mirrored once into CoreWeave object storage next to the
H100s that do the scoring:

| Checkpoint | Source | Path here |
|---|---|---|
| #117 · 1.5B · step 35679 (final) | published `hf/` export, already staged by exp167 | reused as-is |
| #117 · 1.5B · step 33450 (early stop) | **levanter only** → exported here | `export_exp117_early_stop_to_hf.py` |
| #146 · 3B · step 17839 | published `hf/` export | mirrored in-cluster |

**2. Repair what levanter's transformers-5.12 exporter wrote.**
[`prepare_hf_export.py`](prepare_hf_export.py) downgrades `rope_parameters` →
`rope_theta`/`rope_scaling` and replaces the unresolvable
`tokenizer_class: TokenizersBackend`, then recasts fp32 → bf16. Both repairs are
applied *on disk before upload* on purpose, so the eval worker stays
byte-identical to the one that produced the published numbers.

**3. Verify before spending GPU time.**
[`verify_prepared_exports.py`](verify_prepared_exports.py) checks all three share
one vocabulary and one set of special-token ids, that llama3 rope survived the
config downgrade, and that a real contacts-v1 prompt through the real weights
puts its next-token mass on the format-legal tokens.

**4. Score.** [`run_eval_cw.sh`](run_eval_cw.sh) fans exp82's worker out over
3 × 12 single-H100 CoreWeave jobs at batch priority. **The #117 final checkpoint
is re-scored rather than reusing exp167's published matrices** — the headline
comparison is a 0.0076-nat difference between two #117 checkpoints, which deserves
to be measured in one submission, and reproducing the published 0.535 is itself
the harness check.

**5. Report paired.** [`summarize_results.py`](summarize_results.py) reports the
aggregate table *and* the per-protein paired differences. At a 0.008-nat
separation an unpaired mean ± SEM cannot resolve anything: the between-protein
spread of R-precision is ~0.3, so the SEM alone (~0.013) is wider than the effect.
The paired SE is an order of magnitude smaller because protein-to-protein variance
cancels.

### Running

```bash
cd experiments/exp169_evals_selected_checkpoints_117_146
uv venv && uv sync

# 1. checkpoints -> bf16, vLLM-loadable  (the levanter export needs marin's venv)
/home/bizon/git/marin/.venv/bin/python export_exp117_early_stop_to_hf.py \
    --checkpoint-dir <local levanter dir>/checkpoints --step 33450 \
    --output-dir ~/exp169_eval/hf_exp117_step33450_fp32
uv run python prepare_hf_export.py --src ~/exp169_eval/hf_exp117_step33450_fp32 \
    --dst ~/exp169_eval/hf_exp117_step33450_bf16

# 2. mirror weights into CoreWeave S3 (in-cluster where a published HF export exists —
#    42 s, vs 48 min over the ~2 MB/s workstation uplink)
set -a; source ~/.config/marin/cw-rno2a.env; set +a
/home/bizon/git/marin-freshiris/.venv/bin/python stage_from_hf_cw.py \
    --repo open-athena/marinfold-exp146 --path <run>/hf/step-17839 \
    --dst s3://marin-us-east-02a/MarinFold/exp169_eval/model_exp146_3b_step17839
uv run python stage_model_s3.py --src ~/exp169_eval/hf_exp117_step33450_bf16 \
    --dst s3://marin-us-east-02a/MarinFold/exp169_eval/model_exp117_step33450

# 3. gate, then score, then fetch + metric + plot
uv run python verify_prepared_exports.py --model ref=<dir> --model new=<dir>
./run_eval_cw.sh
./finalize.sh
```

## Success criteria

- All 554 `(dataset, stem)` units scored for all three checkpoints, no skips.
- The #117 final row reproduces exp82's published **0.535** R-precision (all) —
  the harness check that licenses comparing the new rows to the old ones.
- A paired verdict on #117 early-stop vs final, and on the 3B vs both, with
  confidence intervals that say whether the differences are resolvable at all.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
