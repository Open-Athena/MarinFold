---
marinfold_experiment:
  issue: 238
  title: 'exp: promote the #199 CoreWeave cooldown to the default contacts-v1 model'
  kind: models
  branch: claude/marinfold-default-model-e035dc
---

# exp: promote the #199 CoreWeave cooldown to the default contacts-v1 model

**Issue:** [#238](https://github.com/Open-Athena/MarinFold/issues/238) · **Kind:** `models` · **Branch:** `claude/marinfold-default-model-e035dc`

## Question

[#234](https://github.com/Open-Athena/MarinFold/pull/234) landed a new best contacts-v1 model — #199's CoreWeave p06 **cooldown**, `prot-exp199-cw-cv1-p06-cool-s01` at step 290,400. It is not published anywhere durable and nothing in the repo points at it. What has to happen for it to *be* MarinFold's default model?

## Hypothesis

Not a hypothesis-testing experiment — a promotion. The one thing that could
fail is the publication: a checkpoint copied without its rope repair, or with a
tokenizer that has drifted, works well enough to look fine and is wrong.

## Background

- [#199](https://github.com/Open-Athena/MarinFold/issues/199) / [#234](https://github.com/Open-Athena/MarinFold/pull/234) — the run and its evaluation. R-precision **0.6307** all / **0.5837** long on the legacy 554, against the current default's (`p06-aug`, step 145,199) 0.6088 / 0.5633 scored by the same CoreWeave worker in the same batch. Validation loss 2.9397 vs 2.9712 (both current scale).
- The cooldown export exists **only in CoreWeave S3**, at
  `s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_continue_contacts_v1_cw/checkpoints/protein/prot-exp199-cw-cv1-p06-cool-s01/2026.08.14.1/hf/step-290400`.
  Nothing outside the cluster can read it, so it cannot be a registry entry as it stands.
- [#180](https://github.com/Open-Athena/MarinFold/issues/180) — the standing frontier tracker; a new frontier point is a refresh, and its head-to-head-vs-Protenix figure pins whichever checkpoint is current.
- [#197](https://github.com/Open-Athena/MarinFold/issues/197) / [#184](https://github.com/Open-Athena/MarinFold/pull/184) — the transformers-5 rope defect. levanter-side exports state rope only as `rope_parameters`; transformers 4.x ignores that key and silently loads the architecture-default `rope_theta`, worth **0.44–0.77 nats/token** depending on the checkpoint. Every checkpoint we publish gets a repaired `config.json` and a `PROVENANCE.md`, and the registry points at *our* copy rather than a third-party export.
- [#226](https://github.com/Open-Athena/MarinFold/issues/226) — eval2 is now the default eval set. #234's evaluation already scored the full 577-unit universe, so the cooldown's eval2 cuts exist and need reporting, not recomputing.

## Approach

1. **Publish.** Copy the six-file export from CoreWeave S3 into the public `open-athena/MarinFold` bucket at `checkpoints/prot-exp199-cw-cv1-p06-cool-s01/hf/step-290400`, cloud-side on a `cw-us-east-02a` iris job (the workstation has no credentials for that bucket, and a ~5.9 GiB round trip over a 2.5 MB/s uplink is not the way). Verify every object against the S3 ETags #234 recorded, repair the rope block, keep the tokenizer co-located.
2. **Verify.** Re-download the published copy from the public bucket, confirm `scripts/repair_checkpoint_config.py --survey` reads it clean, measure the repaired-vs-as-published NLL delta on the three benchmark documents so `PROVENANCE.md` carries a per-checkpoint number, and run `marinfold evaluate` end-to-end against the registry entry.
3. **Promote.** New default entry in `MODELS.yaml`; `README.md`, `UPDATES.md` and the notebooks that name the default nickname follow.
4. **Refresh #180.** Add the cooldown to `RPRECISION_ROWS`, re-point `plot_vs_protenix.py` at the new frontier model, redraw.
5. **Report eval2.** Both cuts, leading with eval2-natural (n=78), against the checkpoint it displaces.

## Success criteria

- `marinfold infer` with no `--model` resolves the cooldown, downloads it from the public bucket with no authentication, and produces contacts.
- The published `config.json` reads `rope_theta = 500000` under transformers 4.x.
- #180's figures show the new frontier point and the running-best staircase steps to it.
- The eval2 numbers for the new default are written down somewhere a reader will find them.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
