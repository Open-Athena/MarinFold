# MarinFold Updates

## Week of August 31, 2026

### Last week

* **Training: Eric's sweep on decontaminated data is stopped** ([#232](https://github.com/Open-Athena/MarinFold/issues/232), [#233](https://github.com/Open-Athena/MarinFold/pull/233); all three cooldowns were abandoned, [reasons here](https://github.com/Open-Athena/MarinFold/blob/main/experiments/exp232_sweep_cv1_decontam/gpu/exp232_cooldown_cw.md#change-record)). Best numbers ([eval snapshot](https://github.com/Open-Athena/MarinFold/blob/main/experiments/exp232_sweep_cv1_decontam/evals/2026-08-24_rollout_v2/README.md), [W&B](https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1)): **R-precision 0.605** on the legacy 554 (0.5517 eval-val), val **2.9737**, against the contaminated [#199](https://github.com/Open-Athena/MarinFold/issues/199) cooldown's **0.631 / 2.9397**.
* **Training:  Soft-target training implementation still being worked out** ([#177](https://github.com/Open-Athena/MarinFold/issues/177)). No good models have been trained yet using this idea. Zack working on figuring out if there are bugs here.
* **Training: Zack is experimenting with reshulffing full documents during training.** Rebuild each document every epoch with a fresh permutation of structure-statement order and endpoint orientation (4 per row) and train stock next-token CE ([branch README](https://github.com/Open-Athena/MarinFold/blob/2e778b060eeaed713826faa1768c21970b957095/experiments/exp177_models_compare_soft_target_contact_loss_against/README.md#contact-order-augmented-ce-arm), not yet on main). Complementary to Eric's `-aug`, which permutes the *sequence* statements. Full 1.5B run on GB200 at **727k tok/s**, val 3.443 at step 15,236 of 71,360 ([r95](https://wandb.ai/open-athena/MarinFold/runs/exp177-cv1-1_5b-e16-lr3p162e-3-wd0p2-bs128-next_token-contact-order-aug-r95-gb200-8x4-full)).
* **Training: Zack also started [#157](https://github.com/Open-Athena/MarinFold/issues/157)** (positional embedding instead of learned residue-position tokens). At matched step 31,220: **3.185** ([fixed-position](https://wandb.ai/open-athena/MarinFold/runs/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-fixed-position-controlmatch-e16-r3-east02-h100x8)) vs **3.205** ([rope_delta](https://wandb.ai/open-athena/MarinFold/runs/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r2-east08-gb200x4n8)). Let's chat about this at the meeting - I'd like to understand it better.
* **Evals: we have 16 proteins with MSA depth < 10** ([#260](https://github.com/Open-Athena/MarinFold/issues/260)). Only 5 of these are from FoldBench (and therefore not included in standard predictor training sets), but the MarinFold training set has been decontaminated against all of them. These should be a focus of our evals going forward. Tim is experimenting with a dashboard for understanding our performance on these, current draft [here](https://openathena.ai/MarinFold/experiments/exp260_evals_msa_depth_stratified/dashboard/index.html).  

### Upcoming
* Zack: soft-target training, full document reshuffling during training, positional encoding changes
* Tim: working on a short report on our results so far
* Jacob: AFDB complex curation ([#145](https://github.com/Open-Athena/MarinFold/issues/145)), decontaminated as in [#225](https://github.com/Open-Athena/MarinFold/issues/225).

---

## Week of August 24, 2026

### Last week

* Pre-training updates
    * **New "decontaminated" eval and training sets are in use**. Last week, Tim defined new datasets to avoid homology / overlap between training and eval data ([#225](https://github.com/Open-Athena/MarinFold/issues/225)). Eric’s decontaminated-data sweep has narrowed to two useful surviving runs, and both are now being used as starting points for cooldown / follow-up training. The lead decontaminated checkpoint is improving with more training — from ~0.592 to ~0.605 R-precision on the legacy eval — but has not yet caught the current #199 cooldown default at ~0.631. ([#232](https://github.com/Open-Athena/MarinFold/issues/232)).
    * **FoldBench was not "flattering" us, but homology still matters**. Tim split the FoldBench evals to eliminate "contamination", and found that MarinFold accuracy did not drop ([#245](https://github.com/Open-Athena/MarinFold/issues/245)). However, proteins with more homology display higher MarinFold (and other model) performance: ([#247](https://github.com/Open-Athena/MarinFold/issues/247))
    * **Alternate pre-training loss function experiment: soft-target/order-marginalized loss is still training CoreWeave**. Zack updated the sparse soft-target loss path, so it is now JAX/GPU-friendly enough to train on GB200 at about **8.6–8.9s/it (~120k tok/s)** after avoiding tensor parallelism for the sparse LM-head gathers. The first long full-LR run improved soft validation loss from **29.8 → 22.1** by step 10,710, then diverged to **75.4** by step 17,136. It was stopped and relaunched from the best checkpoint with LR **1e-3** instead of **3.16e-3**. A learning rate sweep is next (modulo capacity under the Marin hero run). ([#177](https://github.com/Open-Athena/MarinFold/issues/177)).

* Post-training updates
    * **Tim got the first end-to-end RL set-up working for MarinFold!** Tim fine-tuned a MarinFold checkpoint to emit multiple contact map candidates per input, then ran RL on the result with multiple reward functions and KL smoothing. RL-trained models are not yet beating strong baseline models, however. ([#230](https://github.com/Open-Athena/MarinFold/issues/230), follow-up at [#237](https://github.com/Open-Athena/MarinFold/issues/237))

* Other topics
    * **3D Structures: Tim found that Helico already gets essentially all the value it can from top-L binary MarinFold contacts**; adding lower-confidence contacts boosts recall but hurts precision and does not improve folding, so the next lever is confidence/graded contact conditioning rather than bigger contact lists. ([#249](https://github.com/Open-Athena/MarinFold/pull/249))
    * **Inference: seeding rollouts with one high-confidence pair is at most a small oracle-best win.** Tim tested prompting each rollout with a different top-ranked pairwise contact. Consensus is a tie with i.i.d. sampling (**0.5234 vs 0.5217** on eval-val), while oracle best-of-100 improves modestly (**0.5341 vs 0.5199**). One contact is too little conditioning; the signal seems to be in joint partial contact sets ([#254](https://github.com/Open-Athena/MarinFold/issues/254), [#255](https://github.com/Open-Athena/MarinFold/pull/255)).
    * **Data / formats: experimental-PDB data and multi-chain contacts-v1 landed.** Tim built and published Protenix/AF3-style PDB contacts-v1 corpora, including monomers and multimers, and extended contacts-v1 so it can serialize complexes ([#222](https://github.com/Open-Athena/MarinFold/issues/222), [#223](https://github.com/Open-Athena/MarinFold/pull/223)).
    * **Housekeeping / tooling.** Eric split training sweep skills for TRC vs CoreWeave ([#235](https://github.com/Open-Athena/MarinFold/pull/235)). Tim retired the checked-in `experiments/index.md` in favor of `scripts/itemize.py` printing the index ([#253](https://github.com/Open-Athena/MarinFold/pull/253)). Tim also opened an interactive evals exploration notebook for scoreboards, per-protein browsing, contact maps, and two-checkpoint comparisons ([#250](https://github.com/Open-Athena/MarinFold/issues/250), [#252](https://github.com/Open-Athena/MarinFold/pull/252)).

### Upcoming

* Eric: continue #232 training/evals on the decontaminated corpus, especially whether the later m2-p06 checkpoints and cooldown can close the gap to #199 cooldown while staying clean.
* Tim/Zack while Tim is OOO: keep the weekly thread moving; decide what from the eval/decontamination story belongs in the MoML/blog narrative.
* Zack: babysit the reduced-LR soft-target restart (#177/r86) until it produces validation points; if it still diverges, report before relaunching/killing anything large.
* Post-training (when Tim is back): decide whether #237’s small-KL RL window is worth another targeted run, and whether a reward closer to the deployed consensus metric is the main path forward.

---

## Week of August 17, 2026

### Last week

* **New best model, and on the 554-protein eval set we now pass single-sequence Protenix (our end-of-Aug goal!)** ([#238](https://github.com/Open-Athena/MarinFold/issues/238), [#239](https://github.com/Open-Athena/MarinFold/pull/239), from Eric's [#234](https://github.com/Open-Athena/MarinFold/pull/234)). The cooldown resulted in **R-precision 0.6307 / val loss 2.9397** against 0.6088 / 2.9712, i.e. **+0.028 over Protenix-v2 single-sequence (0.6032), 95% CI [+0.001, +0.054]**. Published as [contacts-v1-exp199-cooldown-1.5B](https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/prot-exp199-cw-cv1-p06-cool-s01) and now the default. [#204](https://github.com/Open-Athena/MarinFold/pull/204)'s plateau fit is falsified and nobody has found where this line stops improving.
* **Correction to last week: the previous default scores 0.6088, not the 0.587 I quoted** ([#209](https://github.com/Open-Athena/MarinFold/issues/209), [#212](https://github.com/Open-Athena/MarinFold/issues/212)). Three measurements now agree (0.6103 / 0.6110 / 0.6088); the 0.587 came from #199's own eval pipeline and the discrepancy is unexplained.
* **Eval note**: the mix of de novo designed vs. natural proteins in a given eval set is very consequential to our conclusions. De novo designs score much higher in both MarinFold and in Protenix SS, with MarinFold underperforming Protenix SS. On natural proteins, both predictors score much lower, and MarinFold tends to do a bit better than Protenix SS. So we need to stratify by de novo vs natural when reporting evals. I think natural is the more important subset to focus on since the de novo designed proteins are typically the result of pipelines that filter based on single-sequence folding using AlphaFold-like models, creating a kind of circular logic when benchmarking against those exact predictors on them.
* **Eval issue: our training set includes proteins similar to those in our eval set** ([#213](https://github.com/Open-Athena/MarinFold/issues/213), [#216](https://github.com/Open-Athena/MarinFold/pull/216)). This hasn't been a big deal in the past since our accuracy doesn't seem to correlate with similarity to training set, but as we get closer to writing up some results it is something we need to fix. We generated a new training set based on a 30% sequence identity threshold ([#225](https://github.com/Open-Athena/MarinFold/issues/225), [#229](https://github.com/Open-Athena/MarinFold/pull/229)).
* **Eric kicked off a new sweep on the decontaminated data above** ([#232](https://github.com/Open-Athena/MarinFold/issues/232), [#233](https://github.com/Open-Athena/MarinFold/pull/233)): ten 1.5B CoreWeave trials over the rebuilt corpora (74.5B tokens). Two diverged and were abandoned; eight are still going ([W&B](https://wandb.ai/open-athena/MarinFold/groups/prot-exp232-cw-cv1-decontam-s02)).
* **In the mean time, we made a new eval set ("eval2") that gets rid of proteins that are close to those in train (40% identity threshold) and also adds 23 additional FoldBench monomers** ([#226](https://github.com/Open-Athena/MarinFold/issues/226), [#228](https://github.com/Open-Athena/MarinFold/pull/228)). We can use this as a stop-gap to think through how we might present results while we wait for Eric's new sweep to run. Once Eric's new sweep finishes, we can eval on a much bigger eval set.
* **On eval2 the headline shrinks.** Natural: MarinFold **0.358** vs Protenix single-seq 0.326 (CI crosses zero). De novo: 0.621 vs 0.799.
* **We fine-tuned a contacts to 3D structure model** ([helico#10](https://github.com/Open-Athena/helico/pull/10), [helico#13](https://github.com/Open-Athena/helico/pull/13)). MSA-free on 98 paired FoldBench monomers. Initial impressions: (1) this turned out to be easier than I thought, (2) we can match but not exceed MSA-based prediction if we provide ground truth (oracle) contacts, (3) with current MarinFold-generated contacts, we do seem to get a small but real boost over pure single sequence prediction!
* **We tried some RL on the base model and were not able to improve consensus R-precision** ([#208](https://github.com/Open-Athena/MarinFold/issues/208), [#220](https://github.com/Open-Athena/MarinFold/pull/220)). My read here is that the "language" of the base model (where we generate a single contact set per rollout, with no ability to backtrack) is too simplistic for RL to work. The only free variable the RL is optimizing is the order the contacts are outputted. Further RL efforts are focusing on fine-tuned models that can generate multiple distinct contact sets per rollout.
* **Toward the idea above, we fine-tuned a multi-draft model from last week's best model** ([#230](https://github.com/Open-Athena/MarinFold/issues/230), [#231](https://github.com/Open-Athena/MarinFold/pull/231)). Redid [#163](https://github.com/Open-Athena/MarinFold/issues/163) with more training steps, also now using PDB data. We got a model that speaks our new format correctly, in that we can prompt it to either generate a single generation as before or multiple generations in one rollout (up to the limits of the context length).
* **Early hints that RL might actually do something on the multi-draft model** ([#237](https://github.com/Open-Athena/MarinFold/issues/237), [#240](https://github.com/Open-Athena/MarinFold/pull/240)). Trying lots of ideas here, including rewarding each individual contact generation as well as rewarding rollouts as a whole, e.g. according to `max(F1(generation))`.
* **A rollout is a coherent 3D hypothesis, not a bag of contacts** ([#211](https://github.com/Open-Athena/MarinFold/issues/211), [#217](https://github.com/Open-Athena/MarinFold/pull/217)). This one the agents ran away with and I haven't checked it yet for reasonableness, but the result is what I'd hope / expect. Assuming it is true, I want to watch that we maintain this property in generations as we do RL.
* **Data: we made a PDB corpus, and contacts-v1 can now describe a complex** ([#222](https://github.com/Open-Athena/MarinFold/issues/222), [#223](https://github.com/Open-Athena/MarinFold/pull/223)). This is the first time we have looked at including experimental structures from PDB in our training set. So far it has only been used for fine-tuning in preparation for RL.
* **Housekeeping.** Sweep skill split into TRC and CoreWeave variants ([#235](https://github.com/Open-Athena/MarinFold/pull/235)); eval-checkpoint defaults to eval2 and scores with rollout+resample ([#214](https://github.com/Open-Athena/MarinFold/pull/214), [#215](https://github.com/Open-Athena/MarinFold/pull/215)).

### Upcoming
* Long term idea: we bring helico (contacts to 3D coordinate generator) into the RL loop, so we can give rewards based on real structural accuracy. The helico model weights stay frozen. The MarinFold model is pushed to output contact sets that cause helico to give high accuracy predictions. Not going to work on this until the fall but I think it's a reasonable next step when / if we get clear evidence that RL can work here.
* Eric: finish the [#232](https://github.com/Open-Athena/MarinFold/issues/232) decontamination sweep, then the [#154](https://github.com/Open-Athena/MarinFold/issues/154) summary of the whole contacts-v1 optimization line.
* Tim: more RL ([#237](https://github.com/Open-Athena/MarinFold/issues/237), [#200](https://github.com/Open-Athena/MarinFold/issues/200))
* Tim: think through what we might submit to MoML conference in early Sept and if a blog post makes sense.
* Zack is back: soft-target training ([#177](https://github.com/Open-Athena/MarinFold/issues/177)).
* Jacob is planning to implement the AFDB complex curation from [#145](https://github.com/Open-Athena/MarinFold/issues/145). Note: we need to make sure we avoid any component monomers with >30% sequence identity to our eval set (implemented as in [#225](https://github.com/Open-Athena/MarinFold/issues/225) / [#229](https://github.com/Open-Athena/MarinFold/pull/229)).
* Deferred but worth doing at some point: make a smaller helico model so inference is less onerous (especially if it goes in the RL loop).
---

## Week of August 10, 2026

### Last week
* **Eric's sweep on ESMFold2 distillation data produced our best model yet** ([#199](https://github.com/Open-Athena/MarinFold/issues/199), [#204](https://github.com/Open-Athena/MarinFold/pull/204), [#205](https://github.com/Open-Athena/MarinFold/pull/205)): 1.5B on AFDB + ESM-Atlas. The CoreWeave `p06-aug` run wins on both axes: **contacts-v1 loss 2.971** (≈2.59 old), and **R-precision 0.587 all / 0.542 long** against the #117 control's ≈3.085 / 0.534 / 0.483. That is within **0.016 of Protenix-v2 single-sequence (0.603)**. But the loss gain outran the accuracy gain, and the shape of the curve looks like a plateau.
  * _[Editor's note, 2026-08-17: the R-precision here is wrong. `p06-aug` scores **0.6088** under Eric's own re-run and 0.6103 / 0.6110 under two independent measurements with exp82's worker — the 0.5873 came from #199's eval pipeline ([#209](https://github.com/Open-Athena/MarinFold/issues/209), [#212](https://github.com/Open-Athena/MarinFold/issues/212)). Corrected, that changes the sign of the Protenix single-sequence comparison (from −0.016 to **+0.005**) and the "plateau" reading did not hold: the cooldown reached 0.6307.]_ [Checkpoints](https://huggingface.co/open-athena/marinfold-exp199); W&B [TRC](https://api.wandb.ai/links/eric-czech/582mdeag) / [CW](https://api.wandb.ai/links/eric-czech/g2x1fbj5). This subsumes [#196](https://github.com/Open-Athena/MarinFold/issues/196).
* **Post-training: we fine-tuned a multi-draft generator** ([#163](https://github.com/Open-Athena/MarinFold/issues/163), [#164](https://github.com/Open-Athena/MarinFold/pull/164), [writeup](https://github.com/Open-Athena/MarinFold/blob/main/experiments/exp163_models_teach_contacts_v1_to_refine_a/WRITEUP.md)). The fine-tune taught the model to write ~15 near-disjoint candidate contact maps in a single rollout. We did this while maintaining ordinary contacts-v1 task accuracy (pairwise R-precision 0.3374 vs 0.3357), although we do see some "leakiness" where even in single-generation mode the fine-tuned model occasionally outputs multiple predictions. [Model published](https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF), plus a [demo notebook](https://colab.research.google.com/github/Open-Athena/MarinFold/blob/main/experiments/exp163_models_teach_contacts_v1_to_refine_a/exp163_multidraft_demo.ipynb). This model basically just adheres to the format rather than "reasoning through the possibilities". It gives us something to try to RL though in [#200](https://github.com/Open-Athena/MarinFold/issues/200).
* **Post-training: backtracking is merged but on hold for now** ([#158](https://github.com/Open-Athena/MarinFold/issues/158) → [#159](https://github.com/Open-Athena/MarinFold/issues/159) → [#160](https://github.com/Open-Athena/MarinFold/issues/160) → [#175](https://github.com/Open-Athena/MarinFold/issues/175); PRs [#161](https://github.com/Open-Athena/MarinFold/pull/161), [#171](https://github.com/Open-Athena/MarinFold/pull/171), [#172](https://github.com/Open-Athena/MarinFold/pull/172), [#197](https://github.com/Open-Athena/MarinFold/pull/197); [series status](https://github.com/Open-Athena/MarinFold/blob/main/experiments/STATUS_backtracking.md)). Similar to the multi-draft generation, we fine-tuned a model with a `<contacts-v1.backtracking>` doc-type and got it to adhere to the new format (**0.1 vs 42 retracts per rollout** based on whether you prompt it with `<contacts-v1>` vs. `<contacts-v1.backtracking>`). We saw a slight loss on R-precision, however (-0.015). Whatever kind of RL we figure out for multi-draft mode could probably also be applied here, but for now multi-draft seems more flexible so we are going to focus on that.
* **Started de-risking whether we can go from contacts to 3D structures using a traditional diffusion module**. To do this we are reviving our old [helico](https://github.com/Open-Athena/helico/) repo (an AF3 clone), and revamping it to fold from contacts instead of MSAs ([helico#10](https://github.com/Open-Athena/helico/pull/10), [design doc](https://github.com/Open-Athena/helico/blob/claude/helico-residue-contacts-redesign-4cc1c4/.agents/project/20260806_contact_conditioned_folding.md)). It conditions on a three-state (contact / no-contact / unknown) token × token matrix from pyconfind, trained across 0–100% conditioning with FP/FN corruption so it tolerates imperfect predicted contacts. Training is ongoing.
* **Pause tokens: negative result** ([#124](https://github.com/Open-Athena/MarinFold/issues/124), [#179](https://github.com/Open-Athena/MarinFold/pull/179)). Zack finished the `<think>` run. Think-masked validation improves slightly, but contact accuracy does not: R-precision 0.3365 standard vs 0.3359 / 0.3347 / 0.3353 with 1/2/3 inserted `<think>` tokens.
* **Data: Jacob's multimer survey is done** ([#145](https://github.com/Open-Athena/MarinFold/issues/145), [writeup](https://github.com/jsilter/MarinFold/blob/exp145/multimer-data-survey/experiments/exp145_data_multimer_data_survey/README.md)). Recommendation is to use the AlphaFold DB complex release to get about 1.96M homodimers and 70k heterodimers.
* **Reminder: validation-loss confusion was due to [marin#7209](https://github.com/marin-community/marin/pull/7209)'s padding-target masking (found by Zack).** Losses recorded after that change are not comparable to our older numbers. This was why we thought the GPU training wasn't working. Eric computed an old to new conversion formula over 4 [#166](https://github.com/Open-Athena/MarinFold/issues/166) checkpoints re-scored under the new code ([gist](https://gist.github.com/eric-czech/9c40252457790a513eeb62a6a965c049)): `current = 0.86358 × old + 0.75716`, which is roughly **`old + 0.382`**. Our landmarks on the new scale: [#117](https://github.com/Open-Athena/MarinFold/issues/117) 1.5B E8 `2.713 → ≈3.095`, #117 1.5B E16 `2.704 → ≈3.086`, [#146](https://github.com/Open-Athena/MarinFold/issues/146) 3B E8 `2.7025 → ≈3.084`, [#166](https://github.com/Open-Athena/MarinFold/issues/166) 1.5B AA-aug `2.664 → ≈3.046`.

### Upcoming
* Eric: decide whether to keep training the best CoreWeave model, [prot-exp199-cw-cv1-s02-m1-p06-aug](https://huggingface.co/open-athena/marinfold-exp199/tree/main/prot-exp199-cw-cv1-s02-m1-p06-aug). It only saw 152B tokens (2e21 FLOPs) from scratch, so there may be a lot left in it. Then checkpoint/code cleanup and an update to [#154](https://github.com/Open-Athena/MarinFold/issues/154).
* Tim: get RL working for the multi-draft model ([#200](https://github.com/Open-Athena/MarinFold/issues/200), [#203](https://github.com/Open-Athena/MarinFold/pull/203)).
* Tim: get helico training far enough to say whether contact conditioning can replace an MSA ([helico#10](https://github.com/Open-Athena/helico/pull/10)).
* Jacob is away this week, then planning to implement the AFDB complex curation from [#145](https://github.com/Open-Athena/MarinFold/issues/145).
* Zack is away this week, then planning to wrap up the soft-target work ([#177](https://github.com/Open-Athena/MarinFold/issues/177)).

---

## Week of August 3, 2026

### Last week

* **Training: two new models are modestly better than last week's.** Eric's amino-acid augmentation run ([#166](https://github.com/Open-Athena/MarinFold/issues/166), [#190](https://github.com/Open-Athena/MarinFold/pull/190)) hits **2.664 val loss / 0.562 R-precision** and Tim's run with ESMFold2 distillation data ([#155](https://github.com/Open-Athena/MarinFold/issues/155), [#192](https://github.com/Open-Athena/MarinFold/pull/192)) is just below it at **2.682 / 0.554**, both past #117's 2.704 / 0.534.
* **Measurement confusion**: [marin#7209](https://github.com/marin-community/marin/pull/7209) caused our observed losses to jump by ~0.42 nats. Zack submitted [marin#7921](https://github.com/marin-community/marin/pull/7921) to give us an option to route around this. If the previous behavior was just buggy though we can update our target validation losses accordingly. It seems possible (but not proven) that this was actually the reason we were saying we couldn't train a good model on GPUs.
* **Post-training: both lines are on hold behind bad measurements.** There were basic issues with both post-training approaches as implemented — the backtracking corpus emitted its ground-truth contacts in sorted order ([#159](https://github.com/Open-Athena/MarinFold/issues/159)), and every fine-tuned checkpoint in the rollout-refinement experiment was scored with the wrong rope ([#163](https://github.com/Open-Athena/MarinFold/issues/163), fixed in [#184](https://github.com/Open-Athena/MarinFold/pull/184)) — and then we had limited TPU and GPU availability. I am going to steer these agents a bit more actively. Hoping to have results on it for next week.
* **Structure: The coarse fold is the bottleneck, not refinement** ([#174](https://github.com/Open-Athena/MarinFold/issues/174), [#176](https://github.com/Open-Athena/MarinFold/pull/176)). Our model can't assign atoms to coarse grained voxels (10 angstrom cubed) accurately. I am leaning toward abandoning this direction for now and instead predicting 3D coordinates using a bespoke diffusion model or similar.

### Upcoming

* Resolve the validation loss issue so that all our runs are comparable. This will mean either adjusting previous validation losses to match new behavior or routing around the new behavior.
* Tim is training another 1.5B on two epochs of ESMFold2 distillation data ([#196](https://github.com/Open-Athena/MarinFold/issues/196)), plus the no-crops ablation that tells us whether the crops corpus is contributing anything.
* Tim is working on the post-training experiments: backtracking ([#160](https://github.com/Open-Athena/MarinFold/issues/160)) and rollout refinement ([#163](https://github.com/Open-Athena/MarinFold/issues/163)).
* Considering looking into training a diffusion model to go from predicted contacts to 3D coordinates.
* Zack is working on testing the effect of training on soft targets ([#177](https://github.com/Open-Athena/MarinFold/issues/177)).
* Zack plans to take a look at Eric's set up for doing model sweeps ([#191](https://github.com/Open-Athena/MarinFold/pull/191)) so he can try it on our ESMFold2 distillation dataset.

---

## Week of July 27, 2026

### Last week

* **Training: Eric trained our best-yet 1.5B model.** On our 554-protein contact benchmark, his new best 1.5B ([#117](https://github.com/Open-Athena/MarinFold/issues/117), eval loss 2.7037) gets **R-precision 0.53** vs **0.42** for the [#61](https://github.com/Open-Athena/MarinFold/issues/61) model we were using.
* **Training: Eric is looking at scaling parameters to 3B and 6B** ([#154](https://github.com/Open-Athena/MarinFold/issues/154)). See also [#166](https://github.com/Open-Athena/MarinFold/issues/166) (amino-acid augmentation from the best 8-epoch models vs random init).
* **Training: Jesse (now on parental leave; Zack to take this over) started looking into training on soft-targets instead of one-hot** (similar to how distillation is done in LLMs). See [#162](https://github.com/Open-Athena/MarinFold/pull/162), under [#147](https://github.com/Open-Athena/MarinFold/issues/147).
* **Data: the ESMFold2-Atlas contacts-v1 corpus landed ([#139](https://github.com/Open-Athena/MarinFold/issues/139), [#141](https://github.com/Open-Athena/MarinFold/pull/141)) — 66.8M documents / 71.4B tokens,** ~16× the AFDB set and the "67M instead of 4M" ([#91](https://github.com/Open-Athena/MarinFold/issues/91)). Thanks Jacob for the curation. This time we saved raw pyconfind contacts, so the next document format over this source will be much cheaper to generate.
* **Data: as our first attempt to generate 3D coordinates**, we wrote out *contacts-and-crops-v1* data for the AFDB training set. Tim is training a model on this ([#137](https://github.com/Open-Athena/MarinFold/issues/137); data [#132](https://github.com/Open-Athena/MarinFold/issues/132), format [#130](https://github.com/Open-Athena/MarinFold/issues/130)).
* **Post-training idea 1: backtracking ([#158](https://github.com/Open-Athena/MarinFold/issues/158), [#159](https://github.com/Open-Athena/MarinFold/issues/159), [#160](https://github.com/Open-Athena/MarinFold/issues/160))** — We allow the model to "undo" a contact by emitting a `<retract>` token ([#161](https://github.com/Open-Athena/MarinFold/pull/161)). The generator retracts when the model's own posterior turns against a contact it already emitted. 1.02M documents generated on 48 H100s in ~4.5h. Training next.
* **Post-training idea 2: rollout refinement ([#163](https://github.com/Open-Athena/MarinFold/issues/163), [#164](https://github.com/Open-Athena/MarinFold/pull/164)):** prompt the model with contacts sampled from previous rollouts. Generated 10k samples as a test.
* **Loss functions: Zack started an experiment to look at a different loss function for pre-training ([#156](https://github.com/Open-Athena/MarinFold/issues/156), [#167](https://github.com/Open-Athena/MarinFold/pull/167))**. Trains on the unordered contact set instead of penalizing arbitrary serialization order and orientation. See also [#157](https://github.com/Open-Athena/MarinFold/issues/157) (relative/RoPE-style residue-position embeddings).
* **Inference fix: got rid of the `top_k=50` sampling we were doing when generating rollouts for evals.** This was leading to a bias where rollouts tend to be shorter than a typical ground truth document, since the *end* token usually survives the top_k threshold but many lower probability contact position tokens don't. Thanks to Zack for a discussion that led to this finding. See [#142](https://github.com/Open-Athena/MarinFold/issues/142).
* **Housekeeping:** small bugs fixed in a codebase review ([#148](https://github.com/Open-Athena/MarinFold/pull/148), thanks Sankalp); an inference fix for checkpoints exported by newer transformers ([#165](https://github.com/Open-Athena/MarinFold/pull/165)); an [ESM-Atlas explorer Colab](https://colab.research.google.com/github/Open-Athena/MarinFold/blob/main/notebooks/explore_esm_atlas_distill.ipynb) ([#140](https://github.com/Open-Athena/MarinFold/pull/140)).

### Upcoming

* Tim is training a 1.5B on a mix of contacts-and-crops-v1 (AFDB) and contacts-v1 (AFDB, ESM-Atlas). It is looking promising as 28% of the way in we are at `contacts-v1-val` of 2.809 ([#155](https://github.com/Open-Athena/MarinFold/issues/155)).
* Tim will continue the two new post-training experiments: backtracking ([#160](https://github.com/Open-Athena/MarinFold/issues/160)) and rollout refinement ([#163](https://github.com/Open-Athena/MarinFold/issues/163)).
* Eric: evals and analysis across the scaling ladder ([#154](https://github.com/Open-Athena/MarinFold/issues/154)) and babysit the augmentation experiment ([#166](https://github.com/Open-Athena/MarinFold/issues/166)). Eric will send Tim an issue with paths to checkpoints he wants eval'd.
* Zack is working on getting a readout from the loss experiments in [#156](https://github.com/Open-Athena/MarinFold/issues/156) and [#147](https://github.com/Open-Athena/MarinFold/issues/147) / [#162](https://github.com/Open-Athena/MarinFold/pull/162).
* Jacob is looking into data curation of multi-protein complexes ([#145](https://github.com/Open-Athena/MarinFold/issues/145)).
* The pause-token dataset ([#124](https://github.com/Open-Athena/MarinFold/issues/124)) is unclaimed if anyone wants to train on it. The data is [ready](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/document_structures/contacts_v1_think). Can discuss on the call whether to do this.

---

## Week of July 20, 2026

### Last week

* **Negative result on our first attempt at post-training ([#120](https://github.com/Open-Athena/MarinFold/issues/120), [#122](https://github.com/Open-Athena/MarinFold/pull/122); data gen: [#100](https://github.com/Open-Athena/MarinFold/issues/100), [#101](https://github.com/Open-Athena/MarinFold/pull/101)).** Fine-tuning on generated "only-correct" rollouts is **worse** than simply re-epoching the original data. Still thinking through why this might be.
* That did turn up a **slightly better model** (the re-epoched one; eval loss 2.7566 → 2.744) so we published that as `contacts-v1-exp120-1.5B` and it's now the default in MODELS.yaml. However, Eric has already improved a lot more beyond that — he has a 2.71 eval loss model from his latest sweep ([#117](https://github.com/Open-Athena/MarinFold/issues/117)).
* First steps on getting the LLM to predict coordinates: we generated documents for **contacts-and-coordinates-v1** ([#105](https://github.com/Open-Athena/MarinFold/issues/105), [#121](https://github.com/Open-Athena/MarinFold/issues/121)), but ultimately decided that the 32k context required here is too big and did not launch any training runs on it.
* Instead, we made a new document structure called **contacts-and-crops-v1**, which keeps documents to 8192 tokens. We first give coarse 10 Å boxes for residues, then all atoms at 0.1 Å detail for a handful of spatial crops. Full [corpus](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/document_structures/contacts_and_crops_v1) has **4,213,203 documents, ~34.5B tokens**.
* **Pause tokens ([#123](https://github.com/Open-Athena/MarinFold/issues/123), [#125](https://github.com/Open-Athena/MarinFold/pull/125), [#126](https://github.com/Open-Athena/MarinFold/issues/126)).** Added `<think>` tokens to contacts-v1 and published the [think-augmented corpus](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/document_structures/contacts_v1_think).
* **Bio2Token merged ([#40](https://github.com/Open-Athena/MarinFold/issues/40), [#114](https://github.com/Open-Athena/MarinFold/pull/114))** — Alex figured out an efficient way to tokenize structures using [bio2token](https://arxiv.org/abs/2410.19110) on TPUs. This is for an alternative approach to structure prediction we are trying using neural tokenizers.
* **Productionization:** fixed the Colab/Kaggle notebook so people can actually run the model ([#107](https://github.com/Open-Athena/MarinFold/pull/107), [#115](https://github.com/Open-Athena/MarinFold/pull/115), [#116](https://github.com/Open-Athena/MarinFold/pull/116)), added a `fold-from-contacts` Colab comparing MarinFold vs MSA contacts ([#129](https://github.com/Open-Athena/MarinFold/pull/129)), and an eval-checkpoint skill that takes a checkpoint to R-precision in one step ([#135](https://github.com/Open-Athena/MarinFold/pull/135)).

### Upcoming

* After the negative result, I want to think more about post-training. One idea that came up in discussions last week with Sergey is to make a new document format that allows for back-tracking on predicted contacts (e.g. reviving the correction/non-correction tokens we had in the earlier [contacts-and-distances-v1](https://huggingface.co/datasets/timodonnell/protein-docs)). This way we can just keep rolling out a document to get different contact sets. We could use the existing base model to sample decoys. Still thinking through how this should work - if anyone has ideas or wants to brainstorm let me know.
* Eric is running an expanded tuning sweep on contacts-v1. This has already given us better models and will likely continue to find more improvements this week ([#117](https://github.com/Open-Athena/MarinFold/issues/117)).
* I'd like to kick off some training runs on [contacts-and-crops-v1](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/document_structures/contacts_and_crops_v1). Eric, do you have bandwidth for running a sweep on this data? If not I will try naive things
* The plan is to have our new team member Zack Nichols (welcome!) train some models on [contacts-v1-think](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/document_structures/contacts_v1_think). The best configurations from Eric's sweep ([#117](https://github.com/Open-Athena/MarinFold/issues/117)) are a good starting place for  this.
* Jacob is close to finished on curating ESMFold2 Atlas distillation data ([#91](https://github.com/Open-Athena/MarinFold/issues/91)). Once that is on huggingface, I will kick off document generation to make contacts-v1 docs out of it. Then we will have 67M proteins instead of 4M for training and will hopefully get nicer results from using that rather than epoching.
* Alex is going to be running some experiments using the bio2token neural tokenizer as an alternative to contact prediction ([#133](https://github.com/Open-Athena/MarinFold/issues/133)).

---

## Week of July 6, 2026

### Last week

* First steps toward post-training. We are likely going to start with rejection fine-tuning (RFT). The idea is to use the model to generate some high quality rollouts, and fine tune on those.
* To derisk this, in [exp98](https://github.com/Open-Athena/MarinFold/tree/main/experiments/exp98_data_generate_rollouts_contacts_v1_train) we wanted to see if our best-of-N accuracy is a lot higher than our average single-rollout accuracy. We generated 1000 rollouts for 1000 structures from our training set (1M rollouts total; 4.6 hours on a v5p-8). We see a nice spread in accuracies: best-of-N F1 score goes from 0.12 to 0.34 as N goes from 1 to 1000. The consensus contact prediction across rollouts (our current inference method) has an F1 score of 0.26. Conclusions: (1) this looks promising for post-training, (2) generating a huge dataset (e.g. 1M proteins) would be expensive.
* In [exp100](https://github.com/Open-Athena/MarinFold/issues/100) we're looking at a cheaper alternative to generate high-quality rollouts. Since the model only outputs contacts (rather than a reasoning trace), we can force the model to only emit true contacts, so every regenerated document is perfectly accurate by construction. For each protein we generate 10 rollouts this way and keep the one with the highest likelihood. So what we get is correct documents where the contacts appear in an order that the model is likely to actually predict (as opposed to random order as in our pretraining documents). This is running now to regenerate our full training set and about 30% done - should finish next week.
* Started on a new document format, **contacts-and-coordinates-v1** — wrote the spec ([#104](https://github.com/Open-Athena/MarinFold/pull/104)) and opened the experiment to generate a training set ([#105](https://github.com/Open-Athena/MarinFold/issues/105)). This is a separate line of work from the post-training stuff. The idea here is to play with an idea for how to get the model to predict 3D coordinates rather than just contacts. Still nailing down how this will work. May need to increase the model context length from 8k to 32k for this.
* Productionization: contacts-v1 inference is now graduated into the marinfold CLI ([#92](https://github.com/Open-Athena/MarinFold/pull/92)). More testing is needed to see if people can actually use the model right now.

### Upcoming

* Finish the only-correct constrained-decoding scale-out ([#100](https://github.com/Open-Athena/MarinFold/issues/100)).
* After that, we will do the actual post-training experiment. We will compare using the data from [#100](https://github.com/Open-Athena/MarinFold/issues/100) vs. just re-epoching our existing training data. To be determined: should we just re-heat Eric's best model for this, or do something else?
* Finalize the format and generate the contacts-and-coordinates-v1 training set ([#105](https://github.com/Open-Athena/MarinFold/issues/105)).
* Eric is running a bigger tuning sweep to push accuracy further ([#61](https://github.com/Open-Athena/MarinFold/issues/61), [#75](https://github.com/Open-Athena/MarinFold/issues/75)).
* Jacob is working on ESMFold2 Atlas distillation data ([#91](https://github.com/Open-Athena/MarinFold/issues/91)).
* Allen is looking into if there is anything simple we can say about what differentiates high-accuracy rollouts vs. average accuracy rollouts ([#102](https://github.com/Open-Athena/MarinFold/issues/102)). Tim will send him data for this.

---

## Week of June 29, 2026

### Last week

* Plot twist: the best model trained in Eric's sweep ([#61](https://github.com/Open-Athena/MarinFold/issues/61), [#75](https://github.com/Open-Athena/MarinFold/issues/75)) learned to predict contacts at a meaningful level of accuracy! Evaluated on our 554-protein eval set it gets >0.4 R-precision ([#89](https://github.com/Open-Athena/MarinFold/issues/89)). It appears there is a phase change after ~23B training tokens.
* Inference tuning for accuracy: 100x rollouts > 10x ensemble of P[first contact] > P[first contact] ([#82](https://github.com/Open-Athena/MarinFold/issues/82)).
* Checked that the accuracy is real generalization and not just memorization: a sequence-alignment K-nearest-neighbor null model gives us a memorization baseline to compare against ([#94](https://github.com/Open-Athena/MarinFold/issues/94)).
* Started digging into where model 61 does well vs. poorly ([#96](https://github.com/Open-Athena/MarinFold/issues/96)) — e.g. we do notably worse on viral proteins, which perhaps makes sense since we train on AFDB (AF2 predictions), and AFDB excludes viral proteins.
* Productionization: contacts-v1 inference is now in the marinfold CLI, and there's a Colab notebook for running the model (needs testing).

### Upcoming

* Eric is running a bigger sweep to further improve accuracy ([#61](https://github.com/Open-Athena/MarinFold/issues/61)).
* Jacob is looking into expanding our training set to include ESMFold2 distillation data ([#91](https://github.com/Open-Athena/MarinFold/issues/91)).
* I am starting our first post-training experiment ([#98](https://github.com/Open-Athena/MarinFold/issues/98)) — the goal is to understand if fine tuning on high-accuracy self-generated rollouts does better than just fine tuning on more training data.

---

## Week of June 22, 2026

### Last week

* Looks like our quick-and-dirty model trained on contacts-v1 does not [perform well](https://github.com/Open-Athena/MarinFold/issues/82#issuecomment-4720288663) at all.
* However, [Eric&#39;s sweep](https://github.com/Open-Athena/MarinFold/issues/61#issuecomment-4752161683) generated models with significantly improved eval perplexities than my quick-and-dirty model. So we are evaluating his best model now ([#89](https://github.com/Open-Athena/MarinFold/issues/89)). We will see if this changes the story.
* While we were waiting for @Eric Czech 's sweep, I tried re-heating my quick-and-dirty model and doing another epoch. That improved eval loss somewhat but is still worse than best model from Eric's sweep ([#85](https://github.com/Open-Athena/MarinFold/issues/85))
  * _[Editor's note, 2026-07-30: the re-heat did **not** improve eval loss. It finished at `contacts-v1-val` **2.9801** against #67's **2.9800**, and its five eval points run 2.9825 / 2.9828 / 2.9843 / 2.9820 / 2.9801 — no progress over the checkpoint it restarted from. The rest of the sentence stands: it was well short of Eric's sweep. Found while assembling the progress tracker in [#180](https://github.com/Open-Athena/MarinFold/issues/180).]_
* Implemented a simple inference algorithm for our contacts-v1 models ([#82](https://github.com/Open-Athena/MarinFold/issues/82))
* Evals: we now include ESMFold2 as a comparison ([#78](https://github.com/Open-Athena/MarinFold/issues/78))

### Upcoming

* Figure out if any of our contacts-v1 models show reasonable accuracy
* Assuming the above answer is no, I want to make a new dataset that gets us back to something closer to 40B tokens (our previous document structure) than 4B tokens (our current one). This can be done by changing the document structure and/or adding new proteins to our training set.
* Get @Alex Merose 's other PRs merged
* Sync with @Jacob Silterra @Sankalp Jajee (e/acc) about tasks

---

## Week of June 15, 2026

### Last two weeks

* We have a new document structure ("[contacts-v1](https://github.com/Open-Athena/MarinFold/blob/main/marinfold/marinfold/document_structures/contacts_v1/SPEC.md)"), a new [training set](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/document_structures/contacts_v1) based on it, and a quick and dirty model trained on it ([#67](https://github.com/Open-Athena/MarinFold/issues/67); [wandb](https://wandb.ai/open-athena/MarinFold/runs/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2)).
* In parallel, Eric is running a modeling sweep that is already getting better eval losses than the quick and dirty model ([#61](https://github.com/Open-Athena/MarinFold/issues/61)).
* We now have an expanded eval set that focuses on low-MSA depth proteins ([#65](https://github.com/Open-Athena/MarinFold/issues/65)). We are running Protenix v2 on it now ([#74](https://github.com/Open-Athena/MarinFold/issues/74)).

### Upcoming

* Implement a simple inference algorithm for our new contacts-v1 models
* Run evals on contacts-v1 models
* If results look promising (e.g. are competitive with Protenix in single sequence mode), I will start planning out our first experiments in post-training. Otherwise, I'll look into improving the base model (e.g. by expanding the training dataset).
* Eric is continuing to train better base models
* I'd like to revisit Alex's work and get things merged / wrapped up ([#38](https://github.com/Open-Athena/MarinFold/pull/38) [#39](https://github.com/Open-Athena/MarinFold/pull/39) [#72](https://github.com/Open-Athena/MarinFold/pull/72))

### Shout outs

* Very cool to see how fast Eric was able to [train models](https://github.com/Open-Athena/MarinFold/issues/61#issuecomment-4701658995) using his [schedule-sweep skill](https://github.com/eric-czech/marin-agent-kb/blob/main/skills/schedule-sweep.md) to optimally use available TPUs
