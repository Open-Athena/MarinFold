# MarinFold Updates

## Week of July 27, 2026

### Last week

* **Where we stand on accuracy.** Re-scored the 554-protein contact benchmark with the current best 1.5B from Eric's sweep ([#117](https://github.com/Open-Athena/MarinFold/issues/117), eval loss **2.7037**) alongside the model we have been quoting since [#61](https://github.com/Open-Athena/MarinFold/issues/61) (2.7566). Figure: [`plots/where_we_stand_rprecision.png`](experiments/exp82_evals_contacts_v1_contact_prediction/plots/where_we_stand_rprecision.png).

  **The tuning gains are showing up in accuracy, and not by a little.** R-precision goes **0.42 → 0.53** for a 0.053-nat eval-loss improvement, and long-range R-precision **0.37 → 0.48**. We are still behind every structure predictor on top-K precision, but the gap to Protenix-v2 in single-sequence mode is now 0.53 vs 0.60 rather than 0.42 vs 0.60.

  | predictor | R-precision (all) | R-precision (long) | AUC (all) |
  |---|---|---|---|
  | MarinFold #61, n=100 rollouts | 0.425 | 0.366 | 0.901 |
  | **MarinFold #117 best, n=100 rollouts** | **0.535** | **0.485** | **0.932** |
  | Protenix-v2 · single-seq | 0.603 | 0.572 | 0.830 |
  | Protenix-v2 · MSA | 0.812 | 0.795 | 0.941 |
  | ESMFold | 0.755 | 0.732 | 0.901 |
  | ESMFold2 | 0.786 | 0.769 | 0.923 |

  Worth noticing the AUC column: at **0.932** we are now second only to Protenix-v2 with an MSA, and above ESMFold2. So the model ranks the *whole* contact map about as well as a structure predictor — what it still does poorly is concentrate its confidence in the top L or top R pairs. That is a calibration/precision problem, not a "the model doesn't know the fold" problem, and it points at a different set of fixes than we've been considering.

  Evals ran on **CoreWeave** — 36 single-H100 jobs at batch priority, 12 shards × 3 passes, ~4 minutes of compute where one workstation GPU takes 80 minutes per checkpoint. Recipe and gotchas are in `AGENTS.md`.

* **We stopped truncating the sampling distribution when we generate rollouts.** All our rollout evals had been running with `top_k=50` — an inherited HuggingFace default baked into the checkpoint export, not a choice anyone made. Truncated sampling renormalizes over the kept tokens, which inflates `<end>` and makes the model stop early. Measured on the eval set, paired, same 554 proteins:

  | | contacts asserted per rollout | vs ground truth (165.2) | R-precision (all) | AUC (all) |
  |---|---|---|---|---|
  | top-k 50 | 110.3 | 0.67× | 0.413 | 0.881 |
  | no top-k | 158.2 | **0.96×** | 0.425 | 0.901 |

  So most of the "under-generation" we have been worrying about since [#142](https://github.com/Open-Athena/MarinFold/issues/142) was the decoder after all — the count goes from 0.67× to 0.96× of ground truth. The accuracy it buys is real but modest (+0.011 R-precision, +0.020 AUC), which is consistent with #142's finding that the *withheld* contacts were mostly ones the model was unsure about. Every MarinFold number above is with top-k off, and it's the default in the eval path now.

  **So how much of 0.41 → 0.53 is the better model and how much is the sampling fix?** Ran the full 2×2 to answer it rather than guess:

  | R-precision (all) | top-k 50 | no top-k | top-k effect |
  |---|---|---|---|
  | #61 | 0.413 | 0.425 | +0.011 |
  | #117 best | 0.528 | 0.535 | +0.007 |
  | **model effect** | **+0.115** | **+0.110** | |

  **~91% model, ~7% sampling**, slightly sub-additive. The interaction makes sense: the better model was already under-generating less (0.75× of GT under top-k vs 0.67×), so it has less to gain from untruncating. One exception — for **AUC** the sampling change is ~a third of the gain, because untruncating is what populates the low-confidence tail that AUC integrates over.

  That top-k-50 row is also the control: it reproduces exp82's published HF-transformers number (0.4150 / 0.8814) to within 0.002 on a completely different stack, which is what licenses comparing any of these numbers to the previously published ones.

* **The ESMFold2-Atlas corpus landed — 67M proteins ([#139](https://github.com/Open-Athena/MarinFold/issues/139), [#141](https://github.com/Open-Athena/MarinFold/pull/141)).** **66,759,922 contacts-v1 documents / 71.4B tokens** (41 drops out of 67M), published to the [bucket](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/document_structures/contacts_v1_esm_atlas). That's ~16× the AFDB corpus and it is the "67M instead of 4M" expansion we have been waiting on since [#91](https://github.com/Open-Athena/MarinFold/issues/91) — thanks Jacob for the curation. We also published the **raw pyconfind contacts** as a reusable intermediate (31.9B contacts), so the next document format over this source costs a serialization pass instead of the ~2,850 core-hours pyconfind took. Worth knowing: the GCP CPU pool would have taken ~37 days for this; CoreWeave's reserved CPU pool did it in ~7 hours.

* **Training on the big corpus has started ([#155](https://github.com/Open-Athena/MarinFold/issues/155)).** A 1.5B on a three-way mixture — contacts-and-crops-v1 + contacts-v1 + ESM-Atlas contacts-v1, one epoch of each, size-proportional (it runs out of the [#137](https://github.com/Open-Athena/MarinFold/issues/137) trainer, so the W&B run is named `exp137-crops1ep-cv11ep-esm1ep-…`) — is 28% through and currently reads `contacts-v1-val` **2.809**. Early days (cosine schedule, most of the improvement comes at the end) but it is the best contacts-v1 loss of any of our own multi-corpus runs so far.

* **Backtracking / self-correction ([#158](https://github.com/Open-Athena/MarinFold/issues/158), [#159](https://github.com/Open-Athena/MarinFold/issues/159), [#160](https://github.com/Open-Athena/MarinFold/issues/160)).** This is the idea from the last update — let documents take back a contact they already emitted. All three pieces moved:
  * The `<retract>` statement is implemented ([#161](https://github.com/Open-Athena/MarinFold/pull/161)); with retraction off the generator emits byte-identical documents to today's contacts-v1, and the new token appends to the end of the vocab so no existing checkpoint's token IDs move.
  * The corpus generator works, and the interesting part is *when* it retracts: the trigger is the base model's own collapsing posterior on a contact it already emitted, with no ground truth involved. Across the full corpus it caught **76.4% of the false positives** the model emitted and **never once retracted a true contact**. So "the model can tell, from context alone, which of its own contacts are wrong" is real — which is the whole bet.
  * **1,023,997 documents / 1.08B tokens** generated on 48 H100s at CoreWeave batch priority in ~4.5 hours, zero worker failures, all of them verified to fold back to exactly the ground-truth contact set. Training on it is next.

* **Rollout refinement ([#163](https://github.com/Open-Athena/MarinFold/issues/163), [#164](https://github.com/Open-Athena/MarinFold/pull/164))** — a different post-training angle: show the model K of its own candidate rollouts and train it to emit a better contact set than any of them. Two results worth knowing regardless of whether this works:
  * **Consensus voting over K rollouts is a dead tie with the base model's own calibrated per-pair matrix** (0.224 vs 0.221). Voting is just a Monte-Carlo estimate of the same marginal — it adds nothing the model doesn't already output. So our current inference recipe is not leaving anything on the table that more rollouts would recover.
  * **But the joint signal is real and large:** condition the *untrained* base model on 50% of a protein's true contacts and R-precision on the remaining ones goes **0.145 → 0.556**. Feed it a *noisy* rollout instead and it gets worse (0.179 → 0.092) — it was trained on all-true contact sections and trusts its context. That gap is exactly what a refiner would have to learn. A local LoRA MVP confirms it's learnable (0.017 → 0.244 on identical K=16 input), though the margin over the base model is modest so far and looks capped by fold diversity. The first real training run started today.

* **contacts-and-crops-v1 training ([#137](https://github.com/Open-Athena/MarinFold/issues/137), [#138](https://github.com/Open-Athena/MarinFold/pull/138)).** Two from-scratch 1.5B runs on the 8k coordinate-document format. The crops-only run died at 41% and I haven't restarted it; the 95%-crops/5%-contacts-v1 mix is 69% through at crops-val **2.591** / contacts-v1-val **2.992**. Crops-val isn't comparable to anything we've published (different corpus and vocabulary), so contacts-v1-val is the number to watch — and note the crops-only run reads 3.62 there, i.e. a model that never sees a standalone contacts-v1 document is bad at them, which is why the 5% mix-in exists.

* **First-ever straight reproduction of one of Eric's runs ([#150](https://github.com/Open-Athena/MarinFold/issues/150), [#152](https://github.com/Open-Athena/MarinFold/pull/152)).** We have never actually checked that MarinFold's own training path lands where Eric's marin path lands — every previous run of ours changed the architecture, the data, or the hardware. This one changes nothing (his exp117 config verbatim, token cache verified bit-identical: 4,676,753,425 tokens / 4,129,682 docs on both sides). It's 41% through. If it misses 2.7112 by much, some of what we've been calling "data effects" in #120/#121/#137 is really a harness effect.

* **On-the-fly document generation ([#147](https://github.com/Open-Athena/MarinFold/issues/147), [#144](https://github.com/Open-Athena/MarinFold/pull/144)).** Builds contacts-v1 documents from the saved ESM-Atlas contacts at read time instead of materializing a corpus. Pilot ran clean on a v6e-8 at 14% MFU; the schedule-matched run is 22% through.

* **A caveat on every eval loss we quote.** Chasing an unexplained 0.41-nat gap between the #147 run and Eric's, the audit turned up that the standard cached packed dataset **scores `<pad>` positions in the loss** — **12.3% of the positions in our contacts-v1 validation stream are padding**. So the loss/perplexity numbers in this doc are over a stream that is ~1/8 trivially-predictable padding, and the real-document-token loss is meaningfully higher. Comparisons *between* our runs are still valid (they all share the eval), but the absolute number isn't what we've been saying it is, and a model trained with padding masked out (like #147's) is penalized against one that wasn't. Worth fixing the eval before we quote perplexity anywhere external.

* **Eric is scaling tokens and parameters ([#154](https://github.com/Open-Athena/MarinFold/issues/154)),** and the ladder so far:

  | run | eval loss |
  |---|---|
  | best prior 1.5B / 8 ep ([#75](https://github.com/Open-Athena/MarinFold/issues/75)) | 2.757 |
  | 1.5B / 8 ep, better tuning ([#117](https://github.com/Open-Athena/MarinFold/issues/117)) | 2.713 |
  | 1.5B / 16 ep ([#117](https://github.com/Open-Athena/MarinFold/issues/117)) | 2.704 |
  | 3B / 8 ep ([#146](https://github.com/Open-Athena/MarinFold/issues/146)) | 2.702 |
  | 6B / 8 ep ([#153](https://github.com/Open-Athena/MarinFold/issues/153)) | — (H100/GPU setup seems broken) |

  Two things I'd flag. **3B at 8 epochs ≈ 1.5B at 16 epochs** — at this token budget, doubling parameters buys about what doubling epochs buys, and the whole ladder spans only 0.055 nats, so on loss alone it looks like we're deep into diminishing returns. But the eval above says that same 0.05 nats was worth +0.11 R-precision, so the loss→accuracy slope has *not* flattened and these small deltas are still worth chasing. Which makes the **6B** result the interesting missing entry — all nine runs crashed. If it's the multi-node GPU bootstrap, `AGENTS.md` has our notes from exp108 and from this week's CoreWeave work; happy to dig in.

  He also opened [#166](https://github.com/Open-Athena/MarinFold/issues/166) to test amino-acid augmentation on the six best 8-epoch configs, both from scratch and warm-started from their matching checkpoints.

* **Zack started a greedy latent contact-set loss for contacts-v1 ([#156](https://github.com/Open-Athena/MarinFold/issues/156), [#167](https://github.com/Open-Athena/MarinFold/pull/167)).** This trains on the unordered contact *set* instead of penalizing arbitrary serialization order and pair orientation; it now runs end-to-end in Levanter with tests and an exp156 harness comparing against stock next-token CE. The new loss trains end-to-end on H100s; GB200 runs are still blocked on an eval issue. GPU telemetry shows comparable hardware efficiency to stock next-token CE (~95% average GPU utilization on the current 8×H100 head-to-head), but model-quality results are still inconclusive because the training/eval curves are noisy — the most recent next-token loss fine-tune looks like it's either diverging or spiking. He has also opened [#157](https://github.com/Open-Athena/MarinFold/issues/157), on replacing the learned residue-position embeddings with a relative/RoPE-style one.

* **Housekeeping:** nine real bugs found and fixed in a codebase review ([#148](https://github.com/Open-Athena/MarinFold/pull/148), thanks Sankalp); an inference fix so checkpoints exported by newer transformers still load ([#165](https://github.com/Open-Athena/MarinFold/pull/165)); an [ESM-Atlas dataset explorer Colab](https://colab.research.google.com/github/Open-Athena/MarinFold/blob/main/notebooks/explore_esm_atlas_distill.ipynb) ([#140](https://github.com/Open-Athena/MarinFold/pull/140)).

* **PSA on checkpoints:** two experiments this week concluded that the top #117 checkpoints had been deleted. They haven't — they were moved to `gs://<bucket>/checkpoints/protein/<run-name>/`, in both `marin-us-east5` and `marin-eu-west4`. The #61/#75 Levanter checkpoint is there too. That unblocks the warm-start in [#160](https://github.com/Open-Athena/MarinFold/issues/160) and [#163](https://github.com/Open-Athena/MarinFold/issues/163).

### Upcoming

* Train on the backtracking corpus ([#160](https://github.com/Open-Athena/MarinFold/issues/160)) and find out whether per-rollout self-correction beats rollout voting. Note the bar is high — #163 says voting already extracts everything the marginal has, so backtracking has to reach the *joint* signal to be worth anything.
* Finish the #163 refiner training run and decide whether to scale the corpus from 10k to 1M proteins.
* Let the three long training runs finish — the ESM-Atlas mixture ([#155](https://github.com/Open-Athena/MarinFold/issues/155)), the crops mix ([#137](https://github.com/Open-Athena/MarinFold/issues/137)) and the reproduction ([#150](https://github.com/Open-Athena/MarinFold/issues/150)) — and re-run the accuracy plot on all of them. Right now every accuracy number we quote comes from a model trained on 4M AFDB proteins; none of them has seen the 67M-protein corpus.
* Eric to run evals and analysis on the scaling improvements ([#154](https://github.com/Open-Athena/MarinFold/issues/154)) — the eval harness now turns a checkpoint into R-precision in ~4 minutes on CoreWeave, so this is cheap to do across the whole ladder — and to babysit the augmentation experiment from the best 1.5B/8ep models against random inits ([#166](https://github.com/Open-Athena/MarinFold/issues/166)).
* Zack to finish the 8×H100 [#156](https://github.com/Open-Athena/MarinFold/issues/156) comparison, summarize validation curves + telemetry, and decide whether repeats / LR sweeps are needed to get the noise under control. Also to fix or bypass the GB200 multi-GPU greedy next-token CE eval issue, so greedy-trained models can be compared on the same CE metric there too.
* The pause-token dataset ([#124](https://github.com/Open-Athena/MarinFold/issues/124)) is still unclaimed if anyone wants it — the data is [ready](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/document_structures/contacts_v1_think).
* Allen on why we fall off with protein length ([#96](https://github.com/Open-Athena/MarinFold/issues/96)) — Jandom's suggestion of folding individual subdomains to see whether multidomain chains are the problem is a good first test.
* Jacob on a report of the multimer content of AFDB / ESM-Atlas and what else is out there ([#145](https://github.com/Open-Athena/MarinFold/issues/145)).
* Alex's bio2token baseline ([#133](https://github.com/Open-Athena/MarinFold/issues/133)) — the tokenizer and documents from [#40](https://github.com/Open-Athena/MarinFold/issues/40) are merged and waiting on a model to be trained from them.

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
* However, [Eric's sweep](https://github.com/Open-Athena/MarinFold/issues/61#issuecomment-4752161683) generated models with significantly improved eval perplexities than my quick-and-dirty model. So we are evaluating his best model now ([#89](https://github.com/Open-Athena/MarinFold/issues/89)). We will see if this changes the story.
* While we were waiting for @Eric Czech 's sweep, I tried re-heating my quick-and-dirty model and doing another epoch. That improved eval loss somewhat but is still worse than best model from Eric's sweep ([#85](https://github.com/Open-Athena/MarinFold/issues/85))
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
