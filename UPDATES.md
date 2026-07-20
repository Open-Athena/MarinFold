# MarinFold Updates

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
