# Summary slides — exp: promote the #199 CoreWeave cooldown to the default contacts-v1 model

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

#234 landed a new best contacts-v1 model — #199's CoreWeave p06 cooldown,
prot-exp199-cw-cv1-p06-cool-s01 at step 290,400 — and left it in CoreWeave S3,
where nothing outside that cluster can read it. This publishes it, verifies it,
and makes it MarinFold's default.

## Why

A model nobody can download is not a default model. And publishing a
contacts-v1 checkpoint has two silent failure modes we have already been bitten
by: a levanter export whose rope block transformers 4.x ignores, and a
tokenizer that has to travel with the weights.

## The model

R-precision 0.631 all / 0.584 long on the 554-protein benchmark, against the
previous default's 0.609 / 0.563 scored by the same harness. Validation loss
2.9397 (~2.5580 on the old scale).

It is the first contacts-v1 model measurably ahead of single-sequence
Protenix-v2: +0.028 paired, 95% CI [+0.001, +0.054]. That clears zero by 0.001,
so read it as "ahead, barely".

The cooldown itself was nearly free — no new data, no new hyperparameters,
29,040 further updates with the learning rate annealed to zero.

## eval2

Not re-run: #234 had already scored the full 577-unit universe and checked in
every cut. On the 78 natural proteins under 40% identity to training the model
scores 0.358, against the previous default's 0.337, single-sequence
Protenix-v2's 0.326 and ESMFold2's 0.529.

The gain survives homology control. The 0.17 gap to ESMFold2 there is the
honest answer to "how good is this model".

## The rope repair cost 1.300 nats

Measured on this checkpoint against exp82's three benchmark documents. The same
defect cost the previous default 0.437. Anyone loading the raw export under
transformers 4.x gets a model worse than the #75 checkpoint from June, silently
— which is why the registry points at our repaired copy and why the number is
measured per checkpoint rather than carried over.

## A second result

Refreshing #180 turned up that #234's harness re-scores the *previous* default
at 0.6088, not the 0.5873 published in #204. #208 got 0.6103 independently.
#234's harness reproduces the historical #75 anchor to 0.00017, so the outlier
is that one reading — and it has never been explained.

Three of #180's conclusions changed: the Protenix single-sequence gap reverses,
the loss-to-accuracy exchange rate stops collapsing, and #204's sigmoid fit is
falsified by two checkpoints above its asymptote.
