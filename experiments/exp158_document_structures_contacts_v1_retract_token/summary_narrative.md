# Summary slides — exp: add a `<retract>` statement to contacts-v1 so documents can take back a previously emitted contact

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can the contacts-v1 document format express **retraction** — a statement that a previously emitted `<contact>` is wrong — so that models can later be trained to self-correct partway through a rollout?

## Why

A single new statement type, `<retract> <pX> <pY>`, is sufficient to express backtracking. It can be added without disturbing existing corpora or checkpoints: with retraction disabled the generator emits **byte-identical** documents to today's contacts-v1, and appending the new token to the end of the vocab leaves every existing token ID unchanged.

## Results so far

_(Fill in as results come in.)_
