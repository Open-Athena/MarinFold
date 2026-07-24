---
marinfold_experiment:
  issue: 159
  title: "exp: build a contacts-v1 backtracking corpus by re-conditioning the base model on its own corrected contacts"
  kind: data
  branch: exp159-backtracking-corpus
---

# exp: build a contacts-v1 backtracking corpus by re-conditioning the base model on its own corrected contacts

**Issue:** [#159](https://github.com/Open-Athena/MarinFold/issues/159) · **Kind:** `data` · **Branch:** `exp159-backtracking-corpus` (off `exp158-contacts-v1-retract-token`; depends on the #158 format change)

## Question

Can we build a contacts-v1 training corpus of coherent **self-correction traces** by generating with the base model in the loop — emitting contacts, retracting the wrong ones once the model's own posterior turns against them, and continuously re-conditioning the model on its currently-live (corrected) contact set?

## Hypothesis

Generating incrementally and, after every retraction, **rebuilding the base model's prompt from the live (post-retraction) contact set** keeps every emitted contact on-distribution — unlike a post-hoc splice, which leaves the whole tail conditioned on a context that still contains the mistake. If the *timing* of each retraction is driven by the base model's own collapsing posterior on the queued contact, the corrected-away contacts are enriched for false positives using a signal computable purely from context — what the trained model (#160) needs to learn *when* to retract.

## Background

Depends on #158 (`<retract>` token + `read.py` fold). Feeds #160 (train + eval). Generator = `contacts-v1-exp120-1.5B` (#120). Reuses the #102 HF rollout path (emission order + teacher-forced scoring) and `inference._fwd_matrix` / `_pcontact_from_fwd` for the posterior signal. See the issue for the full method and the twin-corpus/publishing conventions (#53/#105/#126/#132).

## Approach

### Status: engine core built + unit-tested (CPU). GPU pilot is the next step.

**`backtrack_engine.py` — the pure state machine (done).** No torch / no marinfold model; the base model is reached only through two injected callables, so the whole loop is unit-tested with a stub backend (`test_backtrack_engine.py`, 6 tests, CPU, <1s):

- **Two streams.** Builds the output document's structure section as an ordered edit list of `("contact"|"retract", pair)` while the base model is always handed a clean live-only set via `Proposer.propose(live)` — so every proposal is conditioned on the corrected post-retraction state (the re-conditioning is implicit in the interface).
- **Posterior-collapse timing.** A queued false positive is retracted when `Scorer.score(committed, targets)` — the base model's `s(c)` against the committed set (live minus queue) — collapses relative to its peak (`tau`), drops below `s_floor`, or falls out of the top-`rank_factor·|GT|`. `min_delay` forbids immediate retraction; `eval_cadence` bounds scorer calls. Timing never uses GT.
- **Forced noise retraction.** With `noise_retract_prob`, a *true* contact is forcibly retracted (its posterior won't collapse, so this can't be queued) and re-emitted later.
- **Budget-reserved flush = correctness.** The main loop only runs while it can still afford to retract every live non-GT pair and emit every missing GT pair, so the final live set equals GT **exactly** (recall philosophy F). Budget exhaustion sets `truncated` (the one acknowledged failure mode). Loop guard bans a pair after `loop_cap` retract cycles.
- Output folds (via the real `read.live_contacts`) to exactly GT — asserted in every test.

**Interfaces to implement for the pilot (`Proposer` / `Scorer`):**
- `Proposer.propose(live)` — build the clean contacts-v1 prompt (sequence prefix + live contacts) and sample the next `<contact>` from `contacts-v1-exp120-1.5B` (HF transformers, T=1.0/p=0.95/k=50), returning its canonical position pair or `None` on `<end>`.
- `Scorer.score(committed, targets)` — one `_fwd_matrix` pass over the committed-set prompt → `s(c)` for every target pair at once.

### Remaining (not started)
1. **GPU adapter** `gen_backtracking_worker_hf.py` wrapping exp120 into `Proposer`/`Scorer`; assemble the full document (sequence prefix + engine structure + `<end>`) and verify `live_contacts == GT`.
2. **Pilot** on a small protein set: measure per-document wall-clock (the gating result), FP-enrichment of retractions (the go/no-go), retract-position/delay distributions, label-noise estimate. Dump full documents for eyeballing.
3. **10% single-retract probe class** — a policy mode of the same engine (hold one FP to a designated end/random position).
4. **Scale + publish** to `data/document_structures/contacts_v1_backtracking/` (per #126), if the pilot's cost + FP-enrichment justify it.

## Success criteria

- Engine: output always folds to exactly GT (unless `truncated`); posterior trigger produces FP-enriched retractions on the stub; loop terminates under adversarial proposers. **✅ (unit tests).**
- Pilot: retracted contacts strongly FP-enriched vs kept contacts (go/no-go); retraction delays spread (not immediate, not end-clustered); per-document wall-clock measured + extrapolated.
- Corpus: 0-drop 1:1 twin of the other corpora; ~10% single-retract probe class; anonymously readable from the public bucket.

## Results

_(Pilot pending — engine core + tests landed on `exp159-backtracking-corpus`.)_

## Conclusion

_(Fill in after the pilot.)_
