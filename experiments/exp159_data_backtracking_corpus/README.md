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

### Status: engine + adapter + pilot + a 370-doc corpus done (see Results). Scale-up is the open lift.

**`backtrack_engine.py` — the pure state machine (done).** No torch / no marinfold model; the base model is reached only through two injected callables, so the whole loop is unit-tested with a stub backend (`test_backtrack_engine.py`, 6 tests, CPU, <1s):

- **Two streams.** Builds the output document's structure section as an ordered edit list of `("contact"|"retract", pair)` while the base model is always handed a clean live-only set via `Proposer.propose(live)` — so every proposal is conditioned on the corrected post-retraction state (the re-conditioning is implicit in the interface).
- **Posterior-collapse timing.** A queued false positive is retracted when `Scorer.score(committed, targets)` — the base model's `s(c)` against the committed set (live minus queue) — collapses relative to its peak (`tau`), drops below `s_floor`, or falls out of the top-`rank_factor·|GT|`. `min_delay` forbids immediate retraction; `eval_cadence` bounds scorer calls. Timing never uses GT.
- **Forced noise retraction.** With `noise_retract_prob`, a *true* contact is forcibly retracted (its posterior won't collapse, so this can't be queued) and re-emitted later.
- **Budget-reserved flush = correctness.** The main loop only runs while it can still afford to retract every live non-GT pair and emit every missing GT pair, so the final live set equals GT **exactly** (recall philosophy F). Budget exhaustion sets `truncated` (the one acknowledged failure mode). Loop guard bans a pair after `loop_cap` retract cycles.
- Output folds (via the real `read.live_contacts`) to exactly GT — asserted in every test.

**The two model-backed callables (implemented in `backtrack_adapter.py`):**
- `Proposer.propose(live)` — build the clean contacts-v1 prompt (sequence prefix + live contacts) and sample the next `<contact>` from `contacts-v1-exp120-1.5B` (HF transformers, T=1.0/p=0.95/k=50), returning its canonical position pair or `None` on `<end>`.
- `Scorer.score(committed, targets)` — one `_fwd_matrix` pass over the committed-set prompt → `s(c)` for every target pair at once.

**GPU adapter + pilot (done).** `backtrack_adapter.py` wraps exp120 into `Proposer`/`Scorer` (unit-tested GPU-free in `test_backtrack_adapter.py`); `run_pilot.py` runs the engine on real proteins from exp98's `targets.parquet` (GT + sequences, no pyconfind) and reports the gating numbers. Pilot ran on an RTX A5000 — see Results.

### Remaining (not started)
1. **10% single-retract probe class** — a policy mode of the same engine (hold one FP to a designated end/random position).
2. **Scale-up** — the pilot's ~13 s/doc (no KV reuse) is far too slow for a 4.2M-doc corpus; needs KV-cache reuse across the growing prompt, batching across proteins, and likely TPU. This is the real engineering lift before a full run.
3. **Publish** to `data/document_structures/contacts_v1_backtracking/` (per #126) once scaled.

## Success criteria

- Engine: output always folds to exactly GT (unless `truncated`); posterior trigger produces FP-enriched retractions on the stub; loop terminates under adversarial proposers. **✅ (unit tests).**
- Pilot: retracted contacts strongly FP-enriched vs kept contacts (go/no-go); retraction delays spread (not immediate, not end-clustered); per-document wall-clock measured + extrapolated.
- Corpus: 0-drop 1:1 twin of the other corpora; ~10% single-retract probe class; anonymously readable from the public bucket.

## Results

**Pilot: 3 proteins (L 55–69), exp120 on one RTX A5000, T=1.0/p=0.95/k=50, eval_cadence=3, min_delay=3, tau=0.35, s_floor=1e-3.** (`data/pilot_metrics.csv`, sample doc in `data/pilot_docs/`.)

| entry | L | n_gt | wall | FP emitted | FP caught by trigger | trigger false alarms (TP) | folds→GT |
|---|---|---|---|---|---|---|---|
| AF-A0A0E3P1D7-F1 | 55 | 33 | 14.4 s | 24 | 14 | 0 | ✅ |
| AF-A0A0R2QD14-F1 | 65 | 23 | 14.8 s | 29 | 23 | 0 | ✅ |
| AF-A0A0W8FZR3-F1 | 69 | 6 | 9.7 s | 26 | 9 | 0 | ✅ |

- **Correctness holds on real model output:** all 3 documents fold (via `read.live_contacts`) to exactly GT; none truncated.
- **The posterior-collapse trigger is discriminative (the go/no-go):** across the three proteins, **46 of 79** emitted false positives were retracted *by the trigger* (before the flush), with **0** true contacts ever wrongly retracted by the trigger. So on the real base model the "which contact is wrong" signal is real, and — the point of the whole design — it flags false positives using only the visible contact set, never GT. The uncaught FPs (33) are cleaned at the correctness flush.
- **`trigger_recall` tracks context:** it falls when there are few true contacts (0.35 at n_gt=6 vs 0.79 at n_gt=23) — a small committed set gives the posterior little to lean on, so more FPs survive to the flush. Expect richer proteins to be the trigger's strong suit.
- **Cost:** ~13 s/doc at L≈60 on one A5000 with **no KV-cache reuse** (each step re-prefills the growing prompt). That extrapolates to impractical for a 4.2M-doc corpus — scale needs KV reuse + cross-protein batching + probably TPU. The pilot's job was to *measure* this, and it did.
- Base model emits many FPs when sampled one-at-a-time at T=1.0 (24–29 per protein) — plenty of retraction signal per document.

### Modest corpus (370 docs) — the corpus-level go/no-go

`gen_corpus.py` ran the engine over the 370 exp98 targets with L in [30,130] (one document each, `noise_retract_prob=0.05`), ~1.5 h on the A5000. Aggregated to `data/backtracking_corpus.parquet` (370 docs, ~198k tokens):

- **Correctness at scale:** all **370/370** documents fold to exactly GT; **0 truncated**.
- **Trigger is discriminative across the corpus (the real go/no-go):** of **11,489** emitted false positives, **7,098 (61.8%)** were retracted *by the posterior trigger*, with **0** true-contact false alarms — across all 370 proteins the trigger never once retracted a true contact. The remaining 38% of FPs are cleaned at the correctness flush.
- **Delayed, not immediate:** mean trigger delay **6.4 statements** — retractions sit well back from the mistake (the learnable long-range signal #160 needs), and the trigger-catch rate rises with protein size (70%+ at L~110-120, as the committed context grows).
- mean 31.6 retracts + 83.5 contacts per doc — retraction is a substantial, well-distributed fraction of each document.

This is a much stronger read than the 3-protein pilot and gives #160 a real (if small) corpus to train on. It is **not** the full corpus — 370 docs is pipeline-validation + first-signal scale; a 4.2M-doc twin needs the throughput work above.

Note: the exp120 checkpoint ships a marinfold-custom `tokenizer_class` (`TokenizersBackend`) that `AutoTokenizer` can't resolve; `run_pilot._fix_tokenizer_config` relabels it to `PreTrainedTokenizerFast` after each `resolve_model`. #160's eval path will hit the same thing — worth fixing in the marinfold transformers backend.

## Conclusion

The method works as designed on the real base model: model-in-the-loop generation with re-conditioning produces **exactly-correct** documents, and the base model's own posterior collapse is a **discriminative, GT-free** signal for *which* contacts to retract and *when* (46/79 FPs caught, 0 false alarms on 3 pilot proteins). The open risk is no longer "does the trigger flag the right contacts" (it does) but **throughput**: at ~13 s/doc unbatched, a full corpus needs KV-cache reuse + batching + TPU before scaling. Recommended next step is that throughput work (or a modest-scale corpus for a first #160 training signal), plus the 10% single-retract probe class.
