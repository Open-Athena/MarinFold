---
marinfold_experiment:
  issue: 158
  title: "exp: add a `<retract>` statement to contacts-v1 so documents can take back a previously emitted contact"
  kind: document_structures
  branch: exp158-contacts-v1-retract-token
---

# exp: add a `<retract>` statement to contacts-v1 so documents can take back a previously emitted contact

**Issue:** [#158](https://github.com/Open-Athena/MarinFold/issues/158) · **Kind:** `document_structures` · **Branch:** `exp158-contacts-v1-retract-token`

## Question

Can the contacts-v1 document format express **retraction** — a statement that a previously emitted `<contact>` is wrong — so that models can later be trained to self-correct partway through a rollout?

## Hypothesis

A single new statement type, `<retract> <pX> <pY>`, is sufficient to express backtracking, and it can be added without disturbing existing corpora or checkpoints: the generator's output stays **byte-identical**, and appending the new token to the end of the vocab leaves every existing token ID unchanged.

## Background

contacts-v1 is today **strictly additive**: every `<contact> <pX> <pY>` asserts a true contact, absence means "no contact", and there is no way to take one back. The structure section is also deliberately **order-free** — contacts are selected by degree but then shuffled, and each pair's orientation is coin-flipped, so the model does not learn a degree-sorted ordering.

- Spec: `marinfold/marinfold/document_structures/contacts_v1/SPEC.md`
- Vocab: `.../contacts_v1/vocab.py` — 5 native tokens (`<contacts-v1>`, `<n-term>`, `<c-term>`, `<contact>`, `<think>`) + trailing `<contacts-v1.sequence_only>`
- Inference: `.../contacts_v1/inference.py` (`_rollout_score_matrix`)

**Why now.** The settled rollout recipe (#82) commits to each contact irrevocably — a bad early emission stays in the document and costs precision for the rest of the rollout. #142 pointed at the model's inability to revise rather than its stopping rule. Retraction gives it a recovery mechanism. Precedent for the change shape is #123 (`<think>` tokens): a format extension that keeps existing output byte-identical.

This issue is **format only** — it makes the token, the semantic fold, and the inference readout ready. The backtracking corpus (#159) and training/eval (#160) build on it.

## Approach (as built)

Library change in `marinfold/marinfold/document_structures/contacts_v1/`, plus the sibling coordinate format. No experiment-specific launchable scripts — validation is the package test suite.

### 1. Vocab (`vocab.py`)

Mint one token, `<retract>`, as its own trailing group **appended last** (after `<contacts-v1.sequence_only>`), mirroring the sequence-only precedent. It is kept out of `NATIVE_TOKENS` because that list sits *before* the inherited contacts-and-distances-v1 block — adding to it would shift every downstream ID. Appended last, every pre-existing ID (including the sequence-only token's) is unchanged; a checkpoint only grows its embedding by one row. New helpers: `RETRACT_TOKEN`, `RETRACT_TOKENS`, `retract_tokens()`.

### 2. Coordinate-format ID stability (`contacts_and_coordinates_v1/vocab.py`)

The coordinate/crops format inherits contacts-v1's vocab as its leading block and appends 1001 coordinate tokens after — so a naive append to contacts-v1 would shove every coordinate ID up by one and break published coordinate checkpoints (exp105/130/132). Fixed by having its `inherited_tokens()` **exclude** the retraction group (filter, not a fixed slice), pinning the inherited block at the pre-retraction 2844 tokens. The coordinate format has no retraction, so no coordinate ID moves.

### 3. The fold — the semantic contract (`read.py`, new module)

The structure section is read as an ordered **edit list**: `<contact>` adds the canonical (sorted-position) pair, `<retract>` removes a live pair, and the contact set is whatever is **live at `<end>`**. A retract may reference a contact emitted arbitrarily far back. Pure (regex + a set fold), no pyconfind/torch. Public: `live_contacts(text)`, `fold_statements(...)`, `iter_structure_statements(text)`, `FoldResult`. Malformed edit lists (retract of an absent pair, double retract, re-emit, redundant contact) are tolerated for inference robustness but **counted** in `FoldResult`, so authored corpora (#159) can assert they never occur. A document with no `<retract>` folds to exactly its emitted set — a no-op for every existing document/model. (The doc→contacts fold lives here, not in `parse.py`, which is the pyconfind structure-analysis layer.)

### 4. Generator (`generate.py`) — deliberately untouched

No retraction-injection path was added. The backtracking corpus (#159) is synthesised by a model-in-the-loop engine that emits the edit stream directly, and `build_document` only ever holds ground-truth contacts, so an injection path here would be speculative. Leaving `generate.py` unchanged makes the byte-identical guarantee trivially true. *(This trims the originally-filed §4; noted on the issue.)*

### 5. Inference (`inference.py`)

`_rollout_score_matrix` now folds each rollout completion via `read.live_contacts` and votes only the live contacts. Since current models never emit `<retract>`, this is byte-identical to the old regex scan and is ready for retraction-trained models. `_CONTACT_RE` (and the `re` import) were removed in favour of `read.py`.

### 6. SPEC.md

Added a *Retraction* subsection (fold semantics, orientation, malformed handling, vocab/append-only, generator untouched) and a note in *Structure section* that the "no meaningful ordering" property is relaxed within a retraction-bearing document.

## Success criteria

- **Byte-identical when off:** ✅ `generate.py` untouched (`git diff main` empty for it); all generation/think/sequence-only tests pass.
- **Token IDs unchanged:** ✅ `test_retract_token_is_appended_last` / `test_trailing_tokens_take_the_final_ids_only`; contacts-v1 doc type still id 2, c-and-d block still starts at id 7, sequence-only id unchanged, `<retract>` takes the final id.
- **Coordinate IDs unchanged:** ✅ `test_inherited_block_is_contacts_v1_minus_retraction` (inherited stays 2844); coordinate total still 3845.
- **Round-trip / fold:** ✅ `test_read.py` — contact/retract fold, orientation canonicalisation, re-emit, empty section.
- **Long-distance fold:** ✅ `test_long_distance_retract`.
- **Malformed handling:** ✅ counted + tested (`test_read.py`).
- **Inference honors retraction:** ✅ `test_rollout_honors_retraction` — a retracted pair collects no votes.

## Results

Implemented on branch `exp158-contacts-v1-retract-token`. Files: `vocab.py`, `read.py` (new), `inference.py`, `SPEC.md`, `contacts_and_coordinates_v1/vocab.py`; tests `test_read.py` (new), `test_vocab.py`, `test_inference.py`, `contacts_and_coordinates_v1/test_vocab.py`. Full marinfold suite: **298 passed, 7 skipped**. Vocab grows 2846 → 2847 (contacts-v1 tokenizer only).

## Conclusion

Yes — one appended token plus a fold parser is sufficient to express retraction, with zero disruption to existing contacts-v1 or coordinate-format checkpoints (append-only; embedding grows by one row). The format, the semantic fold (`read.py`), and the inference readout are ready for the backtracking corpus (#159) and training (#160). Generation of retraction documents is intentionally left to #159's model-in-the-loop engine.
