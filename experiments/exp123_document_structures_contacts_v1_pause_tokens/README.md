---
marinfold_experiment:
  issue: 123
  title: "exp: generate contacts-v1 training dataset that includes pause tokens"
  kind: document_structures
  branch: claude/exp123-contacts-v1-pause-tokens
---

# exp: generate contacts-v1 training dataset that includes pause tokens

**Issue:** [#123](https://github.com/Open-Athena/MarinFold/issues/123) · **Kind:** `document_structures` · **Branch:** `claude/exp123-contacts-v1-pause-tokens`

## Question

Can we improve contact-prediction accuracy by adding `<think>` (pause) tokens at inference time to the **contacts-v1** document structure, assuming the model was trained on documents that contain them?

## Hypothesis

Same mechanism as #34: at inference time `<think>` tokens let the model spend extra compute before it has to commit to the next token, without conditioning on additional real tokens — potentially boosting accuracy. This issue is the **contacts-v1 analog of #34**, which did this for `contacts-and-distances-v1` (producing `contacts-and-distances-v2`).

## Background

- Original issue this adapts: #34 — "generate training dataset that includes pause tokens" (for `contacts-and-distances-v1`). The reference implementation of the think-token sampling lives in [`experiments/exp34_document_structures_contacts_and_distances_v2/generate.py`](https://github.com/Open-Athena/MarinFold/blob/main/experiments/exp34_document_structures_contacts_and_distances_v2/generate.py) (`_sample_think_overhead`, `_geometric`).
- Pause tokens: Goyal et al. 2023, https://arxiv.org/abs/2310.02226
- contacts-v1 spec: [`marinfold/marinfold/document_structures/contacts_v1/SPEC.md`](https://github.com/Open-Athena/MarinFold/blob/main/marinfold/marinfold/document_structures/contacts_v1/SPEC.md)

**Why contacts-v1 is a cleaner case than #34:** the contacts-v1 vocab **already reserves `<think>`** — it is one of the five native tokens (`vocab.py`), currently *"Unused by the generator"* / *"Reasoning scratch token reserved by SPEC.md"*. So, unlike #34 (which had to mint both a new doc-type token and `<think>`), this experiment needs **no vocab change and no new tokenizer**. Existing contacts-v1 checkpoints (e.g. the tuned 1.5B) stay token-compatible and can be continue-trained on think-augmented data.

## Approach

**Generation only**, exactly as #34 — running the generation at scale, and training on it, are separate follow-up experiments.

Extend the contacts-v1 generator so it can emit `<think>` tokens in the **structure section** (after `<begin_statements>`), positioned **between — never within —** `<contact>` statements. Everything else about the contacts-v1 document is unchanged: the shuffled sequence section, random wrap-around residue indexing, contact selection by degree, randomized statement order, coin-flipped pair order, the 8192-token budget, and the trailing `<end>`.

Think-token sampling uses the **same distributions as #34**:

- Immediately after `<begin_statements>`, with probability **0.75** include k1 `<think>` tokens, where **k1 ~ Geometric(p=0.13)** (support ≥ 1). The other 25% of the time there is no initial run.
- Additionally include **max(int(k2), 0)** further runs, where **k2 ~ Uniform([-4, 4])**, each placed at a random inter-statement slot (before a `<contact>` statement, never splitting one). Each run's length is sampled independently from **Geometric(p=0.25)**.
- Subtract the total `<think>` count from the 8192-token budget **before** choosing how many contacts fit, so documents still end with `<end>` and never overflow the context window.
- When we later train on these (a follow-up experiment), mask the loss on `<think>` tokens so they don't count toward the objective.

Example — the SPEC's example document, now with an initial 2-token run after `<begin_statements>` and one 1-token run before a later contact:

```
<contacts-v1> <begin_sequence> <p22> <PHE> <n-term> <p20> <p21> <ALA> <c-term> <p22> <p20> <ALA> <begin_statements> <think> <think> <contact> <p20> <p21> <think> <contact> <p22> <p21> <end>
```

**Notes for the implementer (differences from #34):**

1. **No new token / tokenizer.** Reuse the already-reserved `<think>`; keep the `<contacts-v1>` doc-type token (see the doc-type decision below).
2. **Uniform statements.** contacts-v1's structure section contains only `<contact> <pX> <pY>` statements (3 tokens each) — there are no distance statements and no in-statement pLDDT token — so "between statements" simply means "before a `<contact>`", and #34's think-before-pLDDT tie-break does not apply.
3. **Structure section only.** Place `<think>` runs only in the structure section (the predicted part), matching #34's "immediately after `<begin_statements>`". The shuffled sequence section is the given prompt and stays think-free. (Extending think tokens into the sequence section is possible future work, out of scope here.)
4. **Reuse #34's sampling code** (`_sample_think_overhead`, `_geometric`) so the two structures draw from identical distributions, and mirror #34's no-contacts edge case (emit the initial run even when the structure section is empty).

**Doc-type decision (confirmed):** We will **not** mint a new `<contacts-v2>` doc type. Instead, add a generation flag (e.g. `GenerationConfig(think=...)`) to the existing contacts-v1 generator so it emits think-augmented `<contacts-v1>` documents, reusing the already-reserved `<think>` token — **no vocab change**. This keeps one tokenizer and one doc type, and lets a mixed corpus (with/without think) train under the existing contacts-v1 checkpoints. (The alternative — mirroring #34 literally by minting a new `<contacts-v2>` doc-type token to signal think-mode via the doc type, at the cost of a one-token vocab append and a new structure name — was considered and rejected, since `<think>` was already reserved for exactly this purpose.)

## Success criteria

- The contacts-v1 generator can emit documents containing `<think>` tokens per the sampling above, reusing the reserved `<think>` token with **no tokenizer change**.
- Generated documents still fit in 8192 tokens and end with `<end>`; `<think>` runs appear only between `<contact>` statements in the structure section, never inside one.
- Empirical `<think>` statistics on a sample match the spec within sampling noise (initial-run gate ≈ 0.75; run lengths ≈ the specified geometric distributions).
- Tests + a smoke run, mirroring #34's deliverable.

## What was built

The think path is a **generation flag on the existing library generator**, not
a vendored copy (contacts-v1 now lives in the `marinfold` package, unlike #34
which predated that). Changes, all under
[`marinfold/marinfold/document_structures/contacts_v1/`](../../marinfold/marinfold/document_structures/contacts_v1/):

- **`generate.py`** — `GenerationConfig` gains `think: bool = False` plus the
  four #34 distribution knobs (`think_initial_prob=0.75`,
  `think_initial_geom_p=0.13`, `think_additional_count_range=(-4, 4)`,
  `think_run_length_geom_p=0.25`). Ported `_geometric` / `_sample_think_overhead`
  verbatim from #34. `build_document` samples think overhead first, subtracts it
  from the budget, and splices `<think>` runs between `<contact>` statements;
  `GenerationResult.think_tokens` records the count.
- **`cli.py`** — `--think` (+ `--think-*` overrides) on `generate` / `view`.
- **`vocab.py` / `SPEC.md` / `README.md`** — `<think>` moves from
  "reserved / unused" to "emitted when `think=True`"; SPEC gains a *Think
  (pause) tokens* section and the extended RNG-order note.

**No vocab or tokenizer change** — `<think>` was already reserved. **`think=False`
is byte-identical to the pre-think generator** (guarded by a test and by the
existing pinned-output tests), so existing corpora / checkpoints are unaffected.

## Results

Tests: **168 pass** in the contacts-v1 suite (144 pre-existing + 24 new in
`test_think.py` / `test_cli.py`). Smoke run (`python smoke_run.py`):

**Empirical `<think>` statistics** over 10,000 synthetic think documents match
the spec within sampling noise (see [`data/think_stats.csv`](data/think_stats.csv)):

| metric | empirical | spec |
| --- | --- | --- |
| initial-run gate rate | 0.7521 | ~0.75 |
| initial-run mean length | 7.58 | ~7.69 (1/0.13) |
| extra-run count mean | 0.7517 | ~0.75 |
| extra-run mean length | 3.96 | ~4.0 (1/0.25) |

Every generated document fit the 8192-token budget, ended with `<end>`, and had
its `<think>` runs strictly between `<contact>` statements (never inside one);
`think=False` emitted zero `<think>` tokens.

**Real generator** (`generate_document` on `1QYS` via pyconfind, 92 residues, 76
contacts): `think=False` → 420 tokens / 0 think; `think=True` → 451 tokens / 31
think, still fitting the budget and ending in `<end>`. Example think-augmented
document: [`data/example_think_document.txt`](data/example_think_document.txt).

## Conclusion

The contacts-v1 generator can now emit spec-conformant `<think>`-augmented
`<contacts-v1>` documents behind a flag, with no tokenizer change and no effect
on the default (non-think) output. All success criteria are met.

**Scope:** generation only, exactly as #34. The three follow-ups are separate
issues: (1) run the generation **at scale** to produce a think-augmented
contacts-v1 corpus; (2) **train** a contacts-v1 model on it with `<think>`
loss-masked; (3) **evaluate** whether inserting `<think>` tokens at inference
improves contact prediction vs. the current best contacts-v1 model.
