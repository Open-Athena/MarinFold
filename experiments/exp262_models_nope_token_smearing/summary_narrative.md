## exp262: does contacts-v1 need RoPE?

Issue [#262](https://github.com/Open-Athena/MarinFold/issues/262). contacts-v1
documents are self-describing and order-randomized — every statement carries its
own `<pXXX>` coordinate and statement order is uniform noise — so the proposal is
to delete RoPE and replace it with a width-3 causal *token smear*: mix the
previous two tokens' embeddings into the current one, as in the nanogpt
speedrun's smear module.

Phase 0 asks the trained default checkpoint whether there is anything to win,
before any training is bought. Phase 1 is a 3-arm ablation at the exp232 1.5B
architecture on a reduced token budget.

## Two is the tight window

Enumerating the contacts-v1 grammar: a token's (statement form, slot) role is
NOT a function of the previous 1 token class — `(POS, POS)` is ambiguous between
a contact's second argument and a residue statement's head — and IS a
deterministic function of the previous 2. Lookback 3 adds nothing. Width-3 smear
is the tight bound.

## What Phase 0 found

Three probes on `contacts-v1-exp199-cooldown-1.5B`, teacher-forced over
ground-truth documents from the exp245 monomers, on a local A5000.

**The smear half is directly motivated.** Layer 1 holds two nearly pure
previous-token heads (0.999 and 0.996 of their attention at offset 1) and a third
splitting 0.75/0.16 across offsets 1 and 2 — a width-3 smear implemented in
attention, at the cost of three heads.

**The NoPE half loses its mechanism but keeps its premise.** Co-referent
retrieval is already distance-uniform out to 2048 tokens, so RoPE is not costing
us long-range reach and the long-protein story is dead. But randomizing every
exact cross-statement distance costs *less* than deterministically rescaling them
at matched stretch, so the model genuinely is not reading those distances.

**What position is actually for: counting.** Keeping the position range fixed but
letting two tokens share an id costs +1.23 nats, ten times any
distance-randomizing intervention. The model uses position as an index, which is
exactly what NoPE would take away — and exactly what the pre-registered
stopping-behaviour guardrail was written to catch.

## Where it stands

Phase 0 gate passed. Arm B (RoPE + smear) is promoted from safe hedge to the arm
the evidence points at; arm C (NoPE + smear) proceeds with its risk localized;
arm D (interleaved NoPE) is deferred as the most invasive build hedging the
weakest hypothesis.
