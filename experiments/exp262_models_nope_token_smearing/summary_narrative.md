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

## The pilot: the two changes only work together

15M-parameter twins on 150M tokens of the real decontaminated corpus, three seeds
per arm at each arm's own best learning rate: smear alone buys 0.034 nats, NoPE
alone COSTS 0.085, and together they buy 0.157 — an interaction of 0.208. The
gain is entirely in the structure section, the shuffled bag where the theory said
it should land. Note this overturns the Phase 0 reading, which had the smear as
the safe half and NoPE as the speculative one.

## Where it stands

Two full-budget 1.5B runs are training on 64 H100s: exp232's usual setup against
NoPE + smear, both at p06, differing in exactly one thing. At 10% of the schedule
the new arm leads by 0.0156 nats on eval and the control is reproducing exp232 to
0.005 nats, which is what makes the gap readable.

Loss is not the deliverable. An accuracy claim needs a rollout R-precision eval,
and that needs an HF exporter first — neither the NoPE config nor the smear
weights have an HF Qwen3 representation.
