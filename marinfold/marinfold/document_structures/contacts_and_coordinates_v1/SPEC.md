# spec: contacts-and-coordinates-v1

This document defines a spec for the *contacts-and-coordinates-v1* document
type. It is the input to a coding agent that will do the implementation.

contacts-and-coordinates-v1 extends [contacts-v1](../contacts_v1/SPEC.md):
same sequence section, same style of `<contact>` statements, plus a new
**coordinate section** that specifies 3D atom positions using a coarse-to-fine
tokenized encoding. The point of the format is to teach a model to both read
and produce a *soft*, progressively-refinable representation of atomic
coordinates — coarse bins early, finer bins later, with enough built-in noise
that the model isn't forced into all-or-nothing precision.

## Example document

An 8-residue chain (the same toy chain used in the contacts-v1 README),
with 2 contacts and a handful of coordinate statements. Shown one statement
per line; the real document is one space-separated token string.

```
<contacts-and-coordinates-v1>
<begin_sequence>
<p23> <PHE>
<n-term> <p20>
<p26> <LYS>
<p25> <THR>
<p27> <VAL>
<p21> <ALA>
<p22> <GLY>
<c-term> <p27>
<p20> <MET>
<p24> <SER>
<begin_statements>
<contact> <p21> <p27>
<contact> <p20> <p26>
<p26> <CA> <xyz-129> <xyz-360> <xyz-883> <xyz-749>
<p20> <CB> <xyz-602> <xyz-188>
<p24> <CA> <xyz-205>
<p22> <N> <xyz-079> <xyz-412> <xyz-098>
<p24> <CA> <xyz-105> <xyz-741> <xyz-552> <xyz-337>
<end>
```

The very first coordinate statement (`<p26> <CA>`) is always full precision
— see *Mention scheduling* for why. `<p24> <CA>` is then mentioned twice:
first at depth 1 (`<xyz-205>`), where noise happened to push its
x-coordinate across the 100/200 Å boundary (true hundreds digit is `1`, not
`2`); a later depth-4 mention (`<xyz-105> <xyz-741> <xyz-552> <xyz-337>`)
reads the corrected, full-precision value. See *Noise model* below for the
exact arithmetic behind this example.

## Details

### Sequence section

Byte-for-byte the same as contacts-v1's sequence section: `<pXXX> <RESIDUE>`
/ `<n-term> <pXXX>` / `<c-term> <pXXX>` statements in random order, residues
numbered from a random wrap-around start index over `<p0>`–`<p1999>`. See
[contacts-v1's SPEC](../contacts_v1/SPEC.md#sequence-section) and
[Residue indexing](../contacts_v1/SPEC.md#residue-indexing) — reused
verbatim, including the single-chain-only restriction (multi-chain is
future work for contacts-v1 and stays future work here).

### Structure section

Starts with `<begin_statements>`, same as contacts-v1. It has two parts, in
a fixed order: first all `<contact>` statements, then all coordinate
statements. (Contacts first because they're cheap and few; coordinates are
what fill out the rest of the document. See *Document length* below.)

A statement's type is unambiguous from its first token: `<contact>` starts a
contact statement, a position token `<pXXX>` starts a coordinate statement.
No extra section-marker token is needed between the two parts.

#### Contacts

Same contact definition as contacts-v1 (pyconfind `native_only=True`
contact degree, `min_seq_separation=6`, `min_contact_degree=0.001` — see
[contacts-v1's SPEC](../contacts_v1/SPEC.md#structure-section) for the full
derivation). The difference is *selection*: contacts-v1 takes the
*strongest* N contacts to fill the budget; here contacts are just a small
conditioning hint, so we take a **uniform random sample**:

- With probability 0.3, N = 0 (no contacts at all).
- Otherwise, N ~ Uniform{1, ..., min(50, num_eligible_contacts)}, and the N
  contacts are a uniform random sample (not strongest-first) from the
  above-threshold pool.

Why 50 as the cap, and why contacts don't compete with coordinates for
budget: each `<contact> <pX> <pY>` statement is 3 tokens, so even 50
contacts is only 150 tokens — under 0.5% of the 32768-token budget. The
upper bound is a modeling choice (how much contact conditioning to expose),
not a length constraint. Coordinates below are what actually fill the
document.

The N selected contacts are listed in random order, and (as in contacts-v1)
each pair's two positions are coin-flipped independently.

#### Coordinates

##### The `<xyz-DDD>` vocabulary

1000 tokens, `<xyz-000>` through `<xyz-999>`, minted fresh (no prior-format
analog). A token's three digits are the (x, y, z) digit *at one decimal
place*, jointly: the first digit is x's, the second is y's, the third is
z's, all at the same place.

We use four decimal places — hundreds, tens, ones, tenths — since real
protein coordinates (in a bounded frame; see below) look like `205.3`. A
coordinate mention reveals a **prefix** of these four places, always
starting from hundreds: depth 1 = hundreds only (1 token), depth 2 =
hundreds+tens (2 tokens), depth 3 = +ones (3 tokens), depth 4 = +tenths (4
tokens, full precision at our 0.1 Å resolution). A mention is never just a
lone tens/ones/tenths digit with no hundreds token — the digits are always
a contiguous run from the coarsest place.

**Digit extraction — do this in integer space, not float division.**
Given true coordinate `v` (already inside `[0, 1000)`, see *Coordinate
frame*):

```
v10 = round(v * 10)                    # quantize once, to the nearest 0.1
hundreds_digit = (v10 // 1000) % 10
tens_digit     = (v10 // 100)  % 10
ones_digit     = (v10 // 10)   % 10
tenths_digit   = v10 % 10
```

Do **not** implement this as `floor(v / place) % 10` with `place` a float
like `0.1` — e.g. `180.2 / 0.1` evaluates to `1801.9999999999998` in IEEE-754
double precision, silently corrupting the tenths digit for values whose
representation rounds down instead of up. Scaling once by 10 to an integer
and then using only integer `//`/`%` avoids this entirely. (This is worth a
unit test with a value chosen to trigger the float bug, e.g. `180.2`.)

Worked check against the spec's own motivating example — position
`(205.3, 180.2, 5.7)`:

```
v10 = (2053, 1802, 57)
hundreds -> (2,1,0) -> <xyz-210>
tens     -> (0,8,0) -> <xyz-080>
ones     -> (5,0,5) -> <xyz-505>
tenths   -> (3,2,7) -> <xyz-327>
```

##### Coordinate frame

Real structure coordinates are signed and unbounded, but `<xyz-DDD>` only
covers `[0, 1000)` Å per axis. Every document therefore applies an affine
transform — **a random rotation, then a random translation** — before
quantizing:

1. **Rotate** all atom coordinates about the structure's centroid (mean
   position of all eligible heavy atoms — see *Atom eligibility*) by a
   rotation drawn uniformly from SO(3) (e.g. a normalized random quaternion,
   or `scipy.spatial.transform.Rotation.random`, seeded — see
   *Determinism*).
2. **Translate** so the rotated structure's bounding box lands at a
   uniformly random position inside `[margin, 1000 − margin)` per axis
   (default `margin = 10.0` Å), rather than always centering it. Compute
   the rotated bounding box `[lo, hi]` per axis; if `hi − lo > 1000 − 2 ×
   margin` the structure is too large to fit and is **skipped** (same
   philosophy as contacts-v1 skipping chains over 2000 residues — this
   should be rare; folded single-chain proteins essentially never approach
   a 980 Å span). Otherwise pick the placement offset uniformly at random
   within the available slack.

Both the rotation and the translation are random *and structure-invariant*
(they don't change any physical distance or contact), so this is free
data augmentation: every document exercises a different region and
orientation inside the cube, rather than the model latching onto
deposition-frame quirks (chain always "near the origin", always the same
orientation). This mirrors contacts-v1's motivation for the random
wrap-around residue-numbering start — expose the model to the whole token
range, not just whatever the input happened to use.

1 token-unit is fixed at 1 Å (no per-document scale factor) — scaling would
make the noise model below (calibrated in Å) meaningless without also
rescaling it per document.

After the frame transform, a noisy coordinate (see *Noise model*) is
clamped to `[0, 999.9]` before digit extraction, so a boundary atom never
produces an out-of-range digit.

##### Atom eligibility

All heavy atoms are eligible — reuse contacts-and-distances-v1's 37-name
`ATOM_NAMES` vocab and its `_vocab_safe_atoms`-style filtering (drop
hydrogens; drop any atom whose name isn't in the vocab). An eligible atom is
identified by its `(position, atom name)` pair, e.g. `<p26> <CA>` — the
sequence section already established the residue identity, so a coordinate
statement doesn't repeat the amino acid. Not every residue has every atom
name (e.g. glycine has no `<CB>`) — eligibility is whatever atoms actually
exist on that residue in the parsed structure.

##### Mention scheduling (coarse-to-fine, with variation)

The coordinate section is generated as a sequence of independent **mention
events**, each producing one statement `<pX> <ATOM> <xyz-...>+`. Events are
sampled and appended until the token budget runs out (see *Document
length*). Each event:

1. **Pick an atom** — uniformly at random from all eligible atoms, *with
   replacement* across events. The same atom can be (and often is) picked
   multiple times — that's how progressive refinement across the document
   happens. There's no bookkeeping requiring an atom's depth to increase
   monotonically across its repeated mentions, or requiring every atom to
   reach depth 4, or even to be mentioned at all — for large structures
   where the atom count exceeds the number of events the budget allows,
   most atoms will get zero or one mention; for small structures where the
   budget allows far more events than there are atoms, some atoms will
   accumulate multiple independent noisy mentions after already reaching
   full depth. Both are fine and intended — extra independent noisy
   mentions of an already-fully-specified atom are useful training signal
   (they reinforce that the noise is calibrated, not that the value
   changes), not wasted tokens.

2. **Pick a depth** (1–4). **Exception: the very first coordinate statement
   in the document (event index 0) always gets depth 4**, regardless of the
   schedule below. Before any coordinate has been given, the document's
   random rotation+translation frame (see *Coordinate frame*) is unknown to
   a reader — a lone coarse digit carries little information with no
   established anchor to interpret it relative to. Revealing one atom's
   exact position first gives everything mentioned afterward, at any depth,
   something concrete to be relative to. (The atom for this first event is
   still chosen the same way as any other — step 1 above, uniformly at
   random — only its depth is fixed; its token cost still counts toward the
   budget bookkeeping like any other event.)

   Every event after the first draws its depth from a distribution that
   shifts from shallow to deep as the coordinate section progresses, so the
   document *trends* hundreds → tens → ones → tenths overall, while still
   allowing an early deep mention or a late shallow one occasionally.
   Concretely: let `t = (coordinate tokens emitted so far) / (coordinate
   section token budget)`, a value in `[0, 1]` (already slightly above 0 by
   the second event, since the forced first event's own tokens count toward
   the numerator). Depths have centers `c = [0, 1/3, 2/3, 1]`; depth `d`'s
   raw weight is `max(0, 1 − 3·|t − c_d|) + ε` with `ε = 0.05` (a floor so
   no depth is ever impossible); normalize the 4 weights to sum to 1 and
   sample. At `t = 0` this puts 87.5% weight on depth 1, with the remaining
   12.5% split evenly across depths 2–4 (~4.2% each); the distribution
   shifts smoothly through depths 2 and 3 and lands symmetrically on depth 4
   (87.5% / 4.2% each elsewhere) by `t = 1`.

3. **Draw fresh noise and emit.** See *Noise model* — a new independent
   noisy coordinate is drawn for *this event*, and digits 1..depth are read
   off it. Because the noise is redrawn per mention rather than per atom,
   two mentions of the same atom can (rarely) disagree even in their
   coarsest digit — that's the mechanism behind the `<p24> <CA>` example's
   `<xyz-205>` → `<xyz-105>` hundreds-digit correction above.

The forced-depth-4 first event is a fixed rule, not a tunable. The
depth-schedule constants that govern every other event (`ε = 0.05`, kernel
centers, kernel width) are reasonable defaults, not load-bearing invariants
— fine to expose as knobs for future tuning.

##### Noise model

Per the motivating ask: when a statement's *finest* revealed digit is, say,
the hundreds place, that digit should be the true bin ~95% of the time when
the true position sits at the exact **center** of its bin (so the model
isn't trained to treat coordinate bins as rigid) — and a later, finer
mention is what's allowed to reveal that the coarse digit was actually off
by one bin.

For a mention at depth `d`, let `w_d` be the bin width at the finest
revealed place: `100, 10, 1, 0.1` Å for `d = 1, 2, 3, 4`. Draw isotropic
Gaussian noise independently per axis, `σ_d = max(w_d / 4, 0.1)` Å, add it
to the true (frame-transformed) coordinate **once per mention**, clamp to
`[0, 999.9]`, and extract all `1..d` digits from that single noisy value
(so a deep mention's coarser digits come from the same noisy draw as its
finest digit — since `σ_d` is small relative to the coarser bins' width,
they essentially never flip at depth ≥ 2).

Why `σ_d = w_d / 4`: for a true value sitting at the exact center of a bin
of width `w`, the nearest boundary is `w/2` away, and Gaussian noise of
scale `σ` stays inside it with probability `2Φ(w/(2σ)) − 1`. At `σ = w/4`
that's `2Φ(2) − 1 ≈ 95.45%` (verified numerically) — close to the 95%
target, and, unlike the previous uniform-position-average calibration this
replaces, it holds at that *same* ~95.45% for any bin width, since `w/σ` is
fixed at 4 regardless of scale. Concretely: `σ_1 = 25.0` Å, `σ_2 = 2.5` Å,
`σ_3 = 0.25` Å for `d = 1, 2, 3`.

`σ_4` breaks the pattern: `w_4 / 4 = 0.1 / 4 = 0.025` Å is clamped up to a
**floor of `0.1` Å**, since noise finer than that would be tighter than most
real structures' coordinates are even known to — false precision. The
floor means the tenths digit does *not* hit ~95%: at `σ_4 = 0.1`, a
bin-centered true value keeps its correct tenths digit only `2Φ(0.5) − 1 ≈
38.3%` of the time. This is an intentional tradeoff — the tenths digit is
deliberately the softest of the four, so the model isn't trained to trust
sub-0.1 Å precision as reliable just because the vocabulary can nominally
express it.

(Off-center true values — not sitting exactly at a bin's midpoint — are, in
general, nearer to *one* boundary than the center is, so their actual
stay-probability is somewhat lower than the center-of-bin figures above;
those figures are a best case, not an average.)

Worked example — the `<p24> <CA>` mentions in the document above, true
position `(175.0, 45.2, 512.6)` after the frame transform:

| mention | depth | σ (Å) | noisy draw | tokens |
|---|---|---|---|---|
| 1st (early, shallow) | 1 | 25.0 | `(202.80, 55.80, 538.52)` | `<xyz-205>` |
| 2nd (later, deep) | 4 | 0.1 | `(175.26, 45.25, 512.71)` | `<xyz-105> <xyz-741> <xyz-552> <xyz-337>` |

The first mention's noise draw happened to push x (175.0) up across the
100/200 boundary to 202.80, reading hundreds-digit `2` instead of the true
`1`; the second draw, at the much tighter depth-4 noise scale, reads all
four digits correctly. (Exact arithmetic per the digit-extraction formula
above, verified numerically while drafting this spec.)

### Document length

Context budget is 32768 tokens — larger than contacts-v1 /
contacts-and-distances-v1's shared 8192, since coordinates need far more
room than contacts ever did (see the coverage table below: even at 32768,
the largest supported chain only gets partial atom coverage). This is a
deliberate divergence from those two formats — token *vocabulary* is shared
(see *Additional tokens*) but context length isn't, and nothing about the
shared-embeddings story depends on the three formats using the same budget.
Costs:

- Frame: 4 tokens (doc-type, `<begin_sequence>`, `<begin_statements>`,
  `<end>`).
- Sequence section: `2L + 4` tokens for an `L`-residue chain (same as
  contacts-v1).
- Contacts: `3N` tokens for `N` contacts (`N ≤ 50`, so ≤ 150 tokens — see
  *Contacts* above for why this doesn't meaningfully compete for budget).
- Coordinates: whatever remains, `B = 32768 − 4 − (2L + 4) − 3N`. Mention
  events (2 tokens for `<pX> <ATOM>`, plus 1–4 for the `<xyz-*>` tokens, so
  3–6 tokens each) are sampled per *Mention scheduling* and appended until
  the next event wouldn't fit in `B`; that last partial event is dropped
  (truncation, same semantics as contacts-v1's budget truncation). If the
  frame + sequence section alone exceeds the budget, the protein is skipped
  with a warning (same as contacts-v1).

This is also the answer to "what should the contact-count upper bound be":
**50**, chosen because at 3 tokens/contact it's cheap regardless (well under
0.5% of budget) — the real budget pressure is entirely in the coordinate
section, so there was no length-based reason to pick a smaller or larger
contact cap. 50 gives reasonable variety in how much contact conditioning a
document can show.

Rough coverage this implies (heavy-atom count estimated at ~7.7 atoms/residue,
a typical average across the 20 canonical amino acids; simulated directly
from the *Mention scheduling* algorithm above, including the forced-depth-4
first event, using `E[N contacts] ≈ 18`):

| chain length L | ~atoms | events that fit | ~fraction of atoms touched ≥1x | avg mentions per touched atom |
|---|---|---|---|---|
| 50 | ~385 | ~7,430 | ~100% | ~19.3 |
| 300 | ~2,310 | ~7,320 | ~96% | ~3.3 |
| 1000 | ~7,700 | ~7,010 | ~60% | ~1.5 |
| 2000 | ~15,400 | ~6,570 | ~35% | ~1.2 |

Coverage is far more generous than an 8192 budget would allow: mid-size
chains (300 residues) are close to total coverage, and even the largest
supported chain (2000 residues, the *Residue indexing* cap) now gets
roughly a third of its atoms touched at least once, versus the ~6% an
8192-token budget would give it. (Numbers are simulation estimates for the
*default* parameters above, not load-bearing — they'll shift if the
defaults are tuned.)

## Additional tokens

New tokens minted by this format:

- `<contacts-and-coordinates-v1>` — the doc-type token.
- `<xyz-000>` … `<xyz-999>` — the 1000 coordinate tokens, in that numeric
  order.

That's it — 1001 new tokens total. Everything else (positions `<p0>` …
`<p1999>`, `<n-term>`, `<c-term>`, `<contact>`, amino acids, atom names,
`<begin_sequence>`, `<begin_statements>`, `<end>`, `<UNK>`, `<think>`, and
the trailing `<contacts-v1.sequence_only>`) is reused by carrying forward
contacts-v1's entire `all_domain_tokens()` list unchanged — so a model
trained on any one of the three formats shares embeddings with the others
and can be fine-tuned across them without a tokenizer change.

Unlike contacts-v1's own precedent (its 5 native tokens come *before* its
inherited contacts-and-distances-v1 block), this format puts the
**inherited contacts-v1 block first and its own 1001 native tokens last**:

```
all_domain_tokens() = [*contacts-v1's full domain vocab (2844 tokens),
                        *this format's own 1001 native tokens]
```

so that every one of contacts-v1's tokens keeps the *exact same numeric id*
it has in contacts-v1's own standalone tokenizer (both prepend
`<pad>`/`<eos>` at ids 0–1, so contacts-v1's tokens occupy ids 2–2845 in
either tokenizer). A pretrained contacts-v1 checkpoint's embedding matrix
can then be warm-started into a contacts-and-coordinates-v1 model by
appending 1001 new rows, rather than remapping every existing embedding to
a new id. Total domain vocab: 3845 tokens (3847 with `<pad>`/`<eos>`).

## Suggested default parameters

| Parameter | Default | Notes |
|---|---|---|
| `context_length` | 32768 | larger than contacts-v1 / contacts-and-distances-v1's 8192 — coordinates need the room |
| `min_seq_separation` | 6 | reused from contacts-v1's contact definition |
| `min_contact_degree` | 0.001 | reused from contacts-v1's contact definition |
| `n_contacts_zero_prob` | 0.3 | P(N = 0) |
| `n_contacts_max` | 50 | see *Document length* |
| `cube_size` | 1000.0 Å | the `<xyz-DDD>` coordinate range per axis |
| `cube_margin` | 10.0 Å | placement margin; structures needing more than `cube_size − 2×margin` Å span in any axis are skipped |
| `noise_divisor` (`w_d / σ_d`) | 4 | gives ~95.45% bin-center reliability at a mention's finest digit, for `d = 1, 2, 3` |
| `noise_sigma_floor` | 0.1 Å | minimum noise stdev; overrides the divisor at `d = 4` (drops that level's bin-center reliability to ~38%) |
| `depth_kernel_epsilon` | 0.05 | floor weight in the depth-scheduling kernel, keeps all depths always possible |
| `force_full_precision_first_event` | true | the document's very first coordinate statement always gets depth 4 |

## Open questions / future work

- **Multi-chain.** Same restriction as contacts-v1 — single chain only, for
  now.
- **Atom-level contacts.** Contacts remain residue-level, exactly as in
  contacts-v1 (pyconfind contact degree between residues); this spec doesn't
  define atom-level contacts, even though coordinates are now atom-level.
- **Metadata schema.** Not pinned down here (contacts-v1's own SPEC also
  left this to implementation) — expect it to mirror contacts-v1's metadata
  table plus coordinate-specific stats (rotation/translation parameters,
  number of mention events, number of distinct atoms mentioned, a histogram
  of depths reached, `truncated`).
- **RNG draw order.** Should be pinned down at implementation time and
  recorded (contacts-v1's *Determinism* note is the template): e.g.
  residue start index → sequence-section shuffle → rotation quaternion →
  translation offset → contact sample + shuffle + pair-order flips →
  mention-event sequence (atom choice, depth, noise draws, in event order).
