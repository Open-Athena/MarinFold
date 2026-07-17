# spec: contacts-and-crops-v1

This document defines the *contacts-and-crops-v1* document type. It is the
authoritative implementation target for issue
[#130](https://github.com/Open-Athena/MarinFold/issues/130).

**The question.** Can we get a *coordinate*-bearing document format that
stays inside an **8192-token** context (like contacts-v1 /
contacts-and-distances-v1), instead of
[contacts-and-coordinates-v1](../contacts_and_coordinates_v1/SPEC.md)
(ccoord, #105/#106) using 32768? The idea: give **every** atom a cheap
coarse 10 Å box first (Pass 1), then spend a small reserved budget revealing
full 0.1 Å detail inside a handful of selected spatial **crops** (Pass 2).

**Why.** ccoord's coordinate section fills the budget with coarse→fine
mention events *with replacement*, so every doc runs to ~32k tokens
regardless of chain length. That's expensive to train and can't be mixed 1:1
with the 8k contacts formats. This format keeps ccoord's sequence + contact
sections and its `<xyz-DDD>` vocabulary, but replaces the coordinate section
with a bounded two-pass scheme designed for 8k. Small/mid proteins get
essentially full structure at 0.1 Å; large proteins get a graceful coarse
sketch plus scattered fine spot-checks. Shared vocab / warm-start is
preserved (see *Additional tokens*).

## Sequence section

Byte-for-byte ccoord/contacts-v1: `<pXXX> <RESIDUE>` / `<n-term> <pXXX>` /
`<c-term> <pXXX>` statements in random order, residues numbered from a random
wrap-around start index over `<p0>`–`<p1999>`. Single chain only (same
restriction as contacts-v1).

## Coordinate frame

Identical to ccoord: rotate all eligible heavy atoms about their centroid by
a seeded uniform SO(3) rotation, then translate so the bounding box lands at
a uniformly random position inside `[margin, 1000−margin)` per axis
(`margin = 10`, `cube_size = 1000` Å). Structures whose span exceeds
`1000−2·margin` in any axis are skipped (never happens for real single
chains). 1 token-unit = 1 Å. This is free data augmentation: no physical
distance changes.

## Structure section

Opens with `<begin_statements>`. Three parts in fixed order: **contacts**,
then **Pass 1 (coarse boxes)**, then **Pass 2 (crops)**. A statement's type
is unambiguous from its first token (`<contact>` / `<pXXX>` / `<crop>`).

### Contacts

Unchanged from ccoord: pyconfind `native_only=True` contact degree,
`min_seq_separation=6`, `min_contact_degree=0.001`. With probability `0.3`,
N=0; else N ~ Uniform{1..min(50, num_eligible)}, a **uniform random sample**
(not strongest-first), listed in random order, each pair's two positions
coin-flipped. 3 tokens/contact.

### The `<xyz-DDD>` vocabulary

Reused from ccoord (1000 tokens `<xyz-000>`..`<xyz-999>`), **place-agnostic**:
a token packs the (x, y, z) digits at *one* decimal place, and which place is
set by the token's position within its mention. Digits are extracted by
quantizing **once** as `n = round(v*10)` (a tenths-resolution integer) and
reading

```
hundreds = (n // 1000) % 10
tens     = (n //  100) % 10
ones     = (n //   10) % 10
tenths   =  n          % 10
```

Noisy coordinates are clamped to `[0, 999.9]` before extraction. (Never
divide by a float `0.1` — `180.2 / 0.1` is `1801.9999999999998` in IEEE-754
and would corrupt the tenths digit.) The **10 Å box index** of a coordinate
along one axis is `n // 100 = hundreds*10 + tens`, in `[0, 99]` — a fixed
100³ grid of boxes tiling the cube.

### Pass 1 — coarse boxes (all atoms, budget-truncated)

One atom per statement: `<pXXX> <ATOM> <xyz-HHH> <xyz-TTT>` — the hundreds and
tens tokens assign the atom to its **10×10×10 Å box** (4 tokens/atom; no
residue grouping).

- **Sampling (with replacement).** Draw atoms directly — atom `a`'s weight is
  `1/(1 + k_{r(a)})`, where `k_r` is the number of times residue `r` has
  already been sampled; increment `k_{r(a)}` after each draw. Sampling atoms
  (not residues) means bigger residues are proportionally more likely; the
  per-residue downweighting spreads coverage across residues as a
  probabilistic trend (not a guarantee). An atom may be redrawn — it just
  gets a second independent noisy box.
- **Box noise.** Before boxing, add fresh isotropic Gaussian noise
  **σ = 2.0 Å** per axis to the (frame-transformed) coordinate, once per
  mention, then quantize. At σ=2 the box is correct ~98.8% of the time at a
  cell's center, so a small fraction of atoms are emitted in the *wrong* box
  — deliberately (Pass 2 later shows them in the correct box).
- **Budget.** Draw until the next mention would exceed the Pass-1 budget =
  `structure_budget − fine_reserve` (`structure_budget = 8192 − frame −
  sequence − contacts`, `fine_reserve ≈ 2000`). Lay the drawn mentions out in
  draw order.

No forced full-precision anchor is needed (unlike ccoord): Pass 1 hands every
atom its hundreds digit, so absolute placement is anchored from the start.

### Pass 2 — crops (a few boxes at 0.1 Å, with progressive refinement)

Each crop: `<crop> <xyz-HHH> <xyz-TTT>` names a box **exactly** (its true
10 Å cell), then for each included atom `<pXXX> <ATOM> <xyz-OOO> <xyz-TTT>` —
**reuse-box**, ones + tenths only (4 tokens/atom); the header supplies
hundreds + tens.

- **Box selection (with replacement, 3-way).** Each step: **45%** the box of
  a uniformly random atom; **45%** a uniform pick from the *frontier* (boxes
  neighboring an already-shown box — 26-neighborhood — extended only by
  **occupied** shown boxes, so empties don't explore vacuum); **10%** re-show
  an already-shown box, chosen with probability **∝ how many times it has
  already been shown** (preferential attachment — concentrates refinement on
  a few boxes). Fall back to a random atom's box when the frontier / shown-set
  is empty.
- **Per-box refinement noise.** On a box's `i`-th appearance (`i = 0,1,2,…`),
  the atoms' noise is Gaussian **σ = 1/(i+1)² Å** → 1.0, 0.25, 0.111,
  0.0625, … So a box's first read is coarse (~1 Å; its tenths digit is
  near-noise) and repeated reads sharpen it toward a crisp tenths. This is
  *why* Pass 2 samples with replacement.
- **Membership via neighbor bleed-in.** For the chosen box `b`, consider
  every atom whose *true* box is `b` **or a neighbor of `b`**; draw that
  atom's σ-noise and **include it iff its noised position floors into `b`**
  (emit its ones + tenths from the *noised* position — automatically
  consistent with the header). A true-in-`b` atom that noise kicks out is
  dropped from this crop; a neighbor atom noise pulls in is included — and a
  later read of the neighbor box corrects it. Apply an independent **0.99
  keep** per candidate on top.
- **Empty crops allowed.** If nothing lands in `b`, emit just the header and
  move on.
- **Big boxes never skipped.** If a box's atoms overflow the remaining
  budget, emit the atoms that fit (partial last crop) rather than skipping —
  skipping would bias against dense boxes.
- Runs until the structure budget is spent. `<end>` terminates the document.

### Worked digit example

Frame-transformed true position of one atom: `(205.3, 71.8, 6.4)` Å (true box
= x∈[200,210), y∈[70,80), z∈[0,10) → header `<xyz-200> <xyz-070>`).

- **Pass 1** (σ=2 draw → `(206.1, 70.4, 7.9)`): box `<xyz-200> <xyz-070>`
  (the noise kept it in the true box; a larger draw could have flipped a
  digit).
- **Pass 2, first read of this box** (σ=1 draw → `(205.6, 72.3, 6.1)`):
  quantize each axis as `round(v*10)` → 2056 / 723 / 61 → ones token
  `<xyz-526>`, tenths token `<xyz-631>`. Full statement
  `<pXXX> <ATOM> <xyz-526> <xyz-631>`; with the header it reconstructs to
  `(205.6, 72.3, 6.1)`.
- A **later read** of the same box (σ = 0.25, 0.11, …) reveals a sharper
  (x, y, z), converging on the true value.

## Additional tokens

Two new tokens only: `<contacts-and-crops-v1>` (doc-type) and `<crop>`. The
tenths place reuses `<xyz-DDD>` (no new vocab). The vocabulary carries
contacts-v1's entire `all_domain_tokens()` first (byte-stable ids), then this
format's native block — **exactly ccoord's native block (doc-type, then the
1000 `<xyz-DDD>` tokens) with `<crop>` appended last**. This ordering is
load-bearing and chosen for warm-start:

- every contacts-v1 token keeps its id (2–2845), so a contacts-v1 checkpoint
  warm-starts by appending rows;
- every `<xyz-DDD>` token keeps the **exact id ccoord gives it** (2847–3846),
  so a ccoord checkpoint's coordinate embeddings transfer at their own ids;
- the doc-type token reuses ccoord's doc-type id slot (2846, a different
  string at the same id — benign on warm-start), and `<crop>` (3847) is the
  single genuinely new embedding row.

Total domain vocab: 3846 tokens (3848 with `<pad>`/`<eos>`).

## Suggested default parameters

| param | default |
|---|---|
| `context_length` | 8192 |
| `fine_reserve` | 2000 |
| Pass-1 box noise σ (`pass1_box_noise_sigma`) | 2.0 Å |
| Pass-1 sampling | atoms w/ replacement, weight `1/(1+k_r)` |
| Pass-2 select probs | 0.45 random / 0.45 frontier / **0.10 re-show ∝ prior count** |
| Pass-2 refine noise σ (`pass2_refine_noise_base`/(i+1)²) | base 1.0 Å |
| Pass-2 keep prob | 0.99 |
| neighborhood | 26 (face+edge+corner), occupied-only frontier extension |
| `min_seq_separation` / `min_contact_degree` | 6 / 0.001 |
| `n_contacts_zero_prob` / `n_contacts_max` | 0.3 / 50 |
| `cube_size` / `cube_margin` | 1000 / 10 Å |

## RNG draw order (load-bearing — do not reorder)

Seeded from the first 8 sha1 hex digits of `entry_id`:

1. residue start index
2. sequence shuffle
3. rotation quaternion (4 Gaussians)
4. translation offset (3 uniforms, x/y/z)
5. contact sample + shuffle + pair flips (zero-vs-nonzero coin drawn first,
   unconditionally)
6. **Pass-1 atom-draw sequence** — per draw: weighted atom choice (one
   uniform), then per-mention box-noise (x, y, z Gaussians)
7. **Pass-2 crop sequence** — per step: select-coin (one uniform, drawn
   unconditionally), box choice (random-atom uniform / frontier uniform /
   re-show weighted, by branch), then per-candidate membership-noise
   (x, y, z Gaussians) and, only when the noised position floors into the
   box, a keep uniform

## Metadata

Mirrors ccoord's frame table plus crop-specific stats: rotation quaternion,
translation, `num_pass1_mentions`, `num_crops`, `num_empty_crops`,
`num_distinct_crop_boxes`, `crop_atoms_emitted`, `max_box_visits`, the full
per-box visit histogram (`box_visit_counts`, in `summary_dict`), and
`truncated`.

## Coverage (measured on 42 real AFDB + CAMEO structures; see #130)

- **Atom density:** 7.85 heavy atoms/residue. An occupied 10 Å box holds mean
  ~22.7 atoms (p90 50, max 69).
- **Fits 8k for every protein up to the 2000-res cap**; the fine reserve
  guarantees ~15–20 crops regardless of size.
- ≤130 res: 100% residues, ~80–100% atoms boxed; whole protein resolvable at
  0.1 Å inside 8k.
- ~150–500 res: 100% residues boxed, 30–70% atoms; ~15–20 crops, 12–40% atoms
  at full precision.
- 500–2500 res: residue coverage degrades 100%→12%, ~15–20 crops always; a
  coarse sketch + scattered fine spot-checks.
- ~20–40% of crops are empty (surface shell).

## Implementation notes & discrepancies

Kept current as implementation and spec evolve (per repo convention).

- **`truncated` semantics.** Set to `True` only when a crop's atoms overflow
  the remaining budget and the crop is emitted partially (the SPEC's "big
  boxes never skipped" case). Pass 2 otherwise runs until no further crop
  *header* fits, which is a clean stop (not flagged), since Pass 2 samples
  boxes with replacement forever and would never "run out" on its own. So
  `truncated` marks a partial *last* crop, not merely that the budget was
  reached (which is the norm). This differs from ccoord, where `truncated`
  marks a dropped mention event.
- **First-read tenths signal.** On a box's first read (σ=1 Å, 20× the tenths
  half-width) the tenths digit is ~uniform noise; it only becomes crisp after
  several re-shows (σ ≤ ~0.025 needs `i ≳ 6`). This is intended
  (soft, integrate-many-reads) but is the parameter most worth watching in a
  pilot; it could later be gated on visit count with no vocab change.
- **Box index uses `round`, not `floor`.** The 10 Å box index is
  `round(v*10) // 100`, consistent with the digit tokens, so a value at a
  box boundary rounds to 0.1 Å first. "Floors into `b`" in the membership
  rule means "this same `round`-based box index equals `b`".
- **Frame / rotation / contact / digit helpers** are copied from ccoord (not
  imported) so the package is self-contained; they are behaviorally identical
  and covered by this package's own `test_digits_and_frame.py`.
- **Pass-1 fills its whole cap** even for small proteins (repeated coarse
  boxes of the same atoms), matching ccoord's "extra independent noisy
  mentions are useful signal, not wasted tokens" rationale.
