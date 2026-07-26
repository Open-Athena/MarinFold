# contacts-and-crops-v1

A coordinate-bearing protein-document format that fits the **8192-token**
context of the contacts formats (vs
[contacts-and-coordinates-v1](../contacts_and_coordinates_v1/) (ccoord),
which uses 32768). It emits contacts-v1's sequence section and `<contact>`
statements verbatim, then a bounded **two-pass coordinate section**.

The authoritative definition is [`SPEC.md`](SPEC.md). This README is the
quick orientation.

## The document

```
<contacts-and-crops-v1>
<begin_sequence>
  … contacts-v1's sequence section, verbatim …
<begin_statements>
  <contact> <pX> <pY>                 # a small uniform-random sample (0..50)
  <pX> <ATOM> <xyz-HHH> <xyz-TTT>     # Pass 1: every atom → its 10 Å box
  <crop> <xyz-HHH> <xyz-TTT>          # Pass 2: a box, named exactly …
  <pX> <ATOM> <xyz-OOO> <xyz-TTT>     #   … then its atoms at 0.1 Å (ones+tenths)
<end>
```

**Pass 1 (coarse boxes)** hands every atom a cheap 4-token mention placing it
in its 10 Å box (sampled with replacement, weighted to spread across
residues, σ=2 Å box noise), budget-truncated for large chains. **Pass 2
(crops)** spends a reserved ~2000-token budget revealing full 0.1 Å detail
inside a handful of selected boxes: each `<crop>` names a box exactly, then
lists the atoms that fall in it (neighbor bleed-in), with a box's noise
shrinking (σ = 1/(i+1)²) each time it's re-shown so repeated reads sharpen it.
Small/mid proteins reach essentially full 0.1 Å structure inside 8k; large
proteins get a coarse sketch plus scattered fine spot-checks. Positions live
in a random rotated + translated frame inside a 1000 Å cube (free data
augmentation — no physical distance changes).

## Vocabulary

Carries forward **contacts-v1's entire `all_domain_tokens()` list unchanged
and first** (every inherited id byte-stable), then this format's native
block: **exactly ccoord's block (doc-type `<contacts-and-crops-v1>`, then
`<xyz-000>`…`<xyz-999>`) with `<crop>` appended last**. So the `<xyz-DDD>`
ids match ccoord's exactly and the format warm-starts from a
contacts-v1/ccoord checkpoint by appending as little as one row (`<crop>`).
Total domain vocab: 3846 tokens.

## Code surface

- `vocab.py` — the token list, `all_domain_tokens()`, `xyz_token(...)`.
- `parse.py` — `analyze_coordinates(...)`: residues + pyconfind contacts +
  per-atom heavy-atom coordinates, aligned by `(chain, resnum)` (reused from
  ccoord).
- `generate.py` — `generate_document(...)` / `build_document(...)`: the
  deterministic frame transform, the two-pass box/crop scheduler, and the
  noise model.
- `cli.py` — `generate` / `view` / `tokenizer` subcommands.

```bash
# Eyeball a document
python -m marinfold.document_structures.contacts_and_crops_v1.cli \
    view --input tests/data/1QYS.cif
```

Generation needs the `contacts-v1` extra (pyconfind); the `tokenizer`
subcommand does not. Single-chain only, same as contacts-v1.
