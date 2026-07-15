# contacts-and-coordinates-v1

A protein-document format that extends
[contacts-v1](../contacts_v1/): the **identical** sequence section and
`<contact>` statements, plus a **coordinate section** teaching a model to
read and produce a soft, progressively-refinable representation of 3D atom
positions.

The authoritative definition is [`SPEC.md`](SPEC.md). This README is the
quick orientation.

## The document

```
<contacts-and-coordinates-v1>
<begin_sequence>
  … contacts-v1's sequence section, verbatim …
<begin_statements>
  <contact> <pX> <pY>            # a small uniform-random sample (0..50)
  <pX> <ATOM> <xyz-…> …          # coordinate mention events, coarse-to-fine
<end>
```

A coordinate mention names a heavy atom by `(position, atom name)` — e.g.
`<p26> <CA>` — then reveals a **prefix** of its (x, y, z) digits: depth 1 =
hundreds only, up to depth 3 = hundreds+tens+ones (1 Å, full precision). Each
`<xyz-DDD>` token packs the x/y/z digit at one decimal place. Positions
live in a random rotated + translated frame inside a 1000 Å cube (free data
augmentation — no physical distance changes), and every mention carries
calibrated Gaussian noise so the model learns bins are soft, refined by
later, finer mentions. See `SPEC.md` for the noise calibration, the
depth schedule, and the forced full-precision first event.

## Vocabulary

Carries forward **contacts-v1's entire `all_domain_tokens()` list
unchanged and first** (so every inherited token id is byte-stable — a
contacts-v1 checkpoint warm-starts by *appending* rows), then this
format's **1001 native tokens last**: the doc-type token
`<contacts-and-coordinates-v1>` and `<xyz-000>` … `<xyz-999>`. Amino-acid,
atom-name, position, and `<contact>` tokens are all reused (emitted, not
re-minted). Total domain vocab: 3845 tokens.

## Code surface

- `vocab.py` — the token list, `all_domain_tokens()`, `xyz_token(...)`.
- `parse.py` — `analyze_coordinates(...)`: residues + pyconfind contacts +
  per-atom heavy-atom coordinates, aligned by `(chain, resnum)`.
- `generate.py` — `generate_document(...)` / `build_document(...)`: the
  deterministic frame transform, noise model, and mention scheduling.
- `cli.py` — `generate` / `view` / `tokenizer` subcommands.

```bash
# Eyeball a document
python -m marinfold.document_structures.contacts_and_coordinates_v1.cli \
    view --input tests/data/1QYS.cif
```

Generation needs the `contacts-v1` extra (pyconfind); the `tokenizer`
subcommand does not. Single-chain only, same as contacts-v1.
