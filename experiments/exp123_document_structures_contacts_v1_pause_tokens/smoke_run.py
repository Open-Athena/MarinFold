# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp123 smoke run — contacts-v1 ``<think>`` (pause) token generation.

Mirrors #34's deliverable. Two parts:

1. **Statistical validation** (self-contained, no pyconfind): generate many
   think-augmented documents from synthetic chains via ``build_document`` and
   check the empirical ``<think>`` statistics against the spec (initial-run
   gate ~ 0.75; k1 ~ Geometric(0.13) mean ~ 7.69; extra-run count mean ~ 0.75;
   run length ~ Geometric(0.25) mean ~ 4), plus the hard invariants (every doc
   fits the budget and ends with ``<end>``; ``<think>`` runs appear only
   between ``<contact>`` statements, never inside one; ``think=False`` emits
   none).

2. **Real-generator demo** (if pyconfind + a bundled test CIF are available):
   run the actual ``generate_document`` path on ``1QYS`` with ``think`` off vs
   on to show a real think-augmented ``<contacts-v1>`` document.

Writes ``data/think_stats.csv`` and ``data/example_think_document.txt``.
Run:  ``python smoke_run.py``
"""

import csv
import random
import statistics
from pathlib import Path

from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    build_document,
    _generation_seed,
    _sample_think_overhead,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, ResidueInfo
from marinfold.document_structures.contacts_v1.vocab import CONTEXT_LENGTH

_AA = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]
DATA_DIR = Path(__file__).parent / "data"


def _synthetic(n_res: int, n_contacts: int, seq_sep: int = 6):
    residues = [
        ResidueInfo(seq_index=i, resname=_AA[i % len(_AA)], resnum=1 + i, chain="A")
        for i in range(n_res)
    ]
    # Deterministic spread of contacts respecting the default seq-sep, with
    # descending degrees so selection-by-strength has something to do.
    contacts = [
        RawContact(i, i + seq_sep + (i % (n_res - seq_sep - 1)), float(n_contacts - i))
        for i in range(n_contacts)
        if i + seq_sep + (i % (n_res - seq_sep - 1)) < n_res
    ]
    return residues, contacts


def _think_runs(document: str) -> list[int]:
    """Lengths of maximal ``<think>`` runs in the structure section; asserts
    every run is followed by ``<contact>`` (or the section end)."""
    toks = document.split()
    struct = toks[toks.index("<begin_statements>") + 1: toks.index("<end>")]
    runs, i = [], 0
    while i < len(struct):
        if struct[i] != "<think>":
            i += 1
            continue
        j = i
        while j < len(struct) and struct[j] == "<think>":
            j += 1
        nxt = struct[j] if j < len(struct) else "<contact>"
        assert nxt == "<contact>", f"think run split a statement (followed by {nxt!r})"
        runs.append(j - i)
        i = j
    return runs


def statistical_validation(n_docs: int = 10_000) -> dict:
    cfg = GenerationConfig(think=True)
    n_with_initial = 0
    initial_lengths: list[int] = []
    extra_run_counts: list[int] = []
    extra_run_lengths: list[int] = []
    total_think = 0
    n_think_docs = 0

    for d in range(n_docs):
        entry = f"exp123-{d}"
        residues, contacts = _synthetic(60, 40)
        res = build_document(entry, residues, contacts, config=cfg)

        # Hard invariants.
        toks = res.document.split()
        assert res.num_tokens <= CONTEXT_LENGTH
        assert toks[-1] == "<end>"
        assert toks.count("<think>") == res.think_tokens
        _think_runs(res.document)  # placement contract

        # Recover the sampled overhead for this doc's seed (think is sampled
        # first, so replaying _sample_think_overhead on a fresh RNG matches).
        rng = random.Random(_generation_seed(entry))
        k1, extra = _sample_think_overhead(rng, cfg)
        if k1 > 0:
            n_with_initial += 1
            initial_lengths.append(k1)
        extra_run_counts.append(len(extra))
        extra_run_lengths.extend(extra)
        total_think += res.think_tokens
        if res.think_tokens:
            n_think_docs += 1

    # think=False emits none.
    off = build_document("exp123-off", *_synthetic(60, 40), config=GenerationConfig())
    assert "<think>" not in off.document.split() and off.think_tokens == 0

    return {
        "n_docs": n_docs,
        "initial_gate_rate": n_with_initial / n_docs,
        "initial_run_mean_len": statistics.mean(initial_lengths),
        "extra_run_count_mean": statistics.mean(extra_run_counts),
        "extra_run_len_mean": statistics.mean(extra_run_lengths),
        "docs_with_any_think": n_think_docs / n_docs,
        "mean_think_per_doc": total_think / n_docs,
    }


def real_generator_demo():
    """Run the real pyconfind-backed generator on a bundled CIF if available."""
    try:
        import pyconfind  # noqa: F401
        from marinfold.document_structures.contacts_v1.generate import generate_document
    except ImportError:
        return None
    cif = Path(__file__).resolve().parents[2] / "marinfold" / "tests" / "data" / "1QYS.cif"
    if not cif.exists():
        return None
    off = generate_document(str(cif), config=GenerationConfig())
    on = generate_document(str(cif), config=GenerationConfig(think=True))
    return off, on


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("=== Statistical validation (10k synthetic think documents) ===")
    stats = statistical_validation()
    targets = {
        "initial_gate_rate": ("~0.75", 0.75),
        "initial_run_mean_len": ("~7.69 (1/0.13)", 1 / 0.13),
        "extra_run_count_mean": ("~0.75", 0.75),
        "extra_run_len_mean": ("~4.0 (1/0.25)", 4.0),
    }
    for k, v in stats.items():
        tgt = f"   [spec {targets[k][0]}]" if k in targets else ""
        print(f"  {k:24s} = {v:.4f}{tgt}")

    with open(DATA_DIR / "think_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "empirical", "spec_target"])
        for k, v in stats.items():
            w.writerow([k, f"{v:.4f}", targets.get(k, ("", ""))[0]])
    print(f"  wrote {DATA_DIR / 'think_stats.csv'}")

    print("\n=== Real generator demo (1QYS via pyconfind) ===")
    demo = real_generator_demo()
    if demo is None:
        print("  skipped (pyconfind or test CIF unavailable)")
    else:
        off, on = demo
        print(f"  1QYS: {off.seq_len} residues, {on.contacts_emitted} contacts emitted")
        print(f"  think=False: {off.num_tokens} tokens, {off.think_tokens} think")
        print(f"  think=True : {on.num_tokens} tokens, {on.think_tokens} think, "
              f"fits={on.num_tokens <= CONTEXT_LENGTH}, ends={on.document.split()[-1]}")
        (DATA_DIR / "example_think_document.txt").write_text(on.document + "\n")
        print(f"  wrote {DATA_DIR / 'example_think_document.txt'}")

    print("\nSmoke run OK.")


if __name__ == "__main__":
    main()
