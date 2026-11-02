# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 4 -- can every eval protein's contact list even fit in the context?

exp82's rollout recipe gives each rollout ``6L + 128`` generation tokens, capped
at whatever is left of the model's 8,192-token window after the prompt. On the
554-protein set that never bound: the longest protein there is 761 residues and
exp82 saw 0 / 55,400 unfinished rollouts. The FoldBench monomers run to 1,596
residues, and the prompt grows with L while the budget shrinks, so the cap can
become the binding constraint rather than the model's willingness to stop.

This measures it before the job runs, with the real tokenizer and the real
document builder, rather than discovering it from truncated scores afterwards:

* ``prompt_tokens`` -- the sequence section the worker actually sends.
* ``max_tokens`` -- ``min(8192 - prompt_tokens, 6L + 128)``, the worker's rule.
* ``gt_statement_tokens`` -- the tokenized statement section of a document built
  from this protein's *true* contacts. A rollout that reproduced the ground
  truth exactly would need this many tokens, so it is the yardstick.

A protein whose ground truth does not fit is not a model failure and should not
be read as one. One monomer -- ``8uxt_A``, 1,596 residues -- is worse than tight:
``build_document`` itself truncates it, emitting 1,664 of its 3,809 contacts,
so the protein is not representable in contacts-v1 at an 8,192-token context at
all. Proteins in that state are marked ``scorable = 0`` here and are excluded
from the published eval inputs by ``publish_eval_inputs.py``, which names them
rather than dropping them quietly.

    uv run --extra gt python check_context_budget.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from marinfold.document_structures.contacts_v1 import (  # noqa: E402
    GenerationConfig,
    build_document,
)
from marinfold.document_structures.contacts_v1.parse import (
    RawContact,
    residues_from_sequence,
)
from marinfold.document_structures.contacts_v1.vocab import BEGIN_STRUCTURE_TOKEN

#: The worker splits the document here; same constant, same name it uses.
BEGIN_STATEMENTS = BEGIN_STRUCTURE_TOKEN

import upstream as U

DATA = U.DATA
REPORT = DATA / "context_budget.csv"
SUMMARY = DATA / "context_budget.summary.json"

#: The corpus tokenizer every contacts-v1 checkpoint carries.
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
CONTEXT = 8_192
CONTACT_MULT = 6
CONTACT_CONSTANT = 128


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", type=Path, default=DATA / "eval_sets.csv")
    parser.add_argument("--universe", type=Path,
                        default=DATA / "gt_universe_foldbench_monomers.jsonl")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    ground_truth = {}
    for line in args.universe.read_text().splitlines():
        record = json.loads(line)
        ground_truth[record["stem"]] = record

    rows = []
    sets = pd.read_csv(args.sets)
    for index, row in enumerate(sets.itertuples(), 1):
        record = ground_truth[row.stem]
        residues = residues_from_sequence(row.sequence)
        contacts = [RawContact(i, j, float(d)) for i, j, d in record["contacts"]]

        # The worker's prompt: a contact-free document truncated after the
        # statements marker. Realizations differ only in N-terminus and
        # statement order, and all share a prompt length.
        empty = build_document(row.stem, residues, [], config=GenerationConfig())
        prompt = empty.document[
            : empty.document.index(BEGIN_STATEMENTS) + len(BEGIN_STATEMENTS)]
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)

        full = build_document(row.stem, residues, contacts, config=GenerationConfig())
        truncated = bool(getattr(full, "truncated", False))
        statements = full.document[
            full.document.index(BEGIN_STATEMENTS) + len(BEGIN_STATEMENTS):]
        statement_tokens = len(tokenizer(statements, add_special_tokens=False).input_ids)

        max_tokens = min(CONTEXT - prompt_tokens,
                         CONTACT_MULT * record["L"] + CONTACT_CONSTANT)
        rows.append({
            "eval_set": row.eval_set, "stem": row.stem, "L": record["L"],
            "n_gt_contacts": len(contacts),
            "prompt_tokens": prompt_tokens,
            "recipe_budget": CONTACT_MULT * record["L"] + CONTACT_CONSTANT,
            "context_headroom": CONTEXT - prompt_tokens,
            "max_tokens": max_tokens,
            "gt_statement_tokens": statement_tokens,
            "gt_fits": int(statement_tokens <= max_tokens),
            "headroom_ratio": round(max_tokens / max(1, statement_tokens), 3),
            "n_contacts_emitted": int(full.contacts_emitted),
            "document_truncated": int(truncated),
            # A truncated document cannot be produced in full by any rollout, so
            # a score for it measures the format's context limit, not the model.
            "scorable": int(not truncated and statement_tokens <= max_tokens),
        })
        if index % 50 == 0:
            print(f"  [{index}/{len(sets)}]", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT, index=False)
    summary = {
        "context": CONTEXT,
        "recipe": f"min({CONTEXT} - prompt, {CONTACT_MULT}L + {CONTACT_CONSTANT})",
        "n": int(len(frame)),
        "n_gt_does_not_fit": int((frame.gt_fits == 0).sum()),
        "n_not_scorable": int((frame.scorable == 0).sum()),
        "not_scorable": frame.loc[frame.scorable == 0,
                                  ["eval_set", "stem", "L", "n_gt_contacts",
                                   "n_contacts_emitted", "gt_statement_tokens",
                                   "max_tokens", "document_truncated"]
                                  ].to_dict(orient="records"),
        "context_bound": int((frame.context_headroom < frame.recipe_budget).sum()),
        "by_set": {
            name: {
                "n": int(len(group)),
                "n_gt_does_not_fit": int((group.gt_fits == 0).sum()),
                "n_not_scorable": int((group.scorable == 0).sum()),
                "min_headroom_ratio": float(group.headroom_ratio.min()),
                "median_headroom_ratio": float(group.headroom_ratio.median()),
            }
            for name, group in frame.groupby("eval_set")
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "not_scorable"}, indent=2))
    for record in summary["not_scorable"]:
        print(f"  not scorable: {record}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
