# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assert the pass-through chat template changes nothing — issue #208.

contacts-v1 is a raw token sequence with no chat format: no role markers, no
`<|im_start|>`, no `<|im_end|>`. Any real template would prepend tokens the model
has never seen in that position and shift every `<pN>` off the position it was
trained at — silently, producing fluent nonsense rather than an error.

So the requirement is exact: rendering a prompt through the template must produce
**byte-identical token ids** to encoding the raw content. This asserts it rather
than assuming it, because the failure mode is invisible at generation time and
would only show up as inexplicably bad contacts.
"""

import sys

from transformers import AutoTokenizer

REPO = sys.argv[1] if len(sys.argv) > 1 else "timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199"
TEMPLATE = "contacts_v1_passthrough.jinja"

PROBES = [
    "<contacts-v1> <begin_sequence> <p17> <ALA> <p18> <GLY> <begin_statements>",
    "<contacts-v1> <n-term> <p1382> <begin_statements> <contact> <p17> <p25> <end>",
]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(REPO)
    template = open(TEMPLATE).read()
    ok = True
    for probe in PROBES:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": probe}], chat_template=template, tokenize=False)
        via_template = tok.encode(rendered, add_special_tokens=False)
        raw = tok.encode(probe, add_special_tokens=False)
        same = via_template == raw
        ok &= same
        print(f"  {'OK  ' if same else 'DIFF'} {len(raw):3d} ids | {probe[:52]}...")
        if not same:
            print(f"       template {via_template[:12]}\n       raw      {raw[:12]}")
    print("PASS-THROUGH TEMPLATE:", "identical" if ok else "ALTERS THE PROMPT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
