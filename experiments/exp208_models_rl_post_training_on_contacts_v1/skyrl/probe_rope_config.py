# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Which rope interpretation does the exp199 export actually want? — issue #208.

The SkyRL run collapses on its first update, and the metric that gives it away is
`rollout_train_logprobs_abs_diff_mean`: **1.33 nats at step 0**, then 0.08 from
step 1 onward. That ordering is the whole story. At step 0 vLLM holds the weights
it loaded from the HF export and the FSDP trainer holds its own; they disagree
wildly. `sync_weights` then pushes the TRAINER's view into vLLM, after which the
two agree — and generation collapses from 160 contacts/rollout to 0.9.

The suspect is the export's rope block, which declares llama3 scaling at
`factor: 8.0` in BOTH `rope_scaling` and `rope_parameters`:

    rope_theta      = 500000
    rope_scaling    = {rope_type: llama3, factor: 8.0, ...}
    rope_parameters = {rope_type: llama3, factor: 8.0, ..., rope_theta: 500000}

This script asks the question empirically rather than by reading loader source:
score real corpus text (the prompts the RL run itself uses) under each candidate
rope config and compare NLL. Position scaling only bites at long context, so the
prompts -- ~1-2k tokens of protein sequence statements -- are exactly the right
probe. The config the model was trained under wins by a wide margin; a wrong rope
costs whole nats/token, not hundredths (cf. the 0.76 nats/token in the
transformers-5 export bug).

Run on a GPU box:

    python probe_rope_config.py --parquet data/skyrl_train_2k.parquet --n 8
"""

import argparse
import copy
import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199"


def _variants(base):
    """Candidate rope interpretations, as full config objects."""
    out = {}

    as_is = copy.deepcopy(base)
    out["as_exported (llama3 factor 8)"] = as_is

    # No scaling: plain rope at theta=500000. This is what the model is trained
    # with if the llama3 block was added by a repair script rather than by the
    # run that produced the weights.
    plain = copy.deepcopy(base)
    for attr in ("rope_scaling", "rope_parameters"):
        if hasattr(plain, attr):
            try:
                setattr(plain, attr, {"rope_type": "default", "rope_theta": 500000.0})
            except Exception:
                pass
    plain.rope_theta = 500000.0
    out["no_scaling (theta 500k)"] = plain
    return out


def _prompt_ids(parquet, tokenizer, n):
    """Real prompts from the RL dataset, tokenized exactly as the run does."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    rows = table.to_pylist()[:n]
    ids = []
    for row in rows:
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        ids.append(tokenizer(text, return_tensors="pt").input_ids)
    return ids


@torch.no_grad()
def _nll(model, ids):
    """Mean per-token NLL (nats), the same units as the published val loss."""
    total, count = 0.0, 0
    for x in ids:
        x = x.to(model.device)
        out = model(x, labels=x)
        n = x.shape[1] - 1
        total += float(out.loss) * n
        count += n
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoConfig.from_pretrained(args.model)
    ids = _prompt_ids(args.parquet, tok, args.n)
    print(f"{len(ids)} prompts, {sum(x.shape[1] for x in ids)} tokens total\n")

    for name, cfg in _variants(base).items():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=cfg, dtype=torch.bfloat16, device_map="cuda:0",
        )
        model.eval()
        nll = _nll(model, ids)
        rope = getattr(model.config, "rope_parameters", None) or getattr(model.config, "rope_scaling", None)
        print(f"{name:34s} NLL = {nll:.4f} nats/token   rope={rope}")
        del model
        torch.cuda.empty_cache()

    print("\nThe trained-under config wins by a wide margin; a difference of whole")
    print("nats/token means the losing config is not the model the weights encode.")


if __name__ == "__main__":
    main()
