# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Do transformers and vLLM agree on the exp199 export? — issue #208.

The SkyRL run collapses on its first weight sync, and a **zero-learning-rate
control collapses identically** (precision 0.259 -> 0.022, pred/gt 1.13 -> 0.005,
response length 489 -> 996). With `lr=0` no gradient can move a weight, so the
update is not the cause. What is left is that the trainer's copy of the model
was never the same as the one vLLM loaded: at step 0 they disagree by 1.33
nats/token, and `sync_weights` then pushes the trainer's copy into the engines,
after which they "agree" at 0.08 and generation is destroyed.

This script takes SkyRL out of the picture and asks the question directly: load
the SAME checkpoint under transformers and under vLLM, score the SAME tokens, and
compare. A large gap localizes the bug to the export/loader pair (and explains
SkyRL as the messenger); agreement means SkyRL's sync is corrupting the weights
and the loaders are fine.

Uses vLLM's `prompt_logprobs`, which scores tokens the model did not generate, so
both stacks are evaluated on identical inputs with no sampling in the loop.

    python probe_hf_vs_vllm.py --parquet data/skyrl_train_2k.parquet --n 4
"""

import argparse
import json

import numpy as np
import torch


def _prompt_texts(parquet, tokenizer, n):
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet).to_pylist()[:n]
    texts = []
    for row in rows:
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        texts.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--model", default="timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = _prompt_texts(args.parquet, tok, args.n)
    id_lists = [tok(t).input_ids for t in texts]
    print(f"{len(texts)} prompts, {sum(len(x) for x in id_lists)} tokens\n")

    # ---- transformers (what the FSDP trainer uses) ----------------------------
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    hf_lp = []
    with torch.no_grad():
        for ids in id_lists:
            x = torch.tensor([ids], device=model.device)
            logits = model(x).logits.float()
            lp = torch.log_softmax(logits[0, :-1], dim=-1)
            tgt = x[0, 1:]
            hf_lp.append(lp.gather(-1, tgt[:, None]).squeeze(-1).cpu().numpy())
    hf_nll = -np.concatenate(hf_lp).mean()
    print(f"transformers NLL = {hf_nll:.4f} nats/token")
    del model
    torch.cuda.empty_cache()

    # ---- vLLM (what the inference engines use) -------------------------------
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.55,
              max_model_len=4096, enforce_eager=True)
    # vLLM >=0.9 takes token prompts as TokensPrompt dicts, not `prompt_token_ids=`.
    outs = llm.generate(
        [{"prompt_token_ids": ids} for ids in id_lists],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
    )
    vllm_lp = []
    for out, ids in zip(outs, id_lists):
        # prompt_logprobs[0] is None (no token precedes the first); align with HF,
        # which scores positions 1..n-1.
        per_tok = []
        for pos, entry in enumerate(out.prompt_logprobs):
            if entry is None:
                continue
            per_tok.append(entry[ids[pos]].logprob)
        vllm_lp.append(np.array(per_tok))
    vllm_nll = -np.concatenate(vllm_lp).mean()
    print(f"vLLM         NLL = {vllm_nll:.4f} nats/token")

    # ---- the comparison SkyRL reports as rollout_train_logprobs_abs_diff -----
    diffs = np.concatenate([np.abs(a[: len(b)] - b[: len(a)]) for a, b in zip(hf_lp, vllm_lp)])
    print(f"\nmean |hf - vllm| = {diffs.mean():.4f} nats   max = {diffs.max():.4f}")
    print(f"NLL gap          = {abs(hf_nll - vllm_nll):.4f} nats/token")
    print(
        "\nSkyRL logs 1.33 nats at step 0. If this probe reproduces a gap of that\n"
        "order the loaders disagree and SkyRL is only the messenger; if it prints\n"
        "~0.01 the two stacks agree and the corruption is in SkyRL's weight sync."
    )


if __name__ == "__main__":
    main()
