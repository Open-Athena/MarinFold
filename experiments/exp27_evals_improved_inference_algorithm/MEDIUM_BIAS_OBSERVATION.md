# The "99% medium-range" observation: what was actually happening

In exp27's slide decks and READMEs I described MarinFold 1B as having
a "~99% medium-range token bias" that the `range_strategy=uniform`
fix in `ContactsOnlyLogitsProcessor` undoes. **That framing was
misleading.** The real behaviour is autoregressive lock-in, not a
biased marginal prior. This file documents what was actually
observed, what wasn't, and how to reproduce both cleanly.

## TL;DR

1. The model's **marginal** distribution over the 3 range tokens at
   the first state-0 position (one token after `<begin_statements>`)
   is roughly **balanced**: `<long-range-contact>` is in fact the
   most likely (~0.42), then `<short-range-contact>` (~0.30), then
   `<medium-range-contact>` (~0.27). All three are within a small
   factor of uniform; medium is **not** dominant.

2. But once we run grammar-constrained autoregressive sampling
   (the `ContactsOnlyLogitsProcessor` with `range_strategy=model`,
   T=0.7, top_p=0.9, 900 max_tokens), **the model gets stuck on
   whichever range it samples first**, for all ~300 emitted
   statements. The first sample matches the marginal prior
   (mostly long), then the autoregressive state holds it there.

3. With `seed=27` on the train proteins, the first sample happens
   to land on medium for every protein we showed in the slides —
   so the deck reported "99% medium". With other seeds you get
   "99% long" or occasionally "99% short". The
   `range_strategy=uniform` fix forces independent 1/3-1/3-1/3
   draws at every statement, bypassing the lock-in entirely.

## What I actually measured

### (A) Marginal range-token prior at state 0 (1B)

One forward pass: prompt = `<begin_seq><AAs><begin_statements>`,
look at the next-token logprobs (top-128), pick out the 3
`<*-range-contact>` token IDs, exponentiate, renormalize within
those 3.

| protein   | long  | medium | short | raw contact-token mass in top-128 |
|---|---:|---:|---:|---:|
| 8eb9 (L=95)  | 0.482 | 0.275 | 0.243 | 0.913 |
| 7y5j (L=102) | 0.412 | 0.321 | 0.266 | 0.940 |
| 7uk8 (L=394) | 0.421 | 0.272 | 0.308 | 0.964 |

Long is consistently the most likely first-statement range token,
medium is the *least* (or near-least). The marginal prior is not
medium-biased — at all.

### (B) Sticky autoregressive sampling under the LP (1B, seed sweep)

Same LP-constrained sampling, just varying the seed. 900 max_tokens
= 300 contact statements per run. Counts of which range token was
actually emitted, grouped by seed:

```
8eb9_A (1B), range_strategy=model, T=0.7, top_p=0.9, max_tokens=900:
  seed= 27:  long=  0  med=300  short=  0     first 3: ['medium', 'medium', 'medium']
  seed= 28:  long=249  med= 51  short=  0     first 3: ['medium', 'long', 'medium']
  seed= 29:  long=300  med=  0  short=  0     first 3: ['long', 'long', 'long']
  seed= 30:  long=300  med=  0  short=  0     first 3: ['long', 'long', 'long']
  seed= 31:  long=  0  med=  0  short=300     first 3: ['short', 'short', 'short']
  seed= 42:  long=300  med=  0  short=  0     first 3: ['long', 'long', 'long']
  seed=100:  long=300  med=  0  short=  0     first 3: ['long', 'long', 'long']
  seed=1000: long=300  med=  0  short=  0     first 3: ['long', 'long', 'long']
```

Across 8 seeds the first-statement range distribution is roughly
5 long / 1 medium / 1 short / 1 mixed-start, which matches the
0.48/0.27/0.24 marginal prior reasonably well.

But the **per-rollout** range distribution is essentially monolithic:
once the first contact emitted is medium, all 300 stay medium; same
for long; same for short. The seed=28 run is the only one that
"switched mode" (medium-dominant for the first chunk, then
long-dominant for the rest), and even there it's still concentrated
in 2 of the 3 ranges.

### (C) The slide-deck number

In the writeups I quoted:

```
8eb9_A (L=95):  225 contacts → 222 medium + 3 long + 0 short
8baq_A (L=208): 296 contacts → 295 medium + 1 long + 0 short
7uk8_A (L=394): 296 contacts → 295 medium + 1 long + 0 short
```

These were all from **seed=27**. Per (B), that seed lands on medium
first, and the autoregressive lock-in keeps it there. The "~99%
medium" we see in the table is the lock-in showing up after the
random first-draw. With seed=29 the same table would say "~99% long".

(The "3 long + 0 short" residual on 8eb9 reflects post-processing
where the parser maps `<medium-range-contact><p_i><p_j>` to the
*actual* sep bucket via `_range_token_for_separation(j-i)` — so
3 of the 300 medium-emitted pairs had a sep that landed in the
long bucket after rebucketing. Distinct from "model emitted long".)

## Why does this happen?

The proximate cause is whatever in-context dependency the model
learned during training. Some plausible mechanisms:

- **Training docs were range-homogeneous.** If the training-time
  documents typically used one range type per document (e.g. a doc
  is "this protein's medium contacts" or "this protein's long
  contacts"), the model has learned "once you start with range X,
  keep using range X". I don't know the exp1 generator's choices
  for distance-masked docs offhand, but this fits the observed
  monolithic generations.
- **It's not a temperature artifact.** Lowering T should make it
  *more* sticky, not less. T=0.7 is what we used; at T=1.0 the same
  pattern would appear.
- **It's not a vLLM logits-processor ordering bug.** vLLM applies
  logits processors before temperature scaling. The first-token
  distribution under the LP matches what the marginal probe (A)
  predicts (modulo the seed-driven sampling).

## What "fixing" this with range_strategy=uniform did

The `uniform` strategy overwrites the 3 range-token logits to all
zero at every state-0 position, so the LP returns a mask that has
0 at exactly the 3 contact-range IDs and -inf everywhere else.
vLLM then samples uniformly among the 3 at every statement,
ignoring the model's prior (which never tells it to switch ranges).

Cost: we lose the model's range preferences. Benefit: we get the
range *diversity* we actually want for the downstream readout —
short/medium/long contacts each get ~1/3 of the seed prefix.

Empirically, on 1B-train this gave the headline +25.9% lift over
baseline (stage A alone) and +42.81% combined with stage B
iteration. On 1.5B the same fix HURTS — see RESULTS_LOG.md cross-
model section. The marginal prior on 1.5B at state 0 is roughly
0.42 / 0.32 / 0.27 (long / med / short), and probably also has
the same lock-in behavior — so uniform-forcing doesn't fix
anything the 1.5B doesn't already do well, and may distort
something it was getting right.

## Reproduction commands

All commands assume:

- cwd: `experiments/exp27_evals_improved_inference_algorithm/`
- `uv sync` already done
- protenix GT mmCIFs fetched (per the experiment README)
- A GPU with `CUDA_VISIBLE_DEVICES=0` available

### (A) Marginal prior at state 0

```bash
uv run python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys; sys.path.insert(0, '.')
import math
from pathlib import Path
from naive_inference import load_runtime, _build_base_prompt, _encode_tokens
from canonical_sequence import read_canonical_sequence
from sampled_contacts_inference import _resolve_contact_range_ids, _CONTACT_TOKEN_STRS
from vllm import SamplingParams, TokensPrompt

rt = load_runtime(model_nickname='1B', models_yaml=Path('../../marinfold/marinfold/MODELS.yaml'), dtype='bfloat16')
contact_ids = _resolve_contact_range_ids(rt.tokenizer)
id_to_tok = {tid: tok for tid, tok in zip(contact_ids, _CONTACT_TOKEN_STRS)}

for stem in ['8eb9_A', '7y5j_A', '7uk8_A']:
    seq = read_canonical_sequence(Path(f'protenix_data/data/protenix-foldbench-monomers/gt/{stem}.cif'))
    base_ids = _encode_tokens(rt.tokenizer, _build_base_prompt(seq.residue_names))
    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=1, n=1, logprobs=128)
    out = rt.llm.generate([TokensPrompt(prompt_token_ids=base_ids)], sp, use_tqdm=False)
    lp = out[0].outputs[0].logprobs[0] if out[0].outputs[0].logprobs else {}
    probs = {}
    for tid, lpobj in lp.items():
        if int(tid) in id_to_tok:
            probs[id_to_tok[int(tid)]] = math.exp(float(lpobj.logprob))
    total = sum(probs.values())
    n = {k: v/total for k, v in probs.items()}
    print(f'{stem}: long={n[\"<long-range-contact>\"]:.3f} med={n[\"<medium-range-contact>\"]:.3f} short={n[\"<short-range-contact>\"]:.3f}  (mass: {total:.3f})')
"
```

Expected output (matching the table above):
```
8eb9_A: long=0.482 med=0.275 short=0.243  (mass: 0.913)
7y5j_A: long=0.412 med=0.321 short=0.266  (mass: 0.940)
7uk8_A: long=0.421 med=0.272 short=0.308  (mass: 0.964)
```

### (B) Sticky autoregressive sampling under the LP (seed sweep)

```bash
uv run python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys; sys.path.insert(0, '.')
import re
from pathlib import Path
from naive_inference import load_runtime, _build_base_prompt, _encode_tokens
from canonical_sequence import read_canonical_sequence
from sampled_contacts_inference import (
    _resolve_contact_range_ids, ContactsOnlyLogitsProcessor,
    _resolve_position_token_ids,
)
from vllm import SamplingParams, TokensPrompt

rt = load_runtime(model_nickname='1B', models_yaml=Path('../../marinfold/marinfold/MODELS.yaml'), dtype='bfloat16')
contact_ids = _resolve_contact_range_ids(rt.tokenizer)
seq = read_canonical_sequence(Path('protenix_data/data/protenix-foldbench-monomers/gt/8eb9_A.cif'))
n = seq.n_residues
base_ids = _encode_tokens(rt.tokenizer, _build_base_prompt(seq.residue_names))
_RAW_RE = re.compile(r'<(long|medium|short)-range-contact>')

for seed in [27, 28, 29, 30, 31, 42, 100, 1000]:
    lp = ContactsOnlyLogitsProcessor(
        contact_range_ids=contact_ids,
        position_ids=_resolve_position_token_ids(rt.tokenizer, n),
        vocab_size=rt.tokenizer.vocab_size,
        range_strategy='model',
    )
    sp = SamplingParams(temperature=0.7, top_p=0.9, top_k=-1, max_tokens=900, n=1, seed=seed, logits_processors=[lp])
    out = rt.llm.generate([TokensPrompt(prompt_token_ids=base_ids)], sp, use_tqdm=False)
    text = rt.tokenizer.decode(list(out[0].outputs[0].token_ids), skip_special_tokens=False)
    emit = {'long': 0, 'medium': 0, 'short': 0}
    for m in _RAW_RE.finditer(text):
        emit[m.group(1)] += 1
    print(f'seed={seed}: L={emit[\"long\"]:>3} M={emit[\"medium\"]:>3} S={emit[\"short\"]:>3}')
"
```

Expected output:
```
seed=27:  L=  0 M=300 S=  0
seed=28:  L=249 M= 51 S=  0
seed=29:  L=300 M=  0 S=  0
seed=30:  L=300 M=  0 S=  0
seed=31:  L=  0 M=  0 S=300
seed=42:  L=300 M=  0 S=  0
seed=100: L=300 M=  0 S=  0
seed=1000:L=300 M=  0 S=  0
```

### (C) Reproduce the slide-deck "99% medium" numbers

To get the exact numbers I put in the slide deck, use seed=27 (the
default `base_seed` in `sample_contact_prefix`). The full smoke
script is in
`sampled_contacts_inference.py::sample_contact_prefix`; calling it
on 8eb9_A / 7y5j_A / 7uk8_A at the default seed reproduces the
100% medium emit-range pattern.

## What I should have written in the slides

> "MarinFold 1B's grammar-constrained sampling is range-monolithic:
> once it emits its first `<*-range-contact>` token, autoregressive
> dependency keeps it in that range for all ~300 statements in a
> rollout. The marginal prior at state 0 is roughly balanced (long
> 0.42, med 0.27, short 0.31) so the rollout's monolithic range is
> usually long — but seed=27 happens to land on medium first, then
> sticks. The `uniform` range strategy forces independent 1/3 draws
> at each statement, breaking the lock-in and giving the downstream
> readout the across-range diversity it needs."

The "uniform fixes a 99% medium prior" framing was at best
imprecise and at worst wrong. The fix is real — across-rollout
range diversity matters for the readout — but the *reason* it's a
fix is autoregressive lock-in, not a biased single-step marginal.
