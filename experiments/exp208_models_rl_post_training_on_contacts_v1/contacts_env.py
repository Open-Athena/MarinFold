# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``MarinEnv`` for exp208: plain ``<contacts-v1>`` rollouts, dense per-contact
reward, and a **consensus-marginal** document term — issue #208.

Derived from exp200's environment. Three things changed, and each is the reason
#208 exists rather than being a refactor of #200:

**1. Plain mode only.** exp200 trained a 50:50 mix of ``<contacts-v1>`` and
exp163's ``<contacts-v1.multi>``. Dropping multi-draft deletes the
spread-across-sections axis that made #200's headline metric a product of
candidate quality and candidate spread — the two moved in opposite directions and
cancelled. The curriculum collapses to one lesson, so exp200's
``minimum_sample_probability=1.0`` pinning is gone with it.

**2. The document reward is a leave-one-out marginal contribution to the GROUP's
consensus**, not the rollout's own F1. MarinFold reports a consensus over 100
rollouts, and #82 records that "over-sharpening collapses the vote". A
precision-only reward is a sharpening operator, so it can raise per-rollout
quality and *lower* the reported metric. Scoring the marginal makes the objective
literally the deployed metric and pays a rollout for contributing what its
siblings missed. ``doc_term`` selects between this, exp200's own-F1 term, and
none, which is what the #208 arms compare.

**3. The diagnostics are GROUP-level.** exp200's collapse detectors were all
within-rollout: ``mean_jaccard`` between candidate sections, ``n_sections``,
``best_f1`` / ``first_f1`` / ``last_f1``. In plain mode every one of them is NaN
or constant by construction, so a straight port would have left diversity
collapse — the exact failure this experiment is built to detect — invisible
during training. :mod:`consensus` supplies the replacements.

WHY THIS DRIVES vLLM DIRECTLY (unchanged from exp200). marin's
``vLLMInferenceContext.batch_completions`` renders every prompt through a chat
template, and the contacts-v1 vocab has neither a chat template nor
``<|im_end|>``. The context class is chosen by a hardcoded
``if inference_type == "vllm"`` in ``rollout_worker.py``, so a subclass cannot be
injected and patching marin is off the table. What IS available:
``inference_ctx.llm`` and ``inference_ctx.tokenizer`` are plain attributes, and
the renderer is touched nowhere except inside ``batch_completions``. So this
environment builds prompt token ids itself and calls ``llm.generate`` with
``TokensPrompt`` — which is what exp163's validated rollout worker does, and what
made exp200's parity check a like-for-like comparison.
"""

import logging
import math
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import fsspec
import jax
import numpy as np
import pyarrow.parquet as pq
from marin.rl.decoding import DecodingConfig
from marin.rl.environments.base import MarinEnv
from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.types import Rollout, RolloutGroup

import consensus as cs
import contact_rewards as cr
from _trace import Tracer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def _candidate_universe(length: int):
    """Module-level so the LRU cannot pin an environment instance alive."""
    return cs.candidate_index(length)


DOC_TERMS = ("consensus", "own_f1", "none")

# exp98's per-target budget for a single contacts-v1 statement section.
PLAIN_TOKENS_PER_RESIDUE = 4
PLAIN_TOKEN_SLACK = 64


def seed_from(prng_key) -> int:
    """Derive an int seed from marin's ``prng_key``, which is a UNION type.

    ``RolloutWorker`` sets ``use_jax_rng = (inference_type == "levanter")`` and
    then either splits a JAX key or draws ``py_rng.randint(0, 2**31 - 1)``. So a
    vLLM rollout worker — which is what exp208 runs — always hands an environment
    a plain Python int, and ``jax.random.randint`` on it dies with "JAX
    encountered invalid PRNG key data". marin's own ``mock_env`` calls
    ``jax.random.randint(prng_key, ...)`` unguarded, so it has presumably only
    ever been exercised against levanter inference. Accept both rather than
    assume the branch we happen to be on.
    """
    if isinstance(prng_key, (int, np.integer)):
        return int(prng_key) % (2**31 - 1)
    return int(jax.random.randint(prng_key, (), 0, 2**31 - 1))


class ContactsV1RLEnv(MarinEnv):
    """Sample plain contacts-v1 rollouts; score them densely and by consensus.

    Args:
        targets_path: Parquet with ``entry_id, L, gt_contacts`` (a list of
            ``[i, j]`` PAIRS in sequence-index space).
        prompts_path: Directory of ``<entry_id>.parquet`` with ``r, L, prefix,
            seq_positions`` — exp163's ``gen_prompts_exp163.py`` output, which
            exp200's pool builder reused.
        doc_term: ``"consensus"`` (the #208 design), ``"own_f1"`` (exp200's
            document term, the arm-F ablation), or ``"none"`` (step-only).
        err_decay: Geometric decay on the penalty for repeat errors in a rollout.
        precision_ema_decay: Smoothing for ``p_bar``. Higher is smoother.
        initial_precision: ``p_bar`` before any rollouts are scored. #208 Phase 0
            measured 0.482 for this model over 10,000 plain rollouts; the default
            is 0.45 because the training pool is AFDB rather than the PDB-derived
            eval set. exp200's 0.23 was exp163 arm F in multi-draft mode.
        max_protein_len: Longest protein in the pool, for the token budget.
        max_model_len: Engine context length; the response budget is clamped to fit.
        eval_fraction: Fraction of proteins held out for ``mode="eval"``.
        seed: Seed for the train/eval split (NOT for sampling, which comes from
            the caller's ``prng_key``).
        limit: Optional cap on proteins, for smoke tests.
        fetch_workers: Concurrency for per-protein prompt reads.
        trace_path: Object-store prefix for diagnostic events. ``iris job logs``
            shows nothing for a RUNNING child, so without this a worker that
            misbehaves without dying is unobservable from outside the cluster.
    """

    def __init__(
        self,
        *,
        targets_path: str,
        prompts_path: str,
        doc_term: str = "consensus",
        err_decay: float = 0.5,
        precision_ema_decay: float = 0.9,
        initial_precision: float = 0.45,
        vocab_size: int | None = None,
        max_protein_len: int = 512,
        max_model_len: int = 8192,
        eval_fraction: float = 0.05,
        seed: int = 0,
        limit: int | None = None,
        fetch_workers: int = 8,
        trace_path: str | None = None,
    ):
        if doc_term not in DOC_TERMS:
            raise ValueError(f"doc_term must be one of {DOC_TERMS}, got {doc_term!r}")
        self.doc_term = doc_term
        self.err_decay = err_decay
        self.precision_ema_decay = precision_ema_decay
        self.vocab_size = vocab_size
        self.max_protein_len = max_protein_len
        self.max_model_len = max_model_len
        self.prompts_path = prompts_path.rstrip("/")
        self.fetch_workers = fetch_workers

        self._p_bar = float(initial_precision)
        self._prompt_cache: dict[str, list[dict]] = {}
        self._trace = Tracer(trace_path, run=f"env-{doc_term}")
        self._calls = 0

        self._targets = self._load_targets(targets_path, limit, seed=seed)
        ids = sorted(self._targets)
        order = np.random.default_rng(seed).permutation(len(ids))
        n_eval = max(1, int(round(eval_fraction * len(ids)))) if eval_fraction > 0 else 0
        self._eval_ids = [ids[i] for i in order[:n_eval]]
        self._train_ids = [ids[i] for i in order[n_eval:]]
        logger.info(
            "[exp208] %d proteins (%d train / %d eval), doc_term=%s",
            len(ids), len(self._train_ids), len(self._eval_ids), doc_term,
        )
        self._trace.event(
            "env_init", doc_term=doc_term, n_proteins=len(ids),
            n_train=len(self._train_ids), n_eval=len(self._eval_ids),
            initial_precision=self._p_bar,
        )

    @staticmethod
    def _load_targets(path: str, limit: int | None, seed: int = 0) -> dict[str, dict]:
        """Load targets, optionally down to a RANDOM ``limit``-sized subset.

        Sampling, not truncation. Target files are grouped by source dataset, so
        taking the first N rows takes whole benchmarks — exp200 read a 50%
        "regression" that was really just a different, harder protein set.
        """
        with fsspec.open(path, "rb") as fh:
            table = pq.read_table(fh, columns=["entry_id", "L", "gt_contacts"])
        out: dict[str, dict] = {}
        for entry_id, length, pairs in zip(
            table["entry_id"].to_pylist(), table["L"].to_pylist(), table["gt_contacts"].to_pylist()
        ):
            # `gt_contacts` is a list of [i, j] PAIRS, not a flat [i0, j0, i1, ...]
            # vector — that is the shape used for PREDICTIONS, and confusing the
            # two silently halves the ground truth and pairs unrelated residues.
            gt = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs}
            gt = {p for p in gt if cr.in_band(p)}
            if not gt:
                continue
            out[entry_id] = {"L": int(length), "gt": gt}
        if not out:
            raise ValueError(f"no usable targets in {path}")
        if limit is not None and 0 < limit < len(out):
            keys = sorted(out)
            picked = np.random.default_rng(seed).choice(len(keys), size=limit, replace=False)
            out = {keys[int(i)]: out[keys[int(i)]] for i in sorted(picked)}
        return out

    @staticmethod
    def _universe(length: int):
        """Candidate pair universe for a protein length, LRU-cached.

        No ``resolved`` restriction: the training pool is AFDB documents, where
        every residue is present. The eval metric restricts to residues resolved
        in the GT structure, which is a property of the eval set rather than of
        the scoring function.

        THE BOUND IS LOAD-BEARING. A universe is ~L^2/2 entries and is a pure
        function of L, so an unbounded dict keyed on length looks free — but the
        exp200 training pool holds **482 distinct lengths** between 31 and 512,
        which is 22.4M dict entries (a few GB in CPython) if every one is
        retained. That grows gradually as the sampler works through 10,000
        proteins, so it would OOM a rollout worker deep into a run rather than at
        startup. 64 entries covers the reuse that matters (a step touches 8
        proteins) at a bounded cost, and a miss is ~0.1 s against a ~37 s
        sampling call.
        """
        return _candidate_universe(length)

    def _prompts_for(self, entry_id: str) -> list[dict]:
        """Realizations for one protein: ``prefix`` text plus its position map."""
        cached = self._prompt_cache.get(entry_id)
        if cached is not None:
            return cached
        with fsspec.open(f"{self.prompts_path}/{entry_id}.parquet", "rb") as fh:
            table = pq.read_table(fh, columns=["r", "prefix", "seq_positions"])
        rows = [
            {"r": int(r), "prefix": prefix, "seq_positions": list(positions)}
            for r, prefix, positions in zip(
                table["r"].to_pylist(), table["prefix"].to_pylist(),
                table["seq_positions"].to_pylist())
        ]
        self._prompt_cache[entry_id] = rows
        return rows

    def _build_prompt_ids(self, tokenizer, prefix: str) -> list[int]:
        """Tokenize a prompt and assert its document sentinel.

        exp200 had to SWAP index 0 to select the multi-draft sentinel by id.
        exp208 is plain-mode only, so the prompt pool's own id 2 is already
        right — this checks rather than rewrites, which is the stronger
        statement.

        Uses ``.encode()``, not the HF ``__call__`` interface: ``inference_ctx.
        tokenizer`` is levanter's ``HfMarinTokenizer`` wrapper, which is NOT
        callable and returns a plain ``list[int]`` rather than a ``BatchEncoding``.
        """
        ids = list(tokenizer.encode(prefix, add_special_tokens=False))
        if not ids:
            raise ValueError("empty prompt tokenization")
        if ids[0] != cr.PLAIN_DOC_ID:
            raise ValueError(
                f"prompt does not start with <contacts-v1> (id {cr.PLAIN_DOC_ID}); got id {ids[0]}. "
                "The prompt pool was built for a different document structure."
            )
        return ids

    def _response_budget(self, max_prompt_len: int, max_length: int, declared: int | None = None) -> int:
        """Response token budget for one batch.

        Three bounds, all real: exp98's per-residue budget is what the model
        actually needs; the context window is what the engine can hold; and
        ``declared`` is the lesson's own ``max_output_tokens``, from which
        ``curriculum.max_seq_len`` is derived — ``train_batch`` raises when a
        padded sequence overruns it.
        """
        budget = PLAIN_TOKENS_PER_RESIDUE * max_length + PLAIN_TOKEN_SLACK
        room = self.max_model_len - max_prompt_len
        if room <= 0:
            raise ValueError(f"prompt of {max_prompt_len} tokens leaves no room in {self.max_model_len}")
        if declared is not None:
            budget = min(budget, declared)
        return int(min(budget, room))

    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        decoding: DecodingConfig,
        prng_key,
        mode: str = "train",
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Generate ``n_generations`` rollouts for each of ``n_examples`` proteins.

        Distinct prompt realizations per generation, not ``n>1`` on one prompt:
        exp82/exp98 settled on resampling the position numbering and statement
        order per rollout, and the eval this experiment optimizes is a consensus
        over exactly that kind of resampled group.
        """
        from vllm import SamplingParams, TokensPrompt

        self._calls += 1
        call_started = time.time()
        self._trace.event(
            "sample_start", call=self._calls, mode=mode, doc_term=self.doc_term,
            n_examples=n_examples, n_generations=n_generations,
            prng_key_type=type(prng_key).__name__, p_bar=self._p_bar,
        )
        try:
            return self._sample(
                inference_ctx, n_examples, n_generations, decoding, prng_key, mode,
                SamplingParams, TokensPrompt, call_started,
            )
        except BaseException as exc:
            self._trace.exception("sample_failed", exc, call=self._calls, mode=mode)
            raise

    def _sample(
        self, inference_ctx, n_examples, n_generations, decoding, prng_key, mode,
        SamplingParams, TokensPrompt, call_started,
    ):
        if n_generations < 2:
            raise ValueError(
                f"n_generations={n_generations}: both the RLOO baseline and the "
                "consensus marginal are defined across a GROUP, and a group of one "
                "carries no document signal at all."
            )
        pool = self._train_ids if mode == "train" else self._eval_ids
        if not pool:
            raise ValueError(f"no proteins available for mode={mode!r}")

        rng = np.random.default_rng(seed_from(prng_key))
        n_take = min(n_examples, len(pool))
        entry_ids = [pool[i] for i in rng.choice(len(pool), size=n_take, replace=False)]

        with ThreadPoolExecutor(max_workers=self.fetch_workers) as pool_exec:
            prompt_rows = list(pool_exec.map(self._prompts_for, entry_ids))

        specs: list[dict] = []
        for entry_id, rows in zip(entry_ids, prompt_rows):
            if not rows:
                logger.warning("[exp208] no prompts for %s; skipping", entry_id)
                continue
            picks = rng.choice(len(rows), size=n_generations, replace=len(rows) < n_generations)
            for pick in picks:
                row = rows[int(pick)]
                specs.append({
                    "entry_id": entry_id,
                    "r": row["r"],
                    "ids": self._build_prompt_ids(inference_ctx.tokenizer, row["prefix"]),
                    "pos_to_seq": {int(p): i for i, p in enumerate(row["seq_positions"])},
                    "gt": self._targets[entry_id]["gt"],
                })
        if not specs:
            raise ValueError("no prompts were built for this step")

        max_prompt_len = max(len(s["ids"]) for s in specs)
        max_length = max(self._targets[s["entry_id"]]["L"] for s in specs)
        applied = replace(
            decoding,
            max_output_tokens=self._response_budget(
                max_prompt_len, max_length, declared=decoding.max_output_tokens),
            stop_token_ids=[cr.END_ID],
            # TPU vLLM rejects per-request seeds ("JAX does not support per-request
            # seed"); engine-level seeding is VLLMEngineConfig.seed.
            seed=None,
        )
        # CONSTRAIN SAMPLING TO THE REAL VOCABULARY. vLLM pads the vocab to a
        # hardware-friendly multiple — 2845 -> 2848 here — and those padding rows
        # are ZERO, so they emit a logit of exactly 0.0 that nothing masks out of
        # the sampling distribution.
        #
        # Whether that matters depends entirely on where a model's logits sit,
        # which softmax makes invisible everywhere else. Measured on the two #208
        # warm starts: exp163 arm F's top logit has median 12.91, so a 0.0 row has
        # probability ~0 and it emitted zero out-of-range ids in 197,251 tokens.
        # exp199's top logit has median 1.16 and dips to -4.03, so a 0.0 row takes
        # ~1.6% per position and is sometimes the argmax — it emitted ids
        # 2845/2846/2847 in **12.4% of all tokens and in 256 of 256 rollouts**.
        # Those ids do not exist in a 2845-row embedding, and the trainer NaNs on
        # the first step trying to score them.
        #
        # Nothing downstream notices: contact_rewards walks for <contact>/<pN> ids
        # and silently ignores anything else, so every generation metric looks
        # healthy while the rollouts are corrupt.
        sampling_kwargs = dict(
            n=1,
            temperature=applied.temperature,
            top_p=applied.top_p,
            top_k=applied.top_k if applied.top_k is not None else -1,
            max_tokens=applied.max_output_tokens,
            stop_token_ids=[cr.END_ID],
            logprobs=1,
        )
        if self.vocab_size:
            sampling_kwargs["allowed_token_ids"] = list(range(self.vocab_size))
        params = SamplingParams(**sampling_kwargs)
        gen_started = time.time()
        outputs = inference_ctx.llm.generate(
            [TokensPrompt(prompt_token_ids=s["ids"]) for s in specs], params, use_tqdm=False)
        gen_elapsed = time.time() - gen_started

        trace = applied.as_trace()
        # Score every rollout first; the document term needs the whole group, so
        # Rollout objects cannot be built until the group is complete.
        scored: dict[str, list[dict]] = {}
        n_empty = 0
        batch_scored = batch_correct = 0

        for spec, output in zip(specs, outputs, strict=True):
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            if not token_ids:
                n_empty += 1
                continue
            if self.vocab_size:
                # Belt and braces: if `allowed_token_ids` is ever dropped or
                # renamed by a vLLM bump, fail LOUDLY here rather than hand the
                # trainer ids it cannot embed. A NaN 25 minutes into a gang is a
                # much worse error message than this.
                worst = max(token_ids)
                if worst >= self.vocab_size:
                    n_oov = sum(t >= self.vocab_size for t in token_ids)
                    raise ValueError(
                        f"sampled {n_oov}/{len(token_ids)} token ids outside the model "
                        f"vocabulary (max id {worst}, vocab_size {self.vocab_size}). vLLM's "
                        "vocab padding is being sampled; `allowed_token_ids` did not take "
                        "effect. Training on these produces NaN on the first step."
                    )

            reward = cr.dense_rewards(
                token_ids, spec["pos_to_seq"], spec["gt"],
                mode="plain", precision_baseline=self._p_bar, err_decay=self.err_decay,
            )
            # vLLM guarantees the sampled token appears in its own logprob dict; a
            # KeyError here means the engine contract changed, and the resulting
            # length mismatch would silently corrupt the importance ratio.
            logprobs = np.array(
                [lp[t].logprob for t, lp in zip(token_ids, completion.logprobs, strict=True)],
                dtype=np.float32,
            )
            scored.setdefault(spec["entry_id"], []).append({
                "spec": spec, "token_ids": token_ids, "logprobs": logprobs,
                "reward": reward, "truncated": completion.finish_reason == "length",
                "pairs": self._emitted_pairs(token_ids, spec),
            })
            batch_scored += reward.diagnostics["n_contacts_scored"]
            batch_correct += reward.diagnostics["n_contacts_correct"]

        groups: list[RolloutGroup] = []
        diagnostics: list[dict[str, float]] = []
        group_diags: list[dict[str, float]] = []
        n_dropped = 0
        for entry_id, items in scored.items():
            # The replay buffer indexes group rewards as a rectangular array
            # (`batch.groups[0].rollouts` sets the width), so a short group is an
            # IndexError at write time rather than a soft failure.
            if len(items) != n_generations:
                n_dropped += 1
                continue
            doc_rewards, gdiag = self._document_rewards(entry_id, items)
            group_diags.append(gdiag)
            rollouts = []
            for item, doc in zip(items, doc_rewards):
                rollouts.append(Rollout(
                    env_name="contacts-v1-plain",
                    env_example_id=f"{item['spec']['entry_id']}:r{item['spec']['r']}",
                    prompt_tokens=np.array(item["spec"]["ids"], dtype=np.int32),
                    response_tokens=np.array(item["token_ids"], dtype=np.int32),
                    response_logprobs=item["logprobs"],
                    token_rewards=item["reward"].token_rewards,
                    # NB: `episode_reward` carries the DOCUMENT TERM, not a return.
                    # marin's Rollout is a fixed dataclass with no spare field and
                    # patching marin is off the table, so this is the only channel.
                    # ContactsDenseLoss applies RLOO centring on top; W&B's reward
                    # panels therefore read ~0 by design, and the raw consensus is
                    # logged separately below.
                    episode_reward=float(doc),
                    decoding=trace,
                    is_truncated=item["truncated"],
                ))
                diagnostics.append(item["reward"].diagnostics)
            groups.append(RolloutGroup(rollouts=rollouts))

        if n_dropped:
            logger.warning("[exp208] dropped %d ragged group(s)", n_dropped)
        if not groups:
            raise ValueError("every rollout group was ragged or empty; nothing to train on")

        p_bar_before = self._p_bar
        self._update_precision(batch_scored, batch_correct)
        metrics = self._metrics(diagnostics, group_diags, n_empty, n_dropped, applied)
        self._trace.event(
            "sample_done", call=self._calls, mode=mode, doc_term=self.doc_term,
            n_specs=len(specs), n_rollouts=len(diagnostics), n_groups=len(groups),
            n_empty=n_empty, n_dropped=n_dropped,
            gen_s=round(gen_elapsed, 1), total_s=round(time.time() - call_started, 1),
            max_output_tokens=applied.max_output_tokens, max_prompt_len=max_prompt_len,
            p_bar_before=round(p_bar_before, 4), p_bar_after=round(self._p_bar, 4),
            # The kill criteria have to be readable DURING training, not only at
            # eval — exp200 shipped a whole sweep with diversity collapse
            # unobservable because its only spread metric was within-rollout.
            consensus=metrics.get("contacts_plain/consensus_rprec"),
            union_over_r=metrics.get("contacts_plain/union_over_r"),
            inter_jaccard=metrics.get("contacts_plain/inter_rollout_jaccard"),
            n_pred=metrics.get("contacts_plain/n_pred"),
            precision=metrics.get("contacts_plain/precision"),
            doc_abs=metrics.get("contacts_plain/doc_reward_abs_mean"),
            doc_integral=metrics.get("contacts_plain/doc_reward_integral_mean"),
            rho_unscaled=metrics.get("contacts_plain/rho_unscaled"),
            step_abs=metrics.get("contacts_plain/step_reward_abs_mean"),
        )
        return groups, metrics

    def _emitted_pairs(self, token_ids, spec) -> set[tuple[int, int]]:
        """The dedup'd pair set this rollout would contribute to a vote matrix.

        Must match ``score_rollout_worker``'s accounting exactly, since the
        consensus scored here is meant to be the deployed metric: sequence-index
        pairs, separation >= 6, deduplicated within the rollout.
        """
        contacts = cr.walk_contacts(token_ids, spec["pos_to_seq"], spec["gt"])
        return {c.pair for c in contacts if c.pair is not None and c.reason == "ok"}

    def _document_rewards(self, entry_id: str, items: list[dict]):
        """The per-rollout document term, plus this group's diagnostics."""
        target = self._targets[entry_id]
        pairs, position = self._universe(target["L"])
        is_true = cs.truth_mask(pairs, target["gt"])
        n_true = int(is_true.sum())
        votes = cs.vote_counts([it["pairs"] for it in items], position, len(pairs))

        consensus, marginals = cs.loo_marginals(votes, is_true, n_true)
        gdiag = cs.group_diagnostics(votes, is_true, n_true)
        gdiag["consensus_rprec"] = consensus

        if self.doc_term == "consensus":
            doc = np.nan_to_num(marginals, nan=0.0)
        elif self.doc_term == "own_f1":
            doc = np.array([it["reward"].episode_reward for it in items], dtype=np.float64)
        else:
            doc = np.zeros(len(items), dtype=np.float64)
        # THE TWO TERMS MUST BE COMPARED PER TOKEN, NOT PER ROLLOUT. `doc` is one
        # scalar per rollout, but `dense_loss` broadcasts it to EVERY response
        # token, so its influence on the gradient is |doc| * n_response_tokens.
        # `token_rewards` is already a per-token array. Comparing the scalar to
        # the summed array understates the document term by the response length
        # (~400 tokens here) -- measured on the first nano: it read rho 0.02 when
        # the integrated ratio was 6.2, i.e. document-DOMINATED, which is the
        # regime that produced `RuntimeError: Loss is NaN`.
        lengths = np.array([len(it["reward"].token_rewards) for it in items], dtype=np.float64)
        gdiag["doc_reward_abs_mean"] = float(np.mean(np.abs(doc)))
        gdiag["doc_reward_integral_mean"] = float(np.mean(np.abs(doc) * lengths))
        gdiag["step_reward_abs_mean"] = float(
            np.mean([np.abs(it["reward"].token_rewards).sum() for it in items]))
        gdiag["mean_response_tokens"] = float(lengths.mean())
        return doc, gdiag

    def _update_precision(self, scored: float, correct: float) -> None:
        """EMA-update ``p_bar`` AFTER scoring, so one step shares one baseline."""
        if scored <= 0:
            return
        observed = correct / scored
        decay = self.precision_ema_decay
        self._p_bar = float(decay * self._p_bar + (1.0 - decay) * observed)

    def _metrics(self, diagnostics, group_diags, n_empty, n_dropped, applied) -> dict[str, float]:
        """Per-step diagnostics, including #208's group-level collapse detectors."""
        prefix = "contacts_plain"
        out = {
            f"{prefix}/precision_baseline": self._p_bar,
            f"{prefix}/n_rollouts": float(len(diagnostics)),
            f"{prefix}/n_empty_responses": float(n_empty),
            f"{prefix}/n_ragged_groups_dropped": float(n_dropped),
            f"{prefix}/max_output_tokens": float(applied.max_output_tokens),
            f"{prefix}/doc_term_is_consensus": float(self.doc_term == "consensus"),
        }
        for key in ("precision", "n_pred", "n_gt", "first_f1",
                    "n_duplicate", "n_malformed", "n_unmapped", "n_too_close", "n_truncated"):
            values = [d[key] for d in diagnostics if not math.isnan(d.get(key, math.nan))]
            if values:
                out[f"{prefix}/{key}"] = float(np.mean(values))
        # `first_f1` is the whole rollout's F1 in plain mode (one section), so
        # name it for what it is rather than carrying exp200's section vocabulary.
        if f"{prefix}/first_f1" in out:
            out[f"{prefix}/rollout_f1"] = out.pop(f"{prefix}/first_f1")

        rename = {"mean_jaccard": "inter_rollout_jaccard"}
        for key in ("consensus_rprec", "union", "union_over_r", "mean_jaccard", "vote_entropy",
                    "mean_vote_top_r", "mean_pairs_per_rollout", "mean_response_tokens",
                    "doc_reward_abs_mean", "doc_reward_integral_mean", "step_reward_abs_mean"):
            values = [g[key] for g in group_diags if not math.isnan(g.get(key, math.nan))]
            if values:
                out[f"{prefix}/{rename.get(key, key)}"] = float(np.mean(values))
        # rho: the measured balance between the two reward terms, INTEGRATED over
        # the response so the broadcast document scalar is counted once per token
        # the way the loss actually applies it. #208's primary axis is this ratio,
        # and raw lambdas do not express it -- the calibrated lam_doc came out ~65x
        # from a plausible-looking guess. Reported UNSCALED by lam_step/lam_doc,
        # which live in the loss: rho = lam_doc * this / lam_step.
        step_abs = out.get(f"{prefix}/step_reward_abs_mean", 0.0)
        if step_abs > 0:
            out[f"{prefix}/rho_unscaled"] = (
                out.get(f"{prefix}/doc_reward_integral_mean", 0.0) / step_abs)
        return out


__all__ = ["ContactsV1RLEnv", "DOC_TERMS", "seed_from"]
