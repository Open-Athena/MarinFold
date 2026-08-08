# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``MarinEnv`` for exp200: contacts-v1 rollouts with a dense per-contact reward.

WHY THIS DRIVES vLLM DIRECTLY. marin's ``vLLMInferenceContext.batch_completions``
renders every prompt through a chat template (``Qwen3Renderer``), and the
contacts-v1 vocab has neither a chat template nor ``<|im_end|>``. The context
class is chosen by a hardcoded ``if inference_type == "vllm"`` in
``rollout_worker.py``, so a subclass cannot be injected — and patching marin is
off the table. What IS available: ``inference_ctx.llm`` (the vLLM engine) and
``inference_ctx.tokenizer`` are plain attributes, and the renderer is touched
nowhere except inside ``batch_completions``. So this environment builds prompt
token ids itself and calls ``llm.generate`` with ``TokensPrompt`` — which is
exactly what exp163's validated ``gen_rollouts_worker_exp163.py`` does, making
the Phase-1 parity check a like-for-like comparison rather than a reimplementation.

Set ``VLLMEngineConfig.canonical_model_name`` to something containing "qwen" so
the unused renderer still constructs (see ``rl_config.py``); ``Renderer.__init__``
only stores the tokenizer, and the ``<|im_end|>`` assertion lives in a property
this path never reaches.

TWO LESSONS FROM #163 ARE BAKED IN. Prompts come from a pre-built pool rather
than being generated here, so the rollout pod needs no ``marinfold`` install; and
the mode sentinel is set BY TOKEN ID, never by string, because the published
contacts-v1 tokenizer and exp163's renamed tokenizer disagree on how id 7 is
spelled.
"""

import logging
import math
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

import contact_rewards as cr

logger = logging.getLogger(__name__)

DOC_TOKEN_BY_MODE = {"plain": cr.PLAIN_DOC_ID, "multi": cr.MULTI_DOC_ID}

# exp98's per-target budget for a single contacts-v1 statement section.
PLAIN_TOKENS_PER_RESIDUE = 4
PLAIN_TOKEN_SLACK = 64
# exp163's per-section budget: 3 tokens per contact plus section overhead.
MULTI_TOKENS_PER_CONTACT = 3
MULTI_SECTION_SLACK = 8


class ContactsV1RLEnv(MarinEnv):
    """Sample contacts-v1 rollouts and score them with a dense per-contact reward.

    One instance serves one lesson. #200 trains a 50:50 mix of the base task and
    the multi-draft task, which is two instances at equal curriculum weight
    rather than one environment with a branch.

    Args:
        targets_path: Parquet with ``entry_id, L, gt_contacts`` (``gt_contacts``
            flat ``[i0, j0, i1, j1, ...]`` in sequence-index space).
        prompts_path: Directory of ``<entry_id>.parquet``, each with ``r, L,
            prefix, seq_positions`` — exp163's ``gen_prompts_exp163.py`` output.
        mode: ``"plain"`` (``<contacts-v1>``) or ``"multi"`` (``<contacts-v1.multi>``).
        max_sections: Sampler-side cap on scored candidate sections. Must match
            the cap used at eval, or the primary metric is not comparable.
        section_contacts: Expected contacts per section, for the token budget.
        err_decay: Geometric decay on the penalty for repeat errors within a section.
        precision_ema_decay: Smoothing for ``p_bar``. Higher is smoother.
        initial_precision: ``p_bar`` before any rollouts have been scored. 0.30 is
            arm F's measured per-contact precision on held-out proteins.
        max_model_len: Engine context length; the response budget is clamped to fit.
        eval_fraction: Fraction of proteins held out for ``mode="eval"``.
        seed: Seed for the train/eval split (NOT for sampling, which comes from
            the caller's ``prng_key``).
        limit: Optional cap on proteins, for smoke tests.
        fetch_workers: Concurrency for per-protein prompt reads.
    """

    def __init__(
        self,
        *,
        targets_path: str,
        prompts_path: str,
        mode: str = "multi",
        max_sections: int = 8,
        section_contacts: int = 220,
        err_decay: float = 0.5,
        precision_ema_decay: float = 0.9,
        initial_precision: float = 0.30,
        max_model_len: int = 8192,
        eval_fraction: float = 0.05,
        seed: int = 0,
        limit: int | None = None,
        fetch_workers: int = 8,
    ):
        if mode not in DOC_TOKEN_BY_MODE:
            raise ValueError(f"mode must be one of {sorted(DOC_TOKEN_BY_MODE)}, got {mode!r}")
        self.mode = mode
        self.doc_token_id = DOC_TOKEN_BY_MODE[mode]
        self.max_sections = max_sections if mode == "multi" else 1
        self.section_contacts = section_contacts
        self.err_decay = err_decay
        self.precision_ema_decay = precision_ema_decay
        self.max_model_len = max_model_len
        self.prompts_path = prompts_path.rstrip("/")
        self.fetch_workers = fetch_workers

        self._p_bar = float(initial_precision)
        self._prompt_cache: dict[str, list[dict]] = {}

        self._targets = self._load_targets(targets_path, limit)
        ids = sorted(self._targets)
        order = np.random.default_rng(seed).permutation(len(ids))
        n_eval = max(1, int(round(eval_fraction * len(ids)))) if eval_fraction > 0 else 0
        self._eval_ids = [ids[i] for i in order[:n_eval]]
        self._train_ids = [ids[i] for i in order[n_eval:]]
        logger.info(
            "[exp200/%s] %d proteins (%d train / %d eval), max_sections=%d",
            mode, len(ids), len(self._train_ids), len(self._eval_ids), self.max_sections,
        )

    @staticmethod
    def _load_targets(path: str, limit: int | None) -> dict[str, dict]:
        with fsspec.open(path, "rb") as fh:
            table = pq.read_table(fh, columns=["entry_id", "L", "gt_contacts"])
        out: dict[str, dict] = {}
        for entry_id, length, pairs in zip(
            table["entry_id"].to_pylist(), table["L"].to_pylist(), table["gt_contacts"].to_pylist()
        ):
            # `gt_contacts` is a list of [i, j] PAIRS in sequence-index space, the
            # schema exp98's select_targets.py writes and exp163's worker reads as
            # `{(int(i), int(j)) for i, j in t["gt_contacts"]}`. It is not a flat
            # [i0, j0, i1, j1, ...] vector — that is the shape the rollout workers
            # use for PREDICTIONS, and confusing the two silently halves the
            # ground truth and pairs up unrelated residues.
            gt = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs}
            gt = {p for p in gt if cr.in_band(p)}
            if not gt:
                continue
            out[entry_id] = {"L": int(length), "gt": gt}
            if limit is not None and len(out) >= limit:
                break
        if not out:
            raise ValueError(f"no usable targets in {path}")
        return out

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
                table["r"].to_pylist(), table["prefix"].to_pylist(), table["seq_positions"].to_pylist()
            )
        ]
        self._prompt_cache[entry_id] = rows
        return rows

    def _build_prompt_ids(self, tokenizer, prefix: str) -> list[int]:
        """Tokenize a prompt and swap in this lesson's doc-type sentinel.

        Mirrors ``gen_rollouts_worker_exp163.py`` exactly: tokenize, then replace
        index 0 by id. The published contacts-v1 tokenizer spells id 7
        ``<contacts-and-distances-v1>`` while exp163's renamed one spells it
        ``<contacts-v1.multi>``, so doing this by string would silently produce a
        different prompt depending on which tokenizer shipped with the weights.

        Uses ``.encode()``, not the HF ``__call__`` interface exp163's worker used:
        ``inference_ctx.tokenizer`` is levanter's ``HfMarinTokenizer`` wrapper, which
        is NOT callable and returns a plain ``list[int]`` rather than a
        ``BatchEncoding``.
        """
        ids = list(tokenizer.encode(prefix, add_special_tokens=False))
        if not ids:
            raise ValueError("empty prompt tokenization")
        if ids[0] != cr.PLAIN_DOC_ID:
            raise ValueError(
                f"prompt does not start with <contacts-v1> (id {cr.PLAIN_DOC_ID}); got id {ids[0]}. "
                "The prompt pool was built for a different document structure."
            )
        return [self.doc_token_id] + ids[1:]

    def _response_budget(self, max_prompt_len: int, max_length: int, declared: int | None = None) -> int:
        """Response token budget for one batch.

        Three bounds, all real. The task formula (exp163's per-section budget for
        multi, exp98's per-residue budget for plain) is what the model actually
        needs; the context window is what the engine can hold; and ``declared`` is
        the lesson's own ``max_output_tokens``, which must not be exceeded because
        ``curriculum.max_seq_len`` is derived from it and ``train_batch`` raises
        when a padded sequence overruns that.
        """
        if self.mode == "multi":
            per_section = MULTI_TOKENS_PER_CONTACT * self.section_contacts + MULTI_SECTION_SLACK
            budget = per_section * self.max_sections
        else:
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
        order per rollout, and #163's candidate spread depends on it.
        """
        from vllm import SamplingParams, TokensPrompt

        pool = self._train_ids if mode == "train" else self._eval_ids
        if not pool:
            raise ValueError(f"no proteins available for mode={mode!r}")

        seed = int(jax.random.randint(prng_key, (), 0, 1_000_000))
        rng = np.random.default_rng(seed)
        n_take = min(n_examples, len(pool))
        entry_ids = [pool[i] for i in rng.choice(len(pool), size=n_take, replace=False)]

        with ThreadPoolExecutor(max_workers=self.fetch_workers) as pool_exec:
            prompt_rows = list(pool_exec.map(self._prompts_for, entry_ids))

        specs: list[dict] = []
        for entry_id, rows in zip(entry_ids, prompt_rows):
            if not rows:
                logger.warning("[exp200/%s] no prompts for %s; skipping", self.mode, entry_id)
                continue
            picks = rng.choice(len(rows), size=n_generations, replace=len(rows) < n_generations)
            for pick in picks:
                row = rows[int(pick)]
                specs.append(
                    {
                        "entry_id": entry_id,
                        "r": row["r"],
                        "ids": self._build_prompt_ids(inference_ctx.tokenizer, row["prefix"]),
                        "pos_to_seq": {int(p): i for i, p in enumerate(row["seq_positions"])},
                        "gt": self._targets[entry_id]["gt"],
                    }
                )
        if not specs:
            raise ValueError("no prompts were built for this step")

        max_prompt_len = max(len(s["ids"]) for s in specs)
        max_length = max(self._targets[s["entry_id"]]["L"] for s in specs)
        applied = replace(
            decoding,
            max_output_tokens=self._response_budget(
                max_prompt_len, max_length, declared=decoding.max_output_tokens
            ),
            stop_token_ids=[cr.END_ID],
            # TPU vLLM rejects per-request seeds ("JAX does not support per-request
            # seed"); engine-level seeding is VLLMEngineConfig.seed.
            seed=None,
        )
        params = SamplingParams(
            n=1,
            temperature=applied.temperature,
            top_p=applied.top_p,
            top_k=applied.top_k if applied.top_k is not None else -1,
            max_tokens=applied.max_output_tokens,
            stop_token_ids=[cr.END_ID],
            logprobs=1,
        )
        outputs = inference_ctx.llm.generate(
            [TokensPrompt(prompt_token_ids=s["ids"]) for s in specs], params, use_tqdm=False
        )

        trace = applied.as_trace()
        by_entry: dict[str, list[Rollout]] = {}
        diagnostics: list[dict[str, float]] = []
        n_empty = 0
        batch_scored = batch_correct = 0

        for spec, output in zip(specs, outputs, strict=True):
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            if not token_ids:
                n_empty += 1
                continue

            reward = cr.dense_rewards(
                token_ids,
                spec["pos_to_seq"],
                spec["gt"],
                mode=self.mode,
                precision_baseline=self._p_bar,
                err_decay=self.err_decay,
                max_sections=self.max_sections,
            )
            # vLLM guarantees the sampled token appears in its own logprob dict;
            # a KeyError here means the engine contract changed and the resulting
            # length mismatch would silently corrupt the importance ratio.
            logprobs = np.array(
                [lp[t].logprob for t, lp in zip(token_ids, completion.logprobs, strict=True)],
                dtype=np.float32,
            )
            by_entry.setdefault(spec["entry_id"], []).append(
                Rollout(
                    env_name=f"contacts-v1-{self.mode}",
                    env_example_id=f"{spec['entry_id']}:r{spec['r']}",
                    prompt_tokens=np.array(spec["ids"], dtype=np.int32),
                    response_tokens=np.array(token_ids, dtype=np.int32),
                    response_logprobs=logprobs,
                    token_rewards=reward.token_rewards,
                    episode_reward=reward.episode_reward,
                    decoding=trace,
                    is_truncated=completion.finish_reason == "length",
                )
            )
            diagnostics.append(reward.diagnostics)
            batch_scored += reward.diagnostics["n_contacts_scored"]
            batch_correct += reward.diagnostics["n_contacts_correct"]

        # The replay buffer indexes group rewards as a rectangular array
        # (``batch.groups[0].rollouts`` sets the width), so a short group is an
        # IndexError at write time rather than a soft failure.
        groups = [RolloutGroup(rollouts=rs) for rs in by_entry.values() if len(rs) == n_generations]
        n_dropped = len(by_entry) - len(groups)
        if n_dropped:
            logger.warning("[exp200/%s] dropped %d ragged group(s)", self.mode, n_dropped)
        if not groups:
            raise ValueError("every rollout group was ragged or empty; nothing to train on")

        self._update_precision(batch_scored, batch_correct)
        return groups, self._metrics(diagnostics, n_empty, n_dropped, applied)

    def _update_precision(self, scored: float, correct: float) -> None:
        """EMA-update ``p_bar`` AFTER scoring, so one step shares one baseline."""
        if scored <= 0:
            return
        observed = correct / scored
        decay = self.precision_ema_decay
        self._p_bar = float(decay * self._p_bar + (1.0 - decay) * observed)

    def _metrics(
        self,
        diagnostics: list[dict[str, float]],
        n_empty: int,
        n_dropped: int,
        applied: DecodingConfig,
    ) -> dict[str, float]:
        """Per-step generation diagnostics — the collapse detectors from #200.

        ``n_pred_per_section`` falling and ``mean_jaccard`` rising are the two
        signatures the experiment is watching for: reward hacking toward silence,
        and diversity collapse.
        """
        prefix = f"contacts_{self.mode}"
        out = {
            f"{prefix}/precision_baseline": self._p_bar,
            f"{prefix}/n_rollouts": float(len(diagnostics)),
            f"{prefix}/n_empty_responses": float(n_empty),
            f"{prefix}/n_ragged_groups_dropped": float(n_dropped),
            f"{prefix}/max_output_tokens": float(applied.max_output_tokens),
        }
        if not diagnostics:
            return out
        for key in (
            "best_f1", "first_f1", "last_f1", "mean_f1", "precision",
            "n_sections", "n_sections_raw", "n_pred", "n_gt",
            "mean_jaccard", "frac_improving",
            "n_duplicate", "n_malformed", "n_unmapped", "n_too_close", "n_truncated",
        ):
            values = [d[key] for d in diagnostics if not math.isnan(d.get(key, math.nan))]
            if values:
                out[f"{prefix}/{key}"] = float(np.mean(values))
        sections = out.get(f"{prefix}/n_sections", 0.0)
        if sections > 0:
            out[f"{prefix}/n_pred_per_section"] = out.get(f"{prefix}/n_pred", 0.0) / sections
        return out


__all__ = ["ContactsV1RLEnv", "DOC_TOKEN_BY_MODE"]
