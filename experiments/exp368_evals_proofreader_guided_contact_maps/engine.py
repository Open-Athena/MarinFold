"""Cached contact-boundary search using the released frozen causal backbone.

Each proposal is sampled from the generator conditional on contact grammar and
eligible, previously unseen endpoints. Termination is sampled once per step from
the boundary distribution and is independent of the proofreader. The control
uses exactly the same sampler with one proposal. A separate unconstrained vLLM
baseline measures any effect of these grammar and uniqueness constraints.

Candidate draws already carry generator probability. Resampling uses only q^beta
or a rank gate, avoiding an accidental second factor of generator probability.
Forks own their cache objects; shared immutable prefix tensors are safe because
the pinned Transformers DynamicLayer appends with torch.cat.
"""

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from transformers import DynamicCache


@dataclass(frozen=True)
class Policy:
    """A complete policy specification, including its compute controls."""

    name: str
    candidates: int = 1
    fraction: float = 1.0
    beta: float = 0.0
    exploration: float = 0.0
    lookahead: int = 0
    repeats: int = 1
    objective: str = 'new'
    particles: int = 1
    block: int = 8
    repair: bool = False
    candidate_batch: int = 16


@dataclass(frozen=True)
class State:
    """Same-length batched prefixes with immutable cached evidence."""

    cache: DynamicCache
    tokens: torch.Tensor
    hidden: torch.Tensor
    prompt_hidden: torch.Tensor
    triples: torch.Tensor
    pairs: torch.Tensor
    prompt_length: int
    finished: bool = False

    @property
    def count(self) -> int:
        return self.pairs.shape[1]


def fork_cache(cache: DynamicCache, indices: torch.Tensor) -> DynamicCache:
    """Select branches without mutating the source cache or its tensors."""
    return DynamicCache([(key.index_select(0, indices), value.index_select(0, indices))
                         for key, value, *_ in cache])


def select(state: State, indices: torch.Tensor) -> State:
    """Return independent cache objects for the selected prefix branches."""
    return replace(state, cache=fork_cache(state.cache, indices),
        tokens=state.tokens.index_select(0, indices), hidden=state.hidden.index_select(0, indices),
        prompt_hidden=state.prompt_hidden.index_select(0, indices),
        triples=state.triples.index_select(0, indices), pairs=state.pairs.index_select(0, indices))


def selection_weights(scores: torch.Tensor, policy: Policy) -> torch.Tensor:
    """Weight sampled proposals without double-counting their generator prior."""
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite candidate scores')
    weights = scores.clamp_min(1e-8).pow(policy.beta)
    if policy.fraction < 1:
        count = max(1, math.ceil(len(scores) * policy.fraction))
        # Stable ordering gives a reproducible tie break; proposal order is random.
        order = torch.argsort(scores, descending=True, stable=True)
        keep = torch.zeros_like(weights)
        keep[order[:count]] = 1
        weights = weights * keep
    weights = weights / weights.sum()
    return (1 - policy.exploration) * weights + policy.exploration / len(scores)


class Engine:
    """Generate and score contact alternatives using one verified backbone."""

    def __init__(self, reader: nn.Module, lm_head: nn.Module, tokenizer, device: str = 'cuda'):
        self.reader = reader.eval()
        self.backbone = reader.backbone
        self.lm_head = lm_head.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.contact_id = tokenizer.convert_tokens_to_ids('<contact>')
        self.end_id = tokenizer.convert_tokens_to_ids('<end>')
        self.position_ids = torch.tensor([tokenizer.convert_tokens_to_ids(f'<p{i}>')
                                         for i in range(2000)], device=device)
        self.generated_tokens = 0
        self.scored_contacts = 0
        self.candidate_diagnostics = []

    def autocast(self):
        """Keep cached search in FP32; BF16 incremental/full GEMMs drifted."""
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                              enabled=False)

    @torch.inference_mode()
    def start(self, prompt: list[int]) -> State:
        """Encode a prompt once; later steps reuse its causal features."""
        ids = torch.tensor([prompt], device=self.device)
        with self.autocast():
            result = self.backbone(input_ids=ids, use_cache=True)
        hidden = result.last_hidden_state[:, -1]
        width = hidden.shape[-1]
        return State(result.past_key_values, ids, hidden, hidden,
                     hidden.new_empty((1, 0, 3 * width)),
                     ids.new_empty((1, 0, 2)), len(prompt))

    @torch.inference_mode()
    def advance(self, state: State, tokens: torch.Tensor) -> tuple[State, torch.Tensor]:
        """Append tokens to an owned branch cache; preserve the supplied state."""
        cache = DynamicCache([(key, value) for key, value, *_ in state.cache])
        with self.autocast():
            result = self.backbone(input_ids=tokens, past_key_values=cache, use_cache=True)
        hidden = result.last_hidden_state
        self.generated_tokens += tokens.numel()
        return replace(state, cache=result.past_key_values,
            tokens=torch.cat([state.tokens, tokens], dim=1), hidden=hidden[:, -1]), hidden

    @torch.inference_mode()
    def probabilities(self, hidden: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """Apply grammar masks before the generator's temperature-1 top-p-.95."""
        with self.autocast():
            logits = self.lm_head(hidden).float()
        logits = logits.masked_fill(~allowed, -torch.inf)
        ordered, indices = logits.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(-1).cumsum(-1)
        remove = cumulative - ordered.softmax(-1) >= .95
        ordered = ordered.masked_fill(remove, -torch.inf)
        return torch.zeros_like(logits).scatter(-1, indices, ordered.softmax(-1))

    def sample(self, hidden: torch.Tensor, allowed: torch.Tensor,
               rng: torch.Generator) -> torch.Tensor:
        """Draw one legal next token per branch with a private random stream."""
        return torch.multinomial(self.probabilities(hidden, allowed), 1, generator=rng)

    def boundary_end(self, state: State, rng: torch.Generator) -> bool:
        """Sample EOS with no proofreader influence and no extra EOS candidates."""
        allowed = torch.zeros((1, self.lm_head.out_features), device=self.device, dtype=torch.bool)
        allowed[:, [self.contact_id, self.end_id]] = True
        return self.sample(state.hidden, allowed, rng).item() == self.end_id

    @torch.inference_mode()
    def extend(self, state: State, length: int, offset: int, rng: torch.Generator) -> State:
        """Sample one complete, legal, distinct contact for every current branch."""
        batch = state.tokens.shape[0]
        position_tokens = self.position_ids[(torch.arange(length, device=self.device) + offset) % 2000]
        residues = torch.arange(length, device=self.device)
        legal = (residues[:, None] - residues[None]).abs() >= 6
        legal = legal[None].expand(batch, -1, -1).clone()
        if state.count:
            rows = torch.arange(batch, device=self.device)[:, None]
            a, b = state.pairs.unbind(-1)
            legal[rows, a, b] = False
            legal[rows, b, a] = False
        if not legal.any(dim=-1).any(dim=-1).all():
            raise ValueError('Contact universe exhausted')
        previous_triples, previous_pairs = state.triples, state.pairs
        current, marker_hidden = self.advance(state, state.tokens.new_full((batch, 1), self.contact_id))
        # The input branch can carry tens of GB of KV tensors. Retain its small
        # readout evidence, but release this local reference before the next append.
        del state
        allowed = torch.zeros((batch, self.lm_head.out_features), device=self.device, dtype=torch.bool)
        allowed[:, position_tokens] = legal.any(dim=-1)
        first = self.sample(current.hidden, allowed, rng)
        inverse = torch.full((self.lm_head.out_features,), -1, device=self.device, dtype=torch.long)
        inverse[position_tokens] = residues
        a = inverse[first[:, 0]]
        current, first_hidden = self.advance(current, first)
        allowed.zero_()
        allowed[:, position_tokens] = legal[torch.arange(batch, device=self.device), a]
        second = self.sample(current.hidden, allowed, rng)
        b = inverse[second[:, 0]]
        current, second_hidden = self.advance(current, second)
        triple = torch.cat([marker_hidden, first_hidden, second_hidden], dim=-1)
        pairs = torch.stack([a, b], dim=-1).sort(dim=-1).values[:, None]
        return replace(current, triples=torch.cat([previous_triples, triple], dim=1),
                       pairs=torch.cat([previous_pairs, pairs], dim=1))

    @torch.inference_mode()
    def score(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute bidirectional judgments on cached causal contact features."""
        batch, contacts = state.pairs.shape[:2]
        if not contacts:
            return state.hidden.new_empty((batch, 0)), state.hidden.new_zeros(batch)
        counts = state.hidden.new_tensor([state.prompt_length, contacts], dtype=torch.float32).log1p()
        features = torch.cat([state.prompt_hidden, state.hidden, counts[None].expand(batch, -1)], dim=-1)
        mask = torch.ones((batch, contacts), dtype=torch.bool, device=self.device)
        with self.autocast():
            logits, pooled = self.reader.contact_head(state.triples, features, mask)
            recall = self.reader.recall_head(pooled).squeeze(-1).float().sigmoid()
        self.scored_contacts += batch * contacts
        return logits.sigmoid(), recall

    def finish(self, state: State) -> State:
        """Append a genuinely sampled end token before final proofreading."""
        current, _ = self.advance(state, state.tokens.new_full((1, 1), self.end_id))
        return replace(current, finished=True)

    def choose(self, state: State, policy: Policy, length: int, offset: int,
               rng: torch.Generator, max_contacts: int) -> State:
        """Branch, optionally look ahead, then commit only one proposed contact."""
        if policy.candidates == 1:
            return self.extend(state, length, offset, rng)
        objectives, completions, pairs, immediate = [], [], [], []
        batch_limit = min(policy.candidate_batch, 4) if policy.lookahead else policy.candidate_batch
        for start in range(0, policy.candidates, batch_limit):
            batch = min(batch_limit, policy.candidates - start)
            candidates = self.extend(select(state, state.tokens.new_zeros(batch)), length, offset, rng)
            q, _ = self.score(candidates)
            immediate.append(q[:, -1].clone())
            objective = q[:, -1] if policy.objective == 'new' else q.sum(-1)
            depth = min(policy.lookahead, max_contacts - candidates.count)
            if depth:
                # Fixed-length future evidence is a diagnostic, not terminal value.
                index = candidates.tokens.new_tensor(list(range(batch)) * policy.repeats)
                future = select(candidates, index)
                for _ in range(depth):
                    future = self.extend(future, length, offset, rng)
                fq, _ = self.score(future)
                scores = fq[:, candidates.count - 1] if policy.objective == 'new' else fq[:, :candidates.count].sum(-1)
                objective = scores.reshape(policy.repeats, batch).mean(0)
                del future, fq
            objectives.append(objective)
            completions.append(candidates.tokens[:, -3:].clone())
            pairs.append(candidates.pairs[:, -1:].clone())
            del candidates, q
        objective = torch.cat(objectives)
        chosen = torch.multinomial(selection_weights(objective, policy), 1, generator=rng)
        if state.count < 4 or state.count % 8 == 0:
            candidate_pairs = torch.cat(pairs)[:, 0].tolist()
            immediate_q = torch.cat(immediate).tolist()
            score_values = objective.tolist()
            selected = chosen.item()
            for index, pair in enumerate(candidate_pairs):
                self.candidate_diagnostics.append(dict(prefix_contacts=state.count, proposal=index,
                    pair=pair, immediate_probability=immediate_q[index], selection_score=score_values[index],
                    selected=index == selected, lookahead=min(policy.lookahead, max_contacts - state.count - 1)))
        # Recompute only the selected triple, keeping memory independent of K.
        committed, hidden = self.advance(state, torch.cat(completions).index_select(0, chosen))
        return replace(committed, triples=torch.cat([state.triples, hidden.flatten(1)[:, None]], dim=1),
                       pairs=torch.cat([state.pairs, torch.cat(pairs).index_select(0, chosen)], dim=1))

    def rollout(self, prompt: list[int], length: int, offset: int, policy: Policy,
                seed: int, contact_limit: int | None = None) -> State:
        """Run one guided trajectory with generator-controlled termination."""
        rng = torch.Generator(device=self.device).manual_seed(seed)
        self.candidate_diagnostics = []
        state = self.start(prompt)
        budget = min(6 * length + 128, 8192 - len(prompt) - 1)
        maximum = min(budget // 3, contact_limit or 8192, (length - 6) * (length - 5) // 2)
        if policy.particles > 1:
            return self.population(state, length, offset, policy, rng, maximum)
        if policy.repair:
            ordinary = self.continue_rollout(state, length, offset, Policy('repair-base'), rng, maximum)
            if ordinary.count < 2:
                return ordinary
            q, _ = self.score(ordinary)
            # The earliest half is where later evidence can change a commitment.
            # Re-encoding the retained prefix counts toward this policy's timing.
            cut = int(q[0, :max(1, ordinary.count // 2)].argmin())
            retained = self.reencode(prompt, ordinary.tokens[0, len(prompt):len(prompt) + 3 * cut],
                                     ordinary.pairs[:, :cut])
            repaired = self.continue_rollout(retained, length, offset, replace(policy, repair=False), rng, maximum)
            return repaired
        return self.continue_rollout(state, length, offset, policy, rng, maximum)

    def continue_rollout(self, state: State, length: int, offset: int, policy: Policy,
                         rng: torch.Generator, maximum: int) -> State:
        """Continue a supplied prefix with ordinary generator stopping."""
        while state.count < maximum:
            if self.boundary_end(state, rng):
                return self.finish(state)
            state = self.choose(state, policy, length, offset, rng, maximum)
        return state

    def reencode(self, prompt: list[int], completion: torch.Tensor, pairs: torch.Tensor) -> State:
        """Rebuild a retained repair prefix from its exact original tokens."""
        state = self.start(prompt)
        if not len(completion):
            return state
        state, hidden = self.advance(state, completion[None])
        return replace(state, triples=hidden.reshape(1, -1, 3 * hidden.shape[-1]), pairs=pairs.clone())

    def utility(self, state: State) -> float:
        """Estimate current F1 for comparing prefixes of different lengths."""
        q, recall = self.score(state)
        if not state.count:
            return 0.0
        precision = q.mean()
        return (2 * precision * recall[0] / (precision + recall[0]).clamp_min(1e-8)).item()

    def population(self, initial: State, length: int, offset: int, policy: Policy,
                   rng: torch.Generator, maximum: int) -> State:
        """Keep diverse prefixes after blocks; return one final map per run.

        Each live parent produces two continuations. Finished prefixes retain
        their observed stopping status. A greedy Jaccard penalty discourages
        collapse without treating descendants as independent map votes.
        """
        live = [initial]
        completed = []
        while live:
            proposals = []
            for parent in live:
                copies = policy.particles if parent.count == 0 else 2
                for _ in range(copies):
                    state = select(parent, parent.tokens.new_tensor([0]))
                    for _ in range(min(policy.block, maximum - state.count)):
                        if self.boundary_end(state, rng):
                            state = self.finish(state)
                            break
                        state = self.extend(state, length, offset, rng)
                    score = self.utility(state)
                    if state.finished or state.count == maximum:
                        completed.append((score, state))
                    else:
                        proposals.append((score, state))
            completed = sorted(completed, key=lambda item: item[0], reverse=True)[:policy.particles]
            live = []
            selected_pairs = []
            while proposals and len(live) < policy.particles:
                sets = [set(map(tuple, state.pairs[0].tolist())) for _, state in proposals]
                adjusted = []
                for (score, _), pairs in zip(proposals, sets, strict=True):
                    overlap = max((len(pairs & other) / len(pairs | other) for other in selected_pairs), default=0.)
                    adjusted.append(score - .1 * overlap)
                index = max(range(len(proposals)), key=adjusted.__getitem__)
                _, state = proposals.pop(index)
                live.append(state)
                selected_pairs.append(sets[index])
        return max(completed, key=lambda item: item[0])[1]
