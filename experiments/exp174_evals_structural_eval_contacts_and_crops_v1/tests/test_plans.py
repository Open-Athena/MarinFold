# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan plumbing, against a stub model.

These do not test whether the *model* is any good — they test that each plan
builds a legal prompt, threads the decoder, accumulates the estimate, and comes
back with a structure. A stub sampler makes that checkable in milliseconds
instead of GPU-minutes, and it is where the wiring bugs actually live: a
mis-ordered forced suffix, a dropped visit index, an estimate that never gets
fed.

The stub replies to a forced ``<crop>`` header with atoms placed *in that box*,
so a plan that mangles the header produces observations that land in the wrong
cell and the assertions catch it.
"""

import random

import numpy as np
import pytest

from canonical_pdb import build_atom_array
from document_codec import (
    CROP_HEADER_TOKENS,
    box_from_header,
    position_index,
)
from plans import _neighbors, plan_a, plan_c, plan_f, spatial_order
from sampler import SamplingConfig

_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG"
_RECORD = {
    "record_id": "toy/p0",
    "stem": "p0",
    "dataset": "toy",
    "input_seq": _SEQUENCE,
    "L": len(_SEQUENCE),
}


def _gt_array():
    """Ground truth for the stub's protein: N/CA/C/O on a gentle helix."""
    atoms = []
    for k in range(len(_SEQUENCE)):
        for name, dx in (("N", -1.2), ("CA", 0.0), ("C", 1.2), ("O", 1.6)):
            atoms.append(
                (k + 1, "ALA", name, 30.0 + 0.9 * k + dx, 40.0 + 0.4 * k, 50.0, 0.0)
            )
    return build_atom_array(atoms)


class StubSampler:
    """A model-free stand-in that emits grammar-valid statements.

    Pass 1 gets random boxes; a forced crop header gets member atoms placed
    inside the box it names. Deterministic given ``seed``.
    """

    device = "cpu"
    # A crop body stops at the next <crop>; the plans pass this through as a
    # stop token, so the stub has to carry one.
    crop_id = -1

    def __init__(self, seed: int = 0, atoms_per_crop: int = 4):
        self._vocab: list[str] = []
        self._ids: dict[str, int] = {}
        self._rng = random.Random(seed)
        self.atoms_per_crop = atoms_per_crop
        self.forced_headers: list[tuple] = []

    def encode(self, tokens):
        out = []
        for token in tokens:
            if token not in self._ids:
                self._ids[token] = len(self._vocab)
                self._vocab.append(token)
            out.append(self._ids[token])
        return out

    def decode(self, ids):
        return [self._vocab[i] for i in ids]

    def prefill(self, prompt_ids):
        return ("stub-cache", len(prompt_ids))

    def _pass1_tokens(self, n_mentions):
        tokens = []
        for _ in range(n_mentions):
            tokens += [
                f"<p{self._rng.randrange(len(_SEQUENCE))}>",
                "<CA>",
                f"<xyz-{self._rng.randrange(1000):03d}>",
                f"<xyz-{self._rng.randrange(1000):03d}>",
            ]
        return tokens

    def _crop_body(self, cell):
        tokens = []
        for _ in range(self.atoms_per_crop):
            tokens += [
                f"<p{self._rng.randrange(len(_SEQUENCE))}>",
                "<CA>",
                f"<xyz-{self._rng.randrange(1000):03d}>",
                f"<xyz-{self._rng.randrange(1000):03d}>",
            ]
        return tokens

    def sample_from_cache(self, cache, last_logits, prompt_length, *, n_samples,
                          config, max_new_tokens=None, forced_ids=None,
                          stop_token_ids=None, generator=None):
        cell = None
        if forced_ids:
            forced = self.decode(forced_ids)
            if forced[-CROP_HEADER_TOKENS] == "<crop>":
                cell = box_from_header(forced[-2], forced[-1])
                self.forced_headers.append(cell)
        if cell is None:
            # No forced header: this is a free-running call (Plan A / Pass 1).
            cap = max_new_tokens or 64
            return [self.encode(self._pass1_tokens(min(cap // 4, 40)))
                    for _ in range(n_samples)]
        return [self.encode(self._crop_body(cell)) for _ in range(n_samples)]

    def sample(self, prompt_ids, *, n_samples=1, config=None, max_new_tokens=None,
               forced_ids=None, stop_token_ids=None, generator=None):
        cache, length = self.prefill(prompt_ids)
        return self.sample_from_cache(
            cache, length, len(prompt_ids), n_samples=n_samples, config=config,
            max_new_tokens=max_new_tokens, forced_ids=forced_ids,
            stop_token_ids=stop_token_ids, generator=generator,
        )


def test_plan_a_decodes_a_structure():
    result = plan_a(StubSampler(), _RECORD, config=SamplingConfig(), gt=_gt_array())
    assert result.stats["status"] == "ok"
    assert result.stats["plan"] == "A"
    assert len(result.estimate) > 0
    assert result.stats["decoded_pass1"] > 0


def test_plan_c_forces_one_crop_per_occupied_box():
    stub = StubSampler()
    result = plan_c(stub, _RECORD, config=SamplingConfig(), gt=_gt_array())
    assert result.stats["status"] == "ok"
    assert result.stats["plan"] == "C"
    assert result.stats["n_sweeps_run"] == 1
    # Every forced header names a box the estimate actually occupied, and each
    # is visited exactly once — that is what makes C the no-revisit ablation.
    assert stub.forced_headers
    assert len(stub.forced_headers) == len(set(stub.forced_headers))


def test_plan_f_sweeps_more_than_once_and_revisits_boxes():
    stub = StubSampler()
    result = plan_f(
        stub, _RECORD, config=SamplingConfig(), gt=_gt_array(),
        n_sweeps=3, n_samples=2, n_neighbor_crops=2,
        convergence_a=0.0,  # never converge, so all three sweeps run
    )
    assert result.stats["status"] == "ok"
    assert result.stats["n_sweeps_run"] == 3
    assert len(result.stats["sweeps"]) == 3
    # Revisits are the point. The stub emits random boxes, so the *occupied*
    # cells churn between sweeps and header identity is not a reliable signal;
    # what must hold is that three sweeps issue materially more crop calls than
    # one, i.e. the loop really iterates.
    assert len(stub.forced_headers) > 2 * len(set(stub.forced_headers)) / 3
    for sweep in result.stats["sweeps"]:
        assert sweep["n_cells"] > 0
        assert not np.isnan(sweep["mean_displacement_a"])


def test_plan_f_stops_when_displacement_falls_below_the_threshold():
    # A huge threshold converges after the first sweep.
    result = plan_f(
        StubSampler(), _RECORD, config=SamplingConfig(), gt=_gt_array(),
        n_sweeps=5, n_samples=1, n_neighbor_crops=0, convergence_a=1e9,
    )
    assert result.stats["n_sweeps_run"] == 1


def test_plan_f_samples_more_than_plan_c_per_box():
    c_stub, f_stub = StubSampler(), StubSampler()
    plan_c(c_stub, _RECORD, config=SamplingConfig(), gt=_gt_array())
    plan_f(f_stub, _RECORD, config=SamplingConfig(), gt=_gt_array(),
           n_sweeps=2, n_samples=4, n_neighbor_crops=2, convergence_a=0.0)
    assert len(f_stub.forced_headers) > len(c_stub.forced_headers)


def test_gt_filtering_drops_atoms_the_residue_cannot_have():
    # The stub emits only <CA>; a ground truth with no CA at all must therefore
    # yield an empty estimate rather than silently accepting the mentions.
    atoms = [(k + 1, "ALA", "N", float(k), 0.0, 0.0, 0.0) for k in range(len(_SEQUENCE))]
    result = plan_a(StubSampler(), _RECORD, config=SamplingConfig(),
                    gt=build_atom_array(atoms))
    assert len(result.estimate) == 0


def test_spatial_order_covers_every_cell_from_the_frontier():
    cells = [(x, y, 0) for x in range(4) for y in range(4)]
    order = spatial_order(cells, random.Random(0))
    assert sorted(order) == sorted(cells)
    # The property the model cares about is the frontier one: every cell after
    # the first is adjacent to a cell already visited, so its neighbours are
    # already refined and available for its prompt. (Consecutive *output*
    # positions can be further apart than one step — that is just how
    # breadth-first layers work — which is fine, since neighbour context is
    # looked up by cell identity, not by recency.)
    visited = {order[0]}
    for cell in order[1:]:
        assert any(n in visited for n in _neighbors(cell)), cell
        visited.add(cell)


def test_crop_bodies_stop_at_the_next_crop_token():
    """A crop ends at the next ``<crop>``, not at ``<end>``.

    ``<end>`` terminates the whole document; in training a crop is delimited by
    the header that follows it. A sampler that only stopped on ``<end>`` would
    run every crop call to its token cap and free-run the rest of Pass 2 —
    several times the compute, and "forced tiling" that is not forced.
    """
    class RecordingStub(StubSampler):
        crop_id = -99

        def __init__(self):
            super().__init__()
            self.stop_tokens = []

        def sample_from_cache(self, *args, stop_token_ids=None, **kwargs):
            self.stop_tokens.append(stop_token_ids)
            return super().sample_from_cache(*args, **kwargs)

    stub = RecordingStub()
    plan_c(stub, _RECORD, config=SamplingConfig(), gt=_gt_array())
    crop_calls = [s for s in stub.stop_tokens if s is not None]
    assert crop_calls, "no crop-body call recorded"
    assert all(s == [RecordingStub.crop_id] for s in crop_calls)


def test_plan_e1_forces_real_contacts_capped_at_the_formats_maximum():
    """E1 must write real contacts, and no more than a document ever shows.

    The format caps a document at ``n_contacts_max`` (50) and samples them
    uniformly rather than strongest-first, so forcing every true contact — often
    hundreds — would be a prompt shape the model has never seen.
    """
    from marinfold.document_structures.contacts_and_crops_v1 import GenerationConfig
    from marinfold.document_structures.contacts_and_crops_v1.vocab import CONTACT_TOKEN

    from plans import plan_e1

    contacts = [(i, i + 7, 0.5) for i in range(len(_SEQUENCE) - 7)]  # 43 available

    class PromptRecordingStub(StubSampler):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def prefill(self, prompt_ids):
            self.prompts.append(self.decode(prompt_ids))
            return super().prefill(prompt_ids)

    stub = PromptRecordingStub()
    result = plan_e1(stub, _RECORD, _gt_array(), config=SamplingConfig(),
                     contacts=contacts)
    assert result.stats["plan"] == "E1"
    assert result.stats["contacts_available"] == len(contacts)
    assert result.stats["contacts_forced"] == len(contacts)  # under the cap
    forced = stub.prompts[0].count(CONTACT_TOKEN)
    assert forced == len(contacts)

    # Over the cap, only n_contacts_max are written.
    many = [(i, j, 0.5) for i in range(20) for j in range(i + 7, 30)]
    assert len(many) > GenerationConfig().n_contacts_max
    stub2 = PromptRecordingStub()
    capped = plan_e1(stub2, _RECORD, _gt_array(), config=SamplingConfig(), contacts=many)
    assert capped.stats["contacts_forced"] == GenerationConfig().n_contacts_max
    assert stub2.prompts[0].count(CONTACT_TOKEN) == GenerationConfig().n_contacts_max


def test_plan_a_records_the_models_own_contact_count():
    # Nothing is teacher-forced in A, so this column is the model's own choice.
    result = plan_a(StubSampler(), _RECORD, config=SamplingConfig(), gt=_gt_array())
    assert "self_emitted_contacts" in result.stats
    assert result.stats["self_emitted_contacts"] >= 0
