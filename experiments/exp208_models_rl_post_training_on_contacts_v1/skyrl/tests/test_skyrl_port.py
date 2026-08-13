# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SkyRL port that do not need SkyRL installed — issue #208.

The port's two riskiest properties are framework-independent and can be pinned
here, on the workstation, without a GPU or a Ray cluster:

* the advantage estimator must NOT collapse per-token rewards the way GRPO does,
  because that discards the per-contact signal #208 exists to deliver;
* a constant-across-tokens advantage must be refused, since it is the SkyRL
  analogue of marin.rl's `np.full` failure mode — a run that trains, logs, and
  reads as "RL didn't help" while carrying no dense signal at all.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="the advantage estimator is a torch function")
# `importorskip` alone is not enough here: this workstation's system python has a
# stub `torch` that imports cleanly and has no `Tensor`, so the guard passed and
# the suite failed at collection instead of skipping. Check for the real thing.
if not hasattr(torch, "Tensor") or not getattr(torch, "__version__", None):
    pytest.skip("a stub `torch` is importable but unusable; run with a real torch env",
                allow_module_level=True)

from advantage import compute_contacts_dense_advantage  # noqa: E402


def test_per_token_structure_is_preserved_not_collapsed():
    """GRPO does `token_level_rewards.sum(dim=-1)` and broadcasts one scalar back.
    That is right for a trajectory reward and destroys #208's signal."""
    rewards = torch.tensor([[0.26, -0.07, 0.26, 0.0], [0.1, 0.1, -0.4, 0.2]])
    mask = torch.ones_like(rewards)
    adv, ret = compute_contacts_dense_advantage(rewards, mask, np.array([0, 0]))
    assert torch.allclose(adv, rewards)
    assert torch.allclose(ret, adv)
    # distinct per-token values survive
    assert adv[0, 0] != adv[0, 1]


def test_mask_zeroes_padding():
    rewards = torch.tensor([[0.3, 0.4, 0.5, 0.6]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    adv, _ = compute_contacts_dense_advantage(rewards, mask, np.array([0]))
    assert adv[0, 2] == 0.0 and adv[0, 3] == 0.0
    assert adv[0, 0] == pytest.approx(0.3)


def test_constant_advantage_is_refused():
    """The failure mode that looks like a null result instead of a bug."""
    rewards = torch.full((2, 5), 0.42)
    mask = torch.ones_like(rewards)
    with pytest.raises(ValueError, match="constant across its response tokens"):
        compute_contacts_dense_advantage(rewards, mask, np.array([0, 0]))


def test_shape_mismatch_is_loud():
    with pytest.raises(ValueError, match="does not match"):
        compute_contacts_dense_advantage(torch.zeros(2, 5), torch.zeros(2, 4), np.array([0, 0]))


def test_science_modules_are_byte_identical_to_the_marin_path():
    """contact_rewards.py and consensus.py must not drift between harnesses.

    They carry the reward definition and the exp89-verified metric; if the SkyRL
    copies diverge from the marin.rl ones, the two paths stop being comparable
    and the port can no longer be validated against the marin.rl reference.
    """
    here = Path(__file__).resolve().parents[1]
    there = here.parent
    for name in ("contact_rewards.py", "consensus.py"):
        assert (here / name).read_bytes() == (there / name).read_bytes(), (
            f"{name} has drifted between the SkyRL port and the marin.rl path"
        )


def test_constant_advantage_is_refused_even_with_padding():
    """The guard must see through padding.

    `advantages * response_mask` zeroes the padded tail, so taking `.std()` across
    the whole row makes a constant-per-token advantage look like it varies. The
    original guard did exactly that and passed arm C -- a purely document-level
    reward whose advantage is constant within each rollout by construction --
    through 125 steps of training without firing.
    """
    rewards = torch.zeros(2, 8)
    mask = torch.zeros(2, 8)
    mask[:, :5] = 1.0            # 5 real tokens, 3 padded
    rewards[:, :5] = 0.031       # constant across the response, as arm C produces
    with pytest.raises(ValueError, match="constant across its response tokens"):
        compute_contacts_dense_advantage(rewards, mask, index=None)


def test_padding_alone_does_not_trip_the_guard():
    """A genuinely dense reward with padding must still pass."""
    rewards = torch.zeros(2, 8)
    mask = torch.zeros(2, 8)
    mask[:, :5] = 1.0
    rewards[0, :5] = torch.tensor([0.2, 0.0, -0.1, 0.0, 0.3])
    rewards[1, :5] = torch.tensor([0.0, 0.4, 0.0, -0.2, 0.0])
    adv, ret = compute_contacts_dense_advantage(rewards, mask, index=None)
    assert adv.shape == rewards.shape
