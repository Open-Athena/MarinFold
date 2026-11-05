"""Verify saved optimizer decoding against actual eight-rank PyTorch ZeRO-1."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.optim import ZeroRedundancyOptimizer

from tpu_state import decode_rank, zero_partitions


def optimizer_worker(rank: int, root: str) -> None:
    """Save distinguishable Adam moments from real distributed partitioning."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=8
    )
    try:
        shapes = [
            (31, 7),
            (31, 7),
            (13, 11),
            (4,),
            (4,),
            (64, 9),
            (32, 7),
            (21, 3),
            (4, 17),
            (8, 32),
            (3,),
            (1,),
        ]
        parameters = [torch.nn.Parameter(torch.zeros(shape)) for shape in shapes]
        optimizer = ZeroRedundancyOptimizer(
            parameters,
            optimizer_class=torch.optim.AdamW,
            lr=2e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            foreach=False,
        )
        for i, parameter in enumerate(parameters):
            parameter.grad = torch.full_like(parameter, (i + 1) / 100)
        optimizer.step()
        torch.save(
            {
                "step": 1,
                "tokens": 479,
                "world": 8,
                "stream": {"rank": rank, "world": 8},
                "optimizer": optimizer.optim.state_dict(),
            },
            Path(root) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_decode_matches_real_zero_partitions(tmp_path: Path) -> None:
    mp.start_processes(
        optimizer_worker,
        args=(str(tmp_path),),
        nprocs=8,
        join=True,
        start_method="spawn",
    )
    sizes = [
        (31, 7),
        (31, 7),
        (13, 11),
        (4,),
        (4,),
        (64, 9),
        (32, 7),
        (21, 3),
        (4, 17),
        (8, 32),
        (3,),
        (1,),
    ]
    shapes = {f"p{i}": shape for i, shape in enumerate(sizes)}
    partitions = zero_partitions(shapes)
    seen = set()
    for rank, names in enumerate(partitions):
        state = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        mu, nu, step, provenance = decode_rank(
            state, names, shapes, {"step": 1, "tokens": 479, "world": 8}, rank
        )
        assert step == 1 and provenance["stream"]["rank"] == rank
        for name in names:
            g = (int(name[1:]) + 1) / 100
            np.testing.assert_allclose(mu[name], 0.1 * g, atol=1e-9, rtol=1e-6)
            np.testing.assert_allclose(nu[name], 0.05 * g * g, atol=1e-10, rtol=1e-6)
            seen.add(name)
        broken = {**state, "tokens": 478}
        with pytest.raises(ValueError, match="frozen parent"):
            decode_rank(
                broken, names, shapes, {"step": 1, "tokens": 479, "world": 8}, rank
            )
    assert seen == set(shapes)
