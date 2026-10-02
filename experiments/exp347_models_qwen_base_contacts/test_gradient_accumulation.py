"""Check bucket-preserving accumulation against a single global batch."""

import contextlib
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel


def accumulation_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    torch.manual_seed(347)
    reference = torch.nn.Linear(3, 2)
    ddp = DistributedDataParallel(torch.nn.Linear(3, 2), gradient_as_bucket_view=True)
    ddp.module.load_state_dict(reference.state_dict())
    expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-5)
    optimizer = ZeroRedundancyOptimizer(
        ddp.parameters(), optimizer_class=torch.optim.AdamW, lr=2e-5
    )
    torch.manual_seed(92)
    for step in range(3):
        xs = [torch.randn(2 + i + step, 3) for i in range(8)]
        ys = [torch.randn(len(x), 2) for x in xs]
        count = sum(y.numel() for y in ys)
        expected_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=False)
        expected = (
            sum((reference(x) - y).square().sum() for x, y in zip(xs, ys, strict=True))
            / count
        )
        expected.backward()
        for i, j in enumerate(range(rank, 8, 2)):
            sync = ddp.no_sync() if i < 3 else contextlib.nullcontext()
            with sync:
                loss = (ddp(xs[j]) - ys[j]).square().sum() * 2 / count
            loss.backward()
        for actual, target in zip(
            ddp.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual.grad, target.grad, rtol=1e-5, atol=1e-6)
        expected_optimizer.step()
        optimizer.step()
        for actual, target in zip(
            ddp.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-6)
    dist.destroy_process_group()


def test_accumulated_updates_match_global_batch(tmp_path: Path) -> None:
    mp.spawn(
        accumulation_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )
