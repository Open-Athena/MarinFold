"""Gradient accumulation with bounded first-update DDP memory."""

import contextlib

from torch.nn.parallel import DistributedDataParallel


def accumulation_context(
    model: DistributedDataParallel,
    microbatch: int,
    microbatches: int,
    first_update: bool,
) -> contextlib.AbstractContextManager:
    """Establish bucket views before accumulating the first update's gradients.

    DDP has no gradient views after construction, including checkpoint restore.
    Starting with no_sync retains a separate full set of gradients until the
    final microbatch. Synchronizing the first microbatch establishes the views
    early. The final reduction still averages the complete accumulated gradient:
    averaging an already averaged contribution leaves that contribution intact.
    """
    if microbatch == microbatches - 1 or (first_update and microbatch == 0):
        return contextlib.nullcontext()
    return model.no_sync()
