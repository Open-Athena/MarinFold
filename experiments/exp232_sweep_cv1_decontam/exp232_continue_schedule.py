# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Learning-rate schedule used by the exp232 continuation."""

from dataclasses import dataclass

import optax
from levanter.optim.config import LrSchedule, LrScheduleContext


@LrSchedule.register_subclass("linear_inclusive")
@dataclass(frozen=True)
class InclusiveLinearLrSchedule(LrSchedule):
    """Linearly decay to the minimum on the last executed decay update."""

    def build(self, ctx: LrScheduleContext):
        if ctx.decay_steps < 2:
            raise ValueError("inclusive linear decay requires at least two updates")
        return optax.linear_schedule(
            ctx.learning_rate,
            ctx.min_lr,
            transition_steps=ctx.decay_steps - 1,
        )


LR_SCHEDULE = InclusiveLinearLrSchedule()
