# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Write diagnostic events to GCS from inside an iris worker — issue #200.

WHY THIS EXISTS. ``iris job logs`` returns output for a FAILED child job but
nothing for a RUNNING one, so a worker that misbehaves without dying is opaque
from outside the cluster. The exp200 nano runs hit exactly that: the rollout
worker's reported ``weight_step`` cycled -1 -> 4 -> -1 repeatedly while the gang
reported ``failures=0 preemptions=0``, and there was no way to ask the process
what it thought it was doing.

Each event is its OWN object. Object storage has no append, and a
read-modify-write of a shared file from a restarting process is exactly the race
this is meant to observe. Names sort chronologically.

THE BOOT ID IS THE POINT. It is generated once per interpreter, so a fresh boot
id in the trace means the process restarted — which is the hypothesis the nano
runs could not test. ``seq`` restarting at 0 under a new boot id says the same
thing a second way.
"""

import atexit
import json
import os
import signal
import socket
import time
import traceback
import uuid

import fsspec

BOOT_ID = uuid.uuid4().hex[:12]
HOSTNAME = socket.gethostname()
PID = os.getpid()


class Tracer:
    """Append-only-ish event log on object storage. Never raises into the caller.

    A diagnostic that can take down the thing it is diagnosing is worse than no
    diagnostic, so every write is guarded. Failures are counted and reported on
    the next successful event rather than silently dropped.
    """

    def __init__(self, path: str | None, run: str = "run"):
        self.path = f"{path.rstrip('/')}/{run}" if path else None
        self.seq = 0
        self.write_failures = 0
        self._t0 = time.time()
        if self.path:
            self.event("boot", pid=PID, host=HOSTNAME)
            atexit.register(self._on_exit)
            self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        """Record preemption. A v5p worker gets SIGTERM before it goes away."""

        def handler(signum, _frame):
            self.event("signal", signal=int(signum), name=signal.Signals(signum).name)
            # Re-raise the default behaviour rather than swallowing the signal.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(PID, signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not the main thread, or the platform disallows it; tracing a
                # signal is a nice-to-have, not a reason to fail construction.
                pass

    def _on_exit(self) -> None:
        self.event("exit", uptime_s=round(time.time() - self._t0, 1))

    def event(self, kind: str, **fields) -> None:
        if not self.path:
            return
        self.seq += 1
        payload = {
            "kind": kind,
            "boot": BOOT_ID,
            "seq": self.seq,
            "t": time.time(),
            "uptime_s": round(time.time() - self._t0, 1),
            "host": HOSTNAME,
            "pid": PID,
            **fields,
        }
        if self.write_failures:
            payload["prior_write_failures"] = self.write_failures
        name = f"{int(payload['t'] * 1e6):020d}_{BOOT_ID}_{self.seq:06d}.json"
        try:
            with fsspec.open(f"{self.path}/{name}", "w") as fh:
                json.dump(payload, fh)
        except Exception:
            self.write_failures += 1

    def exception(self, kind: str, exc: BaseException, **fields) -> None:
        """Record a traceback. The caller is expected to re-raise."""
        self.event(
            kind,
            error=type(exc).__name__,
            message=str(exc)[:2000],
            traceback=traceback.format_exc()[:8000],
            **fields,
        )


__all__ = ["BOOT_ID", "HOSTNAME", "PID", "Tracer"]
