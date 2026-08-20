# Prompt for Claude

You are taking over autonomous operation of the exp232 CoreWeave GPU sweep. Work
from this checkout and branch:

- Repo: `/home/exedev/repos/MarinFold-br/exp232-sweep-cv1-decontam`
- Branch: `exp/232-sweep-cv1-decontam`
- Experiment directory: `experiments/exp232_sweep_cv1_decontam`

First read, completely and in this order:

1. `/home/exedev/repos/MarinFold-br/exp199-optimize-contacts-v1/.agents/skills/run-training-sweep/SKILL.md` and every referenced document.
2. `experiments/exp232_sweep_cv1_decontam/exp232_cw_operations.md`.
3. `experiments/exp232_sweep_cv1_decontam/exp232_cw_claude_handoff.md`.
4. The complete `experiments/exp232_sweep_cv1_decontam/exp232_sweep.py`.

Then take over the scheduled 30-minute W&B-first heartbeat loop and operate the
sweep through verified completion. The authoritative ledger is
`scratch/exp232_cw_s02/exp232_cw_sweep.sqlite`; inspect and integrity-check it
before acting.

Important operator directives:

- `m1-p01-aug` and `m1-p04-aug` are abandoned after divergence. Never restart or
  redispatch either. Only the remaining eight trials are in completion scope.
- W&B is always `open-athena/MarinFold`, group
  `prot-exp232-cw-cv1-decontam-s02`.
- Every Iris submission must use `--cluster marin`, an exact approved
  `--target-cluster`, `--priority batch`, and `--user eczech`. Never deviate from
  batch priority. Iris list/stop/rpc commands do not accept `--user`; address exact
  `/eczech/...` roots for them.
- Maximize use of visible H100 compute on `cw-rno2a` and `cw-us-east-02a`, within
  the 640-GPU cap. Use the documented `list-peers` backend-capacity query, current
  SQLite placement rates, and W&B progress to enlarge gangs when whole-gang
  headroom appears. Preserve one writer, stop and verify before replacement, use
  unique persisted attempts, account for pending gangs, and replan after every
  action. Never use dev priority to solve scarcity.
- Never target `cw-us-west-04a` or unsupported `cw-us-east-08a`.
- Use `/home/exedev/repos/marin-br/main/.venv/bin/iris` and source
  `/home/exedev/marin.env` without printing secrets.
- Do not post to PR #233 unless the operator explicitly asks.

At takeover, jobs are intentionally left running and Codex has scheduled no
further heartbeat. Begin with a fresh complete heartbeat: reread authoritative
state, build the inventory snapshot, query all eight W&B runs first, reconcile
exact Iris roots and capacity, persist observations, decide and act, integrity
check, report, and schedule exactly one next time-based 30-minute pass. Continue
until all eight runs have `run_progress >= 1` and their final checkpoints are
reachable.

