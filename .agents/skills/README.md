# .agents/skills/

Skill files for AI agents working in MarinFold. Each subdir holds a
`SKILL.md` with a YAML frontmatter block (name + description) plus a
prose body describing the workflow.

Agent harnesses that recognize `.agents/skills/` (or that we explicitly
configure to look here) will surface these skills to the model.
Otherwise, agents can read them on demand like any other docs.

The convention is borrowed from `marin-community/marin`'s
`.agents/skills/` tree — many of these skills are direct ports.

## Current skills

| Skill | Purpose | Origin |
|---|---|---|
| [babysit-job](babysit-job/SKILL.md) | Monitor an Iris job, recover on failure | ported from marin |
| [babysit-zephyr](babysit-zephyr/SKILL.md) | Same, for Zephyr pipeline jobs | ported from marin |

## Adapting marin skills

Marin skills tend to assume:

- Iris configs live at `lib/iris/examples/<cluster>.yaml`. In MarinFold,
  iris configs come from the user's clone of marin (or wherever the
  marin-iris wheel ships them) — we don't vendor configs into the
  repo. Substitute the actual config path when invoking commands.
- The repo is `marin-community/marin` on GitHub. In MarinFold, it's
  `Open-Athena/MarinFold`. Update issue / PR URLs and `gh` invocations
  accordingly.
- Marin-specific tooling: `marin-mcp-babysitter`, ferry / canary
  systems, Grug models, `scripts/logscan.py`, `scripts/iris/dev_tpu.py`.
  Some of these live in the marin wheel, others are scripts that
  would need to be ported separately.

When in doubt, follow the spirit of the marin skill but adapt the
specifics to MarinFold.
