## Delta-stream contact documents

This experiment tests a compact residue-level contact format trained with ordinary next-token cross entropy: each residue emits `AA DELTA* STOP`, where each signed delta points to a contacting residue.

The goal is to keep contact supervision deterministic and compact without adding a structured contact head or exploding sequence length.

## Current training run

The first full run is a 1.5B Qwen-style model on fixed 8192-token packed delta-stream documents, global batch 128, planned for 12,000 steps.

The pilot artifacts were launched before this work split from #177, so existing object-store prefixes still use `exp177_contacts_delta_stream_v1`; future reruns default to exp299 prefixes.

## Preliminary R-precision

At step 4000, delta-stream already beats the closest early contacts-v1 baseline on the common 533-protein subset for all/medium/long contacts.

Mean/geomean readout gives all R=0.0473 and long R=0.0300, versus contacts-v1 r60 step-3567 all R=0.0297 and long R=0.0222.

Max/either-side helps short contacts most, suggesting the early long-range signal benefits from bidirectional agreement rather than one-sided evidence alone.

## Next steps

Let the current 12k-step run finish, score later checkpoints with both readouts, and use the cleaned exp299 scripts for larger follow-up sweeps rather than continuing to accumulate new runs under the old #177-style setup.
