# Short-rollout consensus

## Question

Does 1,000-short-rollout consensus improve over 100 full rollouts?
Default exp277 checkpoint, step 266344. Eval-val only: 97 natural proteins.

## Comparison

Caps: 5, 10, 20, floor(L/5), floor(L/2) emitted contact statements.
Common 1,000 resampled trajectories yield every cap; 100-prefix controls.
Fresh 100-full-rollout baseline; fixed temperature 1, top-p .95, top-k off.
Exact exp89 metrics, paired protein bootstrap intervals, timings and raw tokens.

## Status

Contact-cutoff tests and a real-protein GPU smoke pass.
Production submitted on one GB200; all 97 proteins are queued for evaluation.
No accuracy conclusion yet.
