# MarinFold eval utilities

Shared helpers for MarinFold evaluation experiments.

## Poisson bootstrap eval-loss stderr

Use `marinfold_evals.poisson_bootstrap_weighted_mean` when an eval has one row
per independent unit/document with:

- `loss_sum`: summed negative log-likelihood for that document
- `token_count`: number of scored tokens in that document

The point estimate is the usual token-weighted loss,
`sum(loss_sum) / sum(token_count)`, and the reported stderr is the standard
deviation of document-level Poisson(1) bootstrap replicates.

```bash
cd evals
uv run --extra dataframe python -m marinfold_evals.bootstrap_eval_loss \
  --input ../experiments/<exp>/data/per_doc_losses.parquet \
  --output ../experiments/<exp>/data/bootstrap_summary.json
```
