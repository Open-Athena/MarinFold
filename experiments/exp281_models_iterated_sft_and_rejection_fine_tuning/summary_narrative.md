# exp281 — Iterated contact synthesis SFT

## Question

Can a model learn to combine multiple structural hypotheses using prolonged SFT, then improve its hypothesis histories through rejection selection?

## Training design

Warm up the multi format and explicit final-prediction token. Continue with hypothesis loss weight 0.1 and full reference-answer supervision. Generate fresh histories between rounds. Later select whole histories by generated final-answer F1, then train their hypotheses followed by reference contacts.

## Finalization

Teach natural finalization on complete examples. On interrupted examples, insert the final token only between complete contact triples and mask its prediction loss. Train the answer after it at full weight. Compare candidate histories at a shared budget.

## Compute placement

Start with one 8-H100 node on cw-us-east-02a at batch priority. The current decontaminated exp232 checkpoint and AFDB/ESM inputs already have copies in its S3 bucket. Other H100 and GB200 peers are reachable. Iris capacity observations are recorded in data/iris_capacity.json.

## Engineering validation

Behavioral tests and tiny-Qwen CPU DDP / GPU checkpoint-resume checks pass. The H100 Iris smoke and staged-input preflight succeeded. vLLM candidate generation and timing artifacts pass locally. Full 1.5B training and hypothesis accuracy remain untested.

## Next experimental gates

Build the initial corpus, validate format acquisition, measure full-model memory and throughput, then begin synthesis rounds. Compare fixed versus refreshed histories and best versus random rejection selection. Use eval-val for iteration and preserve eval-test.
