# Multi-hypothesis contacts

`<contacts-v1.multi>` uses the contacts-v1 sequence header and contact triples.
`<begin_statements>` opens each hypothesis. `<final-prediction>` opens a separate
composite answer; `<end>` terminates that answer. The final answer is not limited
to contacts present in the hypotheses.

Empty sections represent empty predicted contact sets. They are syntactically
valid; evaluation must distinguish them from hypotheses containing contacts
when measuring whether the model produces multiple structural hypotheses.

The model may emit the final marker naturally. A caller may instead insert it
after any complete contact statement, even within a hypothesis. Use
`truncate_history(tokens, budget)` to obtain a valid prefix; it rounds down to a
statement boundary. Insertion inside a triple is unsupported. `parse_history`
rejects malformed structure rather than extracting convenient fragments.

These functions operate on vocabulary strings, excluding the sequence header.
Parsed positions remain in the contacts-v1 position ring; callers supply the
sequence mapping when scoring. Experiment #281 owns training loss weights,
dataset construction, sampling, and rejection selection.

The tokenizer must contain atomic `<contacts-v1.multi>` and `<final-prediction>`
tokens. #281 appends missing tokens and resizes embeddings without changing
existing vocabulary ids. Save this tokenizer with every model checkpoint.
