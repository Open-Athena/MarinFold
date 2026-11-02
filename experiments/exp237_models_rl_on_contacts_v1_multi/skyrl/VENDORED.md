# Vendored from exp208

These files are **byte-identical copies** of
`experiments/exp208_models_rl_post_training_on_contacts_v1/skyrl/`:

| file | why it is copied rather than imported |
|---|---|
| `contact_rewards.py` | pure numpy; the contacts-v1 token walk and the vocab guard |
| `consensus.py` | pure numpy; the leave-one-out machinery, reused over *sections* instead of *rollouts* |
| `contacts_env_skyrl.py` | the deliberately-thin env that carries per-protein state |
| `export_skyrl_checkpoint.py` | FSDP shard -> HF directory at world_size 1 |
| `contacts_v1_passthrough.jinja` | the identity chat template |

`experiments/AGENTS.md` forbids one experiment importing another — an experiment
is a record of what was run, and a shared mutable dependency would let a later
edit change an earlier experiment's meaning. exp208 vendored exp200's code for
the same reason. Keeping the copies byte-identical means `diff` against exp208 is
empty, so a reader can see at a glance that nothing was quietly changed.

Verify with:

```bash
diff -r --exclude '*.pyc' --exclude tests --exclude VENDORED.md \
  <(ls) ../../exp208_models_rl_post_training_on_contacts_v1/skyrl/
for f in contact_rewards.py consensus.py contacts_env_skyrl.py \
         export_skyrl_checkpoint.py contacts_v1_passthrough.jinja; do
  diff "$f" "../../exp208_models_rl_post_training_on_contacts_v1/skyrl/$f" && echo "OK $f"
done
```
