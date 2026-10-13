# Pinned contacts-v1 tokenizer

These three JSON files are an unmodified snapshot of
`eczech/contacts-v1-tokenizer-5d68a24a899f` at revision
`80f4e411b957641e8c350b07bbe1e2c832697518`.

The pinned Levanter loader accepts local directories but passes a remote
`repo@revision` directly to Hugging Face, which rejects it. Bundling this
64 KB snapshot uses its supported local-directory interface and keeps the
revision explicit without changing the tokenizer or training caches.

Reproduce from the repository root:

```bash
hf download eczech/contacts-v1-tokenizer-5d68a24a899f tokenizer.json tokenizer_config.json special_tokens_map.json --revision 80f4e411b957641e8c350b07bbe1e2c832697518 --local-dir experiments/exp363_models_double_channel_width/tokenizer/contacts-v1@80f4e411b957641e8c350b07bbe1e2c832697518
```
