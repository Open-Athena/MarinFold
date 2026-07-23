"""Exact corpus stats from the analyzed parts + copy the contacts-v1 tokenizer."""
import os, s3fs, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfFileSystem

fs = s3fs.S3FileSystem(anon=False)
base = "marin-us-east-02a/protein-structure/MarinFold/exp139_esm_atlas_contacts_v1/analyzed"
parts = sorted(f for f in fs.ls(base, detail=False) if f.endswith(".parquet"))
print("parts:", len(parts))

def stat(p):
    with fs.open(p, "rb") as fh:
        t = pq.ParquetFile(fh)
        n = t.metadata.num_rows
        tb = t.read(columns=["num_tokens", "num_contacts", "seq_len"])
    return (n, sum(tb.column("num_tokens").to_pylist()),
            sum(tb.column("num_contacts").to_pylist()),
            sum(tb.column("seq_len").to_pylist()))

rows = toks = conts = slen = 0; done = 0
with ThreadPoolExecutor(max_workers=24) as pool:
    for fut in as_completed([pool.submit(stat, p) for p in parts]):
        n, t, c, s = fut.result(); rows += n; toks += t; conts += c; slen += s; done += 1
        if done % 500 == 0: print(f"  {done}/{len(parts)}")
print(f"DOCUMENTS: {rows:,}")
print(f"TOKENS: {toks:,}  ({toks/1e9:.2f} B)  mean {toks/rows:.0f}/doc")
print(f"RAW CONTACTS: {conts:,}  mean {conts/rows:.0f}/structure")
print(f"mean seq_len: {slen/rows:.1f}")

# copy the contacts-v1 tokenizer next to the new corpus
hffs = HfFileSystem(token=os.environ["HF_TOKEN"])
src = "buckets/open-athena/MarinFold/data/document_structures/contacts_v1/tokenizer"
dst = "buckets/open-athena/MarinFold/data/document_structures/contacts_v1_esm_atlas/tokenizer"
for f in hffs.ls(src, detail=False):
    name = f.split("/")[-1]
    data = hffs.cat_file(f)
    with hffs.open(f"{dst}/{name}", "wb") as fh: fh.write(data)
    print("tokenizer copied:", name, len(data), "B")
