"""Wipe partial bucket uploads and re-dedupe analyzed/ to exactly one part per input."""
import os, re, s3fs
from huggingface_hub import HfFileSystem
# 1) wipe both target paths in the bucket (partial of-03363 / stray uploads)
hffs = HfFileSystem(token=os.environ["HF_TOKEN"])
for p in ("data/document_structures/contacts_v1_esm_atlas/train",
          "data/contacts/esm_atlas_esmfold2_distill"):
    root = f"buckets/open-athena/MarinFold/{p}"
    try: files = [f for f in hffs.ls(root, detail=False)]
    except Exception: files = []
    for f in files:
        try: hffs.rm(f)
        except Exception as e: print("rm err", f, repr(e)[:80])
    print(f"wiped {p}: removed {len(files)}")
# 2) re-dedupe analyzed/
fs = s3fs.S3FileSystem(anon=False)
base = "marin-us-east-02a/protein-structure/MarinFold/exp139_esm_atlas_contacts_v1"
srcs = sorted(f.split('/')[-1] for f in fs.ls(f"{base}/source/structures/parts", detail=False) if f.endswith(".parquet"))
missing_parts = sorted(f.split('/')[-1] for f in fs.ls(f"{base}/source_missing", detail=False) if f.endswith(".parquet"))
outs = sorted(f.split('/')[-1] for f in fs.ls(f"{base}/analyzed", detail=False) if f.endswith(".parquet"))
globs = {"analyzed": srcs,
         "resumeA": [n for n in srcs if n.startswith("part_03")],
         "resumeB": [n for n in srcs if n.startswith("part_0299")],
         "resumeC": [n for n in srcs if n in ("part_02981.parquet","part_02985.parquet",
                     "part_02987.parquet","part_02988.parquet","part_02989.parquet")],
         "resumeD": missing_parts}
bypart, bad = {}, []
for o in outs:
    m = re.match(r"(analyzed|resume[ABCD])-(\d+)-of-(\d+)\.parquet$", o)
    if not m: bad.append(o); continue
    lst = globs[m.group(1)]; idx = int(m.group(2))
    if idx >= len(lst): bad.append(o); continue
    bypart.setdefault(lst[idx], []).append(o)
removed = 0
for part, files in sorted(bypart.items()):
    for extra in sorted(files)[1:]:
        fs.rm(f"{base}/analyzed/{extra}"); removed += 1
for b in bad:
    fs.rm(f"{base}/analyzed/{b}"); removed += 1
    print("removed unmappable", b)
outs2 = [f.split('/')[-1] for f in fs.ls(f"{base}/analyzed", detail=False) if f.endswith(".parquet")]
missing = [n for n in srcs if n not in bypart]
tot = sum(fs.info(f"{base}/analyzed/{o}")["size"] for o in outs2)
print(f"removed {removed} redundant/unmappable")
print(f"FINAL: files={len(outs2)} covered={len(bypart)}/{len(srcs)} missing={len(missing)} bytes={tot/1e9:.1f} GB")
print("COMPLETE" if (not missing and len(outs2) == len(srcs)) else "INCOMPLETE")
