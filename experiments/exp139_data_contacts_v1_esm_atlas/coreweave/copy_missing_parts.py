"""Recompute uncovered input parts and copy them to source_missing/ for a final pass."""
import re, s3fs
from concurrent.futures import ThreadPoolExecutor, as_completed
fs = s3fs.S3FileSystem(anon=False)
base = "marin-us-east-02a/protein-structure/MarinFold/exp139_esm_atlas_contacts_v1"
srcs = sorted(f.split('/')[-1] for f in fs.ls(f"{base}/source/structures/parts", detail=False) if f.endswith(".parquet"))
src_idx = {n: i for i, n in enumerate(srcs)}
outs = [f.split('/')[-1] for f in fs.ls(f"{base}/analyzed", detail=False) if f.endswith(".parquet")]
globs = {
  "analyzed": srcs,
  "resumeA":  [n for n in srcs if n.startswith("part_03")],
  "resumeB":  [n for n in srcs if n.startswith("part_0299")],
  "resumeC":  [n for n in srcs if n in ("part_02981.parquet","part_02985.parquet",
               "part_02987.parquet","part_02988.parquet","part_02989.parquet")],
}
covered = set()
for o in outs:
    m = re.match(r"(analyzed|resumeA|resumeB|resumeC)-(\d+)-of-(\d+)\.parquet$", o)
    if not m: continue
    lst = globs[m.group(1)]; idx = int(m.group(2))
    if idx < len(lst): covered.add(src_idx[lst[idx]])
missing = [srcs[i] for i in sorted(set(range(len(srcs))) - covered)]
print("MISSING count:", len(missing))
dst_prefix = f"{base}/source_missing"
existing = set()
try: existing = {f.split('/')[-1] for f in fs.ls(dst_prefix, detail=False)}
except Exception: pass
todo = [n for n in missing if n not in existing]
print("to copy:", len(todo), "already there:", len(missing) - len(todo))
def cp(n):
    fs.copy(f"{base}/source/structures/parts/{n}", f"{dst_prefix}/{n}"); return n
ok = fail = 0
with ThreadPoolExecutor(max_workers=16) as pool:
    for fut in as_completed([pool.submit(cp, n) for n in todo]):
        try: fut.result(); ok += 1
        except Exception as e: fail += 1; print("FAIL", repr(e)[:120])
final = [f.split('/')[-1] for f in fs.ls(dst_prefix, detail=False) if f.endswith(".parquet")]
print(f"copied ok={ok} fail={fail}; source_missing now holds {len(final)} parts")
print("SORTED_MISSING:", ",".join(sorted(final)))
