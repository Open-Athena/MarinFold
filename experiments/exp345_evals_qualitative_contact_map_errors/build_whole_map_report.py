"""Build a standalone viewer for all 19,400 sampled maps and their comparisons."""

import base64
import gzip
import json
from collections import Counter

import pandas as pd

from analyze import HERE, INPUTS, unpack
from analyze_whole_map import CACHE, contact_sets


def main() -> None:
    """Embed complete sampled maps and canonical summaries in an offline report."""
    bundle = json.loads(gzip.decompress(INPUTS.read_bytes()))
    selections = {r["stem"]: r for r in json.loads((HERE / "data/whole_map_selections.json").read_text())}
    metadata = pd.read_csv(HERE.parent / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv").set_index("stem")
    samples = pd.read_csv(HERE / "data/whole_map_sample_metrics.csv.gz")
    proteins = []
    for item in bundle["proteins"]:
        record = item["truth"]
        stem = record["stem"]
        _, truth, i, j = unpack(item)
        true_pairs = [(int(a), int(b)) for a, b in zip(i, j, strict=True) if truth[a, b]]
        frame = pd.read_parquet(CACHE / f"{stem}.parquet").sort_values("rollout")
        maps = contact_sets(frame, record)
        counts = Counter(pair for contacts in maps for pair in contacts)
        pooled = selections[stem]["selections"]["all"]["pooled_200"]
        universe = sorted(set(counts) | set(true_pairs) | {tuple(pair) for pair in pooled})
        ids = {pair: n for n, pair in enumerate(universe)}
        metrics = samples[samples.stem.eq(stem)]
        proteins.append({"stem": stem, "L": record["L"], "resolved": record["resolved"],
                         "title": metadata.loc[stem, "title"], "n_rollouts": 200,
                         "truth": true_pairs, "universe": universe,
                         "maps": [[ids[pair] for pair in contacts] for contacts in maps],
                         "pooled": pooled,
                         "votes": [[a, b, n] for (a, b), n in sorted(counts.items())],
                         "f1": metrics[metrics["range"].eq("all")].sort_values("rollout").f1.round(8).tolist(),
                         "long_f1": metrics[metrics["range"].eq("long")].sort_values("rollout").f1.round(8).tolist(),
                         "selection": selections[stem]["selections"],
                         "diversity": selections[stem]["mean_pairwise_jaccard"]})
    payload = {"proteins": proteins,
               "summary": pd.read_csv(HERE / "data/whole_map_summary.csv").to_dict("records"),
               "deltas": pd.read_csv(HERE / "data/whole_map_deltas.csv").to_dict("records"),
               "validation": json.loads((HERE / "data/whole_map_validation.json").read_text())}
    encoded = base64.b64encode(gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0)).decode()
    template = (HERE / "whole_map_template.html").read_text()
    output = template.replace("__PAYLOAD__", encoded).replace("__STYLE__", (HERE / "report_style.css").read_text())
    output = output.replace("__CANVAS_RENDERER__", (HERE / "map_renderer.js").read_text())
    output = output.replace("__FINDINGS__", (HERE / "whole_map_findings.html").read_text())
    (HERE / "whole_map_report.html").write_text(output)
    print(f"Built {len(output):,} characters; {len(proteins)*200} individual maps")


if __name__ == "__main__":
    main()
