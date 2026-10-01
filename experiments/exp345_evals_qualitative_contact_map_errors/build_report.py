"""Build a fully offline HTML atlas of all 97 eval-val contact maps."""

import base64
import gzip
import json

import numpy as np
import pandas as pd

from analyze import HERE, INPUTS, unpack


def main() -> None:
    """Embed canonical map data, metadata, metrics and case notes in one HTML."""
    bundle = json.loads(gzip.decompress(INPUTS.read_bytes()))
    diagnostics = pd.read_csv(HERE / "data/per_protein.csv").set_index("stem")
    metadata = pd.read_csv(HERE.parent / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv").set_index("stem")
    notes = json.loads((HERE / "data/case_notes.json").read_text())
    proteins = []
    for item in bundle["proteins"]:
        rec = item["truth"]
        stem = rec["stem"]
        score, truth, i, j = unpack(item)
        supported = score[i, j] > 0
        assert np.sum(supported) >= rec["L"]
        pairs = np.column_stack((i[truth[i, j]], j[truth[i, j]])).tolist()
        proteins.append({"stem": stem, "L": rec["L"], "resolved": rec["resolved"], "n_rollouts": 100,
                         "truth": pairs,
                         "votes": np.column_stack((i[supported], j[supported], score[i[supported], j[supported]])).astype(int).tolist(),
                         "metrics": {"stem": stem, **diagnostics.loc[stem].to_dict()},
                         "title": metadata.loc[stem, "title"], "sequence": metadata.loc[stem, "sequence"],
                         "case": notes.get(stem)})
    payload = {"proteins": proteins, "summary": json.loads((HERE / "data/provenance.json").read_text()), "notes": notes}
    packed = base64.b64encode(gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0)).decode()
    template = (HERE / "report_template.html").read_text()
    assert template.count("__PAYLOAD__") == 1
    report = template.replace("__PAYLOAD__", packed).replace("__CANVAS_RENDERER__", (HERE / "map_renderer.js").read_text())
    (HERE / "report.html").write_text(report)
    print(f"Built report.html: {len(report):,} bytes, {len(proteins)} proteins")


if __name__ == "__main__":
    main()
