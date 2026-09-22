# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the numbered figure notebooks — MarinFold issue #250.

Lives in the experiment directory with everything else #250 produced.

The notebooks in this directory come in pairs: ``<n>_make_<name>_data.ipynb`` writes a dataset,
``<n>_plot_<name>.ipynb`` reads that dataset and draws it. Nothing is recomputed at plot time, so
a figure can be redrawn on a laptop, and the numbers in a figure always come from a run you can
point at.

The point of the split is that the dataset carries **provenance**: every ``make`` notebook writes
a ``metadata.json`` beside its data recording what produced it — the checkout and its dirty state,
the machine and GPU, package versions, the exact inference recipe, every input file with its
digest, and every output with its digest. The ``plot`` notebook prints that back before drawing.
When a figure looks different from last week, the diff between two ``metadata.json`` files says
why.

Datasets live in ``notebooks/figures/data/<n>_<name>/`` and figures in
``notebooks/figures/output/``.
"""

import getpass
import hashlib
import os
import json
import platform
import socket
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone  # timezone.utc, not datetime.UTC: 3.10 support
from pathlib import Path

FIGURES = Path(__file__).resolve().parent
DATA = FIGURES / "data"
OUTPUT = FIGURES / "output"
REPO = FIGURES.parents[2]   # <repo>/experiments/exp250_*/figures

#: Public bucket roots the make-notebooks read from. Anonymous; no token anywhere.
BUCKET = "https://huggingface.co/buckets/open-athena/MarinFold/resolve"
EXP245 = f"{BUCKET}/data/contacts-v1-foldbench-monomers-exp245"
EXP89 = f"{BUCKET}/data/contacts-v1-model-eval-exp89"
EXP247 = f"{BUCKET}/data/contacts-v1-protein-properties-exp247"
HELICO = ("https://huggingface.co/buckets/timodonnell/helico-experiments/resolve/"
          "exp14_foldbench_held_out_monomers")

#: Figure 1's two panels, in inches. They share a height so the row reads as one band, and split
#: the width unevenly: the map panel carries the deposited structure and two contact maps, the
#: document panel carries eight lines of text.
#:
#: **Drawn at final size.** The assembler scales a row by one factor so the panels plus their
#: gutter fill the column, so the widths below have to sum to (468 - 2*MARGIN - GUTTER) = 455 pt
#: = 6.32 in for that factor to be 1. Any other split shrinks the type on the page along with the
#: panels — at the 3.2 in this file used to specify, a 9 pt axis label reached the page at 5.9 pt.
FIG1_DOC_PANEL = (2.00, 1.85)
FIG1_MAP_PANEL = (4.32, 1.85)
#: Kept as the name pair 5's panel is drawn at; figure 1 itself uses the two above.
FIG1_PANEL = FIG1_DOC_PANEL
#: The structure and the two maps, in figure coordinates, all on the same horizontal band.
#:
#: The structure sits beside the maps rather than inside one of them. As an inset over the ground
#: truth's upper triangle it covered contacts that are only "drawn twice" in the sense that the
#: matrix is symmetric — the reader still has to find them in the mirror, and the same region of
#: the predicted map beside it is not a mirror of anything.
#:
#: The two map rectangles are identical width and height by construction — `figure.colorbar`
#: carving space out of one axes, and constrained layout sizing each around whatever decorations
#: it carries, have each drawn these two at different widths before, and a size difference
#: between a prediction and its ground truth reads as emphasis.
#:
#: The band stops short of the panel top: the assembler letters each panel over its top-left
#: corner, and without that strip the letter lands on the structure.
#: The rectangles are stated as exact pixel fractions at the 300 dpi the PNGs are written at
#: (4.32 x 1.85 in = 1296 x 555 px), so the two maps are the same size in the raster as well as
#: in the vector: 340 px square, at x = 475 and x = 849, y = 120. Equal fractions are equal in
#: the SVG whatever they are, but a rasteriser rounds each edge on its own, and 0.266 x 1296 =
#: 344.7 px once came out as a 345 px map beside a 344 px one.
#:
#: The 190 px between the structure and the first map is deliberate: about half of it is the
#: map's y-axis label and tick numbers, and the rest is the gap that keeps the fold from reading
#: as part of the left map.
FIG1_STRUCTURE_RECT = (10 / 1296, 120 / 555, 275 / 1296, 340 / 555)
FIG1_MAP_RECTS = ((475 / 1296, 120 / 555, 340 / 1296, 340 / 555),
                  (849 / 1296, 120 / 555, 340 / 1296, 340 / 555))
#: The colourbar, which describes the predicted map only — the ground truth is binary.
FIG1_BAR_RECT = (1197 / 1296, 120 / 555, 18 / 1296, 340 / 555)


#: Colour and line style per Helico arm, shared by every panel that draws them: figure 3's bars
#: and its MSA-depth lines are the same six predictors, and the same predictor drawn grey in one
#: panel and purple in the next is a reader's problem, not a style preference.
#:
#: The bracket on contact conditioning — true contacts above, none below — is dashed and dotted
#: where a line style is available, so it reads as the frame rather than as two more predictors.
ARM_STYLE = {
    "oracle": ("#444444", "--"),
    "mf_L_363k": ("#C44E52", "-"),
    "mf_L": ("#C44E52", "-"),           # the same arm on the older checkpoint
    "mf_L2": ("#D9908F", "-"),
    "mf_L5": ("#E7BEBD", "-"),
    "off": ("#B0B0B0", ":"),
    "protenix_v2_single_seq": ("#7A8DA6", "-"),
    "v2ss": ("#7A8DA6", "--"),
    "esmfold2": ("#8172B2", "-"),
    "esmfold": ("#55A868", "-"),
    "protenix_v2_msa": ("#1F5C8B", "-"),
    "v2msa": ("#1F5C8B", "--"),
}
#: The fallback for an arm with no entry above.
NEUTRAL = "#7A8DA6"


_TRACKED_PACKAGES = ("marinfold", "torch", "transformers", "vllm", "numpy", "pandas", "matplotlib")


def digest(payload: bytes) -> str:
    """sha256 of a blob, the identity used throughout the metadata."""
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str) -> bytes:
    """Read one public file into memory (anonymous)."""
    with urllib.request.urlopen(url) as response:
        return response.read()


class Inputs:
    """Records every file a make-notebook reads, with its digest and size.

    Use :meth:`fetch` in place of a bare download and the record builds itself, so a dataset
    cannot silently be built from a source that changed underneath it.
    """

    def __init__(self):
        self.records: list[dict] = []

    def fetch(self, url: str) -> bytes:
        payload = fetch(url)
        self.records.append({"url": url, "bytes": len(payload), "sha256": digest(payload)})
        return payload

    def add_file(self, path: Path, *, hash_content: bool = True) -> Path:
        """Record a local file (a checkpoint shard, a FASTA) without reading it into memory."""
        path = Path(path)
        record = {"path": str(path), "bytes": path.stat().st_size}
        if hash_content:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    hasher.update(chunk)
            record["sha256"] = hasher.hexdigest()
        self.records.append(record)
        return path


def git_state() -> dict:
    """Commit, branch and dirtiness of the checkout that produced a dataset."""

    def run(*arguments: str) -> str:
        result = subprocess.run(["git", *arguments], cwd=REPO, capture_output=True, text=True,
                                check=False)
        return result.stdout.strip()

    dirty = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": dirty.splitlines()[:20],
    }


def package_versions() -> dict:
    """Versions of the packages whose behaviour can move a number."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def machine() -> dict:
    """Where this ran, including the GPU when there is one.

    ``FIGLIB_MACHINE_LABEL`` replaces the recorded hostname. Datasets from this directory get
    committed to a public repository, and some machines are named after their own address — a
    rented GPU box whose hostname is its public IP should not be published by a figure's
    provenance. The substitution is recorded rather than hidden: ``hostname_overridden`` says a
    label was used, so nobody later reads the label as a real hostname.
    """
    label = os.environ.get("FIGLIB_MACHINE_LABEL")
    record = {
        "hostname": label or socket.gethostname(),
        "hostname_overridden": bool(label),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            record["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"{major}.{minor}",
                "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                "count": torch.cuda.device_count(),
            }
    except (ImportError, AttributeError) as exc:
        # No torch, or a torch that cannot answer (a partial install on a system python, seen on
        # this project's workstation). Recorded rather than swallowed: a dataset that does not
        # know whether it had a GPU should say so, not imply it had none.
        record["gpu_probe_error"] = repr(exc)
    return record


def model_identity(nickname: str) -> dict:
    """What a checkpoint nickname actually resolved to, and the config that will be read.

    `rope_theta` is here because it is the field a transformers-5 export states in a way our
    pinned transformers 4.x ignores — loading theta 10000 where the model was trained with
    500000, silently and with no error (#180). A dataset that recorded 10000 was produced by a
    mis-loaded model.
    """
    from marinfold.registry import resolve_model

    path = Path(resolve_model(nickname))
    config = json.loads((path / "config.json").read_text())
    files = sorted(entry.name for entry in path.iterdir() if entry.is_file())
    return {
        "nickname": nickname,
        "path": str(path),
        "files": files,
        "config_sha256": digest((path / "config.json").read_bytes()),
        "tokenizer_sha256": digest((path / "tokenizer.json").read_bytes())
                            if (path / "tokenizer.json").exists() else None,
        "rope_theta": config.get("rope_theta"),
        "rope_scaling": config.get("rope_scaling"),
        "vocab_size": config.get("vocab_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "weight_bytes": sum((path / name).stat().st_size for name in files
                            if name.endswith(".safetensors")),
    }


def dataset_dir(name: str) -> Path:
    """`data/<name>/`, created."""
    path = DATA / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_dataset(name: str, *, notebook: str, parameters: dict, inputs: Inputs,
                  files: dict, extra: dict = None) -> Path:
    """Write a dataset's files plus the `metadata.json` that says how they were made.

    `files` maps a filename to either bytes (written as-is) or a callable taking the destination
    path (for pandas / numpy writers). Every output is hashed after writing, so the plot notebook
    can tell whether it is looking at the bytes this run produced.
    """
    directory = dataset_dir(name)
    written = {}
    for filename, payload in files.items():
        path = directory / filename
        if callable(payload):
            payload(path)
        elif isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(str(payload))
        written[filename] = {"bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}

    metadata = {
        "dataset": name,
        "notebook": notebook,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": parameters,
        "git": git_state(),
        "machine": machine(),
        "packages": package_versions(),
        "inputs": inputs.records,
        "outputs": written,
        **(extra or {}),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {len(written)} file(s) + metadata.json to {directory}")
    for filename, record in written.items():
        print(f"   {filename:32s} {record['bytes']:>10,d} B  {record['sha256'][:12]}")
    return directory


def load_metadata(name: str) -> dict:
    """The `metadata.json` for a dataset, with a clear error when it has not been generated."""
    path = dataset_dir(name) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run the matching `make` notebook for '{name}' first")
    return json.loads(path.read_text())


def require(name: str, *filenames: str) -> Path:
    """Assert a dataset has the files this plot notebook needs, and say what to do when it does not.

    A dataset written before a `make` notebook changed can be missing a file the plot now reads.
    The bare `FileNotFoundError` from `np.load` does not tell you that; this does, and names the
    notebook to re-run.
    """
    directory = dataset_dir(name)
    missing = [filename for filename in filenames if not (directory / filename).exists()]
    if not missing:
        return directory
    try:
        producer = load_metadata(name)["notebook"]
        when = load_metadata(name)["generated_utc"]
        detail = f"was written by {producer} on {when} and predates them"
    except FileNotFoundError:
        detail = "has never been generated"
    raise FileNotFoundError(
        f"{name} is missing {missing}: the dataset {detail}. Run the matching `make` notebook "
        f"again — the plot notebooks never regenerate data themselves, on purpose.")


def describe(name: str) -> dict:
    """Print what produced a dataset, and return the metadata.

    This is the cell every plot notebook opens with. Read it before believing the figure: it says
    which checkpoint, which recipe, which machine, which inputs, and whether the checkout that
    produced it had uncommitted changes.
    """
    metadata = load_metadata(name)
    machine_record = metadata["machine"]
    gpu = machine_record.get("gpu")
    git = metadata["git"]

    print(f"dataset      {metadata['dataset']}")
    print(f"made by      {metadata['notebook']}")
    print(f"generated    {metadata['generated_utc']}")
    print(f"machine      {machine_record['hostname']} · python {machine_record['python']}"
          + (f" · {gpu['name']} ({gpu['memory_gib']} GiB, cc {gpu['compute_capability']})"
             if gpu else " · no GPU"))
    print(f"checkout     {git['commit'][:12]} on {git['branch']}"
          + ("  ** UNCOMMITTED CHANGES **" if git["dirty"] else ""))
    versions = {k: v for k, v in metadata["packages"].items() if v}
    print(f"packages     " + ", ".join(f"{k} {v}" for k, v in versions.items()))

    if metadata.get("parameters"):
        print("\nparameters")
        for key, value in metadata["parameters"].items():
            print(f"   {key:24s} {value}")
    if metadata.get("model"):
        model = metadata["model"]
        print("\nmodel")
        if isinstance(model, str):
            # A dataset can name the model it is about without having loaded one; only
            # `model_identity` fills in the fields below.
            model = {"nickname": model}
        print(f"   {'nickname':24s} {model['nickname']}")
        if "rope_theta" in model:
            print(f"   {'rope_theta':24s} {model['rope_theta']}"
                  + ("   ** expected 500000 — this dataset came from a mis-loaded model **"
                     if model["rope_theta"] != 500_000 else ""))
            print(f"   {'vocab_size':24s} {model['vocab_size']}")
            print(f"   {'weights':24s} {model['weight_bytes'] / 2**30:.2f} GiB")
    print("\ninputs")
    for record in metadata["inputs"]:
        source = record.get("url") or record.get("path")
        print(f"   {record['sha256'][:12] if 'sha256' in record else '(unhashed)':14s} "
              f"{record['bytes']:>12,d} B  {source}")
    print("\noutputs")
    for filename, record in metadata["outputs"].items():
        print(f"   {record['sha256'][:12]:14s} {record['bytes']:>12,d} B  {filename}")

    current = git_state()
    if current["commit"] != git["commit"]:
        print(f"\nnote: this checkout is at {current['commit'][:12]}, the dataset was made at "
              f"{git['commit'][:12]} — regenerate if the difference touches the generator")
    return metadata


def figure_style(dpi: int = 300) -> None:
    """One typographic style for every figure here, and a default save resolution."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.dpi": dpi, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save_figure(figure, name: str, dpi: int = 300, formats=("png", "pdf", "svg"),
                tight: bool = True) -> Path:
    """Write a panel to `output/` as a high-resolution PNG and, by default, vector PDF and SVG.

    The SVG is what `assemble_figures.py` composes into the multi-panel manuscript figures, so a
    panel that is not written as SVG cannot be assembled.

    No title and no panel letter is ever baked in by these notebooks — captions and lettering
    belong to the document the panel lands in.

    Pass ``formats=("png",)`` for a panel whose content is already a raster — a ray-traced
    structure, say. Wrapping a bitmap in a PDF buys no vector detail and roughly doubles what
    the repository carries.

    ``tight=False`` saves at exactly the figure's declared size instead of cropping to its ink.
    Panels that have to share a footprint in an assembled figure need it: with the default tight
    bounding box, two panels declared the same size come out different sizes, because each is
    cropped to whatever it happens to draw.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        # NOT bbox_inches=None for the untight case: None means "use rcParams", and this module
        # sets savefig.bbox to "tight" there — so passing None cropped anyway. The full-figure
        # Bbox is what actually preserves the declared size.
        figure.savefig(OUTPUT / f"{name}.{suffix}", dpi=dpi,
                       bbox_inches="tight" if tight else figure.bbox_inches)
    print(f"wrote {OUTPUT / name}.{{{','.join(formats)}}} ({dpi} dpi)")
    return OUTPUT / f"{name}.{formats[0]}"


def bootstrap_mean(values, draws: int = 2_000, seed: int = 0):
    """Mean and a percentile bootstrap interval over proteins."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------------------------
# The eval universes and the metric, shared by the make-notebooks.
# --------------------------------------------------------------------------------------------

MIN_DEGREE, MIN_SEPARATION = 0.001, 6
RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}
CUTS = (("L", lambda L, true: L), ("L/2", lambda L, true: max(1, L // 2)),
        ("L/5", lambda L, true: max(1, L // 5)), ("R", lambda L, true: true))


def load_legacy_universe(inputs: Inputs):
    """The historical 554-protein set: `(targets, ground_truth)`.

    Its input sequences are not published as a table; exp94's query FASTA is the same set, checked
    byte-identical against the prompts #89 actually used (554/554).
    """
    import pandas as pd

    fasta = inputs.add_file(
        REPO / "experiments/exp94_evals_sequence_knn_baseline/data/eval_queries.fasta")
    rows = []
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            dataset, stem = line[1:].strip().split("__", 1)
            rows.append({"dataset": dataset, "stem": stem, "input_seq": ""})
        elif line.strip():
            rows[-1]["input_seq"] += line.strip()
    targets = pd.DataFrame(rows)
    targets["L"] = targets.input_seq.str.len()

    ground_truth = {}
    for line in inputs.fetch(f"{EXP89}/gt_universe.jsonl").decode().splitlines():
        record = json.loads(line)
        ground_truth[(record["dataset"], record["stem"])] = record
    return targets, ground_truth


def load_foldbench_universe(inputs: Inputs):
    """#245's 333 FoldBench monomers with their eval-set annotation."""
    import io

    import pandas as pd
    import pyarrow.parquet as pq

    targets = pq.read_table(io.BytesIO(inputs.fetch(
        f"{EXP245}/eval_targets_foldbench_monomers.parquet"))).to_pandas()
    annotation = pd.read_csv(io.BytesIO(inputs.fetch(f"{EXP245}/eval_sets.csv")))
    return targets.merge(
        annotation[["stem", "eval_set", "designed", "is_viral", "kingdom", "deposit_date",
                    "exp199_best_identity", "exp199_stratum"]],
        on="stem", how="left", validate="one_to_one")


def load_helico_per_target(inputs: Inputs, extra_path=None, extra_arm=None):
    """helico exp14's per-target structure scores, plus an arm re-run outside that publication.

    The published table is every arm as exp14 ran them; `extra_path` points at a CSV of the same
    shape holding an arm that was re-run since (currently the MarinFold arm on #232's step-363000
    contacts), and `extra_arm` names the one arm to take from it. Both files are recorded as
    inputs, so a figure cannot quietly be drawn from one of them.

    **Every figure that pools the two sources must come through here.** The published table now
    carries the re-run arm, and a caller that fetches it and concatenates the local copy itself
    counts that arm's targets twice — which reads as a narrower interval, not as an error.
    """
    import io

    import pandas as pd

    frame = pd.read_csv(io.BytesIO(inputs.fetch(f"{HELICO}/scores/per_target.csv")))
    if extra_path is not None:
        extra_path = Path(extra_path)
        if not extra_path.exists():
            raise SystemExit(f"{extra_path} is missing — it holds a re-run Helico arm's "
                             "per-target scores; see the experiment README for how it was made")
        inputs.add_file(extra_path)
        extra = pd.read_csv(extra_path)
        if extra_arm is not None:
            extra = extra[extra.arm == extra_arm]
        missing = [column for column in frame.columns if column not in extra.columns]
        if missing:
            raise SystemExit(f"the re-run arm is missing {missing}; it has to carry the same "
                             "columns as the published table or the two cannot be pooled")
        # Anything the published table already carries wins, and the local copy of it is dropped.
        # helico will eventually republish with the re-run arm included, and a plain concat then
        # gives every one of its targets twice: the means do not move, but n doubles and the
        # bootstrap intervals shrink — an error that shows up as unearned confidence rather than
        # as a crash. This makes the two orderings of that publication equivalent.
        already = frame[["arm", "target_id"]].drop_duplicates()
        marked = extra.merge(already, on=["arm", "target_id"], how="left", indicator=True)
        superseded = int((marked._merge == "both").sum())
        extra = marked[marked._merge == "left_only"].drop(columns="_merge")
        if superseded:
            print(f"note: {superseded} of {superseded + len(extra)} rows in {extra_path.name} are "
                  "already in the published table and were dropped in its favour"
                  + ("; the local copy is now redundant and can be deleted" if extra.empty else ""))
        if not extra.empty:
            frame = pd.concat([frame, extra[frame.columns]], ignore_index=True)
    return frame


def load_protein_features(inputs: Inputs):
    """#247's 75 per-protein features for the 314 natural FoldBench monomers, keyed by stem.

    The de novo designs are deliberately absent from that table: an MSA depth for a sequence
    nobody has ever seen in nature is not the same quantity.
    """
    import io

    import pandas as pd

    return pd.read_csv(io.BytesIO(inputs.fetch(f"{EXP247}/protein_features.csv")))


def load_foldbench_scores(inputs: Inputs):
    """#245's per-protein scores for all nine predictors, keyed by (dataset, stem)."""
    import io

    import pandas as pd

    scores = pd.read_csv(io.BytesIO(inputs.fetch(f"{EXP245}/per_protein.csv.gz")),
                         compression="gzip")
    return (scores.rename(columns={"precision": "value"})
                  .assign(dataset="foldbench_monomer")[["dataset", "stem", "predictor",
                                                        "range", "cut", "value"]])


def true_matrix(length: int, contacts):
    """#89's ground-truth contact matrix: degree >= 0.001, separation >= 6."""
    import numpy as np

    matrix = np.zeros((length, length), bool)
    for i, j, degree in contacts:
        i, j = int(i), int(j)
        if degree >= MIN_DEGREE and (j - i) >= MIN_SEPARATION and i < j < length:
            matrix[i, j] = matrix[j, i] = True
    return matrix


def candidate_pairs(resolved):
    """Upper-triangle pairs of resolved residues, and their sequence separations."""
    import numpy as np

    resolved = np.asarray(resolved)
    left, right = np.triu_indices(len(resolved), k=1)
    i, j = resolved[left], resolved[right]
    return i, j, j - i


def score_metrics(score, record: dict):
    """precision @ {L, L/2, L/5, R} + AUC per separation range — #89's `metric_rows`."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    length = record["L"]
    truth = true_matrix(length, record["contacts"])
    i, j, separation = candidate_pairs(record["resolved"])
    pair_scores, pair_truth = score[i, j], truth[i, j].astype(int)
    rows = []
    for name, (low, high) in RANGES.items():
        in_range = separation >= low
        if high is not None:
            in_range &= separation <= high
        values, labels = pair_scores[in_range], pair_truth[in_range]
        n_candidate, n_true = int(values.size), int(labels.sum())
        ranked = labels[np.argsort(-values, kind="mergesort")] if n_candidate else None
        for cut, size_of in CUTS:
            top = min(int(size_of(length, n_true)), n_candidate)
            rows.append(dict(range=name, cut=cut, n_candidate=n_candidate, n_true=n_true,
                             value=float(ranked[:top].sum()) / top if top > 0 else float("nan")))
        auc = (float(roc_auc_score(labels, values))
               if n_candidate and 0 < n_true < n_candidate else float("nan"))
        rows.append(dict(range=name, cut="AUC", n_candidate=n_candidate, n_true=n_true, value=auc))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Drawing rollouts — #82's rollout + resample readout, with emission order kept.
# --------------------------------------------------------------------------------------------

#: Probability floor for the pairwise log-score. `inference.py`'s `_PROB_FLOOR`, which is private.
_PROB_FLOOR = 1e-12


class RolloutDraw:
    """What ``draw_rollout_votes`` produced for one protein.

    ``statements`` is the whole point: one row per emitted ``<contact> <pX> <pY>``, **in the order
    the model wrote it**, with the sequence indices its two position tokens map to (-1 for a token
    outside the protein's slice of the position ring). ``votes`` and ``score`` are the summary
    ``predict`` would have returned for these same rollouts — ``score`` being ``votes`` plus #82's
    pairwise tie-break, bounded to [0, 0.5) so it only orders pairs already tied on votes.
    """

    def __init__(self, statements, per_rollout, completions, votes, score, seq_len,
                 max_new_tokens):
        self.statements = statements
        self.per_rollout = per_rollout
        self.completions = completions
        self.votes = votes
        self.score = score
        self.seq_len = seq_len
        self.max_new_tokens = max_new_tokens


def draw_rollout_votes(sequence: str, *, entry_id: str, model: str, n_rollouts: int = 100,
                       temperature: float = 1.0, top_p: float = 0.95, top_k: int = -1,
                       seed: int = 0, backend: str = "transformers", dtype: str = "bfloat16",
                       batch_size: int = 64, min_separation: int = MIN_SEPARATION) -> RolloutDraw:
    """Draw ``n_rollouts`` contacts-v1 rollouts for one sequence and count the votes.

    #82's settled rollout + resample: each rollout gets a *fresh document realization* (contacts-v1
    randomizes the N-terminal position index and the statement order), one sampled contact-section
    completion is drawn from each, and every distinct in-band pair a rollout asserts casts one
    vote. The large zero-vote tie mass is then broken by the pairwise log-probability from one
    canonical realization.

    Reproduced from the public pieces of ``marinfold.document_structures.contacts_v1`` rather than
    run through ``predict``, which returns the summed matrix and discards emission order. Every
    step is the one ``_rollout_score_matrix`` takes; the three tie-break lines are ``_fwd_matrix``,
    ``_sym_from_fwd`` and ``_tiebreak``, which are private, so they are stated in full here rather
    than reached into. ``backend.next_token_probs`` — the only model call they make — is public.

    ``backend`` defaults to transformers rather than vLLM because a seeded draw has to come back
    the same on a re-run and vLLM's sampler is not seedable per request.

    Raises:
        SystemExit: the chain cannot be serialized as a contacts-v1 document, a ``<retract>``
            statement was emitted (the vote accumulation assumes the emitted set is the live set),
            or a pair collected more votes than there were rollouts.
    """
    import numpy as np
    import pandas as pd
    from marinfold import load_backend
    from marinfold.document_structures.contacts_v1 import (
        CONTEXT_LENGTH, NUM_POSITION_INDICES, GenerationConfig, build_document,
        iter_structure_statements, residues_from_sequence, sample_contacts)
    from marinfold.document_structures.contacts_v1.vocab import (
        BEGIN_STRUCTURE_TOKEN, CONTACT_TOKEN, position_token)

    engine = load_backend(backend, model=model, dtype=dtype, tail_batch_size=batch_size)
    tokenizer = engine.tokenizer

    residues = residues_from_sequence(sequence)
    prefixes: list[list[int]] = []
    n_term_indices: list[int] = []
    seq_len = 0
    for rollout in range(n_rollouts):
        built = build_document(f"{entry_id}:r{rollout}", residues, [], config=GenerationConfig())
        if built is None:
            raise SystemExit(f"{entry_id} cannot be serialized as a contacts-v1 document")
        document = built.document
        cut = document.index(BEGIN_STRUCTURE_TOKEN) + len(BEGIN_STRUCTURE_TOKEN)
        prefixes.append(list(tokenizer.encode(document[:cut], add_special_tokens=False)))
        n_term_indices.append(built.n_term_index)
        seq_len = built.seq_len
    if seq_len != len(sequence):
        raise SystemExit(f"document length {seq_len} != input sequence length {len(sequence)}")

    # #82's budget: 4L + 64 tokens, sized from L only — never from the ground-truth contact count,
    # which would make the rollout oracle-dependent.
    max_new_tokens = min(CONTEXT_LENGTH - len(prefixes[0]), 4 * seq_len + 64)
    completions = sample_contacts(engine, prefixes, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_p=top_p, top_k=top_k, seed=seed,
                                  batch_size=batch_size)

    # Nothing is filtered in the parse: the near-diagonal band, the repeats and the off-protein
    # statements are all part of what a rollout actually emits, and the caller decides.
    rows, texts = [], []
    for rollout, (token_ids, n_term_index) in enumerate(
            zip(completions, n_term_indices, strict=True)):
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        texts.append(text)
        for order, (kind, pos_a, pos_b) in enumerate(iter_structure_statements(text)):
            mapped = []
            for position in (pos_a, pos_b):
                index = (position - n_term_index) % NUM_POSITION_INDICES
                mapped.append(int(index) if index < seq_len else -1)
            low, high = (min(mapped), max(mapped)) if -1 not in mapped else (-1, -1)
            rows.append(dict(rollout=rollout, order=order, kind=kind, pos_a=pos_a, pos_b=pos_b,
                             seq_i=low, seq_j=high))
    statements = pd.DataFrame(rows)
    per_rollout = pd.DataFrame([
        dict(rollout=rollout, n_term_index=n_term_index, n_tokens=len(token_ids),
             truncated=len(token_ids) >= max_new_tokens,
             n_statements=int((statements.rollout == rollout).sum()))
        for rollout, (token_ids, n_term_index) in enumerate(
            zip(completions, n_term_indices, strict=True))])

    # One vote per *distinct* in-band pair per rollout. Today's models never emit `<retract>`, so
    # the live set is the emitted set; a retracting model would need read.py's edit-list fold here
    # instead, and the check below is what would notice.
    if (statements.kind != "contact").any():
        raise SystemExit("a <retract> statement was emitted: the vote accumulation assumes the "
                         "emitted set is the live set — fold read.py's edit list instead")
    votes = np.zeros((seq_len, seq_len), dtype=np.int64)
    in_band = statements[(statements.seq_i >= 0)
                         & (statements.seq_j - statements.seq_i >= min_separation)]
    for _, pairs in in_band.groupby("rollout"):
        for i, j in pairs[["seq_i", "seq_j"]].drop_duplicates().itertuples(index=False):
            votes[i, j] += 1
            votes[j, i] += 1
    if votes.max() > n_rollouts:
        raise SystemExit(f"{votes.max()} votes from {n_rollouts} rollouts")

    prefix_ids = prefixes[0]
    seq_positions = [(n_term_indices[0] + k) % NUM_POSITION_INDICES for k in range(seq_len)]
    contact_id = tokenizer.convert_tokens_to_ids(CONTACT_TOKEN)
    position_ids = [tokenizer.convert_tokens_to_ids(position_token(p)) for p in seq_positions]
    first = engine.next_token_probs(prefix_ids, [[contact_id]], position_ids)       # (1, L)
    second = engine.next_token_probs(prefix_ids, [[contact_id, pid] for pid in position_ids],
                                     position_ids)                                 # (L, L)
    forward = (np.log(np.clip(np.asarray(first[0], dtype=np.float64), _PROB_FLOOR, None))[:, None]
               + np.log(np.clip(np.asarray(second, dtype=np.float64), _PROB_FLOOR, None)))
    pairwise = 0.5 * (forward + forward.T)
    upper = np.triu_indices(seq_len, k=1)
    low, high = float(pairwise[upper].min()), float(pairwise[upper].max())
    score = votes + (pairwise - low) / (high - low + 1e-9) * 0.5

    return RolloutDraw(statements=statements, per_rollout=per_rollout, completions=texts,
                       votes=votes, score=score, seq_len=seq_len, max_new_tokens=max_new_tokens)


def rollout_accuracy(statements, true_contacts, *, min_separation: int = MIN_SEPARATION):
    """Per-rollout precision / recall / F1 over the distinct in-band pairs each rollout emitted.

    ``statements`` is :attr:`RolloutDraw.statements` (or the CSV of it) and must be indexed the
    same way as ``true_contacts``, a boolean ``[L, L]`` matrix. A rollout is scored on the set it
    votes with: distinct pairs, band-filtered, repeats and off-protein statements dropped.

    Returns a frame indexed by rollout with ``n``, ``hits``, ``precision``, ``recall`` and ``f1``.
    """
    import numpy as np

    counted = statements[(statements.seq_i >= 0)
                         & (statements.seq_j - statements.seq_i >= min_separation)]
    counted = counted.drop_duplicates(subset=["rollout", "seq_i", "seq_j"])
    hits = [bool(true_contacts[i, j]) for i, j in zip(counted.seq_i, counted.seq_j)]
    counted = counted.assign(hit=hits)
    accuracy = counted.groupby("rollout").agg(n=("hit", "size"), hits=("hit", "sum"))
    n_true = int(np.triu(true_contacts, min_separation).sum())
    accuracy["precision"] = accuracy.hits / accuracy.n
    accuracy["recall"] = accuracy.hits / n_true
    accuracy["f1"] = (2 * accuracy.precision * accuracy.recall
                      / (accuracy.precision + accuracy.recall))
    return accuracy


def median_rollout(accuracy) -> int:
    """The typical rollout: F1 closest to the median, ties by statement count closest to median.

    A rule rather than an index, so it moves with the data instead of quietly ceasing to be
    typical when the dataset is regenerated. The tie-break matters: F1 is a ratio and several
    rollouts land on the same one, and among those the one that emitted a median number of
    contacts is the one that is typical in both respects.
    """
    ranked = accuracy.assign(
        f1_distance=(accuracy.f1 - accuracy.f1.median()).abs().round(6),
        size_distance=(accuracy.n - accuracy.n.median()).abs(),
    ).sort_values(["f1_distance", "size_distance"])
    return int(ranked.index[0])
