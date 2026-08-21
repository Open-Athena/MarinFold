# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the numbered figure notebooks — MarinFold issue #250.

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
REPO = FIGURES.parents[1]

#: Public bucket roots the make-notebooks read from. Anonymous; no token anywhere.
BUCKET = "https://huggingface.co/buckets/open-athena/MarinFold/resolve"
EXP245 = f"{BUCKET}/data/contacts-v1-foldbench-monomers-exp245"
EXP89 = f"{BUCKET}/data/contacts-v1-model-eval-exp89"
EXP247 = f"{BUCKET}/data/contacts-v1-protein-properties-exp247"
HELICO = ("https://huggingface.co/buckets/timodonnell/helico-experiments/resolve/"
          "exp14_foldbench_held_out_monomers")

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
        print(f"   {'nickname':24s} {model['nickname']}")
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


def save_figure(figure, name: str, dpi: int = 300, formats=("png", "pdf")) -> Path:
    """Write a panel to `output/` as a high-resolution PNG and, by default, a vector PDF.

    No title and no panel letter is ever baked in by these notebooks — captions and lettering
    belong to the document the panel lands in.

    Pass ``formats=("png",)`` for a panel whose content is already a raster — a ray-traced
    structure, say. Wrapping a bitmap in a PDF buys no vector detail and roughly doubles what
    the repository carries.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        figure.savefig(OUTPUT / f"{name}.{suffix}", dpi=dpi)
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
