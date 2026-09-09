"""Collect small pilot outputs and measure quality and structural redundancy.

This analysis is suitable for the screening set. Full corpus release additionally
requires the complete evaluation-structure screen and training-distribution
comparison in PLAN.md. Requested CATH labels are never treated as measured folds.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import biotite.structure as struc
import gemmi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tmtools import tm_align

from launch import storage_filesystem

HERE = Path(__file__).resolve().parent
FOLDSEEK = Path.home() / ".cache/marinfold/foldseek/foldseek/bin/foldseek"
MMSEQS = Path.home() / ".cache/marinfold/mmseqs/mmseqs/bin/mmseqs"
FIELDS = "query,target,qtmscore,ttmscore,qcov,tcov,evalue"


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a complete, portable table, failing on inconsistent row schemas."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ca_structure(sequence: str, coordinates: np.ndarray) -> gemmi.Structure:
    """Create a C-alpha trace for structural search without inventing backbone atoms."""
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for index, (aa, xyz) in enumerate(zip(sequence, coordinates, strict=True), 1):
        residue = gemmi.Residue()
        residue.name = gemmi.expand_one_letter(aa, gemmi.ResidueKind.AA)
        residue.seqid = gemmi.SeqId(index, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(*xyz)
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    return structure


def secondary_fractions(sequence: str, coordinates: np.ndarray) -> dict:
    """Measure P-SEA composition from C-alpha geometry, independently of labels."""
    atoms = struc.AtomArray(len(sequence))
    atoms.coord = coordinates
    atoms.res_id = np.arange(1, len(sequence) + 1)
    atoms.res_name = [
        gemmi.expand_one_letter(aa, gemmi.ResidueKind.AA) for aa in sequence
    ]
    atoms.atom_name[:] = "CA"
    atoms.chain_id[:] = "A"
    atoms.element[:] = "C"
    sse = struc.annotate_sse(atoms)
    return {
        f"fraction_{name}": float(np.mean(sse == code))
        for code, name in [("a", "alpha"), ("b", "beta"), ("c", "coil")]
    }


def collect(prefixes: list[str], work: Path, report: Path) -> list[dict]:
    """Download a screening set and retain source provenance and per-input timings."""
    fs = storage_filesystem("cw-rno2a")
    records = []
    timings = []
    sources = []
    for prefix in prefixes:
        root = prefix.removeprefix("s3://").rstrip("/")
        if not fs.exists(root + "/complete.json"):
            raise ValueError(f"Refolding is incomplete: {prefix}")
        for path in sorted(fs.glob(root + "/candidates/*.parquet")):
            with fs.open(path, "rb") as handle:
                records.extend(pq.read_table(handle).to_pylist())
            sources.append({"path": "s3://" + path, "bytes": fs.size(path)})
        with fs.open(root + "/timings.csv", "rt") as handle:
            timings.extend(csv.DictReader(handle))
    if not records:
        raise ValueError("No candidate records")
    stems = [row["stem"] for row in records]
    if len(stems) != len(set(stems)):
        raise ValueError("Input prefixes contain duplicate candidates")
    work.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(records), work / "candidates.parquet", compression="zstd"
    )
    write_csv(report / "fold-timings.csv", timings)
    quality = []
    for geometry in ["original", "refolded"]:
        (work / geometry).mkdir(exist_ok=True)
    for row in records:
        sequence = row["sequence"]
        original = np.asarray(row["original_ca"])
        structure = gemmi.read_pdb_string(row["pdb_content"])
        refolded = np.asarray(
            [list(residue["CA"][0].pos) for residue in structure[0][0]]
        )
        case = row["source_prefix"].rsplit("/", 1)[-1]
        summary = {
            key: row[key]
            for key in [
                "stem",
                "source_prefix",
                "scrmsd",
                "plddt",
                "quality_pass",
                "ca_clashes",
                "ca_chain_breaks",
            ]
        }
        summary.update(
            {
                "length": len(sequence),
                "condition": case.split("-", 2)[2],
                "model": case.split("-", 1)[0],
            }
        )
        for name, coordinates in [("original", original), ("refolded", refolded)]:
            summary.update(
                {
                    f"{name}_{key}": value
                    for key, value in secondary_fractions(sequence, coordinates).items()
                }
            )
            ca_structure(sequence, coordinates).write_pdb(
                str(work / name / (row["stem"] + ".pdb"))
            )
        quality.append(summary)
    write_csv(report / "quality.csv", quality)
    groups = defaultdict(list)
    for row in quality:
        groups[(row["model"], row["length"], row["condition"])].append(row)
    aggregates = []
    for (model, length, condition), rows in sorted(groups.items()):
        aggregates.append(
            {
                "model": model,
                "length": length,
                "condition": condition,
                "candidates": len(rows),
                "quality_pass": sum(r["quality_pass"] for r in rows),
                "quality_yield": np.mean([r["quality_pass"] for r in rows]),
                "median_scrmsd": np.median([r["scrmsd"] for r in rows]),
                "median_plddt": np.median([r["plddt"] for r in rows]),
                "mean_alpha": np.mean([r["refolded_fraction_alpha"] for r in rows]),
                "mean_beta": np.mean([r["refolded_fraction_beta"] for r in rows]),
            }
        )
    write_csv(report / "quality-by-case.csv", aggregates)
    (report / "collection.json").write_text(
        json.dumps(
            {
                "candidates": len(records),
                "quality_pass": sum(r["quality_pass"] for r in records),
                "sources": sources,
            },
            indent=2,
        )
        + "\n"
    )
    return records


def run(command: list[str | Path], log: Path) -> None:
    """Persist exact external-tool invocation and fail on nonzero exit."""
    print("Running:", " ".join(map(str, command)), flush=True)
    with log.open("w") as handle:
        handle.write(json.dumps(list(map(str, command))) + "\n")
        handle.flush()
        subprocess.run(
            list(map(str, command)), stdout=handle, stderr=subprocess.STDOUT, check=True
        )


def searches(work: Path, threads: int) -> None:
    """Run both-geometry pilot searches, with TM-align rather than 3Di scores."""
    for geometry in ["original", "refolded"]:
        run(
            [
                FOLDSEEK,
                "easy-search",
                work / geometry,
                work / geometry,
                work / f"{geometry}-pairs.tsv",
                work / f"{geometry}-tmp",
                "--alignment-type",
                "1",
                "--format-output",
                FIELDS,
                "--max-seqs",
                "10000",
                "-e",
                "100",
                "-s",
                "9.5",
                "--threads",
                str(threads),
            ],
            work / f"{geometry}-search.log",
        )


def sequence_screen(work: Path, report: Path, threads: int) -> None:
    """Search both frozen eval references and apply the recorded exclusion rule."""
    records = pq.read_table(work / "candidates.parquet").to_pylist()
    queries = work / "candidate-sequences.fasta"
    queries.write_text(
        "".join(f">{row['stem']}\n{row['sequence']}\n" for row in records)
    )

    reference_root = (
        HERE.parent / "exp225_data_decontaminate_training_corpora/data/reference"
    )
    references = [
        reference_root / name
        for name in ["eval_queries.fasta", "foldbench_all_queries.fasta"]
    ]
    combined = work / "frozen-eval-sequences.fasta"
    text = []
    for index, reference in enumerate(references):
        text.extend(
            f">reference{index}::{line[1:]}" if line.startswith(">") else line
            for line in reference.read_text().splitlines()
        )
    combined.write_text("\n".join(text) + "\n")
    fields = "query,target,fident,qcov,tcov,qlen,tlen,evalue"
    hits = work / "sequence-hits.tsv"
    run(
        [
            MMSEQS,
            "easy-search",
            queries,
            combined,
            hits,
            work / "sequence-tmp",
            "--format-output",
            fields,
            "--alignment-mode",
            "3",
            "--max-seqs",
            "10000",
            "-e",
            "10000",
            "-s",
            "7.5",
            "--threads",
            str(threads),
        ],
        work / "sequence-search.log",
    )
    summarize_sequence_hits(work, report)


def sequence_exclusion_reasons(row: dict) -> dict:
    """Return independent identity and strong-homology exclusion decisions."""
    shorter_coverage = float(
        row["qcov"] if int(row["qlen"]) <= int(row["tlen"]) else row["tcov"]
    )
    return {
        "shorter_coverage": shorter_coverage,
        "identity_rule": float(row["fident"]) >= 0.3 and shorter_coverage >= 0.5,
        "evalue_rule": float(row["evalue"]) <= 1e-3,
    }


def summarize_sequence_hits(work: Path, report: Path) -> None:
    """Enforce the approved identity rule and an additional strong-homology screen.

    The identity/coverage rule has no E-value escape hatch. Significant hits
    below 30% identity are also excluded, following exp225's homology safeguard.
    E-values here refer to a fixed small reference DB, not exp225's corpus DB.
    """
    records = pq.read_table(work / "candidates.parquet").to_pylist()
    reference_root = (
        HERE.parent / "exp225_data_decontaminate_training_corpora/data/reference"
    )
    references = [
        reference_root / name
        for name in ["eval_queries.fasta", "foldbench_all_queries.fasta"]
    ]
    hits = work / "sequence-hits.tsv"
    fields = "query,target,fident,qcov,tcov,qlen,tlen,evalue"
    exclusions = []
    with hits.open() as handle:
        for row in csv.DictReader(handle, fieldnames=fields.split(","), delimiter="\t"):
            reasons = sequence_exclusion_reasons(row)
            if reasons["identity_rule"] or reasons["evalue_rule"]:
                exclusions.append({**row, **reasons})
    write_csv(report / "sequence-exclusions.csv", exclusions)
    (report / "sequence-screen.json").write_text(
        json.dumps(
            {
                "references": {
                    str(path.relative_to(HERE.parent)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in references
                },
                "candidates": len(records),
                "excluded_candidates": len({row["query"] for row in exclusions}),
                "rule": "(identity >=0.30 and shorter coverage >=0.50) or E<=0.001",
                "E_value_scope": "fixed legacy and full FoldBench reference database; not numerically comparable to exp225 corpus-database E-values",
                "mmseqs_version": subprocess.check_output(
                    [MMSEQS, "version"], text=True
                ).strip(),
            },
            indent=2,
        )
        + "\n"
    )


def external_structure_screen(
    work: Path, report: Path, reference: Path, training_db: Path, threads: int
) -> None:
    """Measure evaluation near-duplicates and nearest detected AFDB training reps."""
    for label, target in [("eval", reference), ("training", training_db)]:
        hits = work / f"{label}-structure-hits.tsv"
        run(
            [
                FOLDSEEK,
                "easy-search",
                work / "refolded",
                target,
                hits,
                work / f"{label}-structure-tmp",
                "--alignment-type",
                "1",
                "--format-output",
                FIELDS,
                "--max-seqs",
                "1000" if label == "training" else "10000",
                "-e",
                "100",
                "-s",
                "9.5",
                "--threads",
                str(threads),
            ],
            work / f"{label}-structure-search.log",
        )
        if label == "training":
            summarize_training_hits(
                work, report, training_db.parent.parent / "reps_manifest.csv"
            )
            continue
        best = {}
        exclusions = []
        with hits.open() as handle:
            for row in csv.DictReader(
                handle, fieldnames=FIELDS.split(","), delimiter="\t"
            ):
                stem = row["query"].split(".pdb")[0]
                score = min(float(row["qtmscore"]), float(row["ttmscore"]))
                row = {**row, "stem": stem, "min_tm": score}
                if stem not in best or score > best[stem]["min_tm"]:
                    best[stem] = row
                if score >= 0.8 and min(float(row["qcov"]), float(row["tcov"])) >= 0.8:
                    exclusions.append(row)
        write_csv(report / f"nearest-{label}-structure.csv", list(best.values()))
        if label == "eval":
            write_csv(report / "structure-exclusions.csv", exclusions)
            (report / "structure-screen.json").write_text(
                json.dumps(
                    {
                        "scope": "legacy 554 plus full FoldBench monomer reference; sequence screen also covers all FoldBench protein chains",
                        "reference": str(reference),
                        "excluded_candidates": len({row["stem"] for row in exclusions}),
                        "rule": "both-normalized TM >=0.8 and both coverages >=0.8",
                        "search_method": "Foldseek prefilter + TM-align; not a guarantee of fold-disjointness",
                    },
                    indent=2,
                )
                + "\n"
            )


def summarize_training_hits(work: Path, report: Path, manifest: Path) -> None:
    """Restrict base-AFDB reference matches to the recorded train split."""
    with manifest.open() as handle:
        train = {
            row["representative_id"]
            for row in csv.DictReader(handle)
            if row["split"] == "train"
        }
    best = {}
    with (work / "training-structure-hits.tsv").open() as handle:
        for row in csv.DictReader(handle, fieldnames=FIELDS.split(","), delimiter="\t"):
            if row["target"] not in train:
                continue
            stem = row["query"].split(".pdb")[0]
            score = min(float(row["qtmscore"]), float(row["ttmscore"]))
            if stem not in best or score > best[stem]["min_tm"]:
                best[stem] = {
                    **row,
                    "stem": stem,
                    "min_tm": score,
                    "reference_split": "train",
                }
    write_csv(report / "nearest-training-structure.csv", list(best.values()))
    (report / "training-reference.json").write_text(
        json.dumps(
            {
                "scope": "base AFDB train-split structural-cluster representatives, before exp225 decontamination; not the full current training corpus and not ESM Atlas",
                "n_train_representatives": len(train),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "reported_query_count": len(best),
                "search_cap_per_query": 1000,
                "interpretation": "nearest detected representative; no global fold-novelty claim",
            },
            indent=2,
        )
        + "\n"
    )


def retention_report(work: Path, report: Path) -> None:
    """Apply quality, sequence, structure and cluster-cap filters with an audit row."""
    with (report / "quality.csv").open() as handle:
        quality = list(csv.DictReader(handle))
    sequence_status = json.loads((report / "sequence-screen.json").read_text())
    structure_status = json.loads((report / "structure-screen.json").read_text())
    sequence_exclusions = set()
    if (report / "sequence-exclusions.csv").exists():
        with (report / "sequence-exclusions.csv").open() as handle:
            sequence_exclusions = {row["query"] for row in csv.DictReader(handle)}
    structural_exclusions = set()
    if (report / "structure-exclusions.csv").exists():
        with (report / "structure-exclusions.csv").open() as handle:
            structural_exclusions = {row["stem"] for row in csv.DictReader(handle)}
    if (
        len(sequence_exclusions) != sequence_status["excluded_candidates"]
        or len(structural_exclusions) != structure_status["excluded_candidates"]
    ):
        raise ValueError(
            "Decontamination completion counts disagree with exclusion tables"
        )
    eligible = [
        row
        for row in quality
        if row["quality_pass"] == "True"
        and row["stem"] not in sequence_exclusions | structural_exclusions
    ]
    edges = []
    with (work / "refolded-pairs.tsv").open() as handle:
        for row in csv.DictReader(handle, fieldnames=FIELDS.split(","), delimiter="\t"):
            if (
                min(float(row["qtmscore"]), float(row["ttmscore"])) >= 0.8
                and min(float(row["qcov"]), float(row["tcov"])) >= 0.8
            ):
                edges.append(
                    (row["query"].split(".pdb")[0], row["target"].split(".pdb")[0])
                )
    by_stem = {row["stem"]: row for row in eligible}
    groups = components(list(by_stem), edges)
    selected = set()
    cluster_ids = {}
    for group in groups:
        ranked = sorted(
            group,
            key=lambda stem: (
                -float(by_stem[stem]["plddt"]),
                float(by_stem[stem]["scrmsd"]),
                stem,
            ),
        )
        selected.update(ranked[:5])
        cluster_ids.update({stem: min(group) for stem in group})
    audit = []
    for row in quality:
        stem = row["stem"]
        audit.append(
            {
                "stem": stem,
                "length": row["length"],
                "condition": row["condition"],
                "quality_pass": row["quality_pass"],
                "sequence_excluded": stem in sequence_exclusions,
                "structure_excluded": stem in structural_exclusions,
                "fine_cluster": cluster_ids.get(stem, ""),
                "selected": stem in selected,
            }
        )
    write_csv(report / "retention.csv", audit)
    filtered_quality = [
        {**row, "quality_pass": str(row["stem"] in by_stem)} for row in quality
    ]
    metrics = {
        "candidates": len(quality),
        "quality_pass": sum(row["quality_pass"] == "True" for row in quality),
        "after_decontamination": len(eligible),
        "after_cluster_cap": len(selected),
        "retained_fraction": len(selected) / len(quality),
        "fine_clusters": len(groups),
        "matched_diversity_after_decontamination": matched_comparison(
            filtered_quality, edges
        ),
        "release_status": "provisional: pilot diversity and scale-up gates pending",
    }
    (report / "retention.json").write_text(json.dumps(metrics, indent=2) + "\n")


def components(stems: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Compute conservative connected components, including transitive neighbors."""
    parent = {stem: stem for stem in stems}

    def root(stem: str) -> str:
        while parent[stem] != stem:
            stem = parent[stem]
        return stem

    for a, b in edges:
        if a in parent and b in parent:
            parent[root(b)] = root(a)
    groups = defaultdict(list)
    for stem in stems:
        groups[root(stem)].append(stem)
    return list(groups.values())


def effective_clusters(stems: list[str], edges: list[tuple[str, str]]) -> float:
    """Calculate the exponential entropy of structural component sizes."""
    sizes = np.asarray([len(group) for group in components(stems, edges)])
    probabilities = sizes / sizes.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def matched_comparison(quality: list[dict], edges: list[tuple[str, str]]) -> dict:
    """Repeatedly compare equal-size, length-matched, balanced accepted arms."""
    groups = defaultdict(list)
    for row in quality:
        if row["quality_pass"] == "True":
            groups[(row["length"], row["condition"])].append(row["stem"])
    quotas = {}
    for length in sorted({row["length"] for row in quality}):
        per_class = min(
            len(groups[(length, condition)])
            for condition in ["1.x.x.x", "2.x.x.x", "3.x.x.x"]
        )
        quotas[length] = min(per_class, len(groups[(length, "unconditional")]) // 3)
    rng = np.random.default_rng(278)
    ratios = []
    control_effective = []
    conditioned_effective = []
    n = sum(quotas.values()) * 3
    if n == 0:
        return {
            "matched_n_per_arm": 0,
            "matched_length_counts": json.dumps(
                {length: quota * 3 for length, quota in quotas.items()}, sort_keys=True
            ),
            "ratio_median": None,
            "ratio_resampling_p025": None,
            "ratio_resampling_p975": None,
            "control_effective_median": None,
            "conditioned_effective_median": None,
            "maximum_possible_ratio_median": None,
        }
    for _ in range(100):
        unconditional, conditioned = [], []
        for length, quota in quotas.items():
            unconditional.extend(
                rng.choice(
                    groups[(length, "unconditional")], quota * 3, replace=False
                ).tolist()
            )
            for condition in ["1.x.x.x", "2.x.x.x", "3.x.x.x"]:
                conditioned.extend(
                    rng.choice(
                        groups[(length, condition)], quota, replace=False
                    ).tolist()
                )
        control_count = effective_clusters(unconditional, edges)
        conditioned_count = effective_clusters(conditioned, edges)
        control_effective.append(control_count)
        conditioned_effective.append(conditioned_count)
        ratios.append(conditioned_count / control_count)
    return {
        "matched_n_per_arm": n,
        "matched_length_counts": json.dumps(
            {length: quota * 3 for length, quota in quotas.items()}, sort_keys=True
        ),
        "ratio_median": np.median(ratios),
        "ratio_resampling_p025": np.quantile(ratios, 0.025),
        "ratio_resampling_p975": np.quantile(ratios, 0.975),
        "control_effective_median": float(np.median(control_effective)),
        "conditioned_effective_median": float(np.median(conditioned_effective)),
        "maximum_possible_ratio_median": float(
            np.median(n / np.asarray(control_effective))
        ),
    }


def align_audit_pair(
    pair: tuple[np.ndarray, np.ndarray, str, str, str, str, bool],
) -> dict:
    """Run an independent exact TM-align comparison for the recall audit."""
    x, y, sequence_x, sequence_y, stem_x, stem_y, reported = pair
    result = tm_align(x, y, sequence_x, sequence_y)
    aligned = sum(
        a != "-" and b != "-" for a, b in zip(result.seqxA, result.seqyA, strict=True)
    )
    return {
        "query": stem_x,
        "target": stem_y,
        "min_tm": min(result.tm_norm_chain1, result.tm_norm_chain2),
        "min_coverage": min(aligned / len(sequence_x), aligned / len(sequence_y)),
        "prefilter_reported": reported,
    }


def audit_prefilter(
    work: Path,
    report: Path,
    size: int = 64,
    workers: int = 8,
    thresholds: tuple[float, ...] = (0.5, 0.8),
) -> None:
    """Estimate prefilter recall on a reproducible exhaustive candidate subset."""
    records = pq.read_table(work / "candidates.parquet").to_pylist()
    records = sorted(
        records, key=lambda row: hashlib.sha256(row["stem"].encode()).hexdigest()
    )[:size]
    metrics = []
    for geometry in ["original", "refolded"]:
        reported = set()
        with (work / f"{geometry}-pairs.tsv").open() as handle:
            for row in csv.DictReader(
                handle, fieldnames=FIELDS.split(","), delimiter="\t"
            ):
                reported.add(
                    tuple(
                        sorted(
                            [
                                row["query"].split(".pdb")[0],
                                row["target"].split(".pdb")[0],
                            ]
                        )
                    )
                )
        traces = []
        for row in records:
            structure = gemmi.read_structure(
                str(work / geometry / (row["stem"] + ".pdb"))
            )
            traces.append(
                np.asarray(
                    [list(residue["CA"][0].pos) for residue in structure[0][0]],
                    dtype=np.float64,
                )
            )
        pairs = []
        began = time.perf_counter()
        for i, first in enumerate(records):
            for j in range(i + 1, len(records)):
                second = records[j]
                pairs.append(
                    (
                        traces[i],
                        traces[j],
                        first["sequence"],
                        second["sequence"],
                        first["stem"],
                        second["stem"],
                        tuple(sorted([first["stem"], second["stem"]])) in reported,
                    )
                )
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            audit = list(executor.map(align_audit_pair, pairs, chunksize=16))
        write_csv(report / f"{geometry}-prefilter-audit.csv", audit)
        for threshold in thresholds:
            positives = [
                row
                for row in audit
                if row["min_tm"] >= threshold and row["min_coverage"] >= 0.8
            ]
            metrics.append(
                {
                    "geometry": geometry,
                    "tm_threshold": threshold,
                    "n_candidates": len(records),
                    "n_pairs": len(audit),
                    "true_neighbors": len(positives),
                    "reported_neighbors": sum(
                        row["prefilter_reported"] for row in positives
                    ),
                    "recall": np.mean([row["prefilter_reported"] for row in positives])
                    if positives
                    else None,
                    "elapsed_cpu_wall_seconds": time.perf_counter() - began,
                }
            )
    write_csv(report / "prefilter-recall.csv", metrics)


def summarize_clusters(
    work: Path,
    report: Path,
    thresholds: tuple[tuple[str, float], ...] = (("fine", 0.8), ("broad", 0.5)),
) -> None:
    """Report fine redundancy and broader fold components separately by geometry."""
    with (report / "quality.csv").open() as handle:
        quality = list(csv.DictReader(handle))
    known = {row["stem"] for row in quality}
    summaries = []
    memberships = []
    comparisons = []
    for geometry in ["original", "refolded"]:
        hits = []
        with (work / f"{geometry}-pairs.tsv").open() as handle:
            for row in csv.DictReader(
                handle, fieldnames=FIELDS.split(","), delimiter="\t"
            ):
                a, b = row["query"].split(".pdb")[0], row["target"].split(".pdb")[0]
                if a not in known or b not in known:
                    raise ValueError("Search returned an unrecognized candidate")
                if a != b:
                    hits.append(
                        (
                            a,
                            b,
                            min(float(row["qtmscore"]), float(row["ttmscore"])),
                            min(float(row["qcov"]), float(row["tcov"])),
                        )
                    )
        for label, threshold in thresholds:
            edges = [
                (a, b)
                for a, b, tm, coverage in hits
                if tm >= threshold and coverage >= 0.8
            ]
            comparisons.append(
                {
                    "geometry": geometry,
                    "threshold": label,
                    **matched_comparison(quality, edges),
                }
            )
            for subset in ["all", "quality_pass"]:
                stems = [
                    row["stem"]
                    for row in quality
                    if subset == "all" or row["quality_pass"] == "True"
                ]
                groups = components(stems, edges)
                if not groups:
                    continue
                sizes = np.asarray([len(group) for group in groups])
                probabilities = sizes / sizes.sum()
                summaries.append(
                    {
                        "geometry": geometry,
                        "threshold": label,
                        "subset": subset,
                        "n": len(stems),
                        "clusters": len(groups),
                        "effective_clusters": np.exp(
                            -np.sum(probabilities * np.log(probabilities))
                        ),
                        "unique_cluster_fraction": len(groups) / len(stems),
                        "largest_cluster": int(sizes.max()),
                        "retained_at_cap5": int(np.minimum(sizes, 5).sum()),
                    }
                )
                for group in groups:
                    cluster_id = min(group)
                    memberships.extend(
                        {
                            "geometry": geometry,
                            "threshold": label,
                            "subset": subset,
                            "stem": stem,
                            "cluster": cluster_id,
                        }
                        for stem in group
                    )
    write_csv(report / "cluster-summary.csv", summaries)
    write_csv(report / "cluster-membership.csv", memberships)
    write_csv(report / "matched-diversity.csv", comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--eval-structures", type=Path)
    parser.add_argument(
        "--training-db",
        type=Path,
        default=Path("/data/exp225_decontam/afdb_reps_db/db/targetDB"),
    )
    args = parser.parse_args()
    if not args.skip_collect:
        collect(args.input, args.work, args.report)
    searches(args.work, args.threads)
    sequence_screen(args.work, args.report, args.threads)
    summarize_clusters(args.work, args.report)
    audit_prefilter(args.work, args.report, workers=min(args.threads, 16))
    if args.eval_structures:
        external_structure_screen(
            args.work, args.report, args.eval_structures, args.training_db, args.threads
        )
        retention_report(args.work, args.report)
    provenance = {
        "foldseek_version": subprocess.check_output(
            [FOLDSEEK, "version"], text=True
        ).strip(),
        "foldseek_sha256": hashlib.sha256(FOLDSEEK.read_bytes()).hexdigest(),
        "geometry": "C-alpha",
        "prefilter_recall": "64-candidate exhaustive audit; see prefilter-recall.csv",
        "decontamination": (
            "frozen sequence and evaluation structural-near-duplicate screens complete"
            if args.eval_structures
            else "sequence screened; structural screen pending"
        ),
    }
    (args.report / "analysis-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
