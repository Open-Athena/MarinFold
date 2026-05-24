# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Idea 2 driver: iterative self-distillation.

Pre-flight: same as run_seeded_contacts — needs
``outputs/<stem>/distogram_baseline_naive.npz`` for every train stem
(the round-0 prior).

Worker: ``iterative_inference.predict_one``. Round 1 = seeded
contacts (same as idea 1). Round 2+ uses the prior round's distogram
to extract both new contact statements and modal-distance commits;
prepends both to the prefix, re-reads.

Idempotent on (algorithm, provenance.json) — re-run after any crash.
"""

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
import traceback
from pathlib import Path

_THIS = Path(__file__).resolve().parent
# exp1 path dep: ``canonical_sequence`` (imported transitively from
# ``score_marinfold``) does ``from vocab import AMINO_ACIDS``. The
# worker subprocesses set this up inside ``naive_inference`` after
# pinning CUDA, but the parent process also imports ``score_marinfold``
# in ``_score_and_log`` — so it needs the same path injection.
_EXP1 = _THIS.parent / "exp1_document_structures_contacts_and_distances_v1"
if str(_EXP1) not in sys.path:
    sys.path.insert(0, str(_EXP1))


def _worker(
    gpu_id: int,
    work_queue: "mp.Queue[str]",
    result_queue: "mp.Queue[tuple[str, str, float, str]]",
    *,
    protenix_dir: str,
    out_dir: str,
    model_nickname: str,
    models_yaml: str,
    batch_size: int,
    algorithm: str,
    dtype: str,
    n_rounds: int,
    k_contacts_per_L_per_round: tuple[float, ...],
    k_distances_per_L_per_round: tuple[float, ...],
    min_contact_prob: float,
    min_modal_prob: float,
    sharpen_T_for_modes: float,
    order: str,
    prior_name: str,
) -> None:
    """One worker = one GPU = one persistent vLLM instance.

    Pins the visible GPU **before** importing vLLM (vLLM bakes the
    visible device set into worker config at import time). Then loads
    the model and drains the queue.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(_THIS))
    try:
        # Late imports — must come after CUDA_VISIBLE_DEVICES is set.
        from iterative_inference import load_runtime, predict_one
        rt = load_runtime(
            model_nickname=model_nickname, models_yaml=Path(models_yaml),
            dtype=dtype,
        )
        result_queue.put(("__ready__", f"gpu{gpu_id}", rt.model_load_seconds, ""))
    except Exception as exc:  # noqa: BLE001 — surface init failures
        result_queue.put((
            "__init_error__", f"gpu{gpu_id}", 0.0,
            f"{exc!r}\n{traceback.format_exc()}",
        ))
        return

    while True:
        try:
            stem = work_queue.get(timeout=2)
        except queue_mod.Empty:
            break
        if stem == "__STOP__":
            break
        try:
            elapsed = predict_one(
                rt=rt,
                stem=stem,
                protenix_dir=Path(protenix_dir),
                out_dir=Path(out_dir),
                batch_size=batch_size,
                algorithm=algorithm,
                n_rounds=n_rounds,
                k_contacts_per_L_per_round=k_contacts_per_L_per_round,
                k_distances_per_L_per_round=k_distances_per_L_per_round,
                min_contact_prob=min_contact_prob,
                min_modal_prob=min_modal_prob,
                sharpen_T_for_modes=sharpen_T_for_modes,
                order=order,
                initial_prior_path=Path(out_dir) / stem / prior_name,
            )
            result_queue.put((stem, f"gpu{gpu_id}", elapsed, ""))
        except Exception as exc:  # noqa: BLE001 — surface per-protein failures
            result_queue.put((
                stem, f"gpu{gpu_id}", -1.0,
                f"{exc!r}\n{traceback.format_exc()}",
            ))


def _read_train_stems(train_csv: Path) -> list[tuple[str, int]]:
    """Read frozen train set; return (stem, n_residues) in any order."""
    import csv
    with train_csv.open() as f:
        rows = list(csv.DictReader(f))
    return [(r["stem"], int(r["n_residues"])) for r in rows]


def _already_done(out_dir: Path, stem: str, algorithm: str) -> float | None:
    """Return cached elapsed_seconds if the protein is already complete.

    Match is keyed on (algorithm, distogram.npz exists). If a previous
    run wrote with the same algorithm name, we keep its result and
    skip re-running.
    """
    prov_path = out_dir / stem / "provenance.json"
    npz_path = out_dir / stem / "distogram.npz"
    if not prov_path.exists() or not npz_path.exists():
        return None
    try:
        prov = json.loads(prov_path.read_text())
    except (OSError, ValueError):
        return None
    if prov.get("algorithm") != algorithm:
        return None
    elapsed = prov.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)):
        return None
    return float(elapsed)


def run(
    *,
    train_csv: Path,
    protenix_dir: Path,
    out_dir: Path,
    model_nickname: str,
    models_yaml: Path,
    n_gpus: int,
    batch_size: int,
    algorithm: str,
    dtype: str,
    n_rounds: int,
    k_contacts_per_L_per_round: tuple[float, ...],
    k_distances_per_L_per_round: tuple[float, ...],
    min_contact_prob: float,
    min_modal_prob: float,
    sharpen_T_for_modes: float,
    order: str,
    prior_name: str,
) -> dict:
    """Drive the pool. Returns a small dict of timing + completion info."""
    stems_with_lengths = _read_train_stems(train_csv)
    # Longest first so the bottleneck protein lands on a worker early.
    stems_with_lengths.sort(key=lambda sl: -sl[1])

    pending: list[tuple[str, int]] = []
    cached_elapsed: dict[str, float] = {}
    for stem, n in stems_with_lengths:
        cached = _already_done(out_dir, stem, algorithm)
        if cached is not None:
            cached_elapsed[stem] = cached
            print(
                f"skip {stem} (already complete, "
                f"elapsed_seconds={cached:.1f} from prior run)"
            )
        else:
            pending.append((stem, n))

    print(
        f"Train set: {len(stems_with_lengths)} proteins "
        f"({len(pending)} pending, {len(cached_elapsed)} cached)."
    )

    if not pending:
        print("All proteins already complete; skipping the GPU pool.")
        return {
            "n_proteins": len(stems_with_lengths),
            "n_pending": 0,
            "n_cached": len(cached_elapsed),
            "total_wall_seconds": 0.0,
            "per_protein_elapsed": cached_elapsed,
            "worker_load_seconds": {},
        }

    # Spawn (not fork) so each worker gets a clean process state
    # before CUDA initializes. vLLM is sensitive to this.
    ctx = mp.get_context("spawn")
    work_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    for stem, _ in pending:
        work_queue.put(stem)

    procs: list[mp.Process] = []
    n_workers = min(n_gpus, len(pending))
    print(
        f"Launching {n_workers} workers (1 per GPU) for {len(pending)} stems. "
        f"Starting wall-clock timer NOW."
    )

    t_start = time.time()
    for gpu_id in range(n_workers):
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, work_queue, result_queue),
            kwargs={
                "protenix_dir": str(protenix_dir),
                "out_dir": str(out_dir),
                "model_nickname": model_nickname,
                "models_yaml": str(models_yaml),
                "batch_size": batch_size,
                "algorithm": algorithm,
                "dtype": dtype,
                "n_rounds": n_rounds,
                "k_contacts_per_L_per_round": k_contacts_per_L_per_round,
                "k_distances_per_L_per_round": k_distances_per_L_per_round,
                "min_contact_prob": min_contact_prob,
                "min_modal_prob": min_modal_prob,
                "sharpen_T_for_modes": sharpen_T_for_modes,
                "order": order,
                "prior_name": prior_name,
            },
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Collect results: each worker emits ``__ready__`` once after model
    # load, then one record per protein it processes. Bookkeep both so
    # we can report per-worker load time alongside per-protein elapsed.
    worker_load: dict[str, float] = {}
    per_protein_elapsed: dict[str, float] = {}
    n_done = 0
    n_expected = len(pending)
    init_errors: list[tuple[str, str]] = []
    proto_errors: list[tuple[str, str]] = []
    workers_ready = 0
    while n_done < n_expected:
        try:
            tag, worker_id, val, err = result_queue.get(timeout=5)
        except queue_mod.Empty:
            # Periodically check that workers haven't all died (e.g.
            # init failed on every GPU). If so, bail out.
            alive = [p for p in procs if p.is_alive()]
            if not alive:
                print(
                    f"ERROR: all workers exited with {n_done} / {n_expected} "
                    f"proteins done; aborting."
                )
                break
            continue
        if tag == "__ready__":
            worker_load[worker_id] = val
            workers_ready += 1
            print(f"  [{worker_id}] vLLM ready (load {val:.1f} s).")
            continue
        if tag == "__init_error__":
            print(f"ERROR: {worker_id} init failed:\n{err}")
            init_errors.append((worker_id, err))
            continue
        if val < 0:
            proto_errors.append((tag, err))
            print(f"ERROR on {tag} ({worker_id}):\n{err}")
        else:
            per_protein_elapsed[tag] = val
            print(
                f"  [{worker_id}] {tag}: {val:.1f} s "
                f"(progress {n_done + 1} / {n_expected})"
            )
        n_done += 1

    # Tell any still-alive workers to stop and drain.
    for _ in procs:
        work_queue.put("__STOP__")
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    total_wall_seconds = time.time() - t_start
    print(
        f"\nTotal wall-clock: {total_wall_seconds:.1f} s "
        f"({total_wall_seconds / 60:.1f} min) for {n_done} of {n_expected} proteins."
    )

    all_elapsed = dict(cached_elapsed)
    all_elapsed.update(per_protein_elapsed)
    return {
        "n_proteins": len(stems_with_lengths),
        "n_pending": len(pending),
        "n_cached": len(cached_elapsed),
        "total_wall_seconds": total_wall_seconds,
        "per_protein_elapsed": all_elapsed,
        "worker_load_seconds": worker_load,
        "init_errors": init_errors,
        "proto_errors": proto_errors,
    }


def _score_and_log(
    *,
    train_csv: Path,
    protenix_dir: Path,
    out_dir: Path,
    scores_out: Path,
    experiments_tsv: Path,
    experiment_id: str,
    description: str,
    total_wall_seconds: float,
) -> None:
    """Score the outputs and append a row to data/experiments.tsv."""
    sys.path.insert(0, str(_THIS))
    from score_marinfold import MARINFOLD_BINS, score_one
    import csv

    with train_csv.open() as f:
        train_rows = list(csv.DictReader(f))

    rows = []
    for entry in train_rows:
        stem = entry["stem"]
        pdb_id = entry["pdb_id"]
        chain_id = entry["chain_id"]
        distogram_npz = out_dir / stem / "distogram.npz"
        if not distogram_npz.exists():
            print(f"WARN: {stem} has no distogram — skipping in scoring.")
            continue
        gt_cif = protenix_dir / "gt" / f"{stem}.cif"
        row = score_one(
            distogram_npz=distogram_npz,
            gt_cif=gt_cif,
            pdb_id=pdb_id,
            chain_id=chain_id,
            method=experiment_id,
            bins=MARINFOLD_BINS,
        )
        rows.append(row)

    # Save per-protein scores.
    if rows:
        from score_marinfold import _CSV_FIELDS, _format_value
        scores_out.parent.mkdir(parents=True, exist_ok=True)
        with scores_out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_FIELDS)
            for row in rows:
                writer.writerow([_format_value(row[col]) for col in _CSV_FIELDS])
        print(f"Wrote {scores_out} with {len(rows)} rows.")

    # Compute headline aggregates.
    if rows:
        lddts = [row["lddt_distogram_cb"] for row in rows
                 if isinstance(row["lddt_distogram_cb"], (int, float))
                 and row["lddt_distogram_cb"] == row["lddt_distogram_cb"]]
        lddts.sort()
        mean_lddt = sum(lddts) / len(lddts) if lddts else float("nan")
        median_lddt = lddts[len(lddts) // 2] if lddts else float("nan")
    else:
        mean_lddt = float("nan")
        median_lddt = float("nan")

    # Append row to experiments.tsv. Ratio vs baseline is computed
    # against the existing ``baseline_naive`` row if there is one;
    # for the baseline row itself the ratio is 1.0 by definition.
    from append_experiment_row import upsert_experiment_row
    upsert_experiment_row(
        experiment_id=experiment_id,
        description=description,
        mean_lddt=mean_lddt,
        median_lddt=median_lddt,
        total_wall_seconds=total_wall_seconds,
        tsv_path=experiments_tsv,
    )

    print(
        f"\n=== {experiment_id} ===\n"
        f"  mean_lddt          = {mean_lddt:.4f}\n"
        f"  median_lddt        = {median_lddt:.4f}\n"
        f"  total_wall_seconds = {total_wall_seconds:.1f}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=_THIS / "data" / "train_proteins.csv",
    )
    parser.add_argument(
        "--protenix-dir",
        type=Path,
        default=_THIS / "protenix_data" / "data" / "protenix-foldbench-monomers",
    )
    parser.add_argument(
        "--out", type=Path, default=_THIS / "outputs",
    )
    parser.add_argument(
        "--scores-out",
        type=Path,
        default=_THIS / "data" / "iterative_scores.csv",
    )
    parser.add_argument(
        "--experiments-tsv",
        type=Path,
        default=_THIS / "data" / "experiments.tsv",
    )
    parser.add_argument("--model", default="1B")
    parser.add_argument(
        "--models-yaml",
        type=Path,
        default=_THIS.parent.parent / "marinfold" / "marinfold" / "MODELS.yaml",
    )
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        default="float32",
        help=(
            "vLLM model dtype. V100 (compute 7.0) doesn't support bf16. "
            "fp16 produces NaN logits on this bf16-trained checkpoint "
            "(same dynamic-range failure exp1's README documents for "
            "MPS). float32 reproduces exp20's H100/bf16 LDDT to ~0.001 "
            "on the smoke test."
        ),
    )
    parser.add_argument("--algorithm", default="iterative_v1")
    parser.add_argument(
        "--description",
        default=(
            "Iterative self-distillation: round 1 = seeded contacts; "
            "round 2 = contacts from round-1 distogram + modal distance "
            "commits at high-confidence pairs."
        ),
    )
    parser.add_argument("--n-rounds", type=int, default=2)
    parser.add_argument(
        "--k-contacts-per-L-per-round", nargs="+", type=float,
        default=[1.0, 0.5],
    )
    parser.add_argument(
        "--k-distances-per-L-per-round", nargs="+", type=float,
        default=[0.0, 1.0],
    )
    parser.add_argument("--min-contact-prob", type=float, default=0.3)
    parser.add_argument("--min-modal-prob", type=float, default=0.5)
    parser.add_argument("--sharpen-T-for-modes", type=float, default=0.1)
    parser.add_argument(
        "--order", default="long_med_short",
        choices=["long_med_short", "by_prob"],
    )
    parser.add_argument(
        "--prior-name", default="distogram_baseline_naive.npz",
        help="Snapshot filename in outputs/<stem>/ to use as round-1 prior.",
    )
    args = parser.parse_args()

    info = run(
        train_csv=args.train_csv,
        protenix_dir=args.protenix_dir,
        out_dir=args.out,
        model_nickname=args.model,
        models_yaml=args.models_yaml,
        n_gpus=args.n_gpus,
        batch_size=args.batch_size,
        algorithm=args.algorithm,
        dtype=args.dtype,
        n_rounds=args.n_rounds,
        k_contacts_per_L_per_round=tuple(args.k_contacts_per_L_per_round),
        k_distances_per_L_per_round=tuple(args.k_distances_per_L_per_round),
        min_contact_prob=args.min_contact_prob,
        min_modal_prob=args.min_modal_prob,
        sharpen_T_for_modes=args.sharpen_T_for_modes,
        order=args.order,
        prior_name=args.prior_name,
    )

    print(f"\nWorker model-load times (s): {info.get('worker_load_seconds', {})}")
    print(f"Per-protein elapsed (s): {info.get('per_protein_elapsed', {})}")

    _score_and_log(
        train_csv=args.train_csv,
        protenix_dir=args.protenix_dir,
        out_dir=args.out,
        scores_out=args.scores_out,
        experiments_tsv=args.experiments_tsv,
        experiment_id=args.algorithm,
        description=args.description,
        total_wall_seconds=info["total_wall_seconds"],
    )


if __name__ == "__main__":
    main()
