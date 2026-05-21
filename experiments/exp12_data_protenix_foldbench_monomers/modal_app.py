# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Modal app: run Protenix v2 on FoldBench monomers, capture distograms.

Three Volumes:

- ``protenix-weights`` — Protenix v2 checkpoint + CCD cache, downloaded
  once from `huggingface.co/TMF001/pxdesign-weights`.
- ``protenix-msa`` — pre-computed ColabFold MMseqs2 MSAs per protein
  (one ``{stem}/msa/`` subdir each), persisted so reruns are free
  and deterministic.
- ``foldbench-protenix-runs`` — per-(mode, protein, seed) Protenix
  outputs (CIFs, summary_confidence JSONs, distogram .npz files).

Three top-level entry points (called by ``cli.py run`` or invoked
directly via ``modal run modal_app.py::<fn>``):

- ``setup_weights()`` — one-time bootstrap of the weights Volume.
- ``precompute_msa(stem, sequence)`` — one-time per-protein bootstrap
  of the MSA Volume.
- ``predict_one(job_json_str, stem, mode, seeds, n_sample, n_cycle)``
  — Protenix inference for one protein in one mode. Runs inside a
  worker class so weights stay resident across ``.remote()`` calls.

The distogram capture is via :class:`distogram_hook.DistogramCapture`,
attached to ``runner.model.distogram_head`` after the InferenceRunner
loads weights. Each seed's trunk forward triggers one ``.npz`` write
into the output Volume.
"""

import argparse
import csv
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import modal


APP_NAME = "protenix-foldbench-monomers-exp12"

# Resolved at-import time so the @app.cls decorator below can attach
# Volumes. Names match what cli.py defaults pass in.
WEIGHTS_VOLUME_NAME = "protenix-foldbench-weights"
MSA_VOLUME_NAME = "protenix-foldbench-msa"
OUTPUTS_VOLUME_NAME = "foldbench-protenix-runs"

WEIGHTS_VOL = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
MSA_VOL = modal.Volume.from_name(MSA_VOLUME_NAME, create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)


# We need the distogram_hook.py module visible inside the container.
# Modal will package this with `add_local_python_source` so the worker
# imports it the same way the local CLI does.
PROTENIX_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "wget", "rsync", "curl")
    .pip_install(
        # Pinned-ish — Protenix's released version on PyPI handles its own
        # torch/cuda deps. ``hf_transfer`` accelerates the weights snapshot.
        "protenix",
        "huggingface_hub[hf_transfer]",
        "numpy",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Protenix looks for its CCD cache and checkpoints under
        # $PROTENIX_ROOT_DIR. We mount the weights Volume at /weights
        # and symlink the expected layout in @modal.enter.
        "PROTENIX_ROOT_DIR": "/root/protenix_root",
        # Tell HF tokenization layer to not phone home unnecessarily.
        "TOKENIZERS_PARALLELISM": "false",
        # Protenix's runner/msa_search.py defaults to
        # https://protenix-server.com/api/msa. That endpoint can be
        # unreliable; the underlying helper supports the
        # ColabFold MMseqs2 API as a drop-in, which is the de-facto
        # public service for this workload. We pin it explicitly so
        # MSA pre-compute is deterministic across Modal runs.
        "MMSEQS_SERVICE_HOST_URL": "https://api.colabfold.com",
    })
    .add_local_python_source("distogram_hook")
)

app = modal.App(APP_NAME, image=PROTENIX_IMAGE)


# --------------------------------------------------------------------------
# Local helpers
# --------------------------------------------------------------------------


def _seed_outputs_complete(seed_dir: Path, *, stem: str, n_sample: int) -> bool:
    """Return whether a seed directory contains the full expected payload."""
    if not (seed_dir / f"{stem}_distogram.npz").exists():
        return False
    for sample_idx in range(n_sample):
        if not (seed_dir / f"{stem}_sample_{sample_idx}.cif").exists():
            return False
        if not (seed_dir / f"{stem}_summary_confidence_sample_{sample_idx}.json").exists():
            return False
    return True


def _inject_precomputed_msa_paths(
    job_data: list[dict],
    *,
    stem: str,
    msa_root: Path = Path("/msa"),
) -> bool:
    """Add precomputed MSA paths in-place when present.

    Returns True when at least one precomputed MSA file was found.
    """
    msa_dir = msa_root / stem / "msa"
    base = msa_dir / "0"
    paired = base / "pairing.a3m"
    non_paired = base / "0" / "non_pairing.a3m"

    found_precomputed_msa = False
    for task in job_data:
        for seq in task["sequences"]:
            if "proteinChain" not in seq:
                continue
            if paired.exists():
                seq["proteinChain"]["pairedMsaPath"] = str(paired)
                found_precomputed_msa = True
            if non_paired.exists():
                seq["proteinChain"]["unpairedMsaPath"] = str(non_paired)
                found_precomputed_msa = True

    if not found_precomputed_msa:
        print(
            f"WARN: no pre-computed MSA at {msa_dir} for {stem}; "
            "Protenix will auto-search at inference time."
        )
    return found_precomputed_msa


# --------------------------------------------------------------------------
# Weights bootstrap (one-time)
# --------------------------------------------------------------------------


@app.function(volumes={"/weights": WEIGHTS_VOL}, timeout=60 * 30)
def setup_weights() -> dict:
    """Snapshot the Protenix v2 weights + CCD cache from HF into /weights.

    Idempotent: skips files already present. Returns a small dict
    describing what landed.
    """
    from huggingface_hub import snapshot_download

    target = "/weights"
    print(f"snapshot_download TMF001/pxdesign-weights -> {target}")
    snapshot_download(
        "TMF001/pxdesign-weights",
        local_dir=target,
        allow_patterns=[
            "checkpoint/protenix-v2.pt",
            "ccd_cache/components.v20240608.cif",
            "ccd_cache/components.v20240608.cif.rdkit_mol.pkl",
        ],
    )
    WEIGHTS_VOL.commit()
    sizes = {}
    for p in Path(target).rglob("*"):
        if p.is_file():
            sizes[str(p.relative_to(target))] = p.stat().st_size
    print(f"weights volume contents: {sizes}")
    return {"weights_dir": target, "files": sizes}


# --------------------------------------------------------------------------
# MSA pre-compute (one-time per protein)
# --------------------------------------------------------------------------


@app.function(volumes={"/msa": MSA_VOL}, timeout=60 * 5, cpu=0.5)
def audit_msa(stem: str) -> dict:
    """Tiny CPU function: report whether MSA files exist for ``stem``."""
    base = Path("/msa") / stem / "msa" / "0"
    paired = base / "pairing.a3m"
    non_paired = base / "0" / "non_pairing.a3m"
    return {
        "stem": stem,
        "dir_exists": base.exists(),
        "paired_exists": paired.exists(),
        "non_pairing_exists": non_paired.exists(),
    }


@app.function(volumes={"/msa": MSA_VOL}, timeout=60 * 60, cpu=2.0)
def precompute_msa(stem: str, sequence: str) -> dict:
    """Run Protenix's ColabFold MSA pipeline for one protein and persist.

    Writes a minimal one-task JSON and invokes the same machinery
    ``runner/msa_search.update_seq_msa`` uses internally. Output lives
    at ``/msa/{stem}/msa/...`` with ``pairing.a3m`` / ``non_pairing.a3m``
    (Protenix's expected layout).

    Idempotent: if the target a3m files already exist, returns
    immediately without re-running.

    Path layout from Protenix's colabfold-mode pipeline:
      - ``<msa_res_dir>/0/pairing.a3m``  (stub for monomers; ">query\n<seq>")
      - ``<msa_res_dir>/0/0/non_pairing.a3m``  (the actual unpaired MSA;
        the nested ``0/`` matches ``query_<id>`` numbering in the
        protenix colab_request_utils extractor.)
    Idempotency keys on the real MSA file (``non_pairing.a3m``).
    """
    target_dir = Path("/msa") / stem / "msa"
    unpaired_dir = target_dir / "0" / "0"
    non_paired = unpaired_dir / "non_pairing.a3m"
    if non_paired.exists():
        print(f"[{stem}] MSA already present at {non_paired}; skipping.")
        return {"stem": stem, "skipped": True, "dir": str(unpaired_dir)}

    # Protenix's msa_search machinery operates on the per-task JSON
    # in-place, populating sequence['proteinChain']['pairedMsaPath'] /
    # ['unpairedMsaPath']. We call its internal update_seq_msa helper
    # so the output layout matches what Protenix expects on its end.
    from runner.msa_search import update_seq_msa

    infer_data = {
        "name": stem,
        "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
        "covalent_bonds": [],
    }
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        update_seq_msa(infer_data, str(target_dir), mode="colabfold")
        failed = False
        err = None
    except Exception as e:  # noqa: BLE001 - ColabFold API can fail transiently
        # Don't propagate: starmap would fail-fast on a single error and
        # we'd lose the per-protein status of every other concurrent
        # MSA. The dispatcher script retries (idempotent), and individual
        # failures surface as ``failed: True`` in the result dict.
        print(f"WARN: precompute_msa({stem!r}) raised {type(e).__name__}: {e}")
        failed = True
        err = str(e)
    MSA_VOL.commit()
    return {
        "stem": stem,
        "skipped": False,
        "failed": failed,
        "error": err,
        # The colabfold pipeline lays out:
        #   <target_dir>/0/pairing.a3m       (monomer stub)
        #   <target_dir>/0/0/non_pairing.a3m (real unpaired MSA)
        "paired_exists": (target_dir / "0" / "pairing.a3m").exists(),
        "non_paired_exists": non_paired.exists(),
    }


# --------------------------------------------------------------------------
# Inference worker
# --------------------------------------------------------------------------


@app.cls(
    volumes={
        "/weights": WEIGHTS_VOL,
        "/msa": MSA_VOL,
        "/outputs": OUTPUTS_VOL,
    },
    gpu="H100",
    timeout=60 * 60 * 2,
)
@modal.concurrent(max_inputs=1)
class ProtenixWorker:
    """Long-lived inference worker. Weights load once per container."""

    @modal.enter()
    def setup(self) -> None:
        # Lay out PROTENIX_ROOT_DIR the way Protenix expects:
        #   - checkpoint/protenix-v2.pt        -> /weights/checkpoint/protenix-v2.pt
        #   - common/components.cif            -> /weights/ccd_cache/components.v20240608.cif
        #   - common/components.cif.rdkit_mol.pkl -> /weights/ccd_cache/...rdkit_mol.pkl
        # Protenix's configs/configs_data.py points ccd_components_file at
        # ``$PROTENIX_ROOT_DIR/common/components.cif`` (NOT the versioned
        # ``ccd_cache/`` filename the HF mirror ships), so we hard-link
        # the canonical name in.
        root = Path(os.environ["PROTENIX_ROOT_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        # checkpoint dir: directory symlink is fine.
        ckpt_src = Path("/weights/checkpoint")
        ckpt_dst = root / "checkpoint"
        if ckpt_dst.is_symlink():
            ckpt_dst.unlink()
        elif ckpt_dst.exists():
            shutil.rmtree(ckpt_dst)
        if ckpt_src.exists():
            ckpt_dst.symlink_to(ckpt_src)
            print(f"linked {ckpt_dst} -> {ckpt_src}")
        else:
            print(f"WARN: {ckpt_src} missing")
        # common/ dir: per-file symlinks under canonical names.
        common = root / "common"
        common.mkdir(parents=True, exist_ok=True)
        canonical_map = {
            "components.cif": "/weights/ccd_cache/components.v20240608.cif",
            "components.cif.rdkit_mol.pkl": "/weights/ccd_cache/components.v20240608.cif.rdkit_mol.pkl",
        }
        for canonical_name, src_path in canonical_map.items():
            src = Path(src_path)
            dst = common / canonical_name
            if dst.is_symlink():
                dst.unlink()
            elif dst.exists():
                dst.unlink()
            if src.exists():
                dst.symlink_to(src)
                print(f"linked {dst} -> {src}")
            else:
                print(f"WARN: {src} missing")

    @modal.method()
    def predict_one(
        self,
        *,
        job_json_str: str,
        stem: str,
        mode: str,
        seeds: list[int],
        n_sample: int,
        n_cycle: int,
    ) -> dict:
        """Run Protenix v2 on one (protein, mode); persist outputs to /outputs.

        Returns a small metadata dict with timing + paths produced.
        """
        if mode not in ("single_seq", "msa"):
            raise ValueError(f"unknown mode {mode!r}")

        # Idempotency check: only skip when every seed has the full
        # expected payload. A partial seed (e.g. distogram + sample_0
        # only) must be re-run so select_best still ranks the full
        # N_sample search.
        out_root = Path("/outputs") / mode / stem
        OUTPUTS_VOL.reload()
        all_seeds_done = True
        for seed in seeds:
            seed_dir = out_root / f"seed_{seed}"
            if not _seed_outputs_complete(seed_dir, stem=stem, n_sample=n_sample):
                all_seeds_done = False
                break
        if all_seeds_done:
            print(f"[{mode}/{stem}] already complete on Volume; skipping.")
            return {
                "stem": stem, "mode": mode, "seeds": list(seeds),
                "n_sample": n_sample, "n_distograms_written": 0,
                "output_dir": str(out_root), "skipped": True,
            }

        scratch = Path("/tmp/exp12_scratch") / f"{mode}_{stem}_{int(time.time())}"
        scratch.mkdir(parents=True)
        input_json = scratch / "input.json"
        dump_dir = scratch / "dump"
        dump_dir.mkdir()

        # 1. Prepare the per-protein JSON. If MSA mode, splice in the
        #    pre-computed a3m paths; if single-seq, leave them out and
        #    disable Protenix's auto-search.
        job_data = json.loads(job_json_str)
        if mode == "msa":
            _inject_precomputed_msa_paths(job_data, stem=stem)
        input_json.write_text(json.dumps(job_data, indent=2))

        # 2. Build the Protenix config dict. We mimic runner/inference.py's
        #    run() function, but in-process and parameterized.
        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from protenix.config.config import parse_configs
        from runner.inference import (
            InferenceRunner,
            infer_predict,
            update_inference_configs,  # noqa: F401  # available for live config patching
            update_gpu_compatible_configs,
        )

        # First pass — get model_name to look up its specifics.
        base = {**configs_base, **{"data": data_configs}, **inference_configs}
        arg_str = (
            f"--model_name protenix-v2 "
            f"--input_json_path {input_json} "
            f"--dump_dir {dump_dir} "
            f"--load_checkpoint_dir {os.environ['PROTENIX_ROOT_DIR']}/checkpoint "
            f"--use_msa {'true' if mode == 'msa' else 'false'} "
            f"--seeds {','.join(str(s) for s in seeds)} "
            f"--sample_diffusion.N_sample {n_sample} "
        )
        cfg = parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)
        # Second pass — merge in model-specific overrides for protenix-v2.
        base2 = {**configs_base, **{"data": data_configs}, **inference_configs}
        from collections.abc import Mapping as _Mapping
        def _deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, _Mapping) and k in d and isinstance(d[k], _Mapping):
                    _deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        _deep_update(base2, model_configs[cfg.model_name])
        # Make sure our explicit N_cycle override (if any) lands.
        if "model" in base2 and "N_cycle" in base2["model"]:
            base2["model"]["N_cycle"] = n_cycle
        cfg = parse_configs(configs=base2, arg_str=arg_str, fill_required_with_null=True)
        cfg = update_gpu_compatible_configs(cfg)

        # 3. Construct the runner (loads weights).
        runner = InferenceRunner(cfg)

        # 4. Attach the distogram-capture hook.
        from distogram_hook import DistogramCapture
        capture_out = scratch / "dist_capture"
        capture_out.mkdir()
        capture = DistogramCapture(out_dir=capture_out)
        handle = capture.attach(runner.model.distogram_head)

        # 5. Run inference. Protenix's seed loop calls runner.predict()
        #    once per (seed, sample_in_dataloader); we need
        #    DistogramCapture to know which seed is active for each
        #    forward. The cleanest way is to wrap the seed loop
        #    ourselves rather than calling infer_predict directly.
        try:
            self._run_seed_loop(runner, cfg, capture=capture, stem=stem)
        finally:
            handle.remove()

        # 6. Copy outputs from scratch into the persistent /outputs Volume.
        # Protenix lays them out as:
        #   {dump_dir}/{dataset_name=""}/{stem}/seed_{seed}/predictions/{stem}_sample_*.cif
        #                                                              {stem}_summary_confidence_sample_*.json
        # We flatten the predictions/ subdir away so downstream tools just
        # iterate seed_*/<files>.
        out_root = Path("/outputs") / mode / stem
        out_root.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            src_seed_dir = dump_dir / stem / f"seed_{seed}"   # dataset_name="" elides
            src_predictions = src_seed_dir / "predictions"
            dst_seed_dir = out_root / f"seed_{seed}"
            if not src_predictions.exists():
                print(f"WARN: no Protenix predictions for {mode}/{stem}/seed_{seed}")
                continue
            dst_seed_dir.mkdir(exist_ok=True)
            for f in src_predictions.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_seed_dir / f.name)
                elif f.is_dir():
                    # Sub-dirs (rare; would only be present for multi-sample
                    # full_data dumps). Recursive copy.
                    shutil.copytree(f, dst_seed_dir / f.name, dirs_exist_ok=True)
            # Distogram for this seed (if captured).
            dist_src = capture_out / f"seed_{seed}" / f"{stem}_distogram.npz"
            if dist_src.exists():
                shutil.copy2(dist_src, dst_seed_dir / f"{stem}_distogram.npz")
        OUTPUTS_VOL.commit()
        shutil.rmtree(scratch, ignore_errors=True)
        return {
            "stem": stem,
            "mode": mode,
            "seeds": list(seeds),
            "n_sample": n_sample,
            "n_distograms_written": capture.n_writes,
            "output_dir": str(out_root),
        }

    @staticmethod
    def _run_seed_loop(runner, cfg, *, capture, stem: str) -> None:
        """Per-seed loop mirroring runner.inference.infer_predict.

        Inlined so we can call ``capture.set_current(stem, seed)``
        right before the forward triggers the distogram hook.
        """
        import torch
        from runner.inference import update_inference_configs
        from protenix.data.inference.infer_dataloader import get_inference_dataloader
        from protenix.utils.seed import seed_everything

        with open(cfg.input_json_path, "r") as f:
            json_data = json.load(f)
        seed_in_json = json_data[0].get("modelSeeds")
        if seed_in_json and cfg.use_seeds_in_json:
            seeds = [int(i) for i in seed_in_json]
        else:
            seeds = list(cfg.seeds)

        dataloader = get_inference_dataloader(configs=cfg)
        for seed in seeds:
            seed_everything(seed=seed, deterministic=cfg.deterministic)
            capture.set_current(stem=stem, seed=int(seed))
            for batch in dataloader:
                data, atom_array, data_err = batch[0]
                if data_err:
                    print(f"data error for {stem} seed {seed}: {data_err}")
                    continue
                new_cfg = update_inference_configs(cfg, data["N_token"].item())
                runner.update_model_configs(new_cfg)
                pred = runner.predict(data)
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=stem,
                    seed=seed,
                    pred_dict=pred,
                    atom_array=atom_array,
                    entity_poly_type={
                        k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"
                    },
                )
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Local-side dispatcher (called by cli.py run)
# --------------------------------------------------------------------------


def run_cli(args: argparse.Namespace) -> None:
    """Drive the Modal app from a manifest + jobs dir produced locally.

    Steps:
      1. Read inputs/manifest.csv + inputs/jobs/*.json.
      2. (One-time) call setup_weights.remote() if /weights is empty —
         we ask the user to invoke that explicitly via
         ``modal run modal_app.py::setup_weights`` first, since it's
         a clear $-spending checkpoint.
      3. For MSA mode: precompute_msa.starmap() across all proteins
         that don't already have an MSA on the Volume.
      4. ProtenixWorker.predict_one.starmap() across (protein × mode).
      5. Print a summary.

    Note: this function deploys the app on each run (cheap on Modal,
    happens automatically via the ``with app.run()`` context).
    """
    inputs_dir = Path(args.inputs)
    manifest_csv = inputs_dir / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_csv}")

    with manifest_csv.open() as f:
        manifest = list(csv.DictReader(f))
    if not manifest:
        raise ValueError(f"empty manifest: {manifest_csv}")

    # Optional --stems-file restricts the run to a subset. Used for
    # resumes (e.g. after switching Modal workspaces or hitting a
    # spend limit mid-run).
    stems_filter: set[str] | None = None
    if getattr(args, "stems_file", None):
        stems_filter = {s.strip() for s in Path(args.stems_file).read_text().splitlines() if s.strip()}
        manifest = [r for r in manifest if r["stem"] in stems_filter]
        print(f"restricted to {len(manifest)} stems from {args.stems_file}")
        if not manifest:
            raise ValueError(f"no manifest rows matched {args.stems_file}")

    modes = [m.strip() for m in args.modes.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(
        f"run_cli: {len(manifest)} proteins × {len(modes)} modes "
        f"× {len(seeds)} seeds × n_sample={args.n_sample} (n_cycle={args.n_cycle}, gpu={args.gpu})"
    )

    # Override the Volume name on the output side if the user asked for it.
    global OUTPUTS_VOL  # noqa: PLW0603
    if args.output_volume != OUTPUTS_VOLUME_NAME:
        OUTPUTS_VOL = modal.Volume.from_name(args.output_volume, create_if_missing=True)
        print(f"using output Volume: {args.output_volume!r}")

    with app.run():
        # 1. MSA pre-compute (only for MSA mode, and only for proteins
        #    without an MSA on the Volume already; precompute_msa is
        #    itself idempotent so re-running is cheap).
        if "msa" in modes:
            print("dispatching MSA pre-compute (one-time per protein)...")
            msa_args = []
            for row in manifest:
                stem = row["stem"]
                job_path = inputs_dir / row["job_json"]
                job = json.loads(job_path.read_text())
                seq = job[0]["sequences"][0]["proteinChain"]["sequence"]
                msa_args.append((stem, seq))
            msa_results = list(precompute_msa.starmap(msa_args))
            n_done = sum(1 for r in msa_results if not r.get("skipped"))
            print(f"MSA pre-compute: {n_done} ran, {len(msa_results) - n_done} skipped (already on Volume).")

        # 2. Fan out predictions across (protein × mode).
        worker = ProtenixWorker()
        call_args = []
        for row in manifest:
            stem = row["stem"]
            job_path = inputs_dir / row["job_json"]
            job_json_str = job_path.read_text()
            for mode in modes:
                call_args.append(dict(
                    job_json_str=job_json_str,
                    stem=stem,
                    mode=mode,
                    seeds=seeds,
                    n_sample=args.n_sample,
                    n_cycle=args.n_cycle,
                ))

        print(f"dispatching {len(call_args)} predict_one jobs across modal workers...")
        # `.map` with kwargs: convert each dict to a `.remote.aio` call
        # so we can collect futures and print as they land.
        futures = [worker.predict_one.spawn(**kw) for kw in call_args]
        for fut, kw in zip(futures, call_args):
            try:
                result = fut.get()
                print(f"[done] {kw['mode']}/{kw['stem']}: {result}")
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {kw['mode']}/{kw['stem']}: {e}")

    print("run_cli: done. Outputs on Modal Volume:", args.output_volume)
    print("To sync to local: modal volume get", args.output_volume, ". outputs/")
