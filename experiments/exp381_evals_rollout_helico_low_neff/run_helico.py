"""Run one independent Helico prediction per frozen MarinFold contact set."""

import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("HELICO_REPO", "/home/bizon/git/helico"))
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"
CHECKPOINT = "/ckpts/contacts-msafree-01/contacts-msafree-01-step-6000.pt"
CHECKPOINT_SHA = "779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644"
WORKERS = 4
TAG = "v1"


def sha256(path: str | Path) -> str:
    """Hash inputs and checkpoint contents without loading weights into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict:
    """Check the source and all frozen input hashes before GPU submission."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if revision != HELICO_SHA or subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "pyproject.toml"], cwd=REPO
    ):
        raise ValueError("Helico source is not the clean pinned revision")
    plan = json.loads((HERE / "data/plan.json").read_text())
    for relative, expected in plan["files_sha256"].items():
        if sha256(HERE / relative) != expected:
            raise ValueError(f"frozen input changed: {relative}")
    if sha256(HERE / "PLAN.md") != plan["plan_markdown_sha256"]:
        raise ValueError("frozen plan changed")
    hours = 200 * 20 / 3600 + WORKERS * 0.1
    estimate = hours * 3.9492
    print(
        f"200 predictions; conservative {hours:.2f} H100-hours, ${estimate:.2f} GPU-only",
        flush=True,
    )
    return dict(
        plan_sha256=sha256(HERE / "data/plan.json"),
        runner_sha256=sha256(__file__),
        helico_source_sha=revision,
        estimated_gpu_usd=estimate,
        modal_client_version=importlib.metadata.version("modal"),
    )


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.10.0",
        "cuequivariance-torch==0.8.1",
        "cuequivariance-ops-torch-cu12==0.8.1",
        "biopython==1.85",
        "numpy==2.2.6",
        "scipy==1.15.3",
        "pyyaml==6.0.3",
        "huggingface-hub==1.18.0",
        "requests==2.34.2",
        "tqdm==4.70.1",
        "pyconfind==0.6.0",
        "tmtools==0.3.0",
    )
    .add_local_dir(str(HERE / "inputs"), remote_path="/root/exp381/inputs")
    .add_local_dir(str(REPO / "src"), remote_path="/root/helico/src")
)
app = modal.App("marinfold-exp381-individual-rollout-helico", image=IMAGE)
ckpts = modal.Volume.from_name("helico-checkpoints", create_if_missing=False)
ccd_cache = modal.Volume.from_name("helico-bench-data", create_if_missing=False)
outputs = modal.Volume.from_name("marinfold-exp381-results", create_if_missing=True)


@app.function(volumes={"/ckpts": ckpts}, timeout=600)
def inspect_checkpoint() -> dict:
    """Inspect saved configuration on CPU before selecting the supported loader."""
    sys.path.insert(0, "/root/helico/src")
    import torch
    from helico.model import HelicoConfig

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    fields = {f.name for f in dataclasses.fields(HelicoConfig)}
    return dict(
        step=checkpoint["step"],
        sha256=sha256(CHECKPOINT),
        config=checkpoint["config"],
        non_model_config={
            k: v for k, v in checkpoint["config"].items() if k not in fields
        },
    )


@app.local_entrypoint()
def inspect_saved() -> None:
    """Persist a small checkpoint schema inspection without allocating a GPU."""
    result = inspect_checkpoint.remote()
    (HERE / "data/helico_checkpoint_config.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["non_model_config"], indent=2))


@app.cls(
    gpu="H100",
    timeout=7200,
    max_containers=WORKERS,
    volumes={"/ckpts": ckpts, "/cache/helico-data": ccd_cache, "/results": outputs},
    secrets=[modal.Secret.from_name("helico-hf-modal")],
)
class Predictor:
    """Keep the pinned Helico model resident across independent contact sets."""

    @modal.enter()
    def setup(self) -> None:
        # GPU-only dependencies exist in the Modal image, not the analysis venv.
        os.environ["HELICO_DATA_DIR"] = "/cache/helico-data"
        sys.path.insert(0, "/root/helico/src")
        import torch
        from helico.data import parse_ccd
        from helico.inference import load_model

        started = time.monotonic()
        if sha256(CHECKPOINT) != CHECKPOINT_SHA:
            raise ValueError("checkpoint digest mismatch")
        # The checkpoint stores TrainConfig, including optimizer/schedule fields.
        # Helico's public loader extracts its model fields and strictly loads
        # the state dict. The CPU inspection records the complete saved schema.
        self.model = load_model(CHECKPOINT, device="cuda", dtype=torch.bfloat16)
        if not self.model.config.use_contacts or self.model.config.use_msa:
            raise ValueError("expected contact-conditioned MSA-free Helico")
        self.ccd = parse_ccd()
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(0)
        self.meta = dict(
            model_load_seconds=time.monotonic() - started,
            gpu_name=props.name,
            gpu_total_memory_gb=props.total_memory / 1e9,
            gpu_compute_capability=f"{props.major}.{props.minor}",
            hostname=socket.gethostname(),
            platform=platform.platform(),
            torch_version=str(torch.__version__),
            runner_tag="modal",
            worker_id=os.environ.get("MODAL_TASK_ID", ""),
            model_nickname="helico-contacts-msafree-01-step-6000",
        )
        self.versions = {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "numpy",
                "scipy",
                "cuequivariance-torch",
                "cuequivariance-ops-torch-cu12",
                "biopython",
                "pyconfind",
            )
        }
        print(f"Loaded verified Helico on {props.name}", flush=True)

    @modal.method()
    def predict(self, target: dict, tasks: list[dict], provenance: dict) -> dict:
        import numpy as np
        import torch
        from helico.bench import predict_target, structure_to_chains
        from helico.data import CONTACT_PRESENT, parse_mmcif
        from helico.inference import contacts_from_pairs

        prep_started = time.monotonic()
        stem = target["stem"]
        gt = parse_mmcif(
            Path("/root/exp381/inputs/gt") / f"{stem}.cif.gz",
            max_resolution=float("inf"),
        )
        if gt is None:
            raise ValueError(f"cannot parse {stem}")
        # The monomer CIF may retain waters/ligands. Predict and score only
        # its protein; auxiliary chemical entities are not requested inputs.
        chains = [c for c in structure_to_chains(gt) if c["type"] == "protein"]
        if len(chains) != 1:
            raise ValueError(f"{stem}: expected a single protein chain")
        if chains[0]["sequence"] != target["helico_sequence"]:
            raise ValueError(f"{stem}: residue mapping mismatch")
        cid = chains[0]["id"]
        protein_chain = next(c for c in gt.chains if c.chain_id == cid)
        gt_keys, ref_coords, ref_names, ref_res, ref_elements = [], [], [], [], []
        for pos, residue in enumerate(protein_chain.residues):
            for atom in residue.atoms:
                if atom.element in {"H", "D"}:
                    continue
                gt_keys.append((pos, atom.name))
                ref_coords.append(atom.coords)
                ref_names.append(atom.name)
                ref_res.append(pos)
                ref_elements.append(atom.element)
        if len(set(gt_keys)) != len(gt_keys):
            raise ValueError("duplicate reference atom identities")
        prep_seconds = time.monotonic() - prep_started
        result_files = {}
        rows = []
        for task in tasks:
            folder = Path("/results") / TAG / task["task_id"]
            complete = folder / "result.json"
            if complete.exists():
                row = json.loads(complete.read_text())
                if (
                    row["input_sha256"] != task["input_sha256"]
                    or row["plan_sha256"] != provenance["plan_sha256"]
                ):
                    raise ValueError("cached result input/protocol mismatch")
                for name, expected in row["output_sha256"].items():
                    if sha256(folder / name) != expected:
                        raise ValueError("cached output checksum mismatch")
            else:
                started = time.monotonic()
                seed = task["helico_seed"]
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                pairs = [(cid, i, cid, j) for i, j in task["contacts"]]
                torch.cuda.synchronize()
                infer_started = time.monotonic()
                prediction_result = predict_target(
                    self.model,
                    chains,
                    self.ccd,
                    target_name=stem,
                    n_samples=1,
                    n_cycles=6,
                    max_tokens=2048,
                    msa_server_url=None,
                    single_sequence=True,
                    contact_pairs=pairs,
                )
                torch.cuda.synchronize()
                elapsed = time.monotonic() - infer_started
                if prediction_result is None:
                    raise ValueError(f"prediction failed: {task['task_id']}")
                tokenized, pred = prediction_result
                if tokenized.n_tokens != target["L_helico"]:
                    raise ValueError("unexpected token/residue indexing")
                state = contacts_from_pairs(pairs, tokenized=tokenized)
                effective = int(
                    torch.triu(state == CONTACT_PRESENT, diagonal=1).sum().item()
                )
                if effective != task["n_effective_contacts"]:
                    raise ValueError(
                        f"contact conditioning mismatch: {effective} vs {task['n_effective_contacts']}"
                    )
                if list(pred["all_coords"].shape[:2]) != [1, 1]:
                    raise ValueError("expected exactly one diffusion sample")
                coords = pred["all_coords"][0, 0].cpu().float().numpy()
                plddt = pred["plddt"][0].cpu().float().numpy()
                if not np.isfinite(coords).all():
                    raise ValueError("nonfinite predicted coordinates")
                pred_keys, atom_names, res_ids, elements, res_names = [], [], [], [], []
                for ri, token in enumerate(tokenized.tokens):
                    for name, element in zip(
                        token.atom_names, token.atom_elements, strict=True
                    ):
                        pred_keys.append((ri, name))
                        atom_names.append(name)
                        elements.append(element)
                        res_ids.append(ri)
                        res_names.append(token.res_name)
                if len(pred_keys) != len(coords) or len(set(pred_keys)) != len(
                    pred_keys
                ):
                    raise ValueError("invalid predicted atom identities")
                pred_lookup = dict(zip(pred_keys, coords, strict=True))
                covered = np.array([key in pred_lookup for key in gt_keys])
                aligned = np.array(
                    [pred_lookup.get(key, [np.nan] * 3) for key in gt_keys],
                    dtype=np.float32,
                )
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    folder / "structure.npz",
                    coords=coords,
                    plddt=plddt,
                    atom_names=atom_names,
                    res_ids=res_ids,
                    elements=elements,
                    res_names=res_names,
                    ref_coords=np.array(ref_coords, dtype=np.float32),
                    ref_atom_names=ref_names,
                    ref_res_ids=ref_res,
                    ref_elements=ref_elements,
                    aligned_pred=aligned,
                    covered=covered,
                )
                pdb = []
                for ai, (xyz, name, residue, element, rn) in enumerate(
                    zip(coords, atom_names, res_ids, elements, res_names, strict=True)
                ):
                    x, y, z = xyz
                    aname = name if len(name) == 4 else f" {name:<3}"
                    pdb.append(
                        f"ATOM  {ai + 1:5d} {aname:4} {rn:>3} A{residue + 1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{float(plddt[ai]):6.2f}          {element:>2}  "
                    )
                (folder / "prediction.pdb").write_text("\n".join(pdb) + "\nEND\n")
                row = {
                    **{k: v for k, v in task.items() if k != "contacts"},
                    **provenance,
                    **self.meta,
                    "ranking_score": float(pred["all_ranking_score"][0, 0]),
                    "ptm": float(pred["all_ptm"][0, 0]),
                    "iptm": float(pred["all_iptm"][0, 0]),
                    "has_clash": bool(pred["all_has_clash"][0, 0]),
                    "mean_plddt": float(plddt.mean()),
                    "n_residues": target["L_helico"],
                    "n_pairs": target["L_helico"] * (target["L_helico"] - 1) // 2,
                    "mode": "individual-rollout",
                    "n_samples": 1,
                    "n_cycles": 6,
                    "elapsed_seconds": elapsed,
                    "target_prepare_seconds": prep_seconds,
                    "prediction_and_dump_seconds": time.monotonic() - started,
                    "total_seconds": self.meta["model_load_seconds"]
                    + prep_seconds
                    + time.monotonic()
                    - started,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "versions": self.versions,
                    "output_sha256": {
                        name: sha256(folder / name)
                        for name in ("structure.npz", "prediction.pdb")
                    },
                }
                complete.write_text(json.dumps(row, indent=2) + "\n")
                outputs.commit()
                print(
                    f"{task['task_id']} complete: {effective} contacts, {elapsed:.1f}s",
                    flush=True,
                )
            rows.append(row)
            for name in ("structure.npz", "prediction.pdb", "result.json"):
                result_files[f"{task['task_id']}/{name}"] = (folder / name).read_bytes()
        return dict(stem=stem, rows=rows, files=result_files)


@app.local_entrypoint()
def run(smoke: bool = False) -> None:
    """Run a two-prediction smoke gate, or resume all 200 frozen predictions."""
    provenance = preflight()
    targets = json.loads((HERE / "inputs/targets.json").read_text())
    tasks = json.loads((HERE / "inputs/tasks.json").read_text())
    if smoke:
        targets = targets[:1]
        tasks = [
            t for t in tasks if t["stem"] == targets[0]["stem"] and t["rollout"] == 0
        ]
    elif not (HERE / "data/smoke_passed.json").exists():
        raise ValueError(
            "run --smoke and validate its two structures before the full run"
        )
    pending = [
        t
        for t in tasks
        if not (HERE / "scratch/results" / t["task_id"] / "result.json").exists()
    ]
    jobs = [
        (target, [t for t in pending if t["stem"] == target["stem"]], provenance)
        for target in targets
    ]
    jobs = [job for job in jobs if job[1]]
    failures = []
    predictor = Predictor()
    for result in predictor.predict.starmap(
        jobs, order_outputs=False, return_exceptions=True
    ):
        if isinstance(result, Exception):
            failures.append(repr(result))
            continue
        for relative, content in result["files"].items():
            path = HERE / "scratch/results" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(
            f"Downloaded {result['stem']}: {len(result['rows'])} predictions",
            flush=True,
        )
    if failures:
        raise RuntimeError(f"Failed tasks (durable partials retained): {failures}")
    for task in tasks:
        folder = HERE / "scratch/results" / task["task_id"]
        row = json.loads((folder / "result.json").read_text())
        if (
            row["input_sha256"] != task["input_sha256"]
            or row["plan_sha256"] != provenance["plan_sha256"]
        ):
            raise ValueError("local cached input/protocol mismatch")
        for name, expected in row["output_sha256"].items():
            if sha256(folder / name) != expected:
                raise ValueError("local output checksum mismatch")
    print(f"Complete: {len(tasks)} predictions", flush=True)


if __name__ == "__main__":
    preflight()
    if os.environ.get("HELICO_DRY_RUN") != "1":
        subprocess.run(["modal", "run", f"{__file__}::run", *sys.argv[1:]], check=True)
