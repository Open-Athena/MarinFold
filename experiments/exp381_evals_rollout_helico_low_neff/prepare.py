"""Freeze the lowest-Neff cohort and ten unselected raw rollouts per model."""

import datetime as dt
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "exp373_evals_mpnn_diversity_neff"
SWEEP = HERE.parent / "exp311_evals_helico_exp277_contact_count_sweep"
sys.path.insert(0, str(PREVIOUS))
from contacts import parse_contacts, sample_seed  # noqa: E402

PUBLIC = "buckets/open-athena/MarinFold/data"


def digest(path: Path) -> str:
    """Return the SHA256 of a small artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(fs: HfFileSystem, uri: str, path: Path, expected: str) -> None:
    """Cache one public file and require its previously published digest."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fs.read_bytes(uri.removeprefix("hf://")))
    if digest(path) != expected:
        raise ValueError(f"checksum mismatch: {path}")


def main() -> None:
    """Prepare reproducible inputs without accessing any held-out structures."""
    fs = HfFileSystem(token=False)
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)
    metadata = pd.read_csv(PREVIOUS / "data/metadata.csv")
    val = metadata.loc[metadata.eval_set == "eval-val"]
    assert len(val) == 97 and val.msa_neff.notna().all()
    selected = val.sort_values(["msa_neff", "stem"]).head(10)
    source_manifest = json.loads((SWEEP / "data/input_manifest.json").read_text())
    fetch(
        fs,
        f"{PUBLIC}/exp311-helico-exp277-contact-count-sweep/inputs/targets.csv",
        inputs / "source_targets.csv",
        source_manifest["prepared_targets_sha256"],
    )
    target_source = pd.read_csv(inputs / "source_targets.csv").set_index("target_id")
    new_plan = json.loads((HERE / "data/source/exp373_two_epoch_plan.json").read_text())
    fetch(
        fs,
        f"{PUBLIC}/exp373-mpnn-diversity-neff/v1/inputs/ground_truth.json",
        PREVIOUS / "inputs/ground_truth.json",
        new_plan["input_hashes"]["inputs/ground_truth.json"],
    )
    ground_truth = {
        r["stem"]: r
        for r in json.loads((PREVIOUS / "inputs/ground_truth.json").read_text())
    }
    inventories = pd.concat(
        [
            pd.read_csv(PREVIOUS / "data/raw_inventory.csv"),
            pd.read_csv(HERE / "data/source/exp373_raw_inventory_two_epoch.csv"),
        ]
    )
    tasks, targets, inventory, timings, selected_raw = [], [], [], [], []
    for _, row in selected.iterrows():
        stem = row.stem
        source = target_source.loc[stem]
        resolved = ground_truth[stem]["resolved"]
        sequence = "".join(row.input_seq[i] for i in resolved)
        if sequence != source.input_seq:
            raise ValueError(f"{stem}: resolved sequence does not match Helico")
        mapping = {i: j for j, i in enumerate(resolved)}
        target = dict(
            stem=stem,
            eval_set="eval-val",
            L=int(row.L),
            L_helico=len(sequence),
            msa_neff=float(row.msa_neff),
            msa_depth=int(row.msa_depth),
            is_viral=int(row.is_viral),
            input_seq=row.input_seq,
            helico_sequence=sequence,
            resolved=resolved,
        )
        targets.append(target)
        fetch(
            fs,
            f"{PUBLIC}/exp311-helico-exp277-contact-count-sweep/inputs/gt/{stem}.cif.gz",
            inputs / "gt" / f"{stem}.cif.gz",
            source_manifest["files_sha256"][f"{stem}.cif.gz"],
        )
        paired = {}
        for model in ("default", "mpnn2-lr5e-4"):
            inv = inventories.loc[
                (inventories.model == model)
                & (inventories.stem == stem)
                & (inventories.temperature == 1)
                & (inventories["mode"] == "randomized")
            ]
            assert len(inv) == 1
            inv = inv.iloc[0].to_dict()
            inventory.append(inv)
            root = "raw" if model == "default" else "raw_two_epoch"
            raw_path = (
                PREVIOUS / "scratch" / root / model / f"{stem}-t1.0-randomized.json.gz"
            )
            fetch(fs, inv["uri"], raw_path, inv["sha256"])
            raw = json.loads(gzip.decompress(raw_path.read_bytes()))
            assert raw["target"]["input_seq"] == row.input_seq
            assert raw["target"]["eval_set"] == "eval-val"
            assert (
                raw["model"] == model
                and raw["temperature"] == 1
                and raw["mode"] == "randomized"
            )
            samples = sorted(raw["samples"], key=lambda s: s["rollout"])[:10]
            assert [s["rollout"] for s in samples] == list(range(10))
            timings.append(
                {
                    **raw["timing"],
                    "model": model,
                    "usage": "reused first 10 of measured 100-rollout batch",
                }
            )
            for sample in samples:
                ri = sample["rollout"]
                assert sample["finish_reason"] == "stop"
                parsed = parse_contacts(sample["text"], sample["offset"], int(row.L))
                assert [list(p) for p in parsed["contacts"]] == sample["contacts"]
                assert sample["seed"] == sample_seed(
                    stem, "randomized", sample["block"], ri
                )
                identity = (
                    sample["seed"],
                    sample["offset"],
                    sample["prompt_token_ids"],
                    raw["prompts"][sample["prompt_id"]],
                )
                if model == "default":
                    paired[ri] = identity
                else:
                    assert paired[ri] == identity
                mapped = [
                    (mapping[i], mapping[j])
                    for i, j in sample["contacts"]
                    if i in mapping and j in mapping
                ]
                effective = sorted(set((i, j) for i, j in mapped if j - i >= 6))
                task = dict(
                    model=model,
                    stem=stem,
                    rollout=ri,
                    task_id=f"{stem}__{model}__r{ri:02}",
                    contacts=effective,
                    raw_sha256=inv["sha256"],
                    raw_uri=inv["uri"],
                    prompt_id=sample["prompt_id"],
                    marinfold_seed=sample["seed"],
                    helico_seed=int.from_bytes(
                        hashlib.sha256(f"exp381-helico:{stem}:{ri}".encode()).digest()[
                            :8
                        ],
                        "big",
                    )
                    % (2**63 - 1),
                    n_canonical_contacts=len(sample["contacts"]),
                    n_unresolved_removed=len(sample["contacts"]) - len(mapped),
                    n_mapped_near_removed=len(mapped) - len(effective),
                    n_effective_contacts=len(effective),
                )
                task["input_sha256"] = hashlib.sha256(
                    json.dumps(task, sort_keys=True).encode()
                ).hexdigest()
                tasks.append(task)
                selected_raw.append(
                    dict(
                        model=model,
                        stem=stem,
                        sample=sample,
                        prompt=raw["prompts"][sample["prompt_id"]],
                    )
                )
    (inputs / "tasks.json").write_text(json.dumps(tasks, indent=2))
    (inputs / "targets.json").write_text(json.dumps(targets, indent=2))
    (inputs / "selected_raw.json.gz").write_bytes(
        gzip.compress(json.dumps(selected_raw).encode(), mtime=0)
    )
    pd.DataFrame(targets).drop(columns=["resolved"]).to_csv(
        HERE / "data/targets.csv", index=False
    )
    pd.DataFrame(tasks).drop(columns=["contacts"]).to_csv(
        HERE / "data/contact_inputs.csv", index=False
    )
    pd.DataFrame(inventory).to_csv(HERE / "data/source_raw_inventory.csv", index=False)
    pd.DataFrame(timings).to_csv(
        HERE / "data/reused_marinfold_timings.csv", index=False
    )
    old_plan = json.loads((PREVIOUS / "data/plan.json").read_text())
    plan = dict(
        issue=381,
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        selection="10 lowest msa_neff (n_eff_0.8) among 97 eval-val, tie stem",
        stems=[t["stem"] for t in targets],
        models={
            "default": old_plan["models"]["default"],
            "mpnn2-lr5e-4": new_plan["models"]["mpnn2-lr5e-4"],
        },
        rollouts=list(range(10)),
        temperature=1,
        top_p=0.95,
        top_k=-1,
        mode="randomized",
        helico_source_sha="b10385d736673c81b10e70d1099962af6f2573c0",
        helico_checkpoint="contacts-msafree-01-step-6000.pt",
        helico_checkpoint_sha256="779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644",
        helico_n_samples=1,
        helico_n_cycles=6,
        helico_dtype="bfloat16",
        helico_seed="first 8 bytes SHA256(exp381-helico:{stem}:{rollout}), big-endian, mod (2**63-1)",
        helico_max_tokens=2048,
        selector="max ranking_score; tie lowest rollout",
        tmscore_revision="23f14ffcf8647f2dc1c0e01043db3a7f3a661c28",
        bootstrap_seed=381,
        bootstrap_draws=10000,
        files_sha256={
            str(p.relative_to(HERE)): digest(p)
            for p in sorted(inputs.rglob("*"))
            if p.is_file()
        },
        plan_markdown_sha256=digest(HERE / "PLAN.md"),
        metadata_sha256=digest(PREVIOUS / "data/metadata.csv"),
    )
    plan_path = HERE / "data/plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        plan["frozen_utc"] = existing["frozen_utc"]
        if plan != existing:
            raise ValueError("refusing to change a frozen plan")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Frozen {len(targets)} targets and {len(tasks)} contact sets")


if __name__ == "__main__":
    main()
