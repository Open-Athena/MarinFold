"""Freeze matched baseline sequences, archived MSAs and independent seed budgets."""

import datetime as dt
import hashlib
import json
from pathlib import Path

import modal
import pandas as pd

from baseline_config import AF3_SHA, ESM_REVISION, ESM_SHA

HERE = Path(__file__).resolve().parent


def digest(payload: bytes) -> str:
    """Return a reproducible content hash."""
    return hashlib.sha256(payload).hexdigest()


def project_a3m(text: str, sequence: str, positions: list[int]) -> str:
    """Project match columns onto resolved residues, retaining intervening insertions.

    Insertions after a retained match column are preserved only when the next
    original match column is also retained. Unresolved segments, including their
    insertions, are removed rather than masquerading as homolog insertions.
    Sequence membership and order remain unchanged; AF3 performs its own dedup.
    """
    records = []
    for line in text.splitlines():
        if line.startswith(">"):
            records.append([line, ""])
        elif line and not line.startswith("#"):
            records[-1][1] += line.strip()
    if not records or records[0][1] != sequence:
        raise ValueError("Archived MSA query differs from the full prompt sequence")
    keep = set(positions)
    output = []
    for header, aligned in records:
        column = -1
        trimmed = []
        for char in aligned:
            if char.islower() or char == ".":
                if column in keep and column + 1 in keep:
                    trimmed.append(char)
            else:
                column += 1
                if column in keep:
                    trimmed.append(char)
        if column + 1 != len(sequence):
            raise ValueError("MSA row has the wrong match-column count")
        output.extend((header, "".join(trimmed)))
    return "\n".join(output) + "\n"


def main() -> None:
    """Download only the ten small archived alignments and freeze the extension."""
    targets = json.loads((HERE / "inputs/targets.json").read_text())
    root = HERE / "inputs/baselines"
    root.mkdir(exist_ok=True)
    volume = modal.Volume.from_name("protenix-foldbench-msa")
    rows, requests = [], []
    for target_index, target in enumerate(targets):
        stem = target["stem"]
        path = root / f"{stem}.original.a3m"
        if not path.exists():
            path.write_bytes(
                b"".join(volume.read_file(f"{stem}/msa/0/0/non_pairing.a3m"))
            )
        projected = project_a3m(
            path.read_text(), target["input_seq"], target["resolved"]
        )
        if projected.splitlines()[1] != target["helico_sequence"]:
            raise ValueError("Resolved MSA query mismatch")
        (root / f"{stem}.a3m").write_text(projected)
        row = dict(
            stem=stem,
            sequence=target["helico_sequence"],
            n_residues=target["L_helico"],
            n_msa_rows=sum(x.startswith(">") for x in projected.splitlines()),
            original_msa_sha256=digest(path.read_bytes()),
            msa_sha256=digest(projected.encode()),
            sequence_sha256=digest(target["helico_sequence"].encode()),
        )
        rows.append(row)
        for model in ("esmfold2", "af3_single", "af3_msa"):
            for rollout in range(10):
                requests.append(
                    dict(
                        **row,
                        model=model,
                        rollout=rollout,
                        seed=381000 + 100 * target_index + rollout,
                        task_id=f"{stem}__{model}__r{rollout:02d}",
                    )
                )
    pd.DataFrame(rows).to_csv(HERE / "data/baseline_inputs.csv", index=False)
    (root / "tasks.json").write_text(json.dumps(requests, indent=2) + "\n")
    plan_path = HERE / "data/baseline_plan.json"
    protocol = dict(
        status="frozen_before_baseline_inference",
        n_targets=10,
        samples_per_target=10,
        targets=rows,
        original_plan_sha256=digest((HERE / "data/plan.json").read_bytes()),
        tasks_sha256=digest((root / "tasks.json").read_bytes()),
        af3=dict(
            source_sha=AF3_SHA,
            weights_volume="marinfold-exp325-alphafold",
            weights_sha256="0dd0290af76eec0119f4c0c41f441fbce4ec18aa21a0a02cac4a3af01d841369",
            recycles=10,
            diffusion_samples_per_seed=1,
            templates=False,
            single_sequence_msa="empty unpaired and paired MSA",
            msa="archived ColabFold non_pairing.a3m, projected to resolved residues",
            selector="full-precision native ranking_score",
        ),
        esmfold2=dict(
            source_sha=ESM_SHA,
            hf_repository="biohub/ESMFold2",
            revision=ESM_REVISION,
            weights_volume="esmfold2-weights",
            loops=20,
            sampling_steps=200,
            diffusion_samples_per_seed=1,
            lm_dropout=0.3,
            esmc_precision="bf16",
            selector="full-precision pTM",
            msa=False,
        ),
        analysis="Identical score.py; macro protein mean; oracle per metric, median, native-confidence pick, sample mean; k=1..10; paired protein bootstrap 10000 draws seed381",
        ordering="Fixed cohort order; seed=381000+100*target_index+rollout; paired across three baseline arms",
        timings="Per seed; AF3 inference includes cold-shape JIT, features and output separate; ESM fold API includes feature prep; no speed comparison",
        interpretation="Matched sample count, not matched compute or training exposure; ten-protein diagnostic pilot",
    )
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        protocol["frozen_utc"] = old["frozen_utc"]
        if old != protocol:
            raise ValueError("Frozen baseline protocol would change")
    else:
        protocol["frozen_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        plan_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        f"Frozen {len(requests)} predictions; archived MSA rows: {[r['n_msa_rows'] for r in rows]}"
    )


if __name__ == "__main__":
    main()
