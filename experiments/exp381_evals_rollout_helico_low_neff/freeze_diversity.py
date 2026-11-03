"""Freeze the contact-diversity protocol and verified 100-rollout source prefixes."""

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "exp373_evals_mpnn_diversity_neff"


def digest(path: Path) -> str:
    """Return file identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Write an immutable plan before new contact generation or selection."""
    original = json.loads((HERE / "data/plan.json").read_text())
    prior = json.loads((PREVIOUS / "data/plan.json").read_text())
    targets = json.loads((HERE / "inputs/targets.json").read_text())
    sources = pd.read_csv(HERE / "data/source_raw_inventory.csv").to_dict("records")
    root = HERE / "inputs/diversity"
    root.mkdir(exist_ok=True)
    (root / "targets.json").write_text(json.dumps(targets, indent=2) + "\n")
    plan = dict(
        original_plan_sha256=digest(HERE / "data/plan.json"),
        protocol_sha256=digest(HERE / "DIVERSITY_PLAN.md"),
        targets_sha256=digest(root / "targets.json"),
        models=original["models"],
        marinfold_revision=prior["marinfold_revision"],
        prefix_sources=sources,
        rollout_start=100,
        rollout_stop=1000,
        cluster="cw-us-east-02a",
        artifact_root="hf://buckets/open-athena/MarinFold/data/exp381-rollout-helico-low-neff/v3-diversity",
        temperature=1.0,
        mode="randomized",
        top_p=0.95,
        top_k=-1,
        selector="k-medoids++ Jaccard; 10 starts; max100 alternate updates; lowest distortion; lexicographic ties",
        selection_seed="SHA256(exp381-diversity:{stem}) first8bytes modulo2**32",
        random_control_seed="SHA256(exp381-random10:{stem}) first8bytes modulo2**32",
        n_representatives=10,
        helico_arms=["medoids10", "random10"],
        helico_seed=original["helico_seed"],
        helico_samples=1,
        helico_cycles=6,
        helico_source_sha=original["helico_source_sha"],
        helico_checkpoint_sha256=original["helico_checkpoint_sha256"],
        bootstrap_draws=10000,
        bootstrap_seed=381,
        primary="confidence-selected TM-score: medoids10 minus random10; paired proteins within checkpoint",
    )
    path = HERE / "data/diversity_plan.json"
    if path.exists():
        previous = json.loads(path.read_text())
        plan["frozen_utc"] = previous["frozen_utc"]
        # The historical inventory has missing optional eval_set values encoded
        # as NaN. Compare the persisted representation, since NaN != NaN even
        # when every frozen byte is unchanged. Raw target splits are checked
        # directly during selection; this field does not select the cohort.
        if json.dumps(plan, indent=2) + "\n" != path.read_text():
            raise ValueError("Frozen diversity plan changed")
    else:
        plan["frozen_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        path.write_text(json.dumps(plan, indent=2) + "\n")
    print("Frozen 20,000 contact rollouts and 400 new Helico predictions")


if __name__ == "__main__":
    main()
