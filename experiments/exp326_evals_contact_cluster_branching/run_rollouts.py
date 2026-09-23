"""Generate locally conditioned contacts-v1 continuation rollouts."""

import argparse
import hashlib
import json
import platform
import random
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BEGIN = "<begin_statements>"
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
NUM_POSITIONS = 2000
MIN_SEPARATION = 6
CONTEXT = 8192
HERE = Path(__file__).resolve().parent


def stable_seed(value: str) -> int:
    """Return a reproducible positive 31-bit seed."""
    return (
        int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF
    )


def selected_targets(targets: pd.DataFrame, selection: str) -> pd.DataFrame:
    """Select one preregistered target cohort."""
    if selection == "natural-dev":
        return targets[(targets.cohort == "eval-val") & (targets.split == "dev")]
    if selection == "natural-heldout":
        return targets[(targets.cohort == "eval-val") & (targets.split == "test")]
    if selection == "foldswitch-dev":
        return targets[
            (targets.cohort == "foldswitch")
            & (targets.split == "dev")
            & targets.primary.astype(bool)
        ]
    raise ValueError(f"unknown selection: {selection}")


def parse_rollout(
    text: str, position_to_sequence: dict[int, int]
) -> tuple[list[list[int]], int, int, int]:
    """Parse unique valid contacts and malformed/out-of-range diagnostics."""
    contacts: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    syntactic = CONTACT_RE.findall(text)
    malformed = text.count("<contact>") - len(syntactic)
    out_of_range = 0
    too_close = 0
    for raw_left, raw_right in syntactic:
        left = position_to_sequence.get(int(raw_left))
        right = position_to_sequence.get(int(raw_right))
        if left is None or right is None:
            out_of_range += 1
            continue
        contact = (min(left, right), max(left, right))
        if contact[1] - contact[0] < MIN_SEPARATION:
            too_close += 1
            continue
        if contact in seen:
            continue
        seen.add(contact)
        contacts.append([contact[0], contact[1]])
    return contacts, malformed, out_of_range, too_close


def make_prompt(
    stem: str,
    sequence: str,
    arm: str,
    rollout: int,
    bundle: list[list[int]],
    build_document,
    residues_from_sequence,
    generation_config,
) -> tuple[str, dict[int, int]]:
    """Build one fresh realization and append a coherent contact bundle."""
    document = build_document(
        f"{stem}:{arm}:r{rollout}",
        residues_from_sequence(sequence),
        [],
        config=generation_config(),
    )
    if document is None:
        raise ValueError(f"{stem}: document builder rejected sequence")
    prefix = document.document[: document.document.index(BEGIN) + len(BEGIN)]
    position_to_sequence = {
        (document.n_term_index + index) % NUM_POSITIONS: index
        for index in range(document.seq_len)
    }
    rng = random.Random(stable_seed(f"{stem}:{arm}:{rollout}:orientation"))
    statements = []
    for raw_left, raw_right in bundle:
        left, right = int(raw_left), int(raw_right)
        if not (0 <= left < len(sequence) and 0 <= right < len(sequence)):
            raise ValueError(f"{stem}: bundle contact {(left, right)} is out of range")
        if rng.random() < 0.5:
            left, right = right, left
        left_position = (document.n_term_index + left) % NUM_POSITIONS
        right_position = (document.n_term_index + right) % NUM_POSITIONS
        statements.append(f"<contact> <p{left_position}> <p{right_position}>")
    if statements:
        prefix += " " + " ".join(statements)
    return prefix, position_to_sequence


def main() -> None:
    """Load the checkpoint once and generate resumable per-target arm files."""
    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig,
        build_document,
        residues_from_sequence,
    )
    from torch import __version__ as torch_version
    from torch.cuda import get_device_properties
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=HERE / "data" / "targets.csv")
    parser.add_argument("--plans", type=Path)
    parser.add_argument(
        "--selection",
        choices=["natural-dev", "natural-heldout", "foldswitch-dev"],
        required=True,
    )
    parser.add_argument("--arms", default="cluster_k3,random_k3,cluster_k5,random_k5")
    parser.add_argument("--out", type=Path, default=HERE / "_cache")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gpu-fraction", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    targets = selected_targets(pd.read_csv(args.targets), args.selection)
    targets = targets.sort_values(["L", "stem"])
    if args.limit is not None:
        targets = targets.iloc[: args.limit]
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    expected_arms = {"iid100", "cluster_k3", "random_k3", "cluster_k5", "random_k5"}
    if not set(arms) <= expected_arms:
        raise ValueError(f"unknown arms: {set(arms) - expected_arms}")
    branch_arms = [arm for arm in arms if arm != "iid100"]
    if branch_arms and args.plans is None:
        raise ValueError("--plans is required for conditioned branch arms")
    plans = pd.read_csv(args.plans) if args.plans is not None else pd.DataFrame()
    if not plans.empty:
        plans = plans[plans.selection == args.selection]

    jobs = []
    for target in targets.itertuples():
        for arm in arms:
            raw_path = args.out / arm / target.cohort / f"{target.stem}.parquet"
            timing_path = (
                args.out / arm / target.cohort / f"{target.stem}.timing.parquet"
            )
            if raw_path.exists() and timing_path.exists():
                print(f"[exp326] skip {arm} {target.stem}", flush=True)
                continue
            jobs.append((target, arm, raw_path, timing_path))
    if not jobs:
        print("[exp326] all selected jobs already complete", flush=True)
        return

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    end_id = int(tokenizer.convert_tokens_to_ids("<end>"))
    if end_id == tokenizer.unk_token_id:
        raise ValueError("checkpoint tokenizer lacks <end>")
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=CONTEXT,
        gpu_memory_utilization=args.gpu_fraction,
        enable_prefix_caching=False,
        generation_config="vllm",
        max_num_seqs=args.max_num_seqs,
        seed=326,
        enforce_eager=args.enforce_eager,
    )
    model_load_seconds = time.perf_counter() - load_started
    gpu = get_device_properties(0)
    metadata = {
        "model_nickname": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "runner_tag": "local",
        "gpu_name": gpu.name,
        "gpu_total_memory_gb": gpu.total_memory / 1e9,
        "gpu_compute_capability": f"{gpu.major}.{gpu.minor}",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch_version,
    }

    total_jobs = len(jobs)
    job_number = 0
    for target in targets.itertuples():
        for arm in arms:
            raw_path = args.out / arm / target.cohort / f"{target.stem}.parquet"
            timing_path = (
                args.out / arm / target.cohort / f"{target.stem}.timing.parquet"
            )
            if raw_path.exists() and timing_path.exists():
                continue
            job_number += 1
            if arm == "iid100":
                selected_plans = pd.DataFrame(
                    {
                        "rollout": range(100),
                        "bundle_json": ["[]"] * 100,
                        "source_rollout": [-1] * 100,
                        "cluster_id": [-1] * 100,
                        "cluster_visits": [0] * 100,
                        "used_fallback": [False] * 100,
                    }
                )
            else:
                selected_plans = plans[(plans.stem == target.stem) & (plans.arm == arm)]
                selected_plans = selected_plans.sort_values("rollout")
                if list(selected_plans.rollout.astype(int)) != list(range(50)):
                    raise ValueError(
                        f"{target.stem} {arm}: expected rollout plans 0..49"
                    )

            prompts = []
            position_maps = []
            bundles = []
            sampling = []
            for plan in selected_plans.itertuples():
                bundle = json.loads(plan.bundle_json)
                prompt, position_map = make_prompt(
                    target.stem,
                    target.sequence,
                    arm,
                    int(plan.rollout),
                    bundle,
                    build_document,
                    residues_from_sequence,
                    GenerationConfig,
                )
                prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                max_new = min(CONTEXT - prompt_tokens, 6 * int(target.L) + 128)
                if max_new < 1:
                    raise ValueError(f"{target.stem}: no completion budget remains")
                prompts.append(prompt)
                position_maps.append(position_map)
                bundles.append(bundle)
                sampling.append(
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=-1,
                        max_tokens=max_new,
                        stop_token_ids=[end_id],
                        skip_special_tokens=False,
                        seed=stable_seed(f"{target.stem}:{arm}:{plan.rollout}:sample"),
                    )
                )

            started = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            rows = []
            for plan, bundle, output, position_map, prompt in zip(
                selected_plans.itertuples(),
                bundles,
                outputs,
                position_maps,
                prompts,
                strict=True,
            ):
                completion = output.outputs[0]
                contacts, malformed, out_of_range, too_close = parse_rollout(
                    completion.text, position_map
                )
                rows.append(
                    {
                        "rollout": int(plan.rollout),
                        "mode": arm,
                        "contacts": contacts,
                        "seed_contacts": bundle,
                        "n_contacts": len(contacts),
                        "n_seed_contacts": len(bundle),
                        "n_tokens": len(completion.token_ids),
                        "max_new": sampling[int(plan.rollout)].max_tokens,
                        "finished": completion.finish_reason == "stop",
                        "malformed_contacts": malformed,
                        "out_of_range_contacts": out_of_range,
                        "too_close_contacts": too_close,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "cohort": target.cohort,
                        "dataset": target.dataset,
                        "stem": target.stem,
                        "L": int(target.L),
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "source_rollout": int(plan.source_rollout),
                        "cluster_id": int(plan.cluster_id),
                        "cluster_visits": int(plan.cluster_visits),
                        "used_fallback": bool(plan.used_fallback),
                    }
                )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(raw_path, index=False)
            written = time.perf_counter()
            timing = pd.DataFrame(
                [
                    {
                        "stem": target.stem,
                        "cohort": target.cohort,
                        "n_residues": int(target.L),
                        "n_pairs": int(target.L) * (int(target.L) - 1) // 2,
                        "mode": arm,
                        "elapsed_seconds": elapsed,
                        "model_load_seconds": model_load_seconds / total_jobs,
                        "total_seconds": written
                        - started
                        + model_load_seconds / total_jobs,
                        "n_rollouts": len(rows),
                        "n_finished": sum(bool(row["finished"]) for row in rows),
                        "n_malformed": sum(
                            int(row["malformed_contacts"]) for row in rows
                        ),
                        "generated_tokens": sum(int(row["n_tokens"]) for row in rows),
                        "batch_size": args.max_num_seqs,
                        "bundle_size": 0 if arm == "iid100" else int(arm[-1]),
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        **metadata,
                    }
                ]
            )
            timing.to_parquet(timing_path, index=False)
            print(
                f"[exp326] {job_number}/{total_jobs} {arm} {target.stem} "
                f"L={target.L} finished={timing.iloc[0].n_finished}/{len(rows)} "
                f"tokens={timing.iloc[0].generated_tokens} time={elapsed:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
