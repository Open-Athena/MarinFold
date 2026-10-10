# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared harness for the issue #341 interpretability analyses.

Every intervention in this experiment (layer ablation, residual capture for
probes and the tuned lens, model swaps) is scored by the same quantity: the
pairwise P(contact) readout from
``marinfold.document_structures.contacts_v1.inference``. This module keeps
that readout as the single source of truth and adds only what the analyses
need around it:

- **Model loading.** :func:`load_model` goes through ``load_backend``, so the
  transformers-5 rope repair (``marinfold.inference._config``) is applied by
  the same code the production evaluator uses, and checks that it took.
  :func:`load_unrepaired_model` builds the negative control the plan's
  harness check 1 calls for: the same weights read with the raw exported
  config, which transformers 4.x loads with theta 10000 and no llama3
  scaling.
- **Hooks.** Plain ``torch`` forward hooks on the Qwen3 decoder layers.
  :func:`capture_residuals` records the residual stream after the embedding
  and after every layer; :func:`mean_ablation` replaces chosen attention or
  MLP outputs with a fixed mean vector. Both act on the backend's own model,
  so a hooked readout is the production readout with hooks attached.
- **Eval proteins.** :func:`load_eval_proteins` reads exp245's FoldBench
  monomers (eval-val, eval-denovo) from the committed exp245 data directory.
  eval-test has a logged read budget and is refused here.
- **Metric.** :func:`r_precision` scores a P(contact) matrix over the
  resolved-residue pair universe, the convention exp89's
  ``compute_metrics.py`` and the exp245 headline numbers use (pairs touching
  an unresolved residue are excluded, not counted as negatives).

The plan (section 4.5) named nnsight for hooks. Plain forward hooks cover
residual capture and ablation without an extra dependency pinned against
transformers 4.x; the Jacobian steps need gradients, which the readout's
``inference_mode`` forward does not provide, and will get their own forward
path when they are written.

Backends here are always the transformers backend (``load_backend(
"transformers", ...)``); the functions rely on its ``model`` property and
``from_model`` constructor, which the other backends do not have.
"""

import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from marinfold import Backend, load_backend
from marinfold.document_structures.contacts_v1.inference import (
    ContactStructure,
    InferenceConfig,
    gt_contact_matrix,
    pairwise_score_matrix,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, residues_from_sequence
from marinfold.registry import resolve_model
from transformers import AutoConfig, AutoModelForCausalLM

EXP245_DATA = (
    Path(__file__).resolve().parent.parent / "exp245_evals_foldbench_held_out_monomers" / "data"
)
ALLOWED_EVAL_SETS = ("eval-val", "eval-denovo")
# Every readout in this experiment averages over this many document seeds
# (plan section 4.4); the salts are pairwise_score_matrix's
# f"{entry_id}#cv1ens{k}", so each protein's shuffle and residue offset are
# identical across models and interventions.
DEFAULT_SEEDS = 4
MIN_SEQ_SEPARATION = 6
LONG_RANGE_SEPARATION = 24
TRAINED_ROPE_THETA = 500_000

Site = Literal["attn", "mlp"]


@dataclass(frozen=True)
class EvalProtein:
    """One exp245 monomer with its ground truth and resolved-residue mask."""

    eval_set: str
    structure: ContactStructure
    resolved: np.ndarray  # 0-based sequence indices with resolved coordinates

    @property
    def stem(self) -> str:
        return self.structure.entry_id

    @property
    def length(self) -> int:
        return len(self.structure.residues)


def load_eval_proteins(eval_sets: Sequence[str] = ALLOWED_EVAL_SETS) -> list[EvalProtein]:
    """Read the scorable exp245 proteins in ``eval_sets``, in eval_sets.csv order.

    Contacts and resolved positions in ``gt_universe_scored.jsonl`` are
    already 0-based sequence indices (exp89's ``true_matrix`` indexes them
    directly), so they are used as-is.
    """
    refused = sorted(set(eval_sets) - set(ALLOWED_EVAL_SETS))
    if refused:
        raise ValueError(f"eval sets {refused} are not readable here (eval-test has a read budget)")
    sets = pd.read_csv(EXP245_DATA / "eval_sets.csv")
    sets = sets[sets["eval_set"].isin(eval_sets) & (sets["scorable"] == 1)]
    ground_truth: dict[str, dict] = {}
    with (EXP245_DATA / "gt_universe_scored.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            ground_truth[record["stem"]] = record

    proteins = []
    for row in sets.itertuples(index=False):
        record = ground_truth[row.stem]
        residues = residues_from_sequence(row.sequence)
        if len(residues) != record["L"]:
            raise ValueError(f"{row.stem}: sequence has {len(residues)} residues, ground truth L={record['L']}")
        contacts = tuple(RawContact(i, j, degree) for i, j, degree in record["contacts"])
        structure = ContactStructure(entry_id=row.stem, residues=residues, gt_contacts=contacts)
        resolved = np.asarray(record["resolved"], dtype=np.int64)
        proteins.append(EvalProtein(eval_set=row.eval_set, structure=structure, resolved=resolved))
    return proteins


def load_model(spec: str | None, *, device: str, dtype: str = "bfloat16") -> Backend:
    """Load a MODELS.yaml nickname (``None`` = registry default) with rope repaired.

    Raises if the loaded config does not carry the trained rope (theta
    500000, llama3 scaling): activations from such a model come from a
    different network.
    """
    backend = load_backend("transformers", model=spec, device=device, dtype=dtype)
    config = backend.model.config
    scaling = config.rope_scaling or {}
    if config.rope_theta != TRAINED_ROPE_THETA or scaling.get("rope_type") != "llama3":
        raise ValueError(f"rope repair did not take: theta={config.rope_theta} scaling={scaling}")
    return backend


def load_unrepaired_model(spec: str | None, repaired: Backend) -> Backend:
    """Same weights, raw exported config: the rope-repair negative control.

    ``repaired`` is the :func:`load_model` backend for the same ``spec``; the
    control reuses its tokenizer, device and dtype, and is wrapped by the
    same backend class so it is scored by identical code.

    Raises if the raw config already reads as theta 500000 (an export that
    needs no repair), since the control would then be identical to the
    repaired model.
    """
    directory = resolve_model(spec)
    config = AutoConfig.from_pretrained(str(directory))
    if config.rope_theta == TRAINED_ROPE_THETA:
        raise ValueError(f"{directory} loads with the trained rope as-is; there is no unrepaired control")
    model = AutoModelForCausalLM.from_pretrained(
        str(directory), config=config, dtype=repaired.model.dtype
    ).to(repaired.model.device)
    return type(repaired).from_model(model, repaired.tokenizer)


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """The Qwen3 decoder layer stack (``model.model.layers``)."""
    return model.model.layers


@contextlib.contextmanager
def capture_residuals(model: torch.nn.Module) -> Iterator[list[list[torch.Tensor]]]:
    """Record the residual stream at every layer boundary while active.

    Yields a list with one entry per layer boundary: index 0 is the
    embedding output, index ``k`` the output of decoder layer ``k - 1``.
    Each entry collects one CPU tensor ``[batch, seq, hidden]`` per forward
    call, in call order (the readout calls the model once for the prefix and
    once per chunk of tails). The hooks only read; outputs are unchanged.
    """
    captured: list[list[torch.Tensor]] = [[] for _ in range(len(decoder_layers(model)) + 1)]

    def recorder(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index].append(hidden.detach().to("cpu"))

        return hook

    handles = [model.model.embed_tokens.register_forward_hook(recorder(0))]
    for k, layer in enumerate(decoder_layers(model)):
        handles.append(layer.register_forward_hook(recorder(k + 1)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def mean_ablation(model: torch.nn.Module, means: Mapping[tuple[int, Site], torch.Tensor]) -> Iterator[None]:
    """Replace selected attention / MLP outputs with fixed mean vectors.

    ``means`` maps ``(layer, site)`` to a ``[hidden]`` vector that replaces
    that sublayer's output at every position, before it is added to the
    residual stream. An empty mapping registers no hooks, so the readout is
    the clean one (harness check 2).
    """

    def replacer(mean: torch.Tensor):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (mean.to(output[0]).expand_as(output[0]), *output[1:])
            return mean.to(output).expand_as(output)

        return hook

    layers = decoder_layers(model)
    handles = []
    for (layer, site), mean in means.items():
        module = layers[layer].self_attn if site == "attn" else layers[layer].mlp
        handles.append(module.register_forward_hook(replacer(mean)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def pcontact(backend: Backend, protein: EvalProtein, *, seeds: int = DEFAULT_SEEDS) -> np.ndarray:
    """Seed-averaged ``[L, L]`` P(contact) matrix from the production readout."""
    built = pairwise_score_matrix(backend, protein.structure, InferenceConfig(model=None, ensemble_k=seeds))
    if built is None:
        raise ValueError(f"{protein.stem}: contacts-v1 could not serialize this chain")
    return built[0]


def r_precision(
    score: np.ndarray,
    protein: EvalProtein,
    *,
    min_separation: int = MIN_SEQ_SEPARATION,
    max_separation: int | None = None,
) -> float:
    """R-precision over resolved pairs with separation in the given band.

    R is the number of true contacts (degree >= 0.001) in the band; the score
    is the fraction of the top-R ranked pairs that are true. NaN when the
    band has no true contacts.
    """
    gt = gt_contact_matrix(protein.structure.gt_contacts, protein.length, MIN_SEQ_SEPARATION)
    a, b = np.triu_indices(len(protein.resolved), k=1)
    rows, cols = protein.resolved[a], protein.resolved[b]
    separation = cols - rows
    band = separation >= min_separation
    if max_separation is not None:
        band &= separation <= max_separation
    labels = gt[rows[band], cols[band]]
    n_true = int(labels.sum())
    if n_true == 0:
        return float("nan")
    order = np.argsort(-score[rows[band], cols[band]], kind="mergesort")
    return float(labels[order[:n_true]].mean())
