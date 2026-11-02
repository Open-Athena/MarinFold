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
  harness check 1 calls for: the same weights with the config reduced to
  the bare transformers-5 rope shape, which transformers 4.x loads with
  theta 10000 and no llama3 scaling.
- **Hooks.** Plain ``torch`` forward hooks on the Qwen3 decoder layers.
  :func:`capture_residuals` records the residual stream after the embedding
  and after every layer; :func:`sublayer_means` accumulates the mean
  attention and MLP output of every layer; :func:`mean_ablation` replaces
  chosen attention or MLP outputs with a fixed mean vector. Both act on the backend's own model,
  so a hooked readout is the production readout with hooks attached.
- **Eval proteins.** :func:`load_eval_proteins` reads exp245's FoldBench
  monomers (eval-val, eval-denovo) from the committed exp245 data directory.
  eval-test has a logged read budget and is refused here.
- **Legacy proteins.** :func:`load_legacy_proteins` reads the legacy-554
  subset A2 uses (``a2_select.py``).
- **Held-out panel.** :func:`load_panel_proteins` reads the A3/A4 panel
  (``panel_select.py``) into the same :class:`EvalProtein` shape, so panel
  and eval proteins go through one readout and one metric.
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
import gzip
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
    fwd_matrix,
    gt_contact_matrix,
    pairwise_score_matrix,
    pcontact_from_fwd,
    prefix_and_positions,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, residues_from_sequence
from marinfold.registry import resolve_model
from transformers import AutoConfig, AutoModelForCausalLM

EXP245_DATA = (
    Path(__file__).resolve().parent.parent / "exp245_evals_foldbench_held_out_monomers" / "data"
)
DATA = Path(__file__).resolve().parent / "data"
ALLOWED_EVAL_SETS = ("eval-val", "eval-denovo")
PROTEIN_SETS = ("eval", "legacy-extra", "panel-long")
# panel-long: held-out panel structures longer than this, for B1's length question.
PANEL_LONG_MIN_LENGTH = 300
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


def load_legacy_proteins(path: Path) -> list[EvalProtein]:
    """Read legacy-554 proteins from ``a2_select.py``'s gzipped JSONL; ``eval_set`` is ``legacy-<dataset>``.

    Each record carries exp89's ground truth (``gt_universe.jsonl``: 0-based
    ``contacts`` with continuous degree and ``resolved``, the same convention
    as exp245's file) and exp213's query sequence. The stem is
    ``<dataset>__<stem>``, as in exp213, because legacy stems are not unique
    across datasets.
    """
    proteins = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            residues = residues_from_sequence(record["sequence"])
            if len(residues) != record["L"]:
                raise ValueError(f"{record['key']}: sequence has {len(residues)} residues, ground truth L={record['L']}")
            contacts = tuple(RawContact(i, j, degree) for i, j, degree in record["contacts"])
            structure = ContactStructure(entry_id=record["key"], residues=residues, gt_contacts=contacts)
            proteins.append(EvalProtein(
                eval_set=f"legacy-{record['dataset']}",
                structure=structure,
                resolved=np.asarray(record["resolved"], dtype=np.int64),
            ))
    return proteins


def load_panel_proteins(path: Path) -> list[EvalProtein]:
    """Read the A3/A4 panel parquet; ``eval_set`` is ``panel-removed`` or ``panel-kept``.

    Panel ground truth is the contact list of an untruncated contacts-v1
    document: every pair at degree >= 0.001 and separation >= 6, with the
    degree itself not recorded. Each contact is given degree 1.0, which only
    has to clear the readout's 0.001 floor. AFDB models have coordinates for
    every residue, so all residues count as resolved.
    """
    panel = pd.read_parquet(path)
    proteins = []
    for row in panel.itertuples(index=False):
        residues = residues_from_sequence(row.sequence)
        if len(residues) != row.seq_len:
            raise ValueError(f"{row.entry_id}: sequence has {len(residues)} residues, seq_len={row.seq_len}")
        contacts = tuple(RawContact(int(i), int(j), 1.0) for i, j in row.contacts)
        structure = ContactStructure(
            entry_id=row.entry_id, residues=residues, gt_contacts=contacts, global_plddt=float(row.global_plddt)
        )
        proteins.append(
            EvalProtein(eval_set=f"panel-{row.set}", structure=structure, resolved=np.arange(row.seq_len))
        )
    return proteins


def base_stem(stem: str) -> str:
    """The PDB chain of a stem: legacy stems are ``<dataset>__<chain>``, eval stems are the chain."""
    return stem.split("__")[-1]


def load_protein_set(
    name: str, *, max_length: int | None = None, sample: int | None = None, seed: int = 341
) -> list[EvalProtein]:
    """One of :data:`PROTEIN_SETS`.

    - ``eval``: eval-val and eval-denovo (116 proteins).
    - ``legacy-extra``: ``b1_select_extra.py``'s 246 legacy-554 proteins.
    - ``panel-long``: the A4 panel's removed structures (never in exp277's
      training data) longer than :data:`PANEL_LONG_MIN_LENGTH` residues.

    ``max_length`` drops longer proteins; ``sample`` keeps a seeded random
    subset of that size, in the set's original order.
    """
    if name == "eval":
        proteins = load_eval_proteins()
    elif name == "legacy-extra":
        proteins = load_legacy_proteins(DATA / "b1_extra_gt.jsonl.gz")
    elif name == "panel-long":
        proteins = [p for p in load_panel_proteins(DATA / "panel.parquet")
                    if p.eval_set == "panel-removed" and p.length > PANEL_LONG_MIN_LENGTH]
    else:
        raise ValueError(f"unknown protein set {name!r}; choose from {PROTEIN_SETS}")
    if max_length is not None:
        proteins = [p for p in proteins if p.length <= max_length]
    if sample is not None and sample < len(proteins):
        keep = np.sort(np.random.default_rng(seed).choice(len(proteins), size=sample, replace=False))
        proteins = [proteins[k] for k in keep]
    return proteins


def load_model(
    spec: str | None, *, device: str, dtype: str = "bfloat16", tail_batch_size: int = 64
) -> Backend:
    """Load a MODELS.yaml nickname (``None`` = registry default) with rope repaired.

    ``tail_batch_size`` is the backend's number of ``<contact> <p_i>`` tails
    per cached forward pass; it trades memory for throughput and does not
    change the readout beyond batched-matmul rounding.

    Raises if the loaded config does not carry the trained rope (theta
    500000, llama3 scaling): activations from such a model come from a
    different network.
    """
    backend = load_backend(
        "transformers", model=spec, device=device, dtype=dtype, tail_batch_size=tail_batch_size
    )
    config = backend.model.config
    scaling = config.rope_scaling or {}
    if config.rope_theta != TRAINED_ROPE_THETA or scaling.get("rope_type") != "llama3":
        raise ValueError(f"rope repair did not take: theta={config.rope_theta} scaling={scaling}")
    return backend


def load_unrepaired_model(spec: str | None, repaired: Backend) -> Backend:
    """Same weights, rope read the way an unrepaired 4.x load reads it.

    The negative control for harness check 1. Published exports have had
    ``repair_config_file`` applied, so their ``config.json`` carries both the
    transformers-5 ``rope_parameters`` block and the 4.x ``rope_theta`` /
    ``rope_scaling`` keys and loads correctly as-is. The control restores
    the hazard: it drops the 4.x keys, leaving the bare transformers-5 shape,
    which transformers 4.x ignores in favour of the architecture default
    (theta 10000, no scaling).

    ``repaired`` is the :func:`load_model` backend for the same ``spec``; the
    control reuses its tokenizer, device and dtype, and is wrapped by the
    same backend class so it is scored by identical code.
    """
    directory = resolve_model(spec)
    raw = json.loads((directory / "config.json").read_text())
    bare = {k: v for k, v in raw.items() if k not in ("rope_theta", "rope_scaling")}
    if "rope_parameters" not in bare:
        raise ValueError(f"{directory} has no rope_parameters block; the control would equal the model")
    config = AutoConfig.for_model(**bare)
    if config.rope_theta == TRAINED_ROPE_THETA:
        raise ValueError(f"transformers read theta {config.rope_theta} from the bare config; no control")
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
def query_residuals(model: torch.nn.Module) -> Iterator[list[list[torch.Tensor]]]:
    """Record the residual at each tail's last position, at every layer boundary.

    The readout forwards the prefix once per ``next_token_probs`` call and
    then the tails (``<contact>`` for lp1, ``<contact> <p_i>`` for lp2) in
    chunks; the next-token distribution is read at a tail's last position.
    This keeps exactly those rows: calls whose sequence is at most 2 tokens
    long are tails (a prefix is always longer), and from each the last
    position is kept, on device. Index 0 is the embedding output, index
    ``k`` the output of decoder layer ``k - 1`` (pre final norm). Per
    boundary the entries are ``[batch, hidden]`` in call order, so after one
    ``fwd_matrix`` they concatenate to ``L + 1`` rows: the lp1 query, then
    the lp2 query for each residue in sequence order.
    """
    captured: list[list[torch.Tensor]] = [[] for _ in range(len(decoder_layers(model)) + 1)]

    def recorder(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] <= 2:
                captured[index].append(hidden[:, -1, :].detach())

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
def sublayer_means(
    model: torch.nn.Module,
) -> Iterator[tuple[dict[tuple[int, Site], torch.Tensor], dict[tuple[int, Site], int]]]:
    """Accumulate the mean output of every attention and MLP sublayer while active.

    Yields ``(means, counts)``, filled on exit: ``(layer, site)`` to a float32
    ``[hidden]`` mean, and to the number of positions averaged, over every
    position of every forward call made inside the block (for the readout, the prefix passes and all tail chunks). Sums
    are kept in float64 on the model's device. The hooks only read; outputs
    are unchanged.
    """
    layers = decoder_layers(model)
    sums: dict[tuple[int, Site], torch.Tensor] = {}
    counts: dict[tuple[int, Site], int] = {}

    def accumulator(key: tuple[int, Site]):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            flat = hidden.detach().reshape(-1, hidden.shape[-1]).to(torch.float64)
            sums[key] = flat.sum(dim=0) + sums[key] if key in sums else flat.sum(dim=0)
            counts[key] = counts.get(key, 0) + flat.shape[0]

        return hook

    handles = []
    for k, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(accumulator((k, "attn"))))
        handles.append(layer.mlp.register_forward_hook(accumulator((k, "mlp"))))
    means: dict[tuple[int, Site], torch.Tensor] = {}
    try:
        yield means, counts
    finally:
        for handle in handles:
            handle.remove()
    for key, total in sums.items():
        means[key] = (total / counts[key]).to(torch.float32)


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


def pcontact_per_seed(
    backend: Backend, protein: EvalProtein, *, seeds: int = DEFAULT_SEEDS, seed_offset: int = 0
) -> np.ndarray:
    """``[seeds, L, L]`` P(contact), one matrix per document seed.

    Seed ``k`` is the realization ``pairwise_score_matrix`` draws with salt
    ``f"{entry_id}#cv1ens{k}"``, for ``k`` in ``seed_offset .. seed_offset +
    seeds - 1``. With ``seed_offset=0`` and ``seeds >= 2``, summing the
    matrices in order and dividing by ``seeds`` reproduces :func:`pcontact`
    exactly; a nonzero offset draws realizations the default readout never
    uses, so repeated runs add independent samples.
    """
    matrices = []
    for k in range(seed_offset, seed_offset + seeds):
        built = prefix_and_positions(protein.structure, entry_id=f"{protein.stem}#cv1ens{k}")
        if built is None:
            raise ValueError(f"{protein.stem}: contacts-v1 could not serialize this chain")
        prefix, seq_positions, _ = built
        matrices.append(pcontact_from_fwd(fwd_matrix(backend, prefix, seq_positions)))
    return np.stack(matrices)


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
