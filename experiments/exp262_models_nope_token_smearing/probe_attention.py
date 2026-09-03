# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 attention probes for issue #262.

Two questions, both asked of the trained default checkpoint while it reads
*ground-truth* contacts-v1 documents under teacher forcing.

**(a) Are there previous-token heads?** A width-3 causal smear hands every token
its own predecessors for free. If the trained model is already spending
attention heads on reaching back one or two positions, those heads are doing a
job the smear would do at no cost — the direct motivation for the smear half of
the proposal. Reported as per-(layer, head) attention mass at offsets 1 and 2.

**(b) Does retrieval decay with document distance?** The structure section is a
*randomly ordered* bag of contact statements, so a query at ``<pA>`` should be
able to reach every earlier mention of ``<pA>`` equally well no matter how far
back it sits — under a uniform shuffle a co-referent 800 tokens back is exactly
as informative as one 50 tokens back, by construction. RoPE supplies a locality
prior this format never asked for. We measure the *lift* of attention onto
co-referent tokens (earlier occurrences of the same position token) over the
chance rate, bucketed by distance. Lift that decays with distance is direct
evidence the prior is costing us long-range retrieval, and would predict exactly
the long-protein weakness we already see; flat or rising lift means the NoPE
half of the proposal has less to win than the argument for it assumes.

Two confounds the lift has to survive, both handled here:

* **Section composition.** Numerator and denominator are confined to the query's
  own section. Otherwise the far buckets fill with sequence-section tokens and
  the "decay" is just the section boundary sliding through the histogram.
* **Per-document co-referent density.** A position token appears in about
  ``2 * contacts / L`` statements out of ``3 * contacts``, so co-referent density
  falls as ``1/L``: long documents are sparser. Far-distance buckets are
  reachable only by long documents, so pooling raw counts across documents makes
  chance look rarer at distance and manufactures rising lift out of nothing. The
  chance rate is therefore evaluated **per document** and accumulated as an
  expected mass, making the reported lift a ratio of sums with per-document
  denominators.

Writes ``data/phase0_attention_offsets.csv``, ``data/phase0_attention_lift.csv``
and ``data/phase0_attention_docs.csv``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from probe_common import build_probe_document, load_model, load_targets

# Distance buckets over ``d = query - key``. Bucket 0 is the query attending to
# itself; buckets 1 and 2 are exactly what a width-3 smear covers.
BUCKET_EDGES = [0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097]
BUCKET_LABELS = [
    "0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128",
    "129-256", "257-512", "513-1024", "1025-2048", "2049-4096", "4097+",
]
NUM_BUCKETS = len(BUCKET_LABELS)
# One extra slot absorbs keys excluded from the analysis (wrong section, or
# non-causal); it is dropped when the CSVs are written.
EXCLUDED_BUCKET = NUM_BUCKETS
NUM_SLOTS = NUM_BUCKETS + 1


def bucketize(distance: torch.Tensor) -> torch.Tensor:
    """Map a non-negative token distance onto ``BUCKET_EDGES``."""
    edges = torch.tensor(BUCKET_EDGES[1:], device=distance.device)
    return torch.bucketize(distance, edges, right=False)


class AttentionAccumulator:
    """Sums attention statistics over documents, per layer and head."""

    def __init__(self, layers: int, heads: int):
        self.layers = layers
        self.heads = heads
        self.offset_mass = np.zeros((layers, heads, 4))  # offsets 0,1,2,3
        self.bos_mass = np.zeros((layers, heads))
        self.mass_total = np.zeros((layers, heads, NUM_SLOTS))
        self.mass_coref = np.zeros((layers, heads, NUM_SLOTS))
        # Mass a co-referent-blind reader would put on co-referents, using the
        # chance rate of the document the mass came from.
        self.mass_expected = np.zeros((layers, heads, NUM_SLOTS))
        self.count_all = np.zeros(NUM_SLOTS, dtype=np.int64)
        self.count_coref = np.zeros(NUM_SLOTS, dtype=np.int64)
        self.queries = 0

    def add_document(
        self,
        layer: int,
        attention: torch.Tensor,
        query_index: torch.Tensor,
        bucket: torch.Tensor,
        coref_query: torch.Tensor,
        coref_key: torch.Tensor,
        coref_density: np.ndarray,
    ) -> None:
        """Fold one layer's attention for one document into the running sums.

        ``attention`` is the full ``[1, H, T, T]`` map; everything else is
        precomputed once per document and shared across the layers.
        """
        selected = attention[0][:, query_index, :].float()  # [H, Q, T]
        heads, num_queries, tokens = selected.shape

        for offset in range(4):
            key_index = query_index - offset
            valid = key_index >= 0
            if valid.any():
                gathered = selected[:, valid, :].gather(
                    2, key_index[valid].view(1, -1, 1).expand(heads, -1, 1)
                )
                self.offset_mass[layer, :, offset] += gathered.squeeze(2).sum(1).cpu().numpy()
        self.bos_mass[layer] += selected[:, :, 0].sum(1).cpu().numpy()

        flat = selected.reshape(heads, num_queries * tokens)
        totals = torch.zeros(heads, NUM_SLOTS, device=selected.device)
        totals.index_add_(1, bucket.reshape(-1), flat)
        totals_host = totals.cpu().numpy()
        self.mass_total[layer] += totals_host
        self.mass_expected[layer] += totals_host * coref_density[None, :]

        if coref_query.numel():
            flat_index = coref_query * tokens + coref_key
            corefs = torch.zeros(heads, NUM_SLOTS, device=selected.device)
            corefs.index_add_(1, bucket.reshape(-1)[flat_index], flat[:, flat_index])
            self.mass_coref[layer] += corefs.cpu().numpy()

    def add_document_counts(self, counts_all: np.ndarray, counts_coref: np.ndarray, num_queries: int) -> None:
        self.count_all += counts_all
        self.count_coref += counts_coref
        self.queries += num_queries


def document_bucket_counts(bucket: torch.Tensor, coref_query: torch.Tensor, coref_key: torch.Tensor) -> tuple:
    """Per-document key counts per bucket, and the co-referent chance rate."""
    tokens = bucket.shape[1]
    counts_all = np.bincount(bucket.reshape(-1).cpu().numpy(), minlength=NUM_SLOTS).astype(np.int64)
    if coref_query.numel():
        flat_index = (coref_query * tokens + coref_key).cpu().numpy()
        counts_coref = np.bincount(
            bucket.reshape(-1).cpu().numpy()[flat_index], minlength=NUM_SLOTS
        ).astype(np.int64)
    else:
        counts_coref = np.zeros(NUM_SLOTS, dtype=np.int64)
    density = np.divide(
        counts_coref, counts_all, out=np.zeros(NUM_SLOTS), where=counts_all > 0
    )
    density[EXCLUDED_BUCKET] = 0.0
    return counts_all, counts_coref, density


def select_documents(targets: pd.DataFrame, tokenizer, max_tokens: int, count: int, seed: int) -> list:
    """A length-stratified sample of the exp245 monomer universe."""
    rng = np.random.default_rng(seed)
    ordered = targets.sort_values("L").reset_index(drop=True)
    picks = np.unique(np.linspace(0, len(ordered) - 1, num=min(count * 3, len(ordered))).astype(int))
    rng.shuffle(picks)
    documents = []
    for index in picks:
        row = ordered.iloc[index]
        document = build_probe_document(row.stem, row.input_seq, row.contacts, tokenizer)
        if len(document.token_ids) > max_tokens or document.contact_statements < 8:
            continue
        documents.append(document)
        if len(documents) >= count:
            break
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="MODELS.yaml nickname; default entry if omitted")
    parser.add_argument("--documents", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-queries", type=int, default=384)
    parser.add_argument("--seed", type=int, default=262)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    directory, tokenizer, model = load_model(arguments.model, attn_implementation="eager")
    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    print(f"[probe] model={directory} layers={layers} heads={heads}")

    targets = load_targets()
    documents = select_documents(targets, tokenizer, arguments.max_tokens, arguments.documents, arguments.seed)
    print(f"[probe] {len(documents)} documents, {min(len(d.token_ids) for d in documents)}"
          f"-{max(len(d.token_ids) for d in documents)} tokens")

    accumulators = {group: AttentionAccumulator(layers, heads) for group in ("struct_pos", "seq_pos")}
    rng = np.random.default_rng(arguments.seed)
    rows = []

    for document in documents:
        tokens = document.tokens
        token_ids = torch.tensor([document.token_ids], device="cuda")
        length = len(tokens)
        positions = np.arange(length)
        is_position = np.array([token.startswith("<p") and token[2:-1].isdigit() for token in tokens])
        in_structure = positions >= document.statements_start

        candidates = {
            # Skip the first 16 tokens of a section: a query needs some section
            # context before "what does it retrieve" means anything.
            "struct_pos": np.where(is_position & in_structure & (positions > document.statements_start + 16))[0],
            "seq_pos": np.where(is_position & ~in_structure & (positions > 16))[0],
        }
        # Keys are confined to the query's own section (see module docstring).
        key_floors = {"struct_pos": document.statements_start, "seq_pos": 0}
        key_ceilings = {"struct_pos": length, "seq_pos": document.statements_start}

        prepared = {}
        for group, pool in candidates.items():
            if len(pool) == 0:
                continue
            if len(pool) > arguments.max_queries:
                pool = np.sort(rng.choice(pool, size=arguments.max_queries, replace=False))
            query_index = torch.tensor(pool, device="cuda")
            key_positions = torch.arange(length, device="cuda")
            distance = query_index[:, None] - key_positions[None, :]
            in_section = (key_positions >= key_floors[group]) & (key_positions < key_ceilings[group])
            valid = (distance >= 0) & in_section[None, :]
            bucket = torch.where(
                valid, bucketize(distance.clamp(min=0)), torch.full_like(distance, EXCLUDED_BUCKET)
            )

            # Co-referents: earlier same-section tokens carrying the SAME
            # position token, i.e. the other statements about this residue.
            token_array = np.array(document.token_ids)
            same = token_array[pool][:, None] == token_array[None, :]
            same &= positions[None, :] < pool[:, None]
            same &= (positions >= key_floors[group])[None, :]
            same &= (positions < key_ceilings[group])[None, :]
            coref_query, coref_key = np.nonzero(same)
            coref_query = torch.tensor(coref_query, device="cuda", dtype=torch.long)
            coref_key = torch.tensor(coref_key, device="cuda", dtype=torch.long)

            counts_all, counts_coref, density = document_bucket_counts(bucket, coref_query, coref_key)
            accumulators[group].add_document_counts(counts_all, counts_coref, len(pool))
            prepared[group] = (query_index, bucket, coref_query, coref_key, density, len(pool))

        def make_hook(layer_index: int):
            def hook(module, args, kwargs, output):
                attention = output[1]
                for group, payload in prepared.items():
                    query_index, bucket, coref_query, coref_key, density, _count = payload
                    accumulators[group].add_document(
                        layer_index, attention, query_index, bucket, coref_query, coref_key, density
                    )
                # Drop the map so the model does not accumulate 24 of them.
                return (output[0], None)
            return hook

        handles = [
            model.model.layers[index].self_attn.register_forward_hook(make_hook(index), with_kwargs=True)
            for index in range(layers)
        ]
        with torch.no_grad():
            model(token_ids, output_attentions=True)
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        rows.append(
            {
                "stem": document.stem,
                "residues": document.residue_count,
                "tokens": length,
                "statements_start": document.statements_start,
                "contact_statements": document.contact_statements,
                "struct_queries": prepared.get("struct_pos", (None,) * 6)[5],
                "seq_queries": prepared.get("seq_pos", (None,) * 6)[5],
            }
        )
        print(f"[probe] {document.stem}: {length} tokens, {document.contact_statements} contacts")

    pd.DataFrame(rows).to_csv(arguments.out_dir / "phase0_attention_docs.csv", index=False)

    offset_rows = []
    lift_rows = []
    for group, accumulator in accumulators.items():
        if accumulator.queries == 0:
            continue
        for layer in range(layers):
            for head in range(heads):
                offset_rows.append(
                    {
                        "group": group,
                        "layer": layer,
                        "head": head,
                        "mass_self": accumulator.offset_mass[layer, head, 0] / accumulator.queries,
                        "mass_prev1": accumulator.offset_mass[layer, head, 1] / accumulator.queries,
                        "mass_prev2": accumulator.offset_mass[layer, head, 2] / accumulator.queries,
                        "mass_prev3": accumulator.offset_mass[layer, head, 3] / accumulator.queries,
                        "mass_bos": accumulator.bos_mass[layer, head] / accumulator.queries,
                    }
                )
                for bucket in range(NUM_BUCKETS):  # EXCLUDED_BUCKET deliberately dropped
                    if accumulator.count_all[bucket] == 0:
                        continue
                    lift_rows.append(
                        {
                            "group": group,
                            "layer": layer,
                            "head": head,
                            "bucket": BUCKET_LABELS[bucket],
                            "bucket_index": bucket,
                            "mass_total": accumulator.mass_total[layer, head, bucket] / accumulator.queries,
                            "mass_coref": accumulator.mass_coref[layer, head, bucket] / accumulator.queries,
                            "mass_expected": accumulator.mass_expected[layer, head, bucket] / accumulator.queries,
                            "keys_all": int(accumulator.count_all[bucket]),
                            "keys_coref": int(accumulator.count_coref[bucket]),
                        }
                    )

    offsets = pd.DataFrame(offset_rows)
    offsets["mass_local_window"] = offsets["mass_prev1"] + offsets["mass_prev2"]
    offsets.to_csv(arguments.out_dir / "phase0_attention_offsets.csv", index=False)
    lift = pd.DataFrame(lift_rows)
    lift.to_csv(arguments.out_dir / "phase0_attention_lift.csv", index=False)

    structure = offsets[offsets.group == "struct_pos"]
    print("\n[probe] structure-section queries, top previous-token heads:")
    print(
        structure.nlargest(10, "mass_prev1")[["layer", "head", "mass_prev1", "mass_prev2", "mass_self"]]
        .to_string(index=False)
    )
    print(
        f"\n[probe] mean attention mass on offsets 1-2 (the smear window): "
        f"{structure.mass_local_window.mean():.4f}"
    )
    print(f"[probe] heads with >0.30 mass at offset 1: {(structure.mass_prev1 > 0.30).sum()} / {len(structure)}")
    print(f"[probe] heads with >0.30 mass in the 1-2 window: {(structure.mass_local_window > 0.30).sum()} / {len(structure)}")

    pooled = (
        lift[lift.group == "struct_pos"]
        .groupby(["bucket_index", "bucket"], as_index=False)[["mass_total", "mass_coref", "mass_expected"]]
        .sum()
    )
    pooled["lift"] = pooled.mass_coref / pooled.mass_expected.replace(0, np.nan)
    pooled["share_of_attention"] = pooled.mass_total / pooled.mass_total.sum()
    print("\n[probe] structure-section co-referent retrieval lift by distance:")
    print(pooled[["bucket", "share_of_attention", "lift"]].to_string(index=False))


if __name__ == "__main__":
    main()
