# ESM Atlas — license terms (verified 2026-06-24)

Authoritative findings for exp91 Q5. The bucket itself
(`s3://esm-protein-atlas/`) carries **no `LICENSE` file**, so the terms come from
the publisher's metadata, not the data.

## What each artifact is licensed under

| Artifact | License | Authority |
|----------|---------|-----------|
| **Atlas dataset** — SAE features (6.8B proteins) + 1.1B predicted structures + 7.7M cluster annotations | **CC BY-SA 4.0** | AWS Open Data registry entry [`biohub-esm-atlas`](https://registry.opendata.aws/biohub-esm-atlas/), `License:` field. This is the canonical access point Biohub publishes and the source named in its own "How to Cite" string. |
| ESMFold2 **model + code** | MIT | [`Biohub/esm`](https://github.com/Biohub/esm) `LICENSE.md` (separate repo). Press coverage ("MIT license for commercial and non-commercial use") refers to the *model*, not the dataset. |
| Individual structures via REST API (`/proteins/{hash}` PDB) | **CC BY 4.0** REMARK in the PDB header | Observed in API responses. **Conflicts** with the registry's BY-SA — likely boilerplate carried from the original (CC BY) ESM Atlas. |

Registry citation string (verbatim):
> "ESM Atlas — Protein Features and Structures was accessed on `DATE` from
> https://registry.opendata.aws/biohub-esm-atlas."

Contact for clarification: **support@biohub.org**. Managed by Biohub.

## What this means for republishing a derived subset (Q5)

1. **ShareAlike binds us.** CC BY-SA 4.0 §"Adapted Material": if we redistribute a
   transformed subset (decoded mmCIF + contacts-v1 training docs), it must be under
   **CC BY-SA 4.0** or a CC-compatible license, with attribution. This is stricter
   than `afdb-24M` (CC BY 4.0).
2. **Upstream sources are consistent with BY-SA.** The Atlas derives from UniParc
   (CC BY 4.0), SPIRE, MGnify, IMG, UHGG, UMAG. BY → BY-SA is a permitted one-way
   relicense, so Biohub's BY-SA wrapper is internally valid; our redistribution
   inherits BY-SA regardless of the looser upstream terms.
3. **Resolve the BY-vs-BY-SA conflict in writing before relying on BY.** The
   per-structure CC BY REMARK would, if authoritative, let us publish under the
   looser CC BY 4.0 (matching `afdb-24M`). But the registry — the publisher's own
   stated terms — says BY-SA. Until Biohub confirms otherwise, **treat the dataset
   as CC BY-SA 4.0**.
4. **Model weights are a separate question.** CC licenses govern the data and its
   adaptations; whether they reach across into model weights trained on the data is
   legally unsettled and not something the dataset license cleanly compels. The
   well-defined obligation is on the *republished dataset*, which is what Q5 builds.

## Action items (preconditions to Q5 execution)

- [ ] Email support@biohub.org to confirm the authoritative dataset license and
      reconcile the registry BY-SA vs per-structure BY REMARK.
- [ ] Decide whether CC BY-SA 4.0 is acceptable for the MarinFold-published subset
      (it propagates ShareAlike to anyone who reuses our docs).
- [ ] Include the registry "How to Cite" attribution in the HF dataset card.
