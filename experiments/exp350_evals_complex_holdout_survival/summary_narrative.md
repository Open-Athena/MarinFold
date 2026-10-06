# Summary slides — exp: curate 30%-held-out experimental complexes and evaluate contacts plus Helico structures

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## Question

Can FoldBench PPI support a useful experimental-complex benchmark under a 30% identity threshold?

We compare a strict component-held-out claim with a pair-held-out claim, then freeze a FoldBench-only set released after the inherited Protenix v1 training cutoff.

## Two holdout claims

A qualifying alignment has exact identity >=30% over >=50% of the shorter sequence.

Component-held-out rejects a dimer when either chain has any training homolog. Pair-held-out rejects it only when one complex-training document has two chain instances that can be assigned one-to-one to homologs of both candidate chains.

The pair rule tests a new relationship between chain families. It permits prior single-chain exposure.

## Survival

The strict rule leaves 0/184 scoped FoldBench dimers and 1/1,929 PINDER dimers after locally auditable MarinFold arms; the last PINDER target is present in Helico fine-tuning.

The pair rule leaves **35 FoldBench + 183 PINDER = 218** after both MarinFold complex arms. These include 55 homodimers and 163 heterodimers.

The conservative Helico same-PDB pair screen leaves **32 + 65 = 97** candidates.

## Confirmation and evidence

The broad complex search hit its 100,000-result median cap. We repeated all 220 initial survivors at a 1,000,000-result cap; its median was 86,868 and it found two additional PINDER exclusions.

Exact MMseqs backtraces provide integer identity counts. Commands, hashes, compact witnesses and final status tables are committed. Complete compressed alignments and logs are published in the public MarinFold HF bucket.

## Decision

Use FoldBench under the explicit **pair-held-out** claim. Its release-date design protects the Helico comparison from inherited Protenix v1 pretraining exposure. We still remove three targets with a paired homolog in Helico's later fine-tuning pool.

PINDER is not part of the frozen benchmark.

## Frozen FoldBench evaluation

The 35 MarinFold-clean pairs become **30 natural dimers** after the three Helico fine-tuning exclusions and manual removal of two de novo binder targets.

The set contains 13 homodimers and 17 heterodimers in 26 connected 30%-identity groups. One five-target ubiquitin-related group is kept intact. The deterministic split has **8 development and 22 test targets**; both cuts have the same mean length (448.25 vs 448.27 residues).

Every target ships with its exact FoldBench assembly, canonical two-chain input, label-to-author residue map, resolved cross-chain candidate universe, and contacts-v1 pyconfind ground truth.

## Contact-context eligibility

A full-context diagnostic ran 100 rollouts for every target. Seven homodimers had at least one rollout reach the checkpoint's complete 8,192-token context: 147/3,000 rollouts in total. These remain valid structural targets but cannot support the strict exp82 requirement that every rollout finish.

The primary contact and contact-conditioned structural comparison therefore uses **23 context-complete targets** in 19 homology groups. Its metadata-only split has **6 development / 17 test targets**, 6 homodimers and 17 heterodimers, and mean lengths of 361.33 versus 361.29 residues.

## Evaluation contract

Contact evaluation ranks only resolved cross-chain residue pairs. R is the number of degree >=0.001 pyconfind interface contacts; no within-chain sequence-separation filter is applied across chains. Contact-budget choices use the 6-target development cut only; the 17-target test cut is read once for the final result.

The structural bundle uses FoldBench's native layout for Helico/DockQ and retains the exact two scored chains even when a deposited assembly has extra copies. Model sample selection must use model confidence; ground-truth best-of-N remains an oracle diagnostic.

The complete 12.3 MB bundle is public at `hf://buckets/open-athena/MarinFold/data/evals/exp350_foldbench_pair_holdout/v1`.

The context audit and 23-target contact table are public at `hf://buckets/open-athena/MarinFold/data/evals/exp350_foldbench_pair_holdout/contact_eval_v1`.

## Contact result

The exp343 model completed 2,300/2,300 strict rollouts. On the 17-target test split, mean inter-chain R-precision is **0.0260 [0.0036, 0.0453]** versus a random expectation of **0.0028**. The score is about 9.3 times random, though low in absolute terms.

On the six development targets, all-contact top-L gives the best Helico result: mean pair-specific DockQ **0.1284**, versus 0.0381 at L/2 and 0.0329 at L/5. Top-L is frozen before reading structural test results.

## Structural result

On 17 test targets, mean DockQ is **0.0362** with contacts withheld, **0.0546** with predicted intra-chain contacts, **0.0697** with all predicted contacts, and **0.8010** with oracle contacts.

The paired all-contact minus withheld delta is **+0.0335 [0.0024, 0.0679]**. All-contact minus intra-only is **+0.0151 [0.0002, 0.0266]**. Both improve 12/17 targets, isolating a positive contribution from predicted inter-chain contacts.

Absolute predicted-contact performance remains low: 2/17 targets reach DockQ >=0.23, versus 1/17 with contacts withheld and 17/17 with oracle contacts.

## Does sampling 1,000 times recover the interface?

Exploratory follow-up: 1,000 attempted rollouts on each of the same 23 targets. Ten unfinished attempts receive zero credit; the target set remains fixed. Individual saved maps exactly reconstruct every aggregate vote.

On 17 test targets, oracle best@100 F1 is **0.110**, rising to **0.192** at best@1000 (matched random: 0.033 and 0.054). Oracle R-precision rises from **0.093 to 0.165**. These require experimental truth to select the sample.

No target has a sample with F1 >=0.5. The strongest is 8jca: 9 correct contacts out of 29 predicted and 29 true (31% precision and recall).

## Real contacts occur, but in different samples

Pooling 1,000 maps recovers **85.7%** of true contacts on average. A matched random pool already recovers **67.2%**, so pooled coverage alone is weak evidence.

The best-of-many F1 advantage over random is real, but the best selected maps average only 26.6% precision and 18.3% recall. Ordinary 1,000-sample consensus reaches **0.0373 R-precision**.

The contact model sometimes emits correct fragments of an interface; these samples do not establish reliable interface recovery. Learned selection or structural realization from this diversity remains untested.

## Do simple contact biases explain the sampling advantage?

Two stronger controls use the same saved rollouts. One preserves each map's amino-acid pair counts. The other preserves every individual residue's contact count and shuffles partners through valid bipartite edge swaps.

Test best@1000 F1: **5.4%** for uniform random pairs, **7.3%** for the amino-acid control, **13.5%** for the per-residue-degree control, and **19.2%** for MarinFold.

Best@100 F1 is 3.3%, 4.8%, 8.2%, and 11.0%, respectively. These are oracle-selected contact maps, not an inference-time selection method.

## Pairing signal survives both controls

Against the per-residue-degree control, the paired best@1000 F1 advantage is **+5.8 percentage points [3.9, 8.8]**, positive on 16/17 test targets. R-precision is 16.5% versus 11.3%.

The degree control preserves any learned localization of interface residues; it tests the additional pairing information. Its score is not solely an amino-acid-size effect.

The sampler is approximate. Five times more shuffling yields 13.50% F1 versus 13.48%. Small-graph exact enumeration, degree/duplicate checks and source hashes validate the implementation. Absolute quality remains limited: no original sample reaches F1 50%.
