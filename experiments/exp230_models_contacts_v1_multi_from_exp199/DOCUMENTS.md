# exp230 — what the fine-tuning documents actually look like

Everything below is **measured on the built corpus**, not read off the code.
Where a number is a sample statistic the sample size is given.

The corpus described here is the one at
`/data/exp230_multi/corpus` → `gs://marin-us-central1/protein-structure/MarinFold/exp230_contacts_v1_multi/tokenized`
(698,824 documents from 49,916 proteins, 7 documents per protein per kind). A
second build at higher rollout coverage is in progress and changes **only**
`--docs-per-protein` (7 → 4) and the protein count (49,916 → ~87,000); every
structural fact on this page is identical between the two.

---

## 1. The short answers

| question | answer |
|---|---|
| **Plain contacts-v1 docs mixed in?** | **Yes, exactly 1:1** — 349,412 plain vs 349,400 multi |
| **Same proteins in both halves?** | **Yes, identical sets.** 49,916 proteins, each with 7 plain + 7 multi. 0 proteins appear in only one half |
| **Rollouts per document — fixed or sampled?** | **Sampled.** `K ~ Uniform{0, 1, …, 12}` per document, redrawn independently for every document. Measured mean **5.99**, min 0, max 12, and the histogram is flat (26.4k–27.2k per value) |
| **Contacts per section** | drafts: mean **79.9**, median 62, range 1–250. Final: mean **139.2**, median 108, max 677 |
| **Documents per protein** | **7 multi + 7 plain** (exactly; min = median = max = 7) |
| **What does 0.1 / 1.0 / 1.0 weight?** | header / draft sections / final section of **multi documents only**. Plain documents are uniform **1.0** throughout. See §5 |
| **Are drafts on-policy?** | Yes — sampled from `contacts-v1-exp199-1.5B` itself, 12 rollouts per protein, T=1.0, top-p 0.95, **top-k off**, budget `6L+128` |

---

## 2. The two document types

A **plain** document is an ordinary contacts-v1 training document, byte-for-byte
what exp199 was pretrained on:

```
<contacts-v1> <begin_sequence> …(position, residue) pairs, shuffled…
<begin_statements> <contact> <pI> <pJ> … <end>
```

A **multi-draft** document differs in exactly two ways (#163's format,
unchanged):

```
<contacts-v1.multi> <begin_sequence> …(position, residue) pairs, shuffled…
<begin_statements> …draft 1…          ← NOT closed by <end>
<begin_statements> …draft 2…          ← supersedes draft 1
<begin_statements> …ground truth… <end>
```

1. **`<begin_statements>` may repeat**, and each occurrence means *"discard the
   previous candidate, here is a new one"*. It is the format's own native
   section separator reused, not a new marker. Only the last section is closed
   by `<end>`, so `<end>` keeps its exact existing meaning as the document
   terminator — which is why it remains the generation stop token and no
   inference path had to change.
2. **Token 0 is `<contacts-v1.multi>`** instead of `<contacts-v1>`. This is
   vocab **id 7 renamed in place**: vocab size stays 2,845, every other id is
   unchanged, so there is no embedding resize and no id drift. The id formerly
   spelled `<contacts-and-distances-v1>` — the *other* format's doc-type
   sentinel, which never appears inside a contacts-v1 document and which exp199
   therefore never saw in training, so its embedding row was effectively unused.

**Verified invariant:** a multi document contains exactly `K+1`
`<begin_statements>` and exactly one `<end>` — 100 % of 5,995 sampled multi
documents.

---

## 3. How many drafts, and which ones

`K` is drawn **per document**, not per protein:

```
K ~ Uniform{0, 1, …, 12}
```

Measured over 349,400 multi documents — mean **5.99**, and flat:

| K | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| docs | 27,204 | 26,811 | 26,910 | 26,923 | 26,726 | 26,840 | 26,941 | 27,039 | 27,029 | 26,755 | 26,957 | 26,841 | 26,436 |

The cap is 12 because generation produced **12 rollouts per protein**, so
`K = 12` shows every draft that exists. Which `K` of the 12 are shown is a fresh
random choice per document, and they are presented in **random order** — never
sorted by quality. #163 established why: sorting makes position alone encode
quality, so the model can learn "later = better" without reading the drafts, and
every training context then ends on the best draft so far, which is a state the
model never sees at generation time.

**`K = 0` is deliberate and is 7.8 % of multi documents.** Such a document is
byte-identical to a plain one apart from token 0. It teaches the honest
conditional *"in this mode a single section is also legal"* — the same reasoning
#175 used when it kept the 20.4 % of backtracking documents that contain no
retraction. It is worth knowing this exists when reading Gate B: multi mode is
not trained to *always* emit many sections.

Each shown draft is **subsampled**: `n ~ Uniform{1, …, min(|draft|, 250)}`
contacts are kept. So a draft is a *partial* candidate, which keeps partial
candidates in distribution at inference time.

---

## 4. Section sizes

Measured over 5,995 sampled multi documents (35,912 draft sections):

| section | mean | median | min | max | p95 |
|---|---|---|---|---|---|
| draft | **79.9** | 62 | 1 | 250 | 215 |
| final (ground truth) | **139.2** | 108 | 5 | 677 | — |

Two properties that matter:

- **The final section is always the complete ground truth** — `final == n_gt` in
  **100 %** of documents. Drafts never eat the answer's token budget.
- **`corr(K, |final|) = +0.006`.** The number of drafts shown does not shrink
  the answer, which is the budget invariant #163 required. The builder allocates
  ground-truth tokens *first* and gives drafts only what is left.

A protein's plain and multi documents carry the **same** ground-truth contact
set (`n_gt` identical for 100 % of proteins) — they differ in the drafts and in
the fresh realization, never in the answer.

---

## 5. What the 0.1 / 1.0 / 1.0 weighting applies to

This is the part most worth getting exactly right, because #163 showed it is
what decides whether the model ever emits a second section at all.

### The convention

Levanter computes `target = roll(input_ids, -1)`, so **`weight[i]` supervises the
prediction of `token[i+1]`**. Sections are half-open `[start, stop)`. That
placement is the whole point:

- `weight[start]` → predicts the section's first `<contact>`
- for a **draft**, `weight[stop-1]` → predicts the next `<begin_statements>`,
  i.e. **the decision to restart**
- for the **final** section, `weight[stop-1]` → predicts `<end>`, i.e. **the
  decision to stop**

### Measured on a real K=3 document (3,458 tokens)

| span | token range | weight | supervises predicting |
|---|---|---|---|
| header (doc-type + sequence) | `[0, 870)` | **0.1** | the next sequence token |
| draft 1 | `[870, 1534)` | **1.0** | its contacts; at 1533 → the next `<begin_statements>` |
| draft 2 | `[1534, 1868)` | **1.0** | its contacts; at 1867 → the next `<begin_statements>` |
| draft 3 | `[1868, 2253)` | **1.0** | its contacts; at 2252 → the next `<begin_statements>` |
| final (ground truth) | `[2253, 3457)` | **1.0** | its contacts; at 3456 → `<end>` |
| the `<end>` position itself | `[3457]` | 0.1 | the packing `<eos>` — **zeroed by the packer** |

So concretely:

- **0.1** applies to the sequence header of a multi document — the `<pN> <AA>`
  pairs. It is *not* zero, so the model still gets a little signal on the
  sequence section, but the objective is dominated by the contact sections.
- **1.0 (draft)** applies to every contact token of every draft section,
  **including its last token**, which is the slot that teaches *"emit another
  `<begin_statements>`"*.
- **1.0 (final)** applies to every contact token of the ground-truth section,
  **including its last token**, which is the slot that teaches *"emit `<end>`"*.

### Why the two must be equal

Because the restart decision and the stop decision are supervised by those two
slots, their **ratio** is what the model optimises between. #163 measured this
directly over 600 documents:

| profile (header/draft/final) | w → `<begin_statements>` | w → `<end>` | ratio | sections emitted |
|---|---|---|---|---|
| 0 / 0 / 1 | 0.000 | 1.000 | 0.00 | **1** |
| 0.1 / 0.1 / 1 | 0.100 | 1.000 | 0.10 | **1** |
| 0.1 / 0.3 / 1 | 0.300 | 1.000 | 0.30 | **1** |
| **F — 0.1 / 1.0 / 1.0** | 1.000 | 1.000 | **1.00** | **~15** |
| E — 0.1 / 1.0 / 2.0 | 1.000 | 2.000 | 0.50 | ~15 |

With `w_draft = 0` the "continue" transition receives *exactly zero gradient* —
the model was never taught to do it. At 0.1 and 0.3 it is supervised 10× and
3.3× more weakly than stopping, which is not enough. **F is the simplest profile
where continuing competes with stopping**, and E's extra weight on the final
bought nothing, so F is what this run uses. Confirmed on our own corpus: restart
weight 1.0, stop weight 1.0, ratio **1.00**.

### Plain documents are uniform 1.0

The rehearsal half is weighted **1.0 on every token, header included** — the
complete pretraining objective, unchanged. It is not run through the profile at
all. That matters because ~47 % of contacts-v1 tokens are the sequence section;
down-weighting it to 0.1 in the rehearsal half would quietly change the base
task the run is trying to *preserve*.

### Two packing details

- **The `<eos>` slot is explicitly zeroed.** Documents are packed several to an
  8,192-token row and terminated `… <end> <eos>`. The weight *on* a document's
  `<eos>` would supervise the first token of the **next** document in the row —
  cross-document leakage that is invisible whenever `w_header` is 0, and
  `w_header` is 0.1 here.
- **Padding is zeroed** and cross-document attention is blocked (segment ids are
  derived from those same `<eos>` tokens).

Corpus-level: **89.3 %** of total token-weight is armed, over 1.10 B tokens in
163,838 packed sequences at 82.2 % packing density.

### No microbatching, deliberately

Levanter re-normalises per-token loss weights **per microbatch**. With drafts and
finals carrying different weights that silently changes the effective objective,
so gradient accumulation is not a free memory lever for this run.

---

## 6. Where the drafts come from

Drafts are **exp199's own rollouts** — the model being fine-tuned — not #163's
E8 rollouts and not #98's archive. That is what "on-policy" means here: a draft
is something the policy would actually write.

| | |
|---|---|
| sampler | `contacts-v1-exp199-1.5B` (bf16, rope-repaired) |
| rollouts per protein | 12 |
| recipe | T = 1.0, top-p = 0.95, **top-k disabled**, budget `6L+128` |
| why top-k off | 50 is the HF default that rides in from an export's `config.json`; it inflates `<end>` and costs ~0.011 R-precision (#142) |

Draft quality is genuinely intermediate — mean draft F1 against ground truth is
**~0.27** at corpus level. That is the discrimination task: the model must learn
that a `<begin_statements>` section it has just been shown is a *hypothesis*,
while the one it is being trained to emit last is the *answer*.

---

## 7. Ten uniformly-sampled documents

Sampled uniformly at random over all 698,824 documents (seed 230). Contact lists
are elided after the first four triples; everything else is verbatim.

### Example 1 — `multi` · arm `afdb` · corpus index 5,772

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `afdb:AF-A0A086CHE0-F1:m4` | 84 | 4 | 24 | 663 | [34, 43, 10, 50, 24] | 0.105 |

```
  <contacts-v1.multi> <begin_sequence> <p1225> <VAL> <n-term> <p1142> <p1146> <ASN> <p1151> <PHE> <p1172> <ILE> <p1223> <LEU> ... [84 residues total, shuffled] ... <LEU> <p1177> <ILE>
  <begin_statements>   # draft 1: 34 contacts
    <contact> <p1218> <p1184> <contact> <p1200> <p1207> <contact> <p1217> <p1181> <contact> <p1209> <p1178> ...
  <begin_statements>   # draft 2: 43 contacts
    <contact> <p1143> <p1162> <contact> <p1193> <p1185> <contact> <p1169> <p1200> <contact> <p1163> <p1143> ...
  <begin_statements>   # draft 3: 10 contacts
    <contact> <p1197> <p1179> <contact> <p1183> <p1197> <contact> <p1186> <p1192> <contact> <p1200> <p1179> ...
  <begin_statements>   # draft 4: 50 contacts
    <contact> <p1166> <p1196> <contact> <p1193> <p1211> <contact> <p1199> <p1170> <contact> <p1178> <p1210> ...
  <begin_statements>   # FINAL (ground truth): 24 contacts
    <contact> <p1177> <p1203> <contact> <p1173> <p1203> <contact> <p1185> <p1193> <contact> <p1216> <p1182> ... <end>
```

### Example 2 — `multi` · arm `afdb` · corpus index 66,616

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `afdb:AF-A0A4V0HUC7-F1:m4` | 482 | 11 | 501 | 7211 | [155, 14, 83, 223, 215, 165, 38, 228, 40, 243, 171, 501] | 0.301 |

```
  <contacts-v1.multi> <begin_sequence> <p97> <PHE> <p83> <GLY> <p1992> <THR> <p312> <ASN> <p412> <ARG> <p415> <LEU> ... [482 residues total, shuffled] ... <ILE> <p145> <LEU>
  <begin_statements>   # draft 1: 155 contacts
    <contact> <p223> <p250> <contact> <p303> <p251> <contact> <p10> <p343> <contact> <p1989> <p1947> ...
  <begin_statements>   # draft 2: 14 contacts
    <contact> <p405> <p381> <contact> <p1990> <p164> <contact> <p293> <p354> <contact> <p65> <p81> ...
  <begin_statements>   # draft 3: 83 contacts
    <contact> <p296> <p304> <contact> <p117> <p278> <contact> <p55> <p48> <contact> <p271> <p243> ...
  <begin_statements>   # draft 4: 223 contacts
    <contact> <p146> <p119> <contact> <p347> <p13> <contact> <p254> <p0> <contact> <p237> <p264> ...
  <begin_statements>   # draft 5: 215 contacts
    <contact> <p196> <p224> <contact> <p251> <p225> <contact> <p1984> <p30> <contact> <p1974> <p54> ...
  <begin_statements>   # draft 6: 165 contacts
    <contact> <p57> <p74> <contact> <p85> <p40> <contact> <p312> <p282> <contact> <p98> <p127> ...
  <begin_statements>   # draft 7: 38 contacts
    <contact> <p418> <p399> <contact> <p170> <p143> <contact> <p289> <p317> <contact> <p370> <p344> ...
  <begin_statements>   # draft 8: 228 contacts
    <contact> <p200> <p225> <contact> <p281> <p335> <contact> <p240> <p1999> <contact> <p364> <p424> ...
  <begin_statements>   # draft 9: 40 contacts
    <contact> <p77> <p114> <contact> <p314> <p292> <contact> <p78> <p30> <contact> <p29> <p56> ...
  <begin_statements>   # draft 10: 243 contacts
    <contact> <p143> <p114> <contact> <p264> <p170> <contact> <p92> <p119> <contact> <p10> <p32> ...
  <begin_statements>   # draft 11: 171 contacts
    <contact> <p84> <p0> <contact> <p224> <p1963> <contact> <p133> <p107> <contact> <p106> <p126> ...
  <begin_statements>   # FINAL (ground truth): 501 contacts
    <contact> <p364> <p403> <contact> <p363> <p380> <contact> <p1987> <p34> <contact> <p38> <p62> ... <end>
```

### Example 3 — `multi` · arm `afdb` · corpus index 99,541

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `afdb:AF-A0A7S0FJ28-F1:m1` | 403 | 2 | 305 | 2046 | [33, 72, 305] | 0.127 |

```
  <contacts-v1.multi> <begin_sequence> <p374> <VAL> <p180> <THR> <p477> <HIS> <p148> <LEU> <p442> <VAL> <p284> <THR> ... [403 residues total, shuffled] ... <VAL> <p508> <ASN>
  <begin_statements>   # draft 1: 33 contacts
    <contact> <p341> <p372> <contact> <p367> <p379> <contact> <p457> <p228> <contact> <p368> <p342> ...
  <begin_statements>   # draft 2: 72 contacts
    <contact> <p338> <p143> <contact> <p210> <p251> <contact> <p273> <p331> <contact> <p472> <p449> ...
  <begin_statements>   # FINAL (ground truth): 305 contacts
    <contact> <p318> <p217> <contact> <p287> <p264> <contact> <p355> <p437> <contact> <p382> <p345> ... <end>
```

### Example 4 — `multi` · arm `esm_atlas` · corpus index 280,310

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `esm_atlas:e3b35e3702150e72c400245bf5537d2f:m2` | 185 | 12 | 169 | 4131 | [37, 80, 81, 82, 17, 97, 22, 29, 175, 123, 124, 211, 169] | 0.142 |

```
  <contacts-v1.multi> <begin_sequence> <p1132> <ILE> <p1224> <GLN> <p1246> <ALA> <p1267> <VAL> <p1245> <ASN> <p1279> <ARG> ... [185 residues total, shuffled] ... <MET> <p1194> <VAL>
  <begin_statements>   # draft 1: 37 contacts
    <contact> <p1172> <p1287> <contact> <p1243> <p1298> <contact> <p1250> <p1154> <contact> <p1206> <p1262> ...
  <begin_statements>   # draft 2: 80 contacts
    <contact> <p1132> <p1253> <contact> <p1161> <p1145> <contact> <p1274> <p1230> <contact> <p1194> <p1174> ...
  <begin_statements>   # draft 3: 81 contacts
    <contact> <p1241> <p1208> <contact> <p1306> <p1205> <contact> <p1268> <p1298> <contact> <p1265> <p1211> ...
  <begin_statements>   # draft 4: 82 contacts
    <contact> <p1238> <p1203> <contact> <p1129> <p1203> <contact> <p1209> <p1201> <contact> <p1211> <p1134> ...
  <begin_statements>   # draft 5: 17 contacts
    <contact> <p1168> <p1223> <contact> <p1216> <p1161> <contact> <p1306> <p1159> <contact> <p1169> <p1128> ...
  <begin_statements>   # draft 6: 97 contacts
    <contact> <p1210> <p1200> <contact> <p1307> <p1183> <contact> <p1256> <p1230> <contact> <p1214> <p1264> ...
  <begin_statements>   # draft 7: 22 contacts
    <contact> <p1148> <p1139> <contact> <p1263> <p1230> <contact> <p1165> <p1184> <contact> <p1140> <p1245> ...
  <begin_statements>   # draft 8: 29 contacts
    <contact> <p1179> <p1192> <contact> <p1229> <p1259> <contact> <p1206> <p1291> <contact> <p1245> <p1295> ...
  <begin_statements>   # draft 9: 175 contacts
    <contact> <p1204> <p1308> <contact> <p1210> <p1204> <contact> <p1267> <p1243> <contact> <p1185> <p1179> ...
  <begin_statements>   # draft 10: 123 contacts
    <contact> <p1171> <p1136> <contact> <p1231> <p1280> <contact> <p1243> <p1308> <contact> <p1275> <p1269> ...
  <begin_statements>   # draft 11: 124 contacts
    <contact> <p1186> <p1140> <contact> <p1206> <p1228> <contact> <p1273> <p1138> <contact> <p1155> <p1192> ...
  <begin_statements>   # draft 12: 211 contacts
    <contact> <p1145> <p1214> <contact> <p1220> <p1229> <contact> <p1219> <p1213> <contact> <p1187> <p1156> ...
  <begin_statements>   # FINAL (ground truth): 169 contacts
    <contact> <p1179> <p1161> <contact> <p1205> <p1171> <contact> <p1241> <p1299> <contact> <p1277> <p1300> ... <end>
```

### Example 5 — `multi` · arm `esm_atlas` · corpus index 292,183

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `esm_atlas:5ed3917b30fe800d7b5f7d4db5cd1efa:m3` | 149 | 4 | 161 | 2470 | [103, 134, 145, 177, 161] | 0.493 |

```
  <contacts-v1.multi> <begin_sequence> <p595> <GLN> <p579> <LYS> <p664> <SER> <p563> <LYS> <p580> <VAL> <p690> <ASN> ... [149 residues total, shuffled] ... <ILE> <p575> <ASP>
  <begin_statements>   # draft 1: 103 contacts
    <contact> <p678> <p685> <contact> <p603> <p627> <contact> <p676> <p668> <contact> <p676> <p553> ...
  <begin_statements>   # draft 2: 134 contacts
    <contact> <p653> <p593> <contact> <p641> <p651> <contact> <p582> <p612> <contact> <p622> <p664> ...
  <begin_statements>   # draft 3: 145 contacts
    <contact> <p626> <p675> <contact> <p631> <p625> <contact> <p561> <p567> <contact> <p697> <p691> ...
  <begin_statements>   # draft 4: 177 contacts
    <contact> <p687> <p695> <contact> <p552> <p616> <contact> <p651> <p638> <contact> <p651> <p634> ...
  <begin_statements>   # FINAL (ground truth): 161 contacts
    <contact> <p646> <p604> <contact> <p691> <p566> <contact> <p681> <p557> <contact> <p587> <p566> ... <end>
```

### Example 6 — `plain` · arm `esm_atlas` · corpus index 360,620

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `esm_atlas:f1c3f2aa7710a36b53c580e5c505b638:p1` | 227 | 0 | 219 | 1119 | [219] | n/a |

```
  <contacts-v1> <begin_sequence> <p852> <MET> <p1023> <GLN> <p1022> <ARG> <p1003> <ARG> <p968> <PHE> <p905> <VAL> ... [227 residues total, shuffled] ... <PRO> <p859> <LEU>
  <begin_statements>   # FINAL (ground truth): 219 contacts
    <contact> <p918> <p968> <contact> <p879> <p893> <contact> <p959> <p915> <contact> <p890> <p883> ... <end>
```

### Example 7 — `plain` · arm `esm_atlas` · corpus index 416,158

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `esm_atlas:d4beb465845e1f4e2e983bdcd82e6948:p1` | 327 | 0 | 253 | 1421 | [253] | n/a |

```
  <contacts-v1> <begin_sequence> <p786> <ASN> <p664> <VAL> <p901> <LYS> <p900> <GLY> <p800> <ARG> <p718> <GLY> ... [327 residues total, shuffled] ... <ALA> <p657> <MET>
  <begin_statements>   # FINAL (ground truth): 253 contacts
    <contact> <p838> <p909> <contact> <p878> <p896> <contact> <p914> <p817> <contact> <p819> <p912> ... <end>
```

### Example 8 — `plain` · arm `pdb` · corpus index 615,447

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `pdb:6f03_B:p0` | 163 | 0 | 37 | 445 | [37] | n/a |

```
  <contacts-v1> <begin_sequence> <p411> <THR> <p384> <ALA> <p389> <LEU> <p350> <ALA> <p381> <GLN> <p342> <ILE> ... [163 residues total, shuffled] ... <GLY> <p423> <ALA>
  <begin_statements>   # FINAL (ground truth): 37 contacts
    <contact> <p447> <p342> <contact> <p355> <p405> <contact> <p338> <p434> <contact> <p446> <p437> ... <end>
```

### Example 9 — `plain` · arm `pdb` · corpus index 638,660

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `pdb:3ix7_A:p1` | 131 | 0 | 92 | 546 | [92] | n/a |

```
  <contacts-v1> <begin_sequence> <p210> <ALA> <p271> <LYS> <p295> <HIS> <p194> <ARG> <p275> <VAL> <p198> <VAL> ... [131 residues total, shuffled] ... <LEU> <p291> <VAL>
  <begin_statements>   # FINAL (ground truth): 92 contacts
    <contact> <p249> <p205> <contact> <p310> <p298> <contact> <p218> <p197> <contact> <p248> <p236> ... <end>
```

### Example 10 — `multi` · arm `pdb` · corpus index 693,956

| doc_id | L | K (drafts) | ground-truth contacts | tokens | section sizes | mean draft F1 |
|---|---|---|---|---|---|---|
| `pdb:5wxu_B:m4` | 400 | 0 | 430 | 2098 | [430] | n/a |

```
  <contacts-v1.multi> <begin_sequence> <p1696> <PHE> <p1805> <GLY> <p1897> <GLN> <p1840> <THR> <p1854> <LEU> <p1643> <ASP> ... [400 residues total, shuffled] ... <GLN> <p1866> <ARG>
  <begin_statements>   # FINAL (ground truth): 430 contacts
    <contact> <p1623> <p1800> <contact> <p1613> <p1915> <contact> <p1980> <p1957> <contact> <p1965> <p1979> ... <end>
```
---

## 8. Reproducing these numbers

```bash
# the corpus itself
python build_corpus.py --targets /data/exp230_multi/targets.parquet \
    --rollouts gs://marin-us-east5/.../rollouts --out /data/exp230_multi/corpus \
    --kmax 12 --docs-per-protein 7
# the weights, and the printed STEPS_PER_EPOCH
python tokenize_corpus.py --in /data/exp230_multi/corpus \
    --out /data/exp230_multi/tokenized --tokenizer /data/exp230_multi/tokenizer_multi
```

The structural invariants on this page are pinned as tests in
[`test_corpus.py`](test_corpus.py): that regenerating a rehearsal document from
parsed ground truth is lossless, that a multi document has `K+1` sections and one
`<end>`, that a plain document has exactly one section, and that under profile F
the restart slot and the stop slot carry **equal** weight.
