"""Open Athena palette, with stable identities across the writeup figures.

Palette and Lato plot font inspected at open-athena.github.io commit
0618887cbd74bc89d0e4575716d262cb1c6c179d, scripts/oa_theme.py and
scripts/datakit/export_figures.py. Serif type is reserved for page headings.
"""

PAPER = "#F1E8DF"
INK = "#1F1E1B"
GRID = "#D2C8BC"
PALETTE = ["#385C8F", "#8F6B38", "#388F8D", "#8F386D", "#7E8F38",
           "#47388F", "#8F3839", "#388F59", "#7C388F", "#4A8F38"]
FONT = "Lato, DejaVu Sans, Arial, sans-serif"
TIERS = ["<10", "10–99", "100–999", "≥1000"]
METHODS = {
    "boltz2": ("Boltz-2 + MSA", PALETTE[8], "s"),
    "af2": ("AlphaFold2 + MSA", PALETTE[7], "^"),
    "af3": ("AlphaFold3 + MSA", PALETTE[6], "D"),
    "protenix_msa": ("Protenix-v2 + MSA", PALETTE[0], "o"),
    "protenix_ss": ("Protenix-v2 · single sequence", PALETTE[1], "s"),
    "esmfold2": ("ESMFold2", PALETTE[2], "D"),
    "esmfold": ("ESMFold", PALETTE[4], "v"),
    "marinfold": ("MarinFold", PALETTE[3], "o"),
    "marinfold_helico": ("Helico + MarinFold top-L", PALETTE[3], "o"),
    "knn": ("Sequence KNN · decontaminated", PALETTE[5], "^"),
    "oracle": ("Helico + oracle map*", INK, "D"),
    "no_contacts": ("Helico · no contacts", "#817970", "s"),
    "single": ("Mean single rollout", PALETTE[1], "s"),
    "consensus": ("Consensus of 100", PALETTE[3], "o"),
    "best100": ("Oracle best of 100*", INK, "D"),
}
ORDER = {
    "01_predictors": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "protenix_ss"],
    "02_oracle": ["oracle", "af3", "af2", "boltz2", "protenix_msa", "no_contacts"],
    "04_contacts": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold", "knn", "protenix_ss"],
    "05_folding": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold_helico", "protenix_ss", "no_contacts"],
    "06_sampling": ["single", "consensus", "best100"],
}
TITLES = {
    "01_predictors": "Fewer relatives, less accurate structures",
    "02_oracle": "What if we already knew the contact map?",
    "02b_confidence": "Can confidence recognize an oracle contact map?",
    "03_method": "From one sequence to a contact map",
    "04_contacts": "Contact prediction still tracks MSA depth",
    "05_folding": "Predicted contacts help Helico fold natural proteins",
    "06_sampling": "Does a better contact map appear in the samples?",
}
CAPTIONS = {
    "02b_confidence": "Twenty natural proteins, selected before inference (five per MSA tier). Each oracle is compared with five uniform and five separation-matched random maps; unknown mask, positive/negative counts, and three-sample diffusion budgets are identical. Select each map’s structure by Helico ranking_score. Confidence wins are paired within protein; ties count half. Intervals bootstrap proteins, not samples. Random maps are weak negatives; this does not establish ranking of plausible alternative folds.",
    "01_predictors": "Natural FoldBench monomers; the same 305 proteins in every structural arm. Points are protein means; bars are 95% protein-bootstrap intervals. MSA depth counts sequences, including the query, in the alignment used by Protenix-v2 + MSA.",
    "02_oracle": "*Oracle = ground-truth contacts AND non-contacts supplied to Helico without an MSA. This is an information upper bound, not a competing predictor or a matched-count true-contact experiment. Same 305 natural proteins as Figure 1.",
    "03_method": "Schematic; arrow widths do not encode amounts. Exp277 step 266,344: a scratch-trained 1.47B-parameter Qwen3, one epoch over 232,090,905 contact documents, totaling 248,583,762,834 raw tokens. Native and ProteinMPNN-redesigned sequences use AFDB / ESM Atlas source structures from the decontaminated corpus. Raw tokens differ from padded training slots.",
    "04_contacts": "Exp277 step 266,344, the 248.584B-raw-token model. All-range R-precision on 314 natural monomers (97 validation + 217 test); 100 resampled rollouts and vote ranking. R is the true-contact count in the resolved candidate universe. KNN indexes the native decontaminated corpus, not the additional redesign sequences. Only five natural proteins have depth <10. The menu also exposes long-range R-precision and separate split/design views.",
    "05_folding": "The 248B-token model through Helico: request the top-L predicted contacts, generate three diffusion samples, select by highest Helico confidence at that fixed cut; no MSA. Helico drops pairs closer than six residues after coordinate mapping; requested and effective counts are saved. Same 305 natural proteins in every arm (95 validation + 210 test). GDT-TS and lDDT are separate metrics. See coverage.csv for exclusions.",
    "06_sampling": "Exp277 step 266,344, 314 natural proteins (97 validation + 217 test), 100 iid rollouts per protein. Consensus and oracle use the same sample pool within each protein. Individual maps are ranked by emission order; short maps retain denominator R, and malformed or unfinished individual maps score zero. The diagnostic consensus uses all parsed maps, following exp321. Oracle selection uses ground truth. This measures contact accuracy, not distinct folds. Scatter colors identify MSA tiers in the interactive view.",
}
METRICS = {"gdt_ts": "GDT-TS", "lddt": "lDDT", "r_precision": "R-precision",
           "r_precision_long": "Long-range R-precision"}
COHORTS = {"natural": "Natural · all splits", "eval-val": "Natural · eval-val",
           "eval-test": "Natural · eval-test", "designed": "Designed",
           "viral": "Natural · viral", "nonviral": "Natural · nonviral"}

MSA_BASELINE_CAPTION = " AlphaFold2/3 and Boltz-2 use the same archived MSA queries and alignments underlying these depth bins, with templates disabled and protein-chain-only inputs. Confidence selects among five AF2 pTM models (three recycles), 25 AF3 samples (five seeds, ten recycles), or 25 Boltz-2 samples (one seed, ten recycles). These are controlled shared-MSA runs, not the models’ full default search pipelines."
for _figure in ("01_predictors", "02_oracle", "04_contacts", "05_folding"):
    CAPTIONS[_figure] += MSA_BASELINE_CAPTION
