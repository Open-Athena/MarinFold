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
    "04_contacts_pl5": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold", "knn", "protenix_ss"],
    "04b_knn": ["marinfold", "knn"],
    "05_folding": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold_helico", "protenix_ss", "no_contacts"],
    "06_sampling": ["single", "consensus", "best100"],
    "06_sampling_pl5": ["single", "consensus", "best100"],
}
TITLES = {
    "02e_oracle_budget": "How many true contacts does Helico need?",
    "02f_oracle_budget_context": "Sparse oracle contacts in predictor context",
    "01_predictors": "Fewer relatives, less accurate structures",
    "01b_af3_sampling": "Does more AlphaFold3 sampling recover accurate folds?",
    "01c_af3_context": "Extra AF3 sampling in predictor context",
    "01d_af3_depth_context": "The low-depth comparison changes at greater depth",
    "02_oracle": "What if we already knew the contact map?",
    "02b_confidence": "Where does the oracle rank among plausible maps?",
    "02c_accuracy_confidence": "Does pTM track TM-score?",
    "03_method": "From one sequence to a contact map",
    "04_contacts": "Contact prediction still tracks MSA depth",
    "04_contacts_pl5": "Precision among the top L/5 contacts",
    "04b_knn": "MarinFold versus sequence-neighbor contact transfer",
    "05_folding": "Predicted contacts help Helico fold natural proteins",
    "06_sampling": "Does a better contact map appear in the samples?",
    "06_sampling_pl5": "Does a better top-L/5 contact set appear in the samples?",
}
CAPTIONS = {
    "02e_oracle_budget": "Random subsets of 5, 10, L/5 and L/2 true contacts on the same 305 natural proteins as Figure 02. Unselected pairs stay unknown. Average two subsets within protein after confidence selection among three diffusion samples per map; bootstrap proteins. Fresh all-positive, full positive/negative and no-contact controls. Ground-truth diagnostic, no MSA. See ORACLE_BUDGET.md and oracle_budget_per_protein.csv for counts and source rows.",
    "02f_oracle_budget_context": "The same oracle-budget sweep alongside Figure 02's archived AF3, AF2, Boltz-2 and Protenix + MSA baselines. Every arm uses the same 305 proteins. The selector menu changes Helico sample selection only. Oracle subsets use ground truth. See ORACLE_BUDGET.md for the complete frozen protocol.",
    "04_contacts_pl5": "P@L/5 on 314 natural FoldBench proteins, using k=max(1,floor(L/5)) and L equal to the frozen input sequence length, not the number of resolved residues. Same nine predictors and same archived predictions as the R-precision panel, including AF2, AF3, Boltz-2 and decontaminated sequence-KNN. Ground truth and eligible pairs are unchanged: pyconfind degree >=0.001, resolved residues, separation >=6 for all-range or >=24 for long-range. Structure contacts rank by degree, MarinFold by votes; stable candidate-order ties. Means and 95% protein-bootstrap intervals. The menu exposes both ranges and split/design views. Preprocessing reproduces every archived R-precision score before using its P@L/5 counterpart.",
    "04b_knn": "Same 314 natural proteins and fixed 248B-token MarinFold checkpoint. Sequence-KNN transfers contacts from the ten nearest sequences in the native decontaminated training corpus; it does not index the additional ProteinMPNN redesign sequences. Points are protein means with 95% protein-bootstrap intervals. The menu exposes all-range and long-range P@L/5 and R-precision. Paired MarinFold-minus-KNN differences and their intervals are in pl5_paired_deltas.csv. This is a contact-prediction baseline, not a structure predictor.",
    "06_sampling_pl5": "P@L/5 counterpart of the sampling diagnostic: the exact same 314 natural proteins and first 100 iid rollouts per protein. L is frozen input sequence length; k=max(1,floor(L/5)). Individual maps retain emission order after eligibility filtering and deduplication; unfilled ranks receive zero credit and invalid/unfinished maps score zero. Consensus ranks votes from all parsed maps, matching the original diagnostic. Oracle best-of-100 is reselected using P@L/5 and uses ground truth. The scatter pairs consensus with this newly selected oracle per protein. Main contact-panel votes instead omit unfinished maps. This measures contact accuracy, not distinct folds.",
    "01c_af3_context": "TM-score for the same five natural test proteins with MSA depth <10. Top block: archived predictor entries, including the 248B-token MarinFold top-L contacts through Helico. Middle: 100 or 1,000 fresh AF3 full runs, one diffusion sample per seed, selected by official ranking_score or pTM. Bottom: ground-truth diagnostics, oracle best AF3 TM and Helico conditioned on the oracle map. ESMFold2 is the archived baseline, not a selected member or average of the 100-contact-map decoy pool. AF3 baseline uses five seeds times five diffusion samples, ten recycles, shared archived MSAs and no templates. Expanded runs retain the same inputs and recycles. Compute budgets differ. Hover identifies the exact source CSV cell.",
    "01d_af3_depth_context": "Mean TM-score on the same 305 natural proteins as the main structural figures: 5, 20, 60 and 220 proteins across the four MSA bins. All archived predictors cover exactly this population; each protein receives equal weight. Hover shows 95% protein-bootstrap intervals and success counts; the prepared CSV retains every contributing row. The 1,000-run AF3 study covers only the five <10 proteins; other cells are not run. Stars mark ground-truth diagnostics. Different bins contain different proteins, so this does not estimate the causal effect of adding MSA sequences. All AF3 results use shared benchmark MSAs and no templates, not the native search pipeline. Budgets and training data differ between predictors.",
    "01b_af3_sampling": "Five natural FoldBench eval-test proteins with MSA depth <10. Every point is one full AF3 run with a fresh seed, one diffusion sample, ten recycles, the same archived MSA and no templates. The menu selects a protein; hover exposes the seed and exact scores. Full-precision pTM selects without ground truth; the oracle best TM is an offline diagnostic. Curves follow ascending-seed prefixes. The sampling panel also shows official ranking_score selection and horizontal archived AF3-25 and ESMFold2 references. Dashed line: the prespecified working accuracy threshold TM >=0.8. Five proteins remain five biological examples.",
    "02c_accuracy_confidence": "TM-score versus Helico pTM for five natural proteins with MSA depth <10. Each protein has 100 ESMFold2 predictions; their extracted maps each receive three Helico samples. Highest pTM selects the Helico sample and ranks maps, without ipTM or a clash penalty. The horizontal coordinate is selected Helico pTM. The menu switches the vertical coordinate between original ESMFold2 TM-score and reconstructed Helico TM-score, and can isolate each protein. Diamonds mark oracle contacts; original oracle TM-score is 1 because its source is the experimental reference. TM-score uses matched protein CA atoms; pTM retains all input tokens. Every map is retained. These are five biological examples, not 500 independent proteins.",
    "02b_confidence": "All five natural FoldBench proteins with MSA depth <10 (test split). Each dot is one of 100 single-sequence ESMFold2 predictions, converted to a full contact/non-contact map and folded by Helico; diamonds use the ground-truth oracle map. Every map has the same eligible-pair mask and three Helico diffusion samples. Highest pTM selects a sample per map; that pTM also ranks the 101 maps. No ipTM term, clash penalty or map filtering is used. Contact counts may vary. Rank 1 is highest pTM; ties are shown as rank intervals. Vertical jitter only separates points. These 500 decoys represent five proteins.",
    "01_predictors": "Natural FoldBench monomers; the same 305 proteins in every structural arm. Points are protein means; bars are 95% protein-bootstrap intervals. MSA depth counts sequences, including the query, in the alignment used by Protenix-v2 + MSA.",
    "02_oracle": "*Oracle = ground-truth contacts AND non-contacts supplied to Helico without an MSA. This is an information upper bound, not a competing predictor or a matched-count true-contact experiment. Same 305 natural proteins as Figure 1.",
    "03_method": "Schematic; arrow widths do not encode amounts. Exp277 step 266,344: a scratch-trained 1.47B-parameter Qwen3, one epoch over 232,090,905 contact documents, totaling 248,583,762,834 raw tokens. Native and ProteinMPNN-redesigned sequences use AFDB / ESM Atlas source structures from the decontaminated corpus. Raw tokens differ from padded training slots.",
    "04_contacts": "Exp277 step 266,344, the 248.584B-raw-token model. All-range R-precision on 314 natural monomers (97 validation + 217 test); 100 resampled rollouts and vote ranking. R is the true-contact count in the resolved candidate universe. KNN indexes the native decontaminated corpus, not the additional redesign sequences. Only five natural proteins have depth <10. The menu also exposes long-range R-precision and separate split/design views.",
    "05_folding": "The 248B-token model through Helico: request the top-L predicted contacts, generate three diffusion samples, select by highest Helico confidence at that fixed cut; no MSA. Helico drops pairs closer than six residues after coordinate mapping; requested and effective counts are saved. Same 305 natural proteins in every arm (95 validation + 210 test). GDT-TS and lDDT are separate metrics. See coverage.csv for exclusions.",
    "06_sampling": "Exp277 step 266,344, 314 natural proteins (97 validation + 217 test), 100 iid rollouts per protein. Consensus and oracle use the same sample pool within each protein. Individual maps are ranked by emission order; short maps retain denominator R, and malformed or unfinished individual maps score zero. The diagnostic consensus uses all parsed maps, following exp321. Oracle selection uses ground truth. This measures contact accuracy, not distinct folds. Scatter colors identify MSA tiers in the interactive view.",
}
METRICS = {"gdt_ts": "GDT-TS", "lddt": "lDDT", "r_precision": "R-precision",
           "r_precision_long": "Long-range R-precision", "p_at_l5": "P@L/5", "p_at_l5_long": "Long-range P@L/5"}
COHORTS = {"natural": "Natural · all splits", "eval-val": "Natural · eval-val",
           "eval-test": "Natural · eval-test", "designed": "Designed",
           "viral": "Natural · viral", "nonviral": "Natural · nonviral"}

MSA_BASELINE_CAPTION = " AlphaFold2/3 and Boltz-2 use the same archived MSA queries and alignments underlying these depth bins, with templates disabled and protein-chain-only inputs. Confidence selects among five AF2 pTM models (three recycles), 25 AF3 samples (five seeds, ten recycles), or 25 Boltz-2 samples (one seed, ten recycles). These are controlled shared-MSA runs, not the models’ full default search pipelines."
for _figure in ("01_predictors", "02_oracle", "04_contacts", "05_folding"):
    CAPTIONS[_figure] += MSA_BASELINE_CAPTION
