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
MARINFOLD_COLOR = PALETTE[3]
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
    "marinfold": ("MarinFold", MARINFOLD_COLOR, "o"),
    "marinfold_helico": ("Helico + MarinFold top-L", MARINFOLD_COLOR, "o"),
    "marinfold_helico_L2": ("Helico + MarinFold top-L/2", MARINFOLD_COLOR, "P"),
    "marinfold_helico_L5": ("Helico + MarinFold top-L/5", MARINFOLD_COLOR, "X"),
    "knn": ("Sequence KNN", PALETTE[5], "^"),
    "oracle": ("Helico + ground truth map", INK, "D"),
    "oracle_L2": ("Helico + L/2 ground truth contacts", INK, "D"),
    "no_contacts": ("Helico · no contacts", "#817970", "s"),
    "single": ("Mean single rollout", PALETTE[1], "s"),
    "consensus": ("Consensus of 100", MARINFOLD_COLOR, "o"),
    "best100": ("Oracle best of 100", INK, "D"),
}
ORDER = {
    "01_predictors": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "protenix_ss"],
    "02_oracle": ["oracle_L2", "af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "protenix_ss", "no_contacts"],
    "04_contacts": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold", "knn", "protenix_ss"],
    "04_contacts_pl5": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold", "knn", "protenix_ss"],
    "04_contacts_pl2": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold", "knn", "protenix_ss"],
    "04b_knn": ["marinfold", "knn"],
    "05_folding": ["oracle_L2", "af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold_helico_L2", "protenix_ss", "no_contacts"],
    "06_sampling": ["single", "consensus", "best100"],
    "06_sampling_pl5": ["single", "consensus", "best100"],
    "06_sampling_pl2": ["single", "consensus", "best100", "af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "protenix_ss"],
}
TITLES = {
    "01_predictors": "Fewer relatives, less accurate structures",
    "01b_af3_sampling": "Does more AlphaFold3 sampling recover accurate folds?",
    "01c_af3_context": "Extra AF3 sampling in predictor context",
    "01d_af3_depth_context": "The low-depth comparison changes at greater depth",
    "02_oracle": "What if we knew L/2 true contacts?",
    "02e_oracle_budget": "How many true contacts does Helico need at low MSA depth?",
    "02f_oracle_budget_context": "Sparse oracle contacts at low MSA depth: predictor context",
    "02g_seed_completion": "Can MarinFold extend a few known contacts?",
    "02h_seed_contact_precision": "Are the added contacts correct?",
    "02b_confidence": "Where does the oracle rank among plausible maps?",
    "02c_accuracy_confidence": "Does pTM track TM-score?",
    "03_method": "From one sequence to a contact map",
    "04_contacts": "Contact prediction still tracks MSA depth",
    "04_contacts_pl5": "Precision among the top L/5 contacts",
    "04_contacts_pl2": "Precision among the top L/2 contacts",
    "04b_knn": "MarinFold versus sequence-neighbor contact transfer",
    "05_folding": "Predicted contacts help Helico fold natural proteins",
    "06_sampling": "Does a better contact map appear in the samples?",
    "06_sampling_pl5": "Does a better top-L/5 contact set appear in the samples?",
    "06_sampling_pl2": "Sampling contacts in predictor context",
}
CAPTIONS = {
    "06_sampling_pl2": "Precision @ L/2 on the same 314 natural proteins (97 validation + 217 test) for every method. MarinFold uses the fixed 248B-token model and the original 100 iid samples per protein: mean single, consensus and ground-truth oracle best-of-100. Standard predictors use the same selected archived structures, ranking emitted ConFind-positive contacts by pLDDT_i × pLDDT_j × contact degree. Filter resolved endpoints and separation >=6 (all) or >=24 (long) before selecting floor(input L/2); unfilled ranks score zero. Original sampling consensus votes over all parsed maps; main contact-panel votes omit unfinished maps. Bars are equal-protein means with existing 95% protein-bootstrap error bars. No scores, intervals or predictions were recomputed for this comparison. Sources: pl2_sampling_comparison_summary.csv and pl2_sampling_comparison_rows.csv, which retain pointers into the original sampling and poster-product tables. See SAMPLING_L2.md.",
    "02g_seed_completion": "Same five natural eval-test proteins with MSA depth <10. Both arms receive identical random subsets of 0, 5, 10 or floor(L/5) ground-truth contacts. Direct: Helico sees only those pairs. Completion: the fixed 248B-token MarinFold model receives those pairs in its prompt, then 100 resampled rollouts vote on additions; keep every seed and fill to floor(L/2) total contacts. Unselected pairs stay unknown. Helico uses no MSA, six recycles and three diffusion samples/map. Primary selection uses ranking_score, matching the sparse-oracle comparison; menus also expose pTM. Average two subsets within protein (one map at zero seeds). Hollow points show subsets; cohort intervals bootstrap five proteins. Menus expose all proteins, GDT-TS, TM-score and lDDT. This is an exploratory oracle-seeded diagnostic, not sequence-only prediction. See SEED_COMPLETION.md.",
    "02h_seed_contact_precision": "Precision of MarinFold additions alone versus the complete L/2 contact set including guaranteed-correct seeds. Pairs are selected solely by votes from normally terminated rollouts, with valid residue mappings and separation >=6; ties break by sequence indices. Ground truth is consulted for scoring only after selecting additions. Two random subsets are averaged per protein, then proteins receive equal weight. n=0 has one map. Each point traces to seed_completion_contact_precision.csv; exact selected pairs, votes and correctness labels are in seed_completion_pairs.csv.",
    "02e_oracle_budget": "All five natural Figure 02 proteins with MSA depth <10. Random subsets of 5, 10, floor(L/5) and floor(L/2) true contacts; L is frozen input sequence length. Unselected pairs stay unknown. Two independent subsets per budget, nested across budgets within each draw. Three diffusion samples/map, six recycles, seed42, no MSA. Highest ranking_score selects one sample per subset, then average the two subset accuracies within protein. Hollow dots show both subsets; filled markers show their mean. The sixth panel averages the five proteins with 95% protein-bootstrap intervals. Fresh no-contact, all-positive and full positive/negative controls. All oracle contacts use ground truth. Menus expose each protein and pTM selection. See ORACLE_BUDGET.md and oracle_budget_per_protein.csv for counts and source rows.",
    "02f_oracle_budget_context": "The same five low-depth natural proteins in every arm: sparse oracle conditioning versus archived AF3, AF2, Boltz-2, Protenix-v2 with/without MSA, ESMFold/ESMFold2 and Helico with 248B-token MarinFold contacts. Predictor inputs and sampling budgets retain their original protocols. Faint dots show individual proteins; means and 95% intervals bootstrap the five proteins. Oracle subsets use ground truth and are averaged within protein. The selector menu changes only the new Helico oracle-budget sweep; archived predictors and MarinFold folding retain their original selection. See ORACLE_BUDGET.md.",
    "04_contacts_pl5": "P@L/5 on 314 natural FoldBench proteins, using k=max(1,floor(L/5)) and L equal to the frozen input sequence length, not the number of resolved residues. Same nine predictors and same archived predictions as the R-precision panel, including AF2, AF3, Boltz-2 and decontaminated sequence-KNN. Ground truth and eligible pairs are unchanged: pyconfind degree >=0.001, resolved residues, separation >=6 for all-range or >=24 for long-range. Structure contacts rank by degree, MarinFold by votes; stable candidate-order ties. Means and 95% protein-bootstrap intervals. The menu exposes both ranges and split/design views. Preprocessing reproduces every archived R-precision score before using its P@L/5 counterpart.",
    "04_contacts_pl2": "Precision among the top floor(L/2) contacts on the same 314 natural FoldBench proteins as P@L/5 (97 validation + 217 test); L is the frozen input sequence length, including unresolved input residues. Same nine predictors, archived rankings and fixed 248B-token MarinFold model. True contacts have pyconfind degree >=0.001; eligible pairs have resolved residues and sequence separation >=6 (all-range) or >=24 (long-range). Structure predictors rank contact degree, MarinFold ranks consensus votes, and sequence-KNN uses transferred scores; stable candidate-order ties. Every natural-protein score uses the full L/2 denominator. Mean and 95% protein-bootstrap intervals. The menu exposes both ranges, splits, viral status and designs separately; the designed protein 5sbj_A has only 10 long-range candidates versus 15 requested. All original R-precision scores reproduce before selecting archived L/2 scores. See PL2_ANALYSIS.md.",
    "04b_knn": "Same 314 natural proteins and fixed 248B-token MarinFold checkpoint. Sequence-KNN transfers contacts from the ten nearest sequences in the native decontaminated training corpus; it does not index the additional ProteinMPNN redesign sequences. Points are protein means with 95% protein-bootstrap intervals. The menu exposes all-range and long-range P@L/5 and R-precision. Paired MarinFold-minus-KNN differences and their intervals are in pl5_paired_deltas.csv. This is a contact-prediction baseline, not a structure predictor.",
    "06_sampling_pl5": "P@L/5 counterpart of the sampling diagnostic: the exact same 314 natural proteins and first 100 iid rollouts per protein. L is frozen input sequence length; k=max(1,floor(L/5)). Individual maps retain emission order after eligibility filtering and deduplication; unfilled ranks receive zero credit and invalid/unfinished maps score zero. Consensus ranks votes from all parsed maps, matching the original diagnostic. Oracle best-of-100 is reselected using P@L/5 and uses ground truth. The scatter pairs consensus with this newly selected oracle per protein. Main contact-panel votes instead omit unfinished maps. This measures contact accuracy, not distinct folds.",
    "01c_af3_context": "TM-score for the same five natural test proteins with MSA depth <10. Top block: archived predictor entries, including the 248B-token MarinFold top-L contacts through Helico. Middle: 100 or 1,000 fresh AF3 full runs, one diffusion sample per seed, selected by official ranking_score or pTM. Bottom: ground-truth diagnostics, oracle best AF3 TM and Helico conditioned on the oracle map. ESMFold2 is the archived baseline, not a selected member or average of the 100-contact-map decoy pool. AF3 baseline uses five seeds times five diffusion samples, ten recycles, shared archived MSAs and no templates. Expanded runs retain the same inputs and recycles. Compute budgets differ. Hover identifies the exact source CSV cell.",
    "01d_af3_depth_context": "Mean TM-score on the same 305 natural proteins as the main structural figures: 5, 20, 60 and 220 proteins across the four MSA bins. All archived predictors cover exactly this population; each protein receives equal weight. Hover shows 95% protein-bootstrap intervals and success counts; the prepared CSV retains every contributing row. The 1,000-run AF3 study covers only the five <10 proteins; other cells are not run. The best-TM and ground truth map rows are diagnostics. Different bins contain different proteins, so this does not estimate the causal effect of adding MSA sequences. All AF3 results use shared benchmark MSAs and no templates, not the native search pipeline. Budgets and training data differ between predictors.",
    "01b_af3_sampling": "Five natural FoldBench eval-test proteins with MSA depth <10. Every point is one full AF3 run with a fresh seed, one diffusion sample, ten recycles, the same archived MSA and no templates. The menu selects a protein; hover exposes the seed and exact scores. Full-precision pTM selects without ground truth; the oracle best TM is an offline diagnostic. Curves follow ascending-seed prefixes. The sampling panel also shows official ranking_score selection and horizontal archived AF3-25 and ESMFold2 references. Dashed line: the prespecified working accuracy threshold TM >=0.8. Five proteins remain five biological examples.",
    "02c_accuracy_confidence": "TM-score versus Helico pTM for five natural proteins with MSA depth <10. Each protein has 100 ESMFold2 predictions; their extracted maps each receive three Helico samples. Highest pTM selects the Helico sample and ranks maps, without ipTM or a clash penalty. The horizontal coordinate is selected Helico pTM. The menu switches the vertical coordinate between original ESMFold2 TM-score and reconstructed Helico TM-score, and can isolate each protein. Diamonds mark oracle contacts; original oracle TM-score is 1 because its source is the experimental reference. TM-score uses matched protein CA atoms; pTM retains all input tokens. Every map is retained. These are five biological examples, not 500 independent proteins.",
    "02b_confidence": "All five natural FoldBench proteins with MSA depth <10 (test split). Each dot is one of 100 single-sequence ESMFold2 predictions, converted to a full contact/non-contact map and folded by Helico; diamonds use the ground-truth oracle map. Every map has the same eligible-pair mask and three Helico diffusion samples. Highest pTM selects a sample per map; that pTM also ranks the 101 maps. No ipTM term, clash penalty or map filtering is used. Contact counts may vary. Rank 1 is highest pTM; ties are shown as rank intervals. Vertical jitter only separates points. These 500 decoys represent five proteins.",
    "01_predictors": "Natural FoldBench monomers; the same 305 proteins in every structural arm. Points are protein means; bars are 95% protein-bootstrap intervals. MSA depth counts sequences, including the query, in the alignment used by Protenix-v2 + MSA.",
    "02_oracle": "Ground truth contacts: two random subsets of floor(L/2) true contacts per protein; every unselected pair is unknown and no non-contacts are supplied. L is the frozen input sequence length; counts cap at available positives (see ORACLE_L2.md). Three diffusion samples/subset, highest ranking_score selection, then average both subset accuracies within protein. Helico receives no MSA. Same 305 natural proteins as Figure 1, with all seven archived predictors and no-contact Helico; 95% protein-bootstrap intervals. This uses ground truth and is not a deployable predictor. The full-map oracle remains in separate confidence/budget diagnostics.",
    "03_method": "Schematic; arrow widths do not encode amounts. Exp277 step 266,344: a scratch-trained 1.47B-parameter Qwen3, one epoch over 232,090,905 contact documents, totaling 248,583,762,834 raw tokens. Native and ProteinMPNN-redesigned sequences use AFDB / ESM Atlas source structures from the decontaminated corpus. Raw tokens differ from padded training slots.",
    "04_contacts": "Exp277 step 266,344, the 248.584B-raw-token model. All-range R-precision on 314 natural monomers (97 validation + 217 test); 100 resampled rollouts and vote ranking. R is the true-contact count in the resolved candidate universe. KNN indexes the native decontaminated corpus, not the additional redesign sequences. Only five natural proteins have depth <10. The menu also exposes long-range R-precision and separate split/design views.",
    "05_folding": "The 248B-token MarinFold model supplies exactly floor(L/2) predicted contacts to Helico, using archived 100-rollout consensus rankings. L is the frozen input sequence length. Filter mapped separation <6 before selecting contacts; every predicted map reaches its budget. Three diffusion samples/map, highest ranking_score selection; no MSA or oracle seed contacts. The ground truth arm supplies two random subsets of floor(L/2) ground-truth contacts, selects one of three samples per subset, then averages subset accuracies. All other pairs stay unknown in both contact arms. Same 305 natural proteins (95 validation + 210 test), with 95% protein-bootstrap intervals. Oracle caps at 109 versus 110 requested contacts for one protein, and uses two maps rather than one. See MARINFOLD_L2.md, ORACLE_L2.md and coverage.csv.",
    "06_sampling": "Exp277 step 266,344, 314 natural proteins (97 validation + 217 test), 100 iid rollouts per protein. Consensus and oracle use the same sample pool within each protein. Individual maps are ranked by emission order; short maps retain denominator R, and malformed or unfinished individual maps score zero. The diagnostic consensus uses all parsed maps, following exp321. Oracle selection uses ground truth. This measures contact accuracy, not distinct folds. Scatter colors identify MSA tiers in the interactive view.",
}
METRICS = {"gdt_ts": "GDT-TS", "lddt": "lDDT", "r_precision": "R-precision",
           "r_precision_long": "Long-range R-precision", "p_at_l5": "P@L/5", "p_at_l5_long": "Long-range P@L/5",
           "p_at_l2": "Precision @ L/2 contacts", "p_at_l2_long": "Long-range precision @ L/2 contacts"}
COHORTS = {"natural": "Natural · all splits", "eval-val": "Natural · eval-val",
           "eval-test": "Natural · eval-test", "designed": "Designed",
           "viral": "Natural · viral", "nonviral": "Natural · nonviral"}

MSA_BASELINE_CAPTION = " AlphaFold2/3 and Boltz-2 use the same archived MSA queries and alignments underlying these depth bins, with templates disabled and protein-chain-only inputs. Confidence selects among five AF2 pTM models (three recycles), 25 AF3 samples (five seeds, ten recycles), or 25 Boltz-2 samples (one seed, ten recycles). These are controlled shared-MSA runs, not the models’ full default search pipelines."
for _figure in ("01_predictors", "02_oracle", "04_contacts", "05_folding"):
    CAPTIONS[_figure] += MSA_BASELINE_CAPTION

NEFF_AXIS = "Effective MSA depth (Neff, 80% identity)"
NEFF_TIER_LABELS = ["<10", "10–100", "100–1000", "≥1000"]
NEFF_FIGURES = {
    "01_predictors": ["gdt_ts", "lddt"],
    "02_oracle": ["gdt_ts", "lddt"],
    "04_contacts": ["r_precision", "r_precision_long"],
    "04_contacts_pl5": ["p_at_l5", "p_at_l5_long"],
    "04_contacts_pl2": ["p_at_l2", "p_at_l2_long"],
    "04b_knn": ["p_at_l5", "p_at_l5_long", "r_precision", "r_precision_long"],
    "05_folding": ["gdt_ts", "lddt"],
}
_NEFF_PROTOCOLS = {
    "01_predictors": "Same seven archived structure predictors; 305 matched natural proteins.",
    "02_oracle": "Same 305 natural proteins, all seven predictors and no-contact Helico. Oracle: two random L/2 true-contact subsets, three samples/subset, ranking_score selection then subset averaging; no MSA. See ORACLE_L2.md.",
    "04_contacts": "Same nine contact predictors and 314 natural proteins; fixed 248B-token model; R-precision on resolved pairs with separation >=6 or >=24 for long-range. Same contact definition and rankings as the original panel.",
    "04_contacts_pl5": "Same nine contact predictors and 314 natural proteins; precision among the top floor(L/5) contacts, L=frozen input length. See PL5_ANALYSIS.md.",
    "04_contacts_pl2": "Same nine contact predictors and 314 natural proteins; Precision @ L/2 contacts, L=frozen input length. Every natural row uses the full floor(L/2) denominator. See PL2_ANALYSIS.md.",
    "04b_knn": "Same 314 natural proteins; 248B-token MarinFold versus k=10 sequence-KNN over the native decontaminated corpus. P@L/5 and R-precision, all and long-range; unchanged original metrics.",
    "05_folding": "Same 305 natural proteins. MarinFold supplies exact top-L/2 predicted contacts, with no MSA or oracle seeds; three Helico samples, ranking_score selection. Oracle supplies two random L/2 true-contact subsets, selecting within each and averaging; one target has 109 rather than 110 oracle contacts. All seven predictors and no-contact Helico retained. See MARINFOLD_L2.md and ORACLE_L2.md.",
}
for _base in NEFF_FIGURES:
    _name = _base + "_neff"
    ORDER[_name] = ORDER[_base].copy()
    TITLES[_name] = TITLES[_base] if _base != "04_contacts" else "Contact prediction versus effective MSA depth"
    CAPTIONS[_name] = (_NEFF_PROTOCOLS[_base] + " Neff is recomputed on the exact frozen predictor alignments: sum of inverse neighbor counts at >=80% identity over jointly non-gap columns. Repeated sequences are downweighted; no alignment or prediction is changed. Bins are labeled <10, 10–100, 100–1000 and >=1000; finite upper bounds are exclusive. Means and 95% intervals bootstrap proteins; cohort membership changes relative to raw depth. Split, viral and design views remain separate. Every score, Neff measurement and contributing protein traces through neff_figure_rows.csv, neff_summary.csv and NEFF_ANALYSIS.md.")
TITLES["01d_af3_depth_context_neff"] = "Predictor accuracy across effective MSA depths"
CAPTIONS["01d_af3_depth_context_neff"] = "Mean TM-score regrouped by Neff80 on the same 305 natural proteins and frozen alignments. Archived predictor rows retain their original protocols, including top-L MarinFold contacts and the full-map oracle in this diagnostic. Extended AF3 sampling still covers only the original five raw-depth<10 proteins; its Neff<10 cells explicitly show n=5, a subset of the full tier. Other extended-sampling cells are not run. These are not matched tier-wide AF3 sampling comparisons. Every source score and protein is preserved in neff_af3_context_rows.csv and neff_af3_context_summary.csv. See NEFF_ANALYSIS.md and AF3_SAMPLING.md."

CONFIDENCE_FIGURES = ["04c_pl5_plddt_neff", "04d_pl5_plddt_degree_neff"]
for _name, _title, _ranking in zip(CONFIDENCE_FIGURES,
        ["Top L/5: pLDDT × pLDDT", "Top L/5: pLDDT × pLDDT × contact degree"],
        ["q_i × q_j", "q_i × q_j × degree"], strict=True):
    ORDER[_name] = ORDER["04_contacts_pl5"]
    TITLES[_name] = _title
    CAPTIONS[_name] = (f"Same 314 natural proteins and selected structures, with structural contacts ranked by {_ranking}. "
        "q is mean predicted lDDT over a residue's deposited heavy atoms, normalized to [0,1]. "
        "All ConFind degree>0 contacts are considered; eligible residues must be resolved in the frozen truth. "
        "Filter separation >=6 (all) or >=24 (long) before taking floor(input_L/5); unfilled ranks receive zero credit. "
        "Truth remains ConFind degree>=0.001, and never enters ranking. Stable (i,j) ties. "
        "MarinFold and Sequence KNN rankings are unchanged. Neff80 bins and equal-protein bootstrap intervals match "
        "the original comparison. Every selected contact and confidence traces through the contact-confidence CSV tables "
        "and the published pair-level archive. See CONTACT_CONFIDENCE.md.")
TITLES["04e_pl5_ranking_deltas"] = "Does confidence improve contact ranking?"
CAPTIONS["04e_pl5_ranking_deltas"] = ("Paired within-protein P@L/5 changes relative to the original ConFind-degree ranking, "
    "on the same saved structure for each predictor. Positive means better. Separate all-range (>=6) and "
    "long-range (>=24) panels; uncertainty is 95% from 5000 protein bootstraps, seed325. "
    "The low-Neff companion has 15 natural proteins. No truth is used in contact ranking; "
    "this is an exploratory scoring comparison on previously inspected validation and test sets. "
    "Prepared values and member lists: data/contact_confidence_deltas.csv. See CONTACT_CONFIDENCE.md.")
