
import pandas as pd
import score_baselines as sb, upstream as U

protenix = sb.score(U.DATA / "predictor_manifest_new.csv",
                    ("protenix-v2_single_seq", "protenix-v2_msa"),
                    sb.SCORES / "protenix")
esm = pd.read_csv(U.DATA / "baseline_precision_esm.csv.gz")
combined = pd.concat([esm, protenix], ignore_index=True)
combined.to_csv(U.DATA / "baseline_precision_new.csv.gz", index=False)
print("rows", len(combined), "units", combined.stem.nunique(),
      combined.groupby("model").stem.nunique().to_dict())
