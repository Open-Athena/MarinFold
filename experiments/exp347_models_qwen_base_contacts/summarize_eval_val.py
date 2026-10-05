"""Summarize complete pilot evaluations by the frozen eval-val viral labels."""

import csv
from pathlib import Path


def main() -> None:
    here = Path(__file__).parent
    source = here.parent / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv"
    with source.open() as handle:
        membership = {
            row["stem"]: row
            for row in csv.DictReader(handle)
            if row["eval_set"] == "eval-val"
        }
    output = []
    for document_format in ["contacts_v1", "prompted"]:
        with (
            here / "data/eval_val_pilot" / document_format / "metrics.csv"
        ).open() as handle:
            rows = list(csv.DictReader(handle))
        if {row["stem"] for row in rows} != set(membership):
            raise ValueError("Incomplete eval-val membership")
        for split in ["all", "viral", "nonviral"]:
            for contact_range in ["all", "long"]:
                values = [
                    float(row["precision"])
                    for row in rows
                    if row["range"] == contact_range
                    and row["cut"] == "R"
                    and (
                        split == "all"
                        or (membership[row["stem"]]["is_viral"] == "1")
                        == (split == "viral")
                    )
                ]
                output.append(
                    {
                        "format": document_format,
                        "split": split,
                        "range": contact_range,
                        "n": len(values),
                        "r_precision": sum(values) / len(values),
                    }
                )
    with (here / "data/eval_val_pilot/strata.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
