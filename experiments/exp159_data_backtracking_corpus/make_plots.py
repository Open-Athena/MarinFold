# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Figures for the exp159 write-up / slide deck.

Reads staged corpus shards (``data/publish/train/*.parquet``) and writes PNGs
plus the sidecar metadata ``build_summary.py`` needs for the plot appendix.

    uv run python make_plots.py --shards 2
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

PLOTS = Path("plots")
DATA = Path("data")
SCRIPT = "make_plots.py"


def load(n_shards: int) -> pd.DataFrame:
    paths = sorted(glob.glob("data/publish/train/*.parquet"))[:n_shards]
    if not paths:
        raise SystemExit("no staged shards — run publish_to_hf.py --stage first")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    print(f"loaded {len(df):,} documents from {len(paths)} shard(s)")
    return df


def plot_retracts_per_doc(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = df.n_retract_stmts.clip(upper=120)
    ax.hist(data, bins=60, color="#4C72B0", edgecolor="none")
    mean = df.n_retract_stmts.mean()
    ax.axvline(mean, color="#C44E52", ls="--", lw=1.5, label=f"mean {mean:.1f}")
    ax.set_xlabel("retract statements per document")
    ax.set_ylabel("documents")
    ax.set_title("Retractions per document")
    ax.legend()
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "retracts_per_doc.png", script=SCRIPT,
        caption=(f"Retractions per document (mean {mean:.1f}). "
                 f"{(df.n_retract_stmts > 0).mean():.0%} of documents contain at "
                 "least one retraction."),
    )


def plot_fp_catch(df: pd.DataFrame) -> None:
    """How many of the model's own false positives the posterior trigger caught."""
    fig, ax = plt.subplots(figsize=(7, 4))
    caught = int(df.fp_retracted_by_trigger.sum())
    flushed = int(df.n_fp_emitted.sum()) - caught
    tp = int(df.tp_retracted_by_trigger.sum())
    bars = ax.bar(
        ["caught by\nposterior trigger", "cleaned at\nfinal flush", "true contacts\n(false alarms)"],
        [caught, flushed, tp],
        color=["#55A868", "#8172B2", "#C44E52"],
    )
    for b, v in zip(bars, [caught, flushed, tp], strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("contacts")
    ax.set_title("Fate of the base model's false positives")
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "fp_catch.png", script=SCRIPT,
        caption=(f"Of {caught + flushed:,} false positives the base model emitted, "
                 f"{caught / (caught + flushed):.1%} were retracted by its own "
                 "collapsing posterior; the rest are removed by the correctness "
                 f"flush. The trigger wrongly retracted {tp} true contacts."),
    )


def plot_retract_distance(df: pd.DataFrame, sample: int = 20000) -> None:
    """Statements between a contact and its retraction — the long-range signal."""
    from marinfold.document_structures.contacts_v1.read import iter_structure_statements

    sub = df if len(df) <= sample else df.sample(sample, random_state=0)
    distances: list[int] = []
    for doc in sub["document"]:
        emitted: dict[tuple[int, int], int] = {}
        for idx, (kind, a, b) in enumerate(iter_structure_statements(doc)):
            pair = (a, b) if a <= b else (b, a)
            if kind == "contact":
                emitted[pair] = idx
            elif pair in emitted:
                distances.append(idx - emitted.pop(pair))
    arr = np.array(distances)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(arr, 0, 100), bins=50, color="#937860", edgecolor="none")
    ax.axvline(arr.mean(), color="#C44E52", ls="--", lw=1.5,
               label=f"mean {arr.mean():.1f}")
    ax.axvline(np.median(arr), color="#4C72B0", ls=":", lw=1.5,
               label=f"median {np.median(arr):.0f}")
    ax.set_xlabel("statements between a contact and its retraction")
    ax.set_ylabel("retractions")
    ax.set_title("Retraction is delayed, not immediate")
    ax.legend()
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "retract_distance.png", script=SCRIPT,
        caption=(f"Emit-to-retract distance (mean {arr.mean():.1f}, median "
                 f"{np.median(arr):.0f}); only {(arr <= 1).mean():.1%} are "
                 "immediate. This spread is the long-range self-correction "
                 "signal the corpus exists to teach."),
    )
    pd.DataFrame({"distance": arr}).to_csv(DATA / "retract_distance.csv", index=False)


def plot_length_relationship(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(30, 401, 20)
    df = df.assign(bin=pd.cut(df.seq_len, bins))
    grouped = df.groupby("bin", observed=True).agg(
        retracts=("n_retract_stmts", "mean"),
        catch=("fp_retracted_by_trigger", "sum"),
        fp=("n_fp_emitted", "sum"),
    )
    centers = [b.mid for b in grouped.index]
    ax.plot(centers, grouped.retracts, "o-", color="#4C72B0", label="retracts / doc")
    ax.set_xlabel("protein length (residues)")
    ax.set_ylabel("mean retracts per document", color="#4C72B0")
    ax2 = ax.twinx()
    ax2.plot(centers, grouped.catch / grouped.fp, "s--", color="#55A868",
             label="fraction of FPs caught")
    ax2.set_ylabel("fraction of false positives caught", color="#55A868")
    ax2.set_ylim(0, 1)
    ax.set_title("Longer proteins: more retractions, better catch rate")
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "length_relationship.png", script=SCRIPT,
        caption=("Retractions per document rise with protein length, and so does "
                 "the fraction of false positives the trigger catches — a bigger "
                 "committed set gives the posterior more to lean on."),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=2)
    args = ap.parse_args()
    PLOTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    df = load(args.shards)
    plot_retracts_per_doc(df)
    plot_fp_catch(df)
    plot_retract_distance(df)
    plot_length_relationship(df)
    print("plots written")


if __name__ == "__main__":
    main()
