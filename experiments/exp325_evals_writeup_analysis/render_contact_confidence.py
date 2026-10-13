"""Render cached, paired effects of confidence-based contact rankings."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from build_summary import save_plot_with_meta
from poster_style import save_poster_vectors
from theme import CAPTIONS, COHORTS, FONT, GRID, INK, METHODS, ORDER, PAPER, TITLES

HERE = Path(__file__).resolve().parent
FIGURE = "04e_pl5_ranking_deltas"
RANKINGS = [("plddt_product", "pLDDT × pLDDT", "#8F386D", "o"),
            ("plddt_product_degree", "pLDDT × pLDDT × degree", "#385C8F", "D")]


def static(frame: pd.DataFrame, tier: str, *, poster_dir: Path | None) -> None:
    """Draw matched per-protein changes for one prespecified population."""
    data = frame[(frame.axis == "neff") & (frame.cohort == "natural") & (frame.tier == tier)]
    methods = ORDER["01_predictors"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.3), sharey=True, facecolor=PAPER)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.76, bottom=0.24, wspace=0.15)
    fig.text(0.08, 0.94, TITLES[FIGURE], fontsize=21, weight="bold", color=INK)
    n = int(data.n.iloc[0])
    population = "All natural FoldBench proteins" if tier == "All depths" else "Natural proteins with Neff <10"
    fig.text(0.08, 0.865, f"{population} · n={n} · paired mean and 95% interval", fontsize=12, color=INK)
    low = min(-0.02, float(data.ci_low.min())-0.01)
    high = max(0.02, float(data.ci_high.max())+0.01)
    for ax, region, label in zip(axes, ["all","long"], ["All-range · separation ≥6", "Long-range · separation ≥24"], strict=True):
        ax.set_facecolor(PAPER)
        ax.axvline(0, color=INK, linewidth=1, zorder=1)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(length=0, colors=INK)
        ax.set_xlim(low,high)
        ax.set_title(label, fontsize=12, pad=15)
        ax.set_xlabel("Change in precision @ L/5", labelpad=12)
        for index, (ranking, name, color, marker) in enumerate(RANKINGS):
            group = data[(data["range"] == region) & (data.ranking == ranking)].set_index("method").reindex(methods)
            y = np.arange(len(methods)) + (index-0.5)*0.25
            ax.errorbar(group["mean"],y,xerr=[group["mean"]-group.ci_low,group.ci_high-group["mean"]],
                        fmt=marker,color=color,markersize=6,capsize=3,label=name,zorder=3)
        ax.set_yticks(range(len(methods)), [METHODS[m][0] for m in methods],fontsize=10)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(0.55,0.08),ncol=2,frameon=False,fontsize=11)
    fig.text(0.08,0.025,"Same saved structures and contact definition; only the contact ranking changes.",fontsize=10,color=INK)
    name = FIGURE + ("_low_neff" if tier == "<10" else "")
    if poster_dir is not None:
        save_poster_vectors(fig,poster_dir,name)
    else:
        save_plot_with_meta(fig,HERE/"plots"/f"{name}.png",caption=CAPTIONS[FIGURE],script="render.py",args=[],dpi=180)
        fig.savefig(HERE/"plots"/f"{name}.pdf",bbox_inches="tight")
        fig.savefig(HERE/"plots"/f"{name}.svg",bbox_inches="tight")
    plt.close(fig)


def render(*, poster_dir: Path | None = None) -> go.Figure:
    """Return the interactive matched-delta chart and save its static companions."""
    frame = pd.read_csv(HERE/"data/contact_confidence_deltas.csv")
    for tier in ["All depths","<10"]:
        static(frame,tier,poster_dir=poster_dir)
    fig = make_subplots(rows=1,cols=2,subplot_titles=["All-range · separation ≥6","Long-range · separation ≥24"],horizontal_spacing=0.10)
    methods = ORDER["01_predictors"]
    menus = []
    for cohort in COHORTS:
        for tier in ["All depths","<10","10–99","100–999","≥1000"]:
            data = frame[(frame.axis=="neff") & (frame.cohort==cohort) & (frame.tier==tier)]
            if data.empty:
                continue
            start = len(fig.data)
            for column, region in enumerate(["all","long"],1):
                for ranking,label,color,marker in RANKINGS:
                    group = data[(data["range"]==region)&(data.ranking==ranking)].set_index("method").reindex(methods)
                    custom = group[["n","original_mean","new_mean","ci_low","ci_high","stems"]].to_numpy().tolist()
                    fig.add_trace(go.Scatter(x=group["mean"],y=[METHODS[m][0] for m in methods],mode="markers",
                        name=label,legendgroup=ranking,showlegend=column==1,visible=start==0,
                        marker=dict(color=color,symbol="circle" if marker=="o" else "diamond",size=9),
                        error_x=dict(type="data",symmetric=False,array=group.ci_high-group["mean"],arrayminus=group["mean"]-group.ci_low),
                        customdata=custom,hovertemplate="%{y}<br>Δ precision=%{x:.3f}<br>n=%{customdata[0]}<br>Degree=%{customdata[1]:.3f}<br>New=%{customdata[2]:.3f}<extra>%{fullData.name}</extra>"),row=1,col=column)
            menus.append((cohort,tier,start,int(data.n.iloc[0])))
    buttons=[]
    for cohort,tier,start,n in menus:
        visible=[start<=idx<start+4 for idx in range(len(fig.data))]
        label=f"{COHORTS[cohort]} · {'All Neff' if tier=='All depths' else 'Neff '+tier} · n={n}"
        buttons.append(dict(label=label,method="update",args=[{"visible":visible},{"title.text":label}]))
    fig.update_layout(template="none",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT,color=INK,size=12),height=540,margin=dict(l=175,r=25,t=110,b=100),
                      legend=dict(orientation="h",x=0,y=-0.20),
                      updatemenus=[dict(buttons=buttons,x=0,y=1.25,xanchor="left",yanchor="top",font=dict(size=11))])
    fig.update_xaxes(title="Change in precision @ L/5",zeroline=True,zerolinecolor=INK,gridcolor=GRID)
    fig.update_yaxes(autorange="reversed")
    fig.update_yaxes(showticklabels=False,row=1,col=2)
    return fig
