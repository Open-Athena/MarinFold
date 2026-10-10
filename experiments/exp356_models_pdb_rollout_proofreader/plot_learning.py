"""Plot the fixed validation sample from durable, checkpoint-independent metrics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta
from storage import ROOT, filesystem

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Refresh the small metric table when requested; otherwise plot it offline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    table = HERE/'data'/f'{args.run}_learning.csv'
    if args.refresh:
        fs, root = filesystem(f'{ROOT}/runs/{args.run}')
        rows = [json.loads(fs.cat(path)) for path in fs.glob(root+'/validation-step-*.json')]
        if not rows:
            raise ValueError('No completed validation measurements')
        pd.DataFrame(rows).sort_values('step').to_csv(table,index=False)
    frame = pd.read_csv(table)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    fig.suptitle(args.run,fontsize=11)
    axes[0].plot(frame.step,frame.contact_loss,'o-')
    axes[0].set(xlabel='Optimizer step',ylabel='Contact binary cross-entropy',title='Fixed validation sample')
    axes[1].plot(frame.step,frame.brier,'o-')
    axes[1].set(xlabel='Optimizer step',ylabel='Contact Brier score',title='Correctness probabilities')
    for column,label in [('precision_mae','Precision'),('recall_mae','Recall')]:
        axes[2].plot(frame.step,frame[column],'o-',label=label)
    axes[2].set(xlabel='Optimizer step',ylabel='Mean absolute error',title='Rollout quality estimates')
    axes[2].legend()
    fig.tight_layout()
    save_plot_with_meta(fig,HERE/'plots'/f'{args.run}_learning.png',
        caption='Fixed 256-rollout validation sample at prefixes 1, 4, 16 and full. These training diagnostics are separate from the complete held-out evaluation.',
        script='plot_learning.py',args=['--run',args.run],dpi=150)
    plt.close(fig)
    print(table)


if __name__ == '__main__':
    main()
