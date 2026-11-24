import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ROOT, force, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/plots')
    args = parser.parse_args()
    root = args.root.resolve()
    entries = manifest(root).set_index('trial_id')
    examples = json.loads((root / 'media/representatives.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    for example in examples:
        trial = example['trial_id']
        frame = pd.read_csv(root / entries.loc[trial, 'motion_file'])
        label = force(frame)[1:].astype(np.float32)
        prediction = pd.read_csv(root / 'results/predictions' / (trial + '.csv'))
        t = prediction.time_s.to_numpy()
        fig, axes = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True, layout='constrained')
        for i, axis in enumerate('xyz'):
            axes[i].plot(t, label[:, i], color='#000000', lw=1.25, label='IMU label')
            axes[i].plot(t, prediction[f'pred_imu_{axis}_N'], color='#0000ff', lw=1.05, label='M-imu')
            axes[i].set_ylabel(f'$F_{axis}$ [N]')
            axes[i].grid(alpha=.2)
            axes[i].set_xlim(t[0], t[-1])
        axes[0].legend(loc='lower right', bbox_to_anchor=(1, 1.01), frameon=False, ncol=2, fontsize=9)
        fig.suptitle(f'{trial} | {example["motion_type"]} | {example["total_mass_g"]} g', fontsize=11)
        axes[-1].set_xlabel('Time from motion start [s]')
        fig.savefig(args.output / (trial + '.png'), dpi=180, bbox_inches='tight', pad_inches=.08)
        plt.close(fig)


if __name__ == '__main__':
    main()
