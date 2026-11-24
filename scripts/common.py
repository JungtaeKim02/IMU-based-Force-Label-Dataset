from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
AXES = list('xyz')
FEATURE_COLUMNS = [f'{prefix}{i}' for prefix in ['goal_pos', 'pos', 'load', 'vel', 'volts', 'temp'] for i in range(1, 7)]


def manifest(root=ROOT):
    return pd.read_csv(Path(root) / 'manifest.csv').sort_values('trial_id')


def force(frame, prefix='force_'):
    return frame[[prefix + axis for axis in AXES]].to_numpy(dtype=np.float64)


def lowpass(values, cutoff=2.0):
    b, a = butter(4, cutoff / 50.)
    return filtfilt(b, a, values, axis=0, padtype='odd', padlen=3 * max(len(a), len(b)))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
