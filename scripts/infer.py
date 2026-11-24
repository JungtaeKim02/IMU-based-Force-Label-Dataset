import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_SKIP_CUDA_CONSTRAINTS_CHECK'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.traverse_util import unflatten_dict
from common import ROOT, dump, manifest, windows
from model import ForceTransformer

jax.config.update('jax_default_matmul_precision', 'highest')


def predictor(root, variant):
    metadata = json.loads((root / 'checkpoints' / (variant + '.json')).read_text())
    with np.load(root / 'checkpoints' / (variant + '.npz'), allow_pickle=False) as archive:
        variables = unflatten_dict({key: jnp.asarray(archive[key]) for key in archive.files}, sep='/')
    model = ForceTransformer(**metadata['model_config'])
    predict = jax.jit(lambda x: model.apply(variables, x, training=False)[0])
    return predict, metadata['normalization']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/predictions')
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--trial')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    root = args.root.resolve()
    entries = manifest(root)
    entries = entries[entries.trial_id == args.trial] if args.trial else entries[entries.split == args.split]
    if not len(entries):
        parser.error('No matching trials')
    predictors = {v: predictor(root, v) for v in ['imu', 'grav']}
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for entry in entries.to_dict('records'):
        frame = pd.read_csv(root / entry['motion_file'])
        result = pd.DataFrame({'time_s': frame.timestamp.to_numpy()[1:] - frame.timestamp.iloc[0]})
        for variant, (predict, normalization) in predictors.items():
            x = windows(frame, normalization)
            values = []
            for start in range(0, len(x), args.batch_size):
                batch = x[start:start + args.batch_size]
                count = len(batch)
                batch = np.pad(batch, ((0, args.batch_size - count), (0, 0), (0, 0)))
                values.append(np.asarray(predict(batch))[:count])
            values = np.concatenate(values)
            for i, axis in enumerate('xyz'):
                result[f'pred_{variant}_{axis}_N'] = values[:, i]
        released = root / 'results/predictions' / (entry['trial_id'] + '.csv')
        delta = None
        if released.exists():
            expected = pd.read_csv(released)
            delta = float(np.max(np.abs(result.iloc[:, 1:].to_numpy() - expected.iloc[:, 1:].to_numpy())))
            if delta > 1e-4:
                raise ValueError(f'{entry["trial_id"]}: prediction mismatch {delta}')
        result.to_csv(args.output / (entry['trial_id'] + '.csv'), index=False)
        report.append(dict(trial_id=entry['trial_id'], samples=len(result), max_difference_from_released_N=delta))
        print(entry['trial_id'], len(result), flush=True)
    dump(args.output / 'verification.json', dict(platform='cpu', per_trial=report))


if __name__ == '__main__':
    main()
