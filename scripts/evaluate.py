import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, dump, force, lowpass, manifest


def aggregate(parts):
    result = {}
    for key in ['imu_label', 'fk_imu', 'fk_grav']:
        error = np.concatenate([part[key] for part in parts])
        result[key] = dict(samples=len(error), mae_xyz_N=np.abs(error).mean(axis=0).tolist(), rmse_xyz_N=np.sqrt(np.mean(error ** 2, axis=0)).tolist())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--predictions', type=Path, default=ROOT / 'results/predictions')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/evaluation')
    args = parser.parse_args()
    root = args.root.resolve()
    parts = []
    per_trial = []
    for entry in manifest(root).query('split == "test"').to_dict('records'):
        frame = pd.read_csv(root / entry['motion_file'])
        prediction = pd.read_csv(args.predictions / (entry['trial_id'] + '.csv'))
        expected_time = frame.timestamp.to_numpy()[1:] - frame.timestamp.iloc[0]
        np.testing.assert_allclose(prediction.time_s, expected_time, atol=1e-9, rtol=0)
        imu = prediction[[f'pred_imu_{a}_N' for a in 'xyz']].to_numpy()
        grav = prediction[[f'pred_grav_{a}_N' for a in 'xyz']].to_numpy()
        target = force(frame)[1:].astype(np.float32).astype(np.float64)
        reference = lowpass(force(frame, 'force_fkref_')[1:])
        errors = dict(imu_label=imu-target, fk_imu=lowpass(imu)-reference, fk_grav=lowpass(grav)-reference)
        part = dict(trial_id=entry['trial_id'], motion_type=entry['motion_type'], total_mass_g=entry['total_mass_g'], **errors)
        parts.append(part)
        per_trial.append(dict(trial_id=entry['trial_id'], motion_type=entry['motion_type'], object_mass_g=entry['object_mass_g'], total_mass_g=entry['total_mass_g'], **aggregate([part])))
    grouped = []
    for style in ['lateral_u', 'fore_aft_s']:
        for mass in [286, 421, 556]:
            group = [p for p in parts if p['motion_type'] == style and p['total_mass_g'] == mass]
            grouped.append(dict(motion_type=style, total_mass_g=mass, trials=len(group), **aggregate(group)))
    report = dict(test_trials=len(parts), sample_weighted=True, target_alignment='next_timestamp',
                  imu_label_metric='M-imu against 5 Hz-preprocessed IMU labels; no additional filtering',
                  fk_metric='Actual model XYZ outputs and kinematic FK reference, each filtered at 2 Hz',
                  aggregate=aggregate(parts), per_condition=grouped, per_trial=per_trial)
    dump(args.output / 'metrics.json', report)
    table = []
    for row in grouped + [dict(motion_type='all', total_mass_g='all', **report['aggregate'])]:
        for key, method in [('imu_label', 'M-imu vs IMU label'), ('fk_imu', 'M-imu vs FK'), ('fk_grav', 'M-grav vs FK')]:
            table.append(dict(motion_type=row['motion_type'], total_mass_g=row['total_mass_g'], method=method, **{f'mae_{a}_N': row[key]['mae_xyz_N'][i] for i, a in enumerate('xyz')}))
    pd.DataFrame(table).to_csv(args.output / 'mae.csv', index=False)
    print(report['aggregate'])


if __name__ == '__main__':
    main()
