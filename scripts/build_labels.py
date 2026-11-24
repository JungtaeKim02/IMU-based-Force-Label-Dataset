import argparse
import csv
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, dump, force, manifest
from processing import FK, relabel


def generate(entry, root, fk):
    source = root / entry['synchronized_file']
    frame = pd.read_csv(source)
    corrected, raw, original, reference, info = relabel(frame, fk)
    if abs(info['m'] - entry['total_mass_g'] / 1000) > 2e-6:
        raise ValueError(entry['trial_id'] + ': inconsistent mass')
    with source.open(newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    added = [f'force_{tag}_{axis}' for axis in 'xyz' for tag in ['fkatt', 'fkref']]
    full_header = header + added
    index = {name: i for i, name in enumerate(full_header)}
    full_rows = []
    for k, source_row in enumerate(rows):
        row = source_row.copy() + [''] * len(added)
        for i, axis in enumerate('xyz'):
            row[index['force_' + axis]] = str(corrected[k, i])
            row[index['force_raw_' + axis]] = str(raw[k, i])
            row[index['force_fkatt_' + axis]] = source_row[header.index('force_' + axis)]
            row[index['force_fkref_' + axis]] = str(reference[k, i])
        full_rows.append(row)
    selected = [row.copy() for row in full_rows if int(row[index['phase']]) == 2]
    zero = Decimal(selected[0][index['timestamp']])
    for row in selected:
        row[index['timestamp']] = format(Decimal(row[index['timestamp']]) - zero, 'f')
    return full_header, full_rows, selected, info


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/labels')
    parser.add_argument('--trial')
    args = parser.parse_args()
    root = args.root.resolve()
    fk = FK(str(root / 'kinematics/scene.xml'))
    entries = manifest(root)
    if args.trial:
        entries = entries[entries.trial_id == args.trial]
    if not len(entries):
        parser.error('Unknown trial ID')
    report = []
    for entry in entries.to_dict('records'):
        header, full, selected, info = generate(entry, root, fk)
        trial = entry['trial_id']
        write(args.output / 'full' / (trial + '.csv'), header, full)
        target = args.output / 'motion' / (trial + '.csv')
        write(target, header, selected)
        generated = pd.read_csv(target)
        released = pd.read_csv(root / entry['motion_file'])
        checked_columns = ['force_' + a for a in 'xyz'] + ['force_raw_' + a for a in 'xyz'] + ['force_fkref_' + a for a in 'xyz']
        delta = float(np.max(np.abs(generated[checked_columns].to_numpy() - released[checked_columns].to_numpy())))
        np.testing.assert_allclose(generated[checked_columns], released[checked_columns], atol=2e-7, rtol=0)
        np.testing.assert_array_equal(generated.timestamp, released.timestamp)
        gravity = generated[['timestamp']].copy()
        gravity['force_x'] = gravity['force_y'] = -999.
        gravity['force_z'] = -entry['total_mass_g'] / 1000 * 9.81
        (args.output / 'gravity').mkdir(parents=True, exist_ok=True)
        gravity.to_csv(args.output / 'gravity' / (trial + '.csv'), index=False)
        report.append(dict(trial_id=trial, motion_samples=len(selected), max_difference_N=delta, **info))
    dump(args.output / 'verification.json', dict(trials=len(report), max_difference_N=max(r['max_difference_N'] for r in report), per_trial=report))
    print(f'Regenerated and verified {len(report)} trials')


if __name__ == '__main__':
    main()
