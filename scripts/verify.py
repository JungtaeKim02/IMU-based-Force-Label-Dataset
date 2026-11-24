import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tokenize
import numpy as np
import pandas as pd
from common import ROOT, FEATURE_COLUMNS, dump, force, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    entries = manifest(root)
    assert len(entries) == 60 and entries.trial_id.is_unique
    assert entries.split.value_counts().to_dict() == dict(train=44, validation=8, test=8)
    assert entries.groupby(['motion_type', 'total_mass_g']).size().eq(10).all()
    assert entries.total_mass_g.eq(entries.object_mass_g + 16).all()
    for split in ['train', 'validation', 'test']:
        expected = entries.loc[entries.split == split, 'trial_id'].tolist()
        assert (root / 'splits' / (split + '.txt')).read_text().splitlines() == expected
    for entry in entries.to_dict('records'):
        raw = root / entry['raw_dir']
        assert all((raw / name).is_file() for name in ['arm.csv', 'imu.csv', 'att.csv', 'meta.json', 'replay.json'])
        full = pd.read_csv(root / entry['synchronized_file'])
        motion = pd.read_csv(root / entry['motion_file'])
        assert len(full) == entry['synchronized_samples']
        assert len(motion) == entry['motion_samples']
        assert (motion.phase == 2).all() and (motion.imu_valid == 1).all()
        assert all(np.any(full.phase == phase) for phase in [0, 1, 2, 3])
        assert motion.timestamp.iloc[0] == 0 and (np.diff(motion.timestamp) > 0).all()
        assert np.isfinite(motion[FEATURE_COLUMNS].to_numpy()).all() and np.isfinite(force(motion)).all()
        np.testing.assert_array_equal(motion[FEATURE_COLUMNS], full.loc[full.phase == 2, FEATURE_COLUMNS])
        gravity = pd.read_csv(root / entry['gravity_file'])
        assert gravity[['force_x', 'force_y']].eq(-999).all().all()
        np.testing.assert_allclose(gravity.force_z, -entry['total_mass_g'] / 1000 * 9.81, atol=1e-10)
    for path in (root / 'scripts').glob('*.py'):
        source = path.read_text()
        assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(source).readline))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node) is None
    checksums = root / 'SHA256SUMS'
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            digest, relative = line.split('  ', 1)
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    result = dict(status='passed', trials=60, split=dict(train=44, validation=8, test=8),
                  motion_samples=int(entries.motion_samples.sum()), test_prediction_samples=int(entries.loc[entries.split == 'test', 'prediction_samples'].sum()),
                  model_input_channels=42, model_inputs_match_synchronized=True, code_comments=0)
    dump(root / 'outputs/verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
