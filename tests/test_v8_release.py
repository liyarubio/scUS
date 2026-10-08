"""Release provenance, input guards and archived numerical-oracle execution."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'analysis/v8'
spec = importlib.util.spec_from_file_location('v8_reproduce', ROOT / 'reproduce.py')
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)


def test_archive_integrity_and_syntax():
    assert v8.verify_archive() > 0
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for row in v8.read_json(ROOT / 'protocols/readout_protocols.json')['producer_files']:
        assert v8.digest(ROOT / 'archive' / row['snapshot'].removeprefix('sources/')) == row['sha256']


def test_input_hash_failure_creates_no_output(tmp_path):
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='missing:'):
        v8.recompute(tmp_path / 'absent_inputs', output)
    assert not output.exists()
    data = tmp_path / 'value.csv'
    data.write_text('modified input')
    with pytest.raises(ValueError, match='checksum mismatch'):
        v8.verify_files(tmp_path, [{'path': data.name, 'sha256': '0' * 64}])
    with pytest.raises(ValueError, match='escapes'):
        v8.verify_files(tmp_path, [{'path': '../external.csv', 'sha256': '0' * 64}])


def test_archived_distance_and_transform_oracles(tmp_path):
    pytest.importorskip('pandas')
    pytest.importorskip('scipy')
    pytest.importorskip('sklearn')
    code = tmp_path / 'code'
    code.mkdir()
    for name in ['common.py', 'distances.py', 'transforms.py', 'test_math.py', 'test_transforms.py']:
        shutil.copy2(ROOT / 'archive/benchmark/code' / name, code / name)
    for script in ['test_math.py', 'test_transforms.py']:
        result = subprocess.run([sys.executable, str(code / script)], cwd=tmp_path,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    for name in ['math_cpu.json', 'transforms.json']:
        assert json.loads((tmp_path / 'tests' / name).read_text())['passed']


def test_archived_timecourse_tests(tmp_path):
    pytest.importorskip('pandas')
    pytest.importorskip('scipy')
    pytest.importorskip('statsmodels')
    for name in ['framework_stats.py', 'test_framework.py']:
        shutil.copy2(ROOT / 'archive/timecourse/code' / name, tmp_path / name)
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(tmp_path / 'test_framework.py')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
