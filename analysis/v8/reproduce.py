#!/usr/bin/env python3
"""Recompute v8 numerical summaries from an explicitly supplied frozen input bundle.

Never trains a model, recomputes embeddings or runs the archived GPU producers.
Input tables are checksum-verified before any output is created.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def contained(root: Path, relative: str) -> Path:
    rel = Path(relative)
    target = (root / rel).resolve()
    if rel.is_absolute() or not target.is_relative_to(root.resolve()):
        raise ValueError(f'Input path escapes its root: {relative}')
    return target


def verify_files(root: Path, records: list[dict]) -> None:
    errors = []
    for row in records:
        path = contained(root, row['path'])
        if not path.is_file():
            errors.append(f'missing: {row["path"]}')
        elif digest(path) != row['sha256']:
            errors.append(f'checksum mismatch: {row["path"]}')
    if errors:
        raise ValueError('Input verification failed:\n' + '\n'.join(errors))


def verify_archive() -> int:
    records = read_json(ROOT / 'SOURCE_MANIFEST.json')['files']
    verify_files(ROOT, records)
    return len(records)


def recompute(source: Path, output: Path | None, check_only: bool = False) -> dict:
    source = source.resolve()
    code_count = verify_archive()
    expected = read_json(ROOT / 'protocols/statistics_inputs.json')['files']
    verify_files(source, expected)
    report = {
        'status': 'inputs_verified', 'input_files': len(expected),
        'archive_files': code_count,
        'scope': 'Saved-output numerical recomputation only; no training, inference or clustering.',
    }
    if check_only:
        return report
    if output is None:
        raise ValueError('--output-dir is required unless --check-only is used')
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Input and output directories must be separate, non-nested trees')
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError('Write generated data outside the versioned analysis/v8 directory')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be absent or an empty directory; existing results are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    work = output / 'work'
    for row in expected:
        target = work / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contained(source, row['path']), target)
    for name in ['revision_data', 'web_supplement', 'logs']:
        (work / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    for script in ['revision_tables.py', 'timecourse_tables.py']:
        shutil.copy2(ROOT / 'archive/manuscript' / script, work / script)
        with (work / 'logs' / (script + '.log')).open('w') as stream:
            subprocess.run([sys.executable, str(work / script)], cwd=work,
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    import pandas as pd
    comparisons = []
    for relative in read_json(ROOT / 'protocols/statistics_inputs.json')['reference_outputs']:
        produced, reference = work / relative, source / relative
        if not reference.is_file():
            comparisons.append({'path': relative, 'status': 'reference_not_supplied'})
            continue
        pd.testing.assert_frame_equal(
            pd.read_csv(produced), pd.read_csv(reference), check_dtype=False,
            check_exact=False, rtol=1e-10, atol=1e-12,
        )
        comparisons.append({'path': relative, 'status': 'matches_saved_v8',
                            'reference_sha256': digest(reference), 'produced_sha256': digest(produced)})
    lps = read_json(work / 'revision_data/lps_validation.json')
    rank = read_json(work / 'revision_data/rpl23_rank_validation.json')
    report.update({
        'status': 'recomputed', 'reference_comparisons': comparisons,
        'all_supplied_references_match': True,
        'all_reference_outputs_supplied': all(r['status'] == 'matches_saved_v8' for r in comparisons),
        'lps': {'candidate_records': lps['candidate_records'],
                'candidate_genes': lps['candidate_genes'], 'joint_US_records': lps['joint_US_records']},
        'rpl23_tests': {'status': rank['status'], 'unit': rank['unit'], 'adjustment': rank['adjustment']},
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy']},
        'generated_files': [{'path': str(p.relative_to(work)), 'sha256': digest(p)}
                            for p in sorted((work / 'revision_data').iterdir()) if p.is_file()],
        'limitations': ['Exploratory cell-level time-course tests; independent cultures unverified.',
                        'Context summaries use overlapping strata, not independent biological replicates.',
                        'This command does not reproduce model inference, clustering or multiomic export.'],
    })
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Path to the v8 manuscript input bundle (not distributed with this code release)')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Verify hashes without creating outputs')
    args = parser.parse_args()
    try:
        result = recompute(args.input_dir, args.output_dir, args.check_only)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError, AssertionError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
