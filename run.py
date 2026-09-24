"""Run every stage with the same Python interpreter, on Windows or POSIX."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=120000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--csv', type=Path, help='Load this schema-compatible CSV instead of generating')
    parser.add_argument('--train-end-day', type=int, default=27)
    parser.add_argument('--test-start-day', type=int, default=36)
    args = parser.parse_args()
    if args.rows < 100000 and args.csv is None:
        parser.error('The full pipeline requires at least 100,000 transactions')
    if not 0 < args.train_end_day < args.test_start_day:
        parser.error('Require 0 < train-end-day < test-start-day')
    args.data_dir = args.data_dir.resolve()
    args.out = args.out.resolve()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    database_path = args.data_dir / 'wallet.duckdb'
    csv_path = args.csv.resolve() if args.csv else args.data_dir / 'transactions.csv'
    stages = [[sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']]
    if args.csv is None:
        stages.append([sys.executable, 'scripts/generate.py', '--rows', str(args.rows),
                       '--seed', str(args.seed), '--output', str(csv_path)])
    shared = ['--db', str(database_path), '--out', str(args.out)]
    stages.extend([
        [sys.executable, 'scripts/load.py', '--csv', str(csv_path), '--db', str(database_path)],
        [sys.executable, 'scripts/eda.py', *shared, '--train-end-day', str(args.train_end_day)],
        [sys.executable, 'scripts/features.py', *shared],
        [sys.executable, 'scripts/train.py', *shared, '--train-end-day', str(args.train_end_day),
         '--test-start-day', str(args.test_start_day), '--seed', str(args.seed)],
        [sys.executable, 'scripts/verify.py', *shared],
    ])
    manifest = {
        'status': 'running', 'python': sys.version, 'platform': platform.platform(),
        'seed': args.seed, 'input_source': 'external CSV' if args.csv else 'simulated',
        'packages': {name: importlib.metadata.version(name) for name in
                     ['duckdb', 'numpy', 'pandas', 'scikit-learn', 'xgboost', 'matplotlib', 'joblib']},
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path)
                          for path in [ROOT / 'run.py', *sorted((ROOT / 'scripts').glob('*.py')),
                                       *sorted((ROOT / 'tests').glob('*.py'))]},
        'commands': [],
    }
    manifest_path = args.out / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    pipeline_start = time.perf_counter()
    try:
        with (args.out / 'pipeline.log').open('w', encoding='utf-8') as log:
            for command in stages:
                print(subprocess.list2cmdline(command), flush=True)
                log.write(subprocess.list2cmdline(command) + '\n')
                log.flush()
                started = time.perf_counter()
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                manifest['commands'].append({
                    'argv': command, 'exit_code': completed.returncode,
                    'seconds': round(time.perf_counter() - started, 2),
                })
                completed.check_returncode()
        manifest['status'] = 'passed'
        manifest['input_sha256'] = file_hash(csv_path)
    except subprocess.CalledProcessError:
        manifest['status'] = 'failed'
        print(f'Failure details: {args.out / "pipeline.log"}', file=sys.stderr)
        raise
    finally:
        manifest['seconds'] = round(time.perf_counter() - pipeline_start, 2)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Complete in {manifest["seconds"]} seconds. Results: {args.out / "metrics.json"}')


if __name__ == '__main__':
    main()
