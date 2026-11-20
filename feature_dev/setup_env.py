"""Install and verify a private Snakemake environment for lxplus development."""
import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / '.feature-dev-snakemake'
VERSION = '9.17.2'
MARKER = '.snd-ml-feature-env.json'


def isolated_environment():
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PIP_CONFIG_FILE=os.devnull)
    return env


def atomic_write(path, text):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify(prefix, env):
    python = prefix / 'bin/python'
    executable = prefix / 'bin/snakemake'
    subprocess.run([str(python), '-m', 'pip', 'check'], env=env, check=True)
    actual = subprocess.check_output([str(executable), '--version'], env=env, text=True).strip()
    if actual != VERSION:
        raise ValueError(f'Expected Snakemake {VERSION}, found {actual}')
    return executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, help='New environment directory; default is your AFS work area')
    parser.add_argument('--check', action='store_true', help='Verify existing setup without installing or changing configuration')
    parser.add_argument('--batch', action='store_true', help='Install and configure the official CERN HTCondor profile')
    parser.add_argument('--batch-profile', type=Path, help='New private profile directory (default: inside environment)')
    parser.add_argument('--batch-jobs', type=int, default=20, help='Concurrency for a newly generated profile')
    args = parser.parse_args()
    if (args.batch_profile or args.batch_jobs != 20) and not args.batch:
        parser.error('Batch profile settings require --batch')
    if args.batch_jobs < 1: parser.error('--batch-jobs must be positive')
    sys.path.insert(0, str(ROOT))
    user = getpass.getuser()
    work = Path('/afs/cern.ch/work') / user[0] / user
    if args.prefix:
        prefix = args.prefix.expanduser().resolve()
    elif CONFIG.is_file():
        prefix = Path(CONFIG.read_text().strip()).parent.parent
    else:
        if not work.is_dir():
            raise ValueError('No AFS work area found. Use setup --prefix /your/writable/path/shower-snakemake')
        prefix = work / 'venvs' / f'shower-snakemake-{VERSION}-lcg107'
    if '\n' in str(prefix) or '\r' in str(prefix):
        raise ValueError('Environment path must not contain a newline')
    # Verify analysis dependencies with the same LCG interpreter as the wrapper.
    import numpy, uproot, yaml, matplotlib
    print(f'Analysis Python: {sys.executable}', flush=True)
    print('Analysis packages: ' + ', '.join(f'{m.__name__} {m.__version__}' for m in (numpy, uproot, yaml, matplotlib)), flush=True)
    env = isolated_environment()
    if args.check:
        executable = verify(prefix, env)
        if args.batch:
            from feature_production.batch import setup
            setup(prefix, check=True, profile=args.batch_profile)
        print(f'Setup verified: {executable}')
        return
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lock_path = prefix.parent / ('.feature-env-' + hashlib.sha256(str(prefix).encode()).hexdigest()[:16] + '.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = prefix / MARKER
        expected = {'owner': 'snd-ml-feature-dev', 'snakemake': VERSION, 'python': sys.executable}
        if prefix.exists():
            if not marker.is_file() or json.loads(marker.read_text()) != expected:
                raise ValueError(f'Refusing to modify an existing unmanaged/different environment: {prefix}. Choose a new --prefix.')
        else:
            prefix.mkdir()
            atomic_write(marker, json.dumps(expected) + '\n')
        # Repeating setup verifies a healthy installation without contacting PyPI.
        healthy = False
        if (prefix / 'bin/snakemake').exists():
            try:
                verify(prefix, env)
                healthy = True
            except (OSError, ValueError, subprocess.CalledProcessError):
                print('Repairing this managed environment.', flush=True)
        if not healthy:
            subprocess.run([sys.executable, '-m', 'venv', str(prefix)], env=env, check=True)
            subprocess.run([str(prefix / 'bin/python'), '-m', 'pip', 'install', f'snakemake=={VERSION}'], env=env, check=True)
        executable = verify(prefix, env)
        if args.batch:
            from feature_production.batch import setup
            setup(prefix, profile=args.batch_profile, jobs=args.batch_jobs)
        freeze = subprocess.check_output([str(prefix / 'bin/python'), '-m', 'pip', 'freeze'], env=env, text=True)
        atomic_write(prefix / 'installed-packages.txt', freeze)
        # Store data, not executable shell code. Only save after successful checks.
        atomic_write(CONFIG, str(executable) + '\n')
    print(f'Configured this checkout: {executable}')
    print('Future terminals: run ./run_feature_dev.sh directly; no activation/export needed.')
    if os.environ.get('FEATURE_DEV_SNAKEMAKE') or os.environ.get('FEATURE_DEV_PYTHON'):
        print('Unset old FEATURE_DEV_SNAKEMAKE and FEATURE_DEV_PYTHON overrides to use the saved setup.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'Setup failed: {error}\nFix the reported issue and rerun setup. No shell startup files were changed.')
