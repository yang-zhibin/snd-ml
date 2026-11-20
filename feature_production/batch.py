"""Private official-profile deployment and batch preflight for the production CLI."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / '.feature-production-profile'
UPSTREAM = 'https://github.com/Snakemake-Profiles/htcondor.git'
REVISION = 'a5b7e6c73d9b9305abe4a04a24056050461627d3'
PACKAGES = ['cookiecutter==2.7.1', 'htcondor==25.0.14',
            'snakemake-executor-plugin-cluster-generic==1.0.9']


def clean_env():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PIP_CONFIG_FILE=os.devnull)
    return env


def atomic_text(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        stream.write(value); name = stream.name
    os.replace(name, path)


def profile_config(profile):
    """Resolve the versioned profile for our pinned Snakemake major (9)."""
    profile = Path(profile).expanduser().resolve()
    candidates = []
    for p in profile.glob('config.v*+.yaml'):
        match = re.fullmatch(r'config\.v(\d+)\+\.yaml', p.name)
        if match and int(match[1]) <= 9: candidates.append((int(match[1]), p))
    path = max(candidates)[1] if candidates else profile/'config.yaml'
    if not path.is_file(): raise ValueError(f'Missing profile configuration: {profile}')
    cfg = yaml.safe_load(path.read_text()) or {}
    executor = cfg.get('executor')
    if executor == 'cluster-generic':
        if not all(cfg.get(k) for k in ('cluster-generic-submit-cmd','cluster-generic-status-cmd')):
            raise ValueError('cluster-generic requires submit and status commands; use setup --batch')
    elif executor != 'htcondor':
        raise ValueError('Batch profile must use htcondor or a configured cluster-generic submission route')
    return path, cfg


def choose_profile(explicit, sample, batch=False):
    if explicit: return Path(explicit).expanduser().resolve()
    if sample == 'full' or batch:
        if SAVED.is_file(): return Path(SAVED.read_text().strip()).expanduser().resolve()
    return None


def check_profile(profile, executable, credentials=False):
    path, cfg = profile_config(profile)
    module = 'snakemake_executor_plugin_' + cfg['executor'].replace('-', '_')
    python = Path(executable).parent/'python'
    subprocess.run([str(python), '-c', f'import {module}'], env=clean_env(), check=True)
    marker = Path(profile)/'deployment.json'
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if metadata['python'] != str(python):
            raise ValueError('Saved profile belongs to a different Snakemake environment; rerun setup --batch with its --prefix')
        # The generated hooks use this interpreter, not /usr/bin/python3.
        subprocess.run([str(python), '-c', 'import htcondor2; print(htcondor2.version())'], env=clean_env(), check=True)
        log = Path(metadata['log_directory'])
        if not log.is_dir() or not os.access(log, os.W_OK | os.X_OK):
            raise ValueError(f'Batch log directory is not writable: {log}')
    if credentials:
        subprocess.run(['klist', '-s'], check=True)
        subprocess.run(['condor_ping', '-type', 'schedd', 'WRITE'], check=True)
    return dict(profile=str(Path(profile).resolve()), config=str(path), executor=cfg['executor'],
                config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def setup(prefix, check=False, profile=None, jobs=20):
    """Deploy once; preserve existing profiles and user customizations on repeats."""
    prefix = Path(prefix)
    python = prefix/'bin/python'
    executable = prefix/'bin/snakemake'
    target = (Path(profile).expanduser().resolve() if profile else
              Path(SAVED.read_text().strip()) if SAVED.is_file() else prefix/'profiles/htcondor')
    if jobs < 1: raise ValueError('Batch jobs must be positive')
    if check:
        result = check_profile(target, executable)
        print('Batch setup verified:', result['profile']); return
    if target.exists():
        if not (target/'deployment.json').is_file():
            raise ValueError(f'Refusing to overwrite an unmanaged profile: {target}; choose --batch-profile')
        check_profile(target, executable)
        atomic_text(SAVED, str(target)+'\n')
        print('Existing batch profile preserved:', target); return
    subprocess.run([str(python), '-m', 'pip', 'install', 'snakemake==9.17.2', *PACKAGES], env=clean_env(), check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], env=clean_env(), check=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.profile-', dir=target.parent) as tmp:
        tmp = Path(tmp); repo = tmp/'upstream'
        subprocess.run(['git', 'clone', '--quiet', UPSTREAM, str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--quiet', REVISION], check=True)
        log = prefix/'htcondor_logs'; log.mkdir(parents=True, exist_ok=True)
        script = ('from cookiecutter.main import cookiecutter; import sys,json; '
                  'cookiecutter(sys.argv[1],no_input=True,accept_hooks=False,output_dir=sys.argv[2],'
                  'extra_context=json.loads(sys.argv[3]))')
        subprocess.run([str(python), '-c', script, str(repo), str(tmp/'generated'),
                        json.dumps(dict(profile_name='htcondor',htcondor_log_dir=str(log),location_cern=True))],
                       env=clean_env(), check=True)
        generated = tmp/'generated/htcondor'
        # Keep the official deployment as provenance, then apply the CERN adapter.
        vanilla = generated/'upstream'; vanilla.mkdir()
        for p in list(generated.iterdir()):
            if p.is_file(): p.rename(vanilla/p.name)
        (generated/'adapter.py').write_text((ROOT/'feature_production/batch_adapter.py').read_text())
        (generated/'grid-jobscript.sh').write_text((vanilla/'grid-jobscript.sh').read_text())
        command = shlex.join([str(python), str(target/'adapter.py')])
        cfg = dict(executor='cluster-generic', jobs=jobs, **{'local-cores':1, 'latency-wait':120,
            'max-jobs-per-second':2, 'max-status-checks-per-second':5, 'restart-times':0,
            'jobscript':str(target/'grid-jobscript.sh'),
            'cluster-generic-submit-cmd':command+' submit '+shlex.quote(str(log)),
            'cluster-generic-status-cmd':command+' status '+shlex.quote(str(log)),
            'cluster-generic-cancel-cmd':command+' cancel '+shlex.quote(str(log))})
        (generated/'config.v8+.yaml').write_text(yaml.safe_dump(cfg))
        (generated/'deployment.json').write_text(json.dumps(dict(upstream=UPSTREAM,revision=REVISION,
            python=str(python),log_directory=str(log),packages=PACKAGES,runtime_unit='minutes'),indent=2)+'\n')
        check_profile(generated, executable)
        generated.rename(target)
    check_profile(target, executable)
    atomic_text(SAVED, str(target)+'\n')
    print('Batch profile configured:', target)
