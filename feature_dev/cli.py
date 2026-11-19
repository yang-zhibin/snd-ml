"""Plan immutable development runs and execute their Snakemake DAG."""
import argparse
import fcntl
import getpass
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from engine import ROOT, schema, load_events, sha, write_json, read_json
from datasets import PHYSICS_DATASETS, DATASETS, resolve_dataset, check_size
from sources import build_plan, resolve_prepared
from exposure import resolve_normalization


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def find_snakemake(work):
    """Use an explicit executable, then this user's installations, then PATH."""
    configured = os.environ.get('FEATURE_DEV_SNAKEMAKE')
    if configured:
        executable = shutil.which(os.path.expanduser(configured))
        if executable:
            return executable
        raise ValueError(
            f'FEATURE_DEV_SNAKEMAKE is not an accessible executable: {configured}. '
            'Install Snakemake in your own account; see README_SHOWER_FEATURES.md '
            '(run ./run_feature_dev.sh setup). Access to a private environment is not required.'
        )
    for candidate in (work/'venvs/shower-snakemake/bin/snakemake',
                      work/'miniconda3/envs/snakemake/bin/snakemake'):
        executable = shutil.which(str(candidate))
        if executable:
            return executable
    executable = shutil.which('snakemake')
    if executable:
        return executable
    raise ValueError(
        'Snakemake was not found. See README_SHOWER_FEATURES.md '
        '(run ./run_feature_dev.sh setup), or set FEATURE_DEV_SNAKEMAKE '
        'to your executable. The analysis Python environment is provided separately by LCG.'
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',default='baseline')
    parser.add_argument('--schema',type=Path)
    parser.add_argument('--dependency',type=Path,action='append',default=[],help='Additional imported algorithm source file; repeat as needed')
    parser.add_argument('--sample',default='small')
    parser.add_argument('--normalization', choices=('auto', 'shape', 'yield'), default='auto')
    parser.add_argument('--exposure-manifest', type=Path, help='Audited parent-source coverage JSON for a prepared full sample')
    parser.add_argument('--dataset', choices=('synthetic','all')+DATASETS)
    parser.add_argument('--dataset-config', type=Path, default=ROOT/'feature_dev/datasets.yaml')
    parser.add_argument('--sample-root', type=Path, help='Use existing prepared samples instead of automatic source preparation')
    parser.add_argument('--files-per-group', type=int, help='Override automatic small-sample source cap per catalogue stratum')
    parser.add_argument('--events-per-file', type=int, help='Override automatic small-sample selected-event cap per source')
    parser.add_argument('--prepare-profile', type=Path, help='Snakemake profile for preparation (e.g. HTCondor); analysis remains local')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare and validate inputs without running feature analysis')
    parser.add_argument('--parameters',default='{}',help='JSON object of algorithm parameters')
    parser.add_argument('--run',required=True)
    user=getpass.getuser();work=Path('/afs/cern.ch/work')/user[0]/user
    parser.add_argument('--output-root',type=Path,default=work/'snd-ml-outputs/unified_shower_features_v1/feature_dev' if work.exists() else Path.cwd()/'feature-dev-output')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.normalization == 'yield' and (args.sample != 'full' or args.dataset not in ('mc-neutrino','mc-muon','real-data')):
        parser.error('Yield requires a single prepared full MC/data dataset')
    if args.exposure_manifest and (args.dataset == 'all' or args.sample != 'full' or args.normalization == 'shape' or args.dataset in (None,'synthetic','testbeam')):
        parser.error('--exposure-manifest requires a single full MC/data yield run')
    if args.dataset == 'all' and args.sample == 'full' and args.normalization != 'shape':
        parser.error('Run full yield datasets individually with their exposure audits')

    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',args.run):parser.error('Invalid run name')
    if args.prepare_only and args.dataset not in (*DATASETS, 'all'):
        parser.error('--prepare-only requires a physics dataset')
    if (args.files_per_group is not None or args.events_per_file is not None) and args.dataset not in (*DATASETS, 'all'):
        parser.error('Sampling overrides require a physics dataset')
    if args.prepare_profile and not args.prepare_profile.is_dir():
        parser.error('--prepare-profile must name an existing Snakemake profile directory')
    if not args.prepare_only:
        builtin = args.algorithm in ('baseline', 'gap_clusters')
        algorithm = ROOT/'feature_dev/algorithms'/f'{args.algorithm}.py' if builtin else Path(args.algorithm)
        if not algorithm.is_file():
            parser.error(f'Algorithm does not exist: {algorithm}')
        if not builtin and not args.schema:
            parser.error('External algorithms require --schema')
        schema(args.schema or ROOT/'feature_dev/schemas'/f'{args.algorithm}.yaml')
        if not isinstance(json.loads(args.parameters), dict):
            parser.error('--parameters must be a JSON object')
        for dependency in args.dependency:
            if not dependency.is_file():
                parser.error(f'Dependency does not exist: {dependency}')
    if args.dataset == 'all':
        # Resolve every sample before creating outputs or starting any child DAG.
        selections = {name: selection_request(args, name) for name in PHYSICS_DATASETS}
        manifests = []
        import copy
        for name in PHYSICS_DATASETS:
            child = copy.copy(args);child.dataset = name
            manifests.append(run_one(child, selections[name]))
        if not args.dry_run and not args.prepare_only:
            out = args.output_root.resolve()/'all'/args.run
            out.mkdir(parents=True, exist_ok=True)
            write_json(out/'manifest.json', {'datasets': manifests, 'sample': args.sample})
            links = ''.join('<li><a href="../../'+name+'/'+args.run+'/plot/report.html">'+html.escape(name)+'</a></li>' for name in PHYSICS_DATASETS)
            (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Dataset reports</h1><p>Independent unit-normalized shape comparisons.</p><ul>'+links+'</ul>')
            print(f'Dataset index: {out / "index.html"}')
    else:
        run_one(args)



def selection_request(args, name):
    import yaml
    config = yaml.safe_load(args.dataset_config.read_text())
    entry = config.get('datasets', {}).get(name, {})
    if args.sample_root is not None or 'sources' not in entry:
        if args.files_per_group is not None or args.events_per_file is not None:
            raise ValueError('Sampling overrides require automatic source preparation')
        return ('prepared', resolve_dataset(name, args.sample, args.dataset_config,
                                             args.sample_root or args.output_root/'samples'))
    if args.sample != 'small':
        raise ValueError('Automatic EOS preparation supports --sample small only; full campaigns require a sharded production workflow. Use --sample-root for existing bounded exports.')
    return ('automatic', build_plan(name, args.dataset_config, args.output_root,
                                    args.files_per_group, args.events_per_file))


def prepare_selection(args, request, work):
    kind, value = request
    if kind == 'prepared':
        return value
    plan = value
    cached = resolve_prepared(plan)
    if cached is not None:
        print(f'Reusing validated {plan["dataset"]} sample: {cached[0]}')
        return cached
    base = Path(plan['directory'])
    print(f'Preparing {plan["dataset"]}: {len(plan["jobs"])} source files, at most '
          f'{sum(j["max_events"] for j in plan["jobs"])} selected events; cache: {base}', flush=True)
    executable = find_snakemake(work)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    def execute(planfile):
        command = [executable, '--snakefile', str(ROOT/'feature_dev/PrepareSnakefile'),
                   '--cores', '1', '--rerun-triggers', 'mtime', '--config',
                   f'plan={planfile}', f'wrapper={ROOT/"run_feature_dev.sh"}']
        if args.prepare_profile:
            command += ['--profile', str(args.prepare_profile.resolve())]
        if args.dry_run:
            command.append('--dry-run')
        else:
            write_json(base/'launch.json', {'command': command, 'cwd': str(base)})
        subprocess.run(command, cwd=planfile.parent, env=env, check=True)
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix='feature-dev-preparation-') as temporary:
            path = Path(temporary)/'plan.json'
            write_json(path, plan)
            execute(path)
        print('Dry run: preparation DAG only; analysis is planned after the sample is available. No result directories created.')
        return None
    base.mkdir(parents=True, exist_ok=True)
    with (base/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = resolve_prepared(plan)
        if cached is not None:
            return cached
        write_json(base/'plan.json', plan)
        execute(base/'plan.json')
        prepared = resolve_prepared(plan)
        if prepared is None:
            raise ValueError('Preparation finished without a validated sample receipt')
        return prepared


def run_one(args, prepared=None):
    user=getpass.getuser();work=Path('/afs/cern.ch/work')/user[0]/user
    builtin=args.algorithm in ('baseline','gap_clusters')
    algorithm=ROOT/'feature_dev/algorithms'/f'{args.algorithm}.py' if builtin else Path(args.algorithm).resolve()
    if not builtin and not args.schema:raise ValueError('External algorithms require --schema')
    definition=args.schema or ROOT/'feature_dev/schemas'/f'{args.algorithm}.yaml'
    dataset_info = None
    if args.dataset in DATASETS:
        request = prepared or selection_request(args, args.dataset)
        selection = prepare_selection(args, request, work)
        if selection is None:
            return
        sample, dataset_info = selection
        if args.prepare_only:
            print(f'Prepared sample: {sample}')
            return
    elif args.dataset == 'synthetic':
        if args.sample not in ('small','synthetic'):raise ValueError('Synthetic dataset has only the small fixture')
        sample=ROOT/'feature_dev/samples/small.jsonl'
        dataset_info={'name':'synthetic','sample':'small','version':'small-v1','normalization':'unit_area'}
    else:
        # Preserve existing commands, but keep the synthetic alias explicit in output.
        sample=ROOT/'feature_dev/samples/small.jsonl' if args.sample in ('small','synthetic') else Path(args.sample).resolve()
        if args.sample in ('small','synthetic'):print('Using the synthetic software fixture (legacy --sample small alias).')
    if args.prepare_only:
        raise ValueError('--prepare-only requires a physics dataset')
    check_size([sample])
    events,sample_manifest=load_events(sample)
    exposure_audit=read_json(args.exposure_manifest) if args.exposure_manifest else None
    normalization=resolve_normalization(args.normalization, args.dataset or 'synthetic', args.sample,
        exposure_audit, sha(sample), events, sample_manifest['provenance'])
    parameters=json.loads(args.parameters)
    if not isinstance(parameters,dict):raise ValueError('--parameters must be a JSON object')
    source_files=[ROOT/'feature_dev/root_metadata.py',ROOT/'feature_dev/exposure.py',ROOT/'evaluation_region_partitions/normalization.py',algorithm,ROOT/'convertData/scifi_plane_clustering.py',ROOT/'feature_dev/engine.py',ROOT/'feature_dev/selected_hits.py',*sorted((ROOT/'feature_dev/algorithms').glob('*.py')),ROOT/'feature_dev/cli.py',ROOT/'feature_dev/datasets.py',ROOT/'feature_dev/sources.py',ROOT/'feature_dev/Snakefile',ROOT/'run_feature_dev.sh',*[p.resolve() for p in args.dependency]]
    hashes={str(p):sha(p) for p in source_files}
    hashes[str(sample)]=sha(sample);hashes[str(sample)+'.manifest.json']=sha(str(sample)+'.manifest.json')
    import numpy, uproot, yaml, matplotlib, platform
    runtime={m.__name__:m.__version__ for m in (numpy,uproot,yaml,matplotlib)}
    runtime['python']=platform.python_version()
    plan={'runtime':runtime,'sample':str(sample),'source_hashes':hashes,
          'baseline':{'algorithm':str(ROOT/'feature_dev/algorithms/baseline.py'),'parameters':{},'schema':schema(ROOT/'feature_dev/schemas/baseline.yaml')},
          'candidate':{'algorithm':str(algorithm),'parameters':parameters,'schema':schema(definition)}}
    plan['normalization']=normalization
    if args.exposure_manifest:
        plan['exposure_audit']=exposure_audit
    if dataset_info is not None:plan['dataset']=dataset_info
    # Histogram settings do not enter the calculation signature.
    calculation={k:v for k,v in plan.items() if k not in ('baseline','candidate','normalization','exposure_audit')}
    for name in ('baseline','candidate'):
        calculation[name]={**plan[name],'schema':{n:{k:v for k,v in d.items() if k not in ('histogram','description')} for n,d in plan[name]['schema']['features'].items()}}
    calc=digest(calculation);valid=digest([calc,'validation-v1'])
    hist=digest([valid,plan['candidate']['schema'],normalization,plan.get('exposure_audit')]);plot=digest([hist,'report-v1'])
    export=digest([valid,normalization,plan.get('exposure_audit'),'root-metadata-v1'])
    keys=dict(calculate=calc,validate=valid,export=export,histogram=hist,plot=plot)
    output=args.output_root.resolve()
    if args.dataset:output=output/args.dataset
    run=output/args.run
    plan['paths']={name:str(output/'.cache'/name/key) for name,key in keys.items()}
    signature=digest(plan)
    manifest={'signature':signature,'stage_keys':keys,'plan':plan,'sample_provenance':sample_manifest}
    executable=find_snakemake(work)
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    def execute(planfile,dry=False):
        command=[executable,'--snakefile',str(ROOT/'feature_dev/Snakefile'),'--cores','1','--rerun-triggers','mtime','--config',f'plan={planfile}',f'wrapper={ROOT / "run_feature_dev.sh"}']
        if dry:command.append('--dry-run')
        # Snakemake bookkeeping belongs with outputs, never in the checkout.
        subprocess.run(command,cwd=Path(planfile).parent,env=env,check=True)
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix='feature-dev-plan-') as temporary:
            p=Path(temporary)/'plan.json';write_json(p,plan);execute(p,True)
        print('Dry run: no result directories created.');return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.feature-dev.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if run.exists():
            if not (run/'manifest.json').exists() or read_json(run/'manifest.json')['signature']!=signature:
                raise ValueError('Run name already exists with different inputs/code/settings; choose a new --run')
        # Detect edited cached products before they can be reused.
        for path in plan['paths'].values():
            receipt=Path(path)/'complete.json'
            if receipt.exists():
                for file,expected in read_json(receipt).items():
                    if not (Path(path)/file).exists() or sha(Path(path)/file)!=expected:raise ValueError(f'Cached output changed: {path}/{file}')
        run.mkdir(exist_ok=True)
        write_json(run/'manifest.json',manifest);write_json(run/'plan.json',plan)
        snapshot=run/'code';snapshot.mkdir(exist_ok=True)
        for i,p in enumerate(source_files):shutil.copyfile(p,snapshot/f'{i:02d}-{p.name}')
        execute(run/'plan.json')
        for name,path in plan['paths'].items():
            link=run/name
            if not link.exists():link.symlink_to(path,target_is_directory=True)
        print(f'Report: {run / "plot/report.html"}')
        published=run/'features.root'
        if not published.exists():published.symlink_to(Path(plan['paths']['export'])/'features.root')
        print(f'Features: {published}')
        return {'dataset':args.dataset,'manifest':str(run/'manifest.json'),'signature':signature}

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as error:raise SystemExit(str(error))
