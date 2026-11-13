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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',default='baseline')
    parser.add_argument('--schema',type=Path)
    parser.add_argument('--dependency',type=Path,action='append',default=[],help='Additional imported algorithm source file; repeat as needed')
    parser.add_argument('--sample',default='small')
    parser.add_argument('--dataset', choices=('synthetic','all')+DATASETS)
    parser.add_argument('--dataset-config', type=Path, default=ROOT/'feature_dev/datasets.yaml')
    parser.add_argument('--sample-root', type=Path, help='Prepared samples directory; default <output-root>/samples')
    parser.add_argument('--parameters',default='{}',help='JSON object of algorithm parameters')
    parser.add_argument('--run',required=True)
    user=getpass.getuser();work=Path('/afs/cern.ch/work')/user[0]/user
    parser.add_argument('--output-root',type=Path,default=work/'snd-ml-outputs/unified_shower_features_v1/feature_dev' if work.exists() else Path.cwd()/'feature-dev-output')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',args.run):parser.error('Invalid run name')
    if args.dataset == 'all':
        # Resolve every sample before creating outputs or starting any child DAG.
        selections = {name: resolve_dataset(name, args.sample, args.dataset_config,
                      args.sample_root or args.output_root/'samples') for name in PHYSICS_DATASETS}
        manifests = []
        import copy
        for name in PHYSICS_DATASETS:
            child = copy.copy(args);child.dataset = name
            manifests.append(run_one(child, selections[name]))
        if not args.dry_run:
            out = args.output_root.resolve()/'all'/args.run
            out.mkdir(parents=True, exist_ok=True)
            write_json(out/'manifest.json', {'datasets': manifests, 'sample': args.sample})
            links = ''.join('<li><a href="../../'+name+'/'+args.run+'/plot/report.html">'+html.escape(name)+'</a></li>' for name in PHYSICS_DATASETS)
            (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Dataset reports</h1><p>Independent unit-normalized shape comparisons.</p><ul>'+links+'</ul>')
            print(f'Dataset index: {out / "index.html"}')
    else:
        run_one(args)


def run_one(args, prepared=None):
    user=getpass.getuser();work=Path('/afs/cern.ch/work')/user[0]/user
    builtin=args.algorithm in ('baseline','gap_clusters')
    algorithm=ROOT/'feature_dev/algorithms'/f'{args.algorithm}.py' if builtin else Path(args.algorithm).resolve()
    if not builtin and not args.schema:raise ValueError('External algorithms require --schema')
    definition=args.schema or ROOT/'feature_dev/schemas'/f'{args.algorithm}.yaml'
    dataset_info = None
    if args.dataset in DATASETS:
        sample, dataset_info = prepared or resolve_dataset(args.dataset, args.sample, args.dataset_config,
                                           args.sample_root or args.output_root/'samples')
    elif args.dataset == 'synthetic':
        if args.sample not in ('small','synthetic'):raise ValueError('Synthetic dataset has only the small fixture')
        sample=ROOT/'feature_dev/samples/small.jsonl'
        dataset_info={'name':'synthetic','sample':'small','version':'small-v1','normalization':'unit_area'}
    else:
        # Preserve existing commands, but keep the synthetic alias explicit in output.
        sample=ROOT/'feature_dev/samples/small.jsonl' if args.sample in ('small','synthetic') else Path(args.sample).resolve()
        if args.sample in ('small','synthetic'):print('Using the synthetic software fixture (legacy --sample small alias).')
    check_size([sample])
    _,sample_manifest=load_events(sample)
    parameters=json.loads(args.parameters)
    if not isinstance(parameters,dict):raise ValueError('--parameters must be a JSON object')
    source_files=[algorithm,ROOT/'convertData/scifi_plane_clustering.py',ROOT/'feature_dev/engine.py',ROOT/'feature_dev/selected_hits.py',*sorted((ROOT/'feature_dev/algorithms').glob('*.py')),ROOT/'feature_dev/cli.py',ROOT/'feature_dev/datasets.py',ROOT/'feature_dev/Snakefile',ROOT/'run_feature_dev.sh',*[p.resolve() for p in args.dependency]]
    hashes={str(p):sha(p) for p in source_files}
    hashes[str(sample)]=sha(sample);hashes[str(sample)+'.manifest.json']=sha(str(sample)+'.manifest.json')
    import numpy, uproot, yaml, matplotlib, platform
    runtime={m.__name__:m.__version__ for m in (numpy,uproot,yaml,matplotlib)}
    runtime['python']=platform.python_version()
    plan={'runtime':runtime,'sample':str(sample),'source_hashes':hashes,
          'baseline':{'algorithm':str(ROOT/'feature_dev/algorithms/baseline.py'),'parameters':{},'schema':schema(ROOT/'feature_dev/schemas/baseline.yaml')},
          'candidate':{'algorithm':str(algorithm),'parameters':parameters,'schema':schema(definition)}}
    if dataset_info is not None:plan['dataset']=dataset_info
    # Histogram settings do not enter the calculation signature.
    calculation={k:v for k,v in plan.items() if k not in ('baseline','candidate')}
    for name in ('baseline','candidate'):
        calculation[name]={**plan[name],'schema':{n:{k:v for k,v in d.items() if k not in ('histogram','description')} for n,d in plan[name]['schema']['features'].items()}}
    calc=digest(calculation);valid=digest([calc,'validation-v1'])
    hist=digest([valid,plan['candidate']['schema']]);plot=digest([hist,'report-v1'])
    keys=dict(calculate=calc,validate=valid,histogram=hist,plot=plot)
    output=args.output_root.resolve()
    if args.dataset:output=output/args.dataset
    run=output/args.run
    plan['paths']={name:str(output/'.cache'/name/key) for name,key in keys.items()}
    signature=digest(plan)
    manifest={'signature':signature,'stage_keys':keys,'plan':plan,'sample_provenance':sample_manifest}
    executable=os.environ.get('FEATURE_DEV_SNAKEMAKE')
    if not executable:
        local=work/'miniconda3/envs/snakemake/bin/snakemake'
        executable=str(local) if local.exists() else shutil.which('snakemake')
    if not executable:raise ValueError('Set FEATURE_DEV_SNAKEMAKE to a Snakemake executable')
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
        print(f'Features: {run / "calculate/features.root"}')
        return {'dataset':args.dataset,'manifest':str(run/'manifest.json'),'signature':signature}

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as error:raise SystemExit(str(error))
