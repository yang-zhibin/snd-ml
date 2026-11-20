"""Plan a per-user production campaign and run its Snakemake DAG."""
import argparse
import csv
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from collections import defaultdict
from common import ROOT, digest, fingerprint, plugin_spec, source_key, stratum, read_json, write_json, sha
from cli import find_snakemake
from sources import build_plan
from plot_settings import load_settings


def catalogue_rows(dataset, config_path):
    import yaml
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get('version') != 1:
        raise ValueError('Dataset registry version must be 1')
    entry = config['datasets'][dataset]
    catalogues, rows, seen = [], [], set()
    from datasets import SAMPLE_TYPES
    for location in entry['sources']['catalogues']:
        path = Path(location).expanduser()
        if not path.is_absolute():path=Path(config_path).resolve().parent/path
        catalogues.append(fingerprint(path,content=True))
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['sample_type'].lower() not in SAMPLE_TYPES[dataset]:
                    raise ValueError('Wrong sample type in catalogue')
                key=source_key(row)
                if not row['source_uid'] or key in seen:raise ValueError('Duplicate catalogue source: '+key)
                seen.add(key);rows.append(row)
    if not rows:raise ValueError('No catalogue sources')
    return catalogues, rows


def make_plan(args):
    spec=plugin_spec(args.plugin,json.loads(args.parameters))
    catalogues, all_rows=catalogue_rows(args.dataset,args.dataset_config)
    if args.sample=='small':
        prepared=build_plan(args.dataset,args.dataset_config,args.output_root,args.files_per_group,args.events_per_file)
        jobs=[dict(j) for j in prepared['jobs']]
    else:
        jobs=[dict(record=r,max_events=0) for r in all_rows]  # includes empty sources
    for job in jobs:
        row=job['record']
        if not row.get('source_geometry_id'):
            row['source_geometry_id']=row['geo_path']
        job['source_key']=source_key(row)
    selection=dict(dataset=args.dataset,sample=args.sample,catalogues=catalogues,
                   jobs=[dict(source_key=j['source_key'],record=j['record'],max_events=j['max_events']) for j in jobs])
    campaign_sha=digest(selection)
    audit=read_json(args.exposure_manifest) if args.exposure_manifest else None
    shape_only=args.sample=='small' or args.dataset=='testbeam'
    mode='shape' if args.normalization=='shape' or (args.normalization=='auto' and shape_only) else 'yield'
    if mode=='yield' and shape_only:raise ValueError('Capped/testbeam samples require shape normalization')
    if audit is not None and mode!='yield':raise ValueError('Exposure audit is only used for yield runs')
    if mode=='yield':
        if not audit or audit.get('campaign_sha256')!=campaign_sha:
            raise ValueError('Yield requires --exposure-manifest with matching campaign_sha256='+campaign_sha+'; first run --normalization shape to create coverage.json')
    settings=load_settings(getattr(args,"hist_config",None),getattr(args,"plot_config",None))
    code_paths=[*sorted((ROOT/'convertData').glob('*.py')), *sorted((ROOT/'feature_production').glob('*.py')),
                ROOT/'feature_production/Snakefile',ROOT/'feature_dev/run_export.sh',ROOT/'run_shower_features.sh',
                ROOT/'snakemake/envs/env_sndsw_nue.sh',ROOT/'snakemake/envs/env_lcg.sh',
                *[ROOT/'evaluation_region_partitions'/name for name in ('make_group_hist.py','make_group_hist2d.py','plot_comparison.py','plot_group_hist2d.py')],
                ROOT/'feature_dev/exposure.py',ROOT/'feature_dev/sources.py',ROOT/'feature_dev/datasets.py',
                ROOT/'feature_dev/engine.py',ROOT/'feature_dev/selected_hits.py',ROOT/'feature_dev/prepare.py',ROOT/'feature_dev/schemas/baseline.yaml',ROOT/'evaluation_region_partitions/normalization.py',
                *map(Path,spec['source_hashes'])]
    plan=dict(version=1,implementation_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),**{k:v for k,v in selection.items() if k!='jobs'},campaign_sha256=campaign_sha,plugin=spec,jobs=jobs,
              normalization=mode,exposure_audit=audit,plot_settings=settings,code={str(p):sha(p) for p in sorted(set(code_paths))})
    conversion_code={p:h for p,h in plan['code'].items() if '/convertData/' in p or p in spec['source_hashes'] or
                     p.endswith(('run_export.sh','env_sndsw_nue.sh','sources.py','selected_hits.py'))}
    plan['conversion_signature']=digest(dict(selection=selection,plugin=spec,code=conversion_code,
                                              normalization=mode,audit=audit))
    histogram_code={p:h for p,h in plan['code'].items() if not p.endswith(('plot_comparison.py','plot_group_hist2d.py'))}
    plan['histogram_signature']=digest(dict(conversion=plan['conversion_signature'],settings=settings['histograms'],code=histogram_code))
    plan['plot_signature']=digest(dict(histograms=plan['histogram_signature'],settings=settings['plotting'],code=plan['code']))
    # Execution profile/output directories are not physics settings.
    plan['signature']=digest(plan)
    plan['directory']=str(args.output_root.resolve()/args.dataset/args.run)
    plan['source_directory']=str((args.feature_root.resolve()/args.dataset/args.run/'sources') if args.feature_root else Path(plan['directory'])/'sources')
    if getattr(args,'reuse_sources',None):
        previous=read_json(args.reuse_sources/'plan.json')
        for key in ('campaign_sha256','plugin','normalization','exposure_audit'):
            if previous.get(key)!=plan.get(key):raise ValueError('Incompatible reusable sources: '+key)
        if any(previous['code'].get(p)!=h for p,h in conversion_code.items()):
            raise ValueError('Converter code changed since reusable source run')
        plan['source_directory']=previous['source_directory']
        plan['reuse_sources']=str(args.reuse_sources.resolve())
        if previous.get('histogram_signature')==plan['histogram_signature']:
            plan['reuse_histograms']=str(args.reuse_sources.resolve())
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['mc-neutrino','testbeam','real-data','mc-muon'],required=True)
    parser.add_argument('--plugin',default='silvia',help='Plugin name or directory')
    parser.add_argument('--parameters',default='{}')
    parser.add_argument('--sample',choices=['small','full'],default='small')
    parser.add_argument('--run',required=True)
    user=getpass.getuser();work=Path('/afs/cern.ch/work')/user[0]/user
    parser.add_argument('--output-root',type=Path,default=work/'snd-ml-outputs/shower-features')
    parser.add_argument('--feature-root',type=Path,help='Optional writable EOS root for source ROOT products; logs/plans remain under output-root')
    parser.add_argument('--dataset-config',type=Path,default=ROOT/'feature_dev/datasets.yaml')
    parser.add_argument('--hist-config',type=Path)
    parser.add_argument('--plot-config',type=Path)
    parser.add_argument('--reuse-sources',type=Path,help='Reuse validated source ROOTs from an existing compatible run directory')
    parser.add_argument('--files-per-group',type=int,default=1)
    parser.add_argument('--events-per-file',type=int,default=20)
    parser.add_argument('--normalization',choices=['auto','shape','yield'],default='auto')
    parser.add_argument('--exposure-manifest',type=Path)
    parser.add_argument('--profile',type=Path,help='Snakemake HTCondor profile, required for full execution')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--check-access',action='store_true',help='Read-only preflight every chosen input and runtime; do not convert')
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',args.run):parser.error('Invalid run name')
    if args.profile and not args.profile.is_dir():parser.error('Missing profile directory')
    if args.sample=='full' and not args.profile and not (args.dry_run or args.check_access):
        parser.error('Full production requires --profile; use --dry-run first')
    if args.sample=='full' and args.profile and not args.dry_run:
        import yaml
        configs=list(args.profile.glob('config*.yaml'))
        #if not any((yaml.safe_load(p.read_text()) or {}).get('executor')=='htcondor' for p in configs):
        if not any((yaml.safe_load(p.read_text()) or {}).get('executor') in ('htcondor', 'cluster-generic') for p in configs):
            parser.error('Full execution requires a profile with executor: htcondor')
    plan=make_plan(args)
    print('Campaign identity:',plan['campaign_sha256'])
    print('Sources:',len(plan['jobs']),'Output:',plan['directory'])
    if args.check_access:
        from worker import preflight
        preflight(plan)
        return
    executable=find_snakemake(work)
    def execute(path):
        command=[executable,'--snakefile',str(ROOT/'feature_production/Snakefile'),'--cores','1',
                 '--rerun-triggers','mtime','--config','plan='+str(path),'wrapper='+str(ROOT/'run_shower_features.sh')]
        if args.profile:command+=['--profile',str(args.profile.resolve())]
        if args.dry_run:command+=['--dry-run']
        env=os.environ.copy();env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
        subprocess.run(command,cwd=path.parent,env=env,check=True)
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix='production-plan-') as temporary:
            path=Path(temporary)/'plan.json';write_json(path,plan);execute(path)
        return
    import fcntl
    args.output_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/'.production.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run=Path(plan['directory'])
        if (run/'plan.json').exists():
            old=read_json(run/'plan.json')
            if old!=plan:raise ValueError('Run already exists with different inputs; choose a new --run')
        elif run.exists():
            raise ValueError('Unrecognized existing run directory: '+str(run))
        from common import verify_receipt
        if plan.get('reuse_sources'):
            for index in range(len(plan['jobs'])):
                verify_receipt(Path(plan['source_directory'])/str(index))
        # Verify reusable products before scheduling.
        from common import verify_receipt
        for receipt in Path(plan['source_directory']).glob('*/complete.json'):
            verify_receipt(receipt.parent)
            from sources import verify_inputs
            verify_inputs({'inputs':read_json(receipt.parent/'inputs.json')})
        if (run/'plots/complete.json').exists():verify_receipt(run/'plots')
        run.mkdir(parents=True,exist_ok=True)
        write_json(run/'plan.json',plan);write_json(run/'plugin.json',plan['plugin'])
        (run/'logs').mkdir(exist_ok=True)
        if plan.get('reuse_histograms'):
            previous=Path(plan['reuse_histograms'])
            import shutil, yaml
            for stage in ('config','histograms'): verify_receipt(previous/stage)
            if not (run/'config').exists():
                shutil.copytree(previous/'config',run/'config')
                (run/'config/plotting.yaml').write_text(yaml.safe_dump(plan['plot_settings']['plotting']))
                write_json(run/'config/complete.json',{str(p.relative_to(run/'config')):sha(p)
                    for p in (run/'config').rglob('*') if p.is_file() and p.name!='complete.json'})
            if not (run/'histograms').exists():
                (run/'histograms').mkdir()
                for p in (previous/'histograms').iterdir():
                    if p.name!='complete.json': (run/'histograms'/p.name).symlink_to(p,target_is_directory=p.is_dir())
                write_json(run/'histograms/complete.json',read_json(previous/'histograms/complete.json'))
        execute(run/'plan.json')
        print('ROOT-backend PDFs and CSV summaries:',run/'plots')
        print('Production ROOT files:',plan['source_directory'])

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as error:raise SystemExit(str(error))
