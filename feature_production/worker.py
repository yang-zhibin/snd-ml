"""Bounded-memory production workers; converter/validator run in isolated SNDSW."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from common import ROOT, canonical, read_json, write_json, sha, check_code, verify_receipt, fingerprint
from sources import converter_arguments, source_group, verify_inputs


def source_inputs(job):
    row=job['record']
    inputs={name:fingerprint(row[name],content=name!='digi_path')
            for name in ('digi_path','geo_path','cutflow_path') if row.get(name)}
    if row['selection_mode']!='direct_input' and 'cutflow_path' not in inputs:
        raise ValueError('Missing cutflow: '+job['source_key'])
    expected=row.get('selection_index_sha256')
    if expected and inputs['cutflow_path']['sha256']!=expected:
        raise ValueError('Changed frozen selection index: '+job['source_key'])
    return inputs


def preflight(plan):
    check_code(plan)
    for job in plan['jobs']:source_inputs(job)
    for output in (plan['directory'],plan['source_directory']):
        destination=Path(output)
        while not destination.exists():destination=destination.parent
        if not os.access(destination,os.W_OK|os.X_OK):raise ValueError('Output parent is not writable: '+str(destination))
    subprocess.run(['bash',str(ROOT/'feature_dev/run_export.sh'),'-c',
                    'import ROOT, SndlhcGeo; print("SNDSW account preflight: PASS")'],check=True)
    print('Input and output account access: PASS; batch submission is not tested')


def exposure_context(plan,job):
    row=job['record']
    lumi=float(row.get('source_lumi_per_file') or row.get('lumi_per_file') or 0)
    probability=float(row.get('sampling_probability') or 1)
    kind=row.get('exposure_kind','unknown')
    # Original exposure is preserved, but event_count_only is not approved for yields.
    valid=plan['sample']=='full' and kind in ('simulated_luminosity_fb','recorded_luminosity_fb') and math.isfinite(lumi) and lumi>0 and 0<probability<=1
    return dict(version=1,source_uid=row['source_uid'],audit_source_id=job['source_key'],
                parent_luminosity_fb=lumi if math.isfinite(lumi) else None,
                sampling_probability=probability,exposure_kind=kind,exposure_valid=valid,
                effective_luminosity_fb=lumi*probability if valid else None,
                campaign_sha256=plan['campaign_sha256'],normalization=plan['normalization'],
                coverage_status='requires_campaign_validation',units='fb^-1',
                exposure_audit=plan.get('exposure_audit'))


def attach_exposure(path,context):
    import numpy as np
    import uproot
    with uproot.update(path) as f:
        f['production_exposure_json']=canonical(context)
        f.mktree('source_exposure',dict(source_uid='string',audit_source_id='string',luminosity_fb='float64',
                                     sampling_probability='float64',effective_luminosity_fb='float64',exposure_valid='bool'))
        f['source_exposure'].extend(dict(source_uid=[context['source_uid']],audit_source_id=[context['audit_source_id']],
            luminosity_fb=np.array([context['parent_luminosity_fb'] if context['parent_luminosity_fb'] is not None else float('nan')]),
            sampling_probability=np.array([context['sampling_probability']]),
            effective_luminosity_fb=np.array([context['effective_luminosity_fb'] if context['effective_luminosity_fb'] is not None else float('nan')]),
            exposure_valid=np.array([context['exposure_valid']])))
    with uproot.open(path) as f:
        if str(f['production_exposure_json'])!=canonical(context):raise ValueError('Exposure metadata mismatch')


def validator_arguments(plan,job,path,output,receipt):
    row=job['record']
    command=[str(ROOT/'convertData/validate_unified_shower_feature_file.py'),str(path),
        '--manifest-version','production_plugins_v1','--expected-source-uid',row['source_uid'],
        '--expected-source-domain-id',str(row['source_domain_id']),
        '--expected-source-geometry-id',row['source_geometry_id'],
        '--expected-input-event-count',str(int(float(row['input_event_count']))),
        '--sample-type',row['sample_type'],'--selection-mode',row['selection_mode'],
        '--input-tree',row['input_tree'],'--expected-max-events',str(job['max_events']),
        '--input-digi-path',row['digi_path'],'--published-feature-path',str(output),
        '--converter-commit',plan['implementation_commit'],'--validation-record',str(receipt),
        '--feature-plugin-spec',str(Path(plan['directory'])/'plugin.json')]
    for field,flag in [('cutflow_path','--input-cutflow-path'),('source_geometry_checksum','--expected-source-geometry-checksum')]:
        if row.get(field):command += [flag,row[field]]
    if row.get('selected_event_count'):command+=['--expected-selected-event-count',str(int(float(row['selected_event_count'])))]
    if int(row['source_domain_id']) in (4,5,6):
        command+=['--expected-source-pdg-code',str(int(float(row['source_pdg_code']))),'--expected-source-energy-gev',str(float(row['source_energy_gev']))]
    return command


def source(plan,index):
    job=plan['jobs'][index];row=job['record'];base=Path(plan['directory']);destination=Path(plan['source_directory'])/str(index)
    if (destination/'complete.json').exists():
        verify_receipt(destination)
        verify_inputs({'inputs':read_json(destination/'inputs.json')})
        return
    if plan.get('reuse_sources'):
        raise ValueError('Missing immutable reused source product')
    inputs=source_inputs(job)
    if job.get('inputs') and inputs!=job['inputs']:raise ValueError('Input changed since planning')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():destination.rmdir() # only Snakemake's empty placeholder
    log=base/'logs'/('source-'+str(index)+'.log');log.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+str(index)+'-',dir=destination.parent) as temporary:
        temporary=Path(temporary)
        converter_job=dict(job,group=source_group(plan['dataset'],row))
        args=converter_arguments(converter_job,temporary)
        if plan['sample']=='full':
            pos=args.index('--feature-dev-export');del args[pos:pos+2]
        args+=['--feature-plugin-spec',str(base/'plugin.json')]
        with log.open('w') as stream:
            command=['bash',str(ROOT/'feature_dev/run_export.sh'),*map(str,args)]
            stream.write('Command: '+repr(command)+'\n');stream.flush()
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            attach_exposure(temporary/'features.root',exposure_context(plan,job))
            command=['bash',str(ROOT/'feature_dev/run_export.sh'),*validator_arguments(plan,job,temporary/'features.root',destination/'features.root',temporary/'validation.json')]
            stream.write('Validation: '+repr(command)+'\n');stream.flush()
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if plan['sample']=='small':
            from prepare import validate_export
            write_json(temporary/'baseline-parity.json',validate_export(temporary/'hits.jsonl',temporary/'features.root'))
        verify_inputs({'inputs':inputs})
        write_json(temporary/'inputs.json',inputs)
        write_json(temporary/'complete.json',{p.name:sha(p) for p in temporary.iterdir() if p.is_file()})
        os.replace(temporary,destination)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('plan',type=Path);parser.add_argument('stage',choices=['source','ranges','histograms','plots','distributions'])
    parser.add_argument('index',nargs='?',type=int)
    args=parser.parse_args();plan=read_json(args.plan);check_code(plan)
    if args.stage=='source':
        if args.index is None or not 0<=args.index<len(plan['jobs']):raise ValueError('Invalid source index')
        source(plan,args.index)
    else:
        from distributions import distributions, resolve, fill, render
        {'ranges':resolve,'histograms':fill,'plots':render,'distributions':distributions}[args.stage](plan)

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as error:raise SystemExit(str(error))
