"""CERN adapters for the official cluster-generic HTCondor profile.

Deployed into a private profile. Snakemake runtime is minutes; MaxRuntime seconds.
No import of the repository, ROOT or SNDSW is needed on the submit host.
"""
import json
import math
from pathlib import Path
import re
import sys
import uuid


def properties(path):
    with open(path) as stream:
        for _, line in zip(range(100), stream):
            if line.startswith('# properties = '): return json.loads(line.split('=', 1)[1])
    raise ValueError('Missing Snakemake job properties')


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0: raise ValueError('Resource must be finite and positive')
    return math.ceil(value)


def description(jobscript, props, directory):
    resources = props.get('resources', {})
    # HTCondor new-style quoted arguments, including spaces and literal quotes.
    argument = str(Path(jobscript).resolve()).replace("'", "''").replace('"', '""')
    return dict(executable='/bin/bash', arguments='"\''+argument+'\'"',
        initialdir=str(Path.cwd()), log=str(directory/'condor.log'),
        output=str(directory/'condor.out'), error=str(directory/'condor.err'),
        should_transfer_files='NO', transfer_executable='False', getenv='True', max_retries='0',
        request_cpus=str(positive(props.get('threads',1))),
        request_memory=str(positive(resources.get('mem_mb',2500)))+'MB',
        request_disk=str(positive(resources.get('disk_mb',5000)))+'MB',
        **{'+MaxRuntime':str(positive(resources.get('runtime',60))*60),
           'requirements':'(TARGET.OpSysAndVer =?= "AlmaLinux9")', 'MY.SendCredential':'True'})


def parse_id(value):
    match = re.fullmatch(r'([0-9a-f]{32})_(\d+)', value)
    if not match: raise ValueError('Invalid job identifier')
    return match[1], int(match[2])


def event_status(events, types):
    state = 'running'
    for event in events:
        if event.type in (types.JOB_HELD, types.JOB_ABORTED, types.EXECUTABLE_ERROR): state='failed'
        elif event.type == types.JOB_TERMINATED:
            state='success' if event.get('TerminatedNormally', True) and event.get('ReturnValue',-1)==0 else 'failed'
    return state


def main():
    import htcondor2 as htcondor
    mode, logdir, *args = sys.argv[1:]; root = Path(logdir)
    if mode == 'submit':
        jobscript, = args
        token = uuid.uuid4().hex; directory = root/token; directory.mkdir(parents=True)
        props = properties(jobscript)
        spec = description(jobscript, props, directory)
        (directory/'submission.json').write_text(json.dumps(dict(properties=props,submit=spec),indent=2))
        htcondor.Credd().add_user_cred(htcondor.CredTypes.Kerberos, None)
        result = htcondor.Schedd().submit(htcondor.Submit(spec))
        cluster = result.cluster()
        (directory/'job.json').write_text(json.dumps(dict(cluster=cluster,jobscript=jobscript)))
        print(f'{token}_{cluster}')
    elif mode == 'status':
        token, cluster = parse_id(args[0]); path = root/token/'condor.log'
        if not path.exists(): print('running'); return
        log = htcondor.JobEventLog(str(path))
        print(event_status(log.events(stop_after=0), htcondor.JobEventType))
    elif mode == 'cancel':
        for value in args:
            token, cluster = parse_id(value)
            record = json.loads((root/token/'job.json').read_text())
            if record['cluster'] != cluster: raise ValueError('Job receipt mismatch')
            htcondor.Schedd().act(htcondor.JobAction.Remove, [str(cluster)+'.0'])
    else: raise ValueError('Unknown adapter mode')


if __name__ == '__main__': main()
