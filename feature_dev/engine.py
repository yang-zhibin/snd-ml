"""Portable selected-hit calculation, validation, histograms and report stages."""
import copy
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'convertData'),str(ROOT/'feature_dev/algorithms')]
from selected_hits import sha
from scifi_plane_clustering import scalar_feature_dtypes


def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'w') as f:json.dump(data,f,allow_nan=False,sort_keys=True);f.write('\n')
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def schema(path,seen=None):
    import yaml
    path=Path(path).resolve();seen=set() if seen is None else set(seen)
    if path in seen:raise ValueError('Schema inheritance cycle')
    seen.add(path);cfg=yaml.safe_load(path.read_text());features={}
    if cfg.get('extends'):features=schema(path.parent/cfg['extends'],seen)['features']
    if cfg.get('inherit'):
        if cfg['inherit']!='scifi_orientation_clusters_v1':raise ValueError('Unknown inherited schema')
        features.update({n:{'dtype':'int64' if t=='i' else 'float64','description':n.replace('_',' '),'missing_value':-999} for n,t in scalar_feature_dtypes()})
    for name,settings in cfg.get('features',{}).items():features[name]={**features.get(name,{}),**settings}
    if not cfg.get('version') or not features:raise ValueError('Schema requires version and features')
    import re
    for name,s in features.items():
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',name) or name in {'runId','eventId','eventIndex','source_uid'}:raise ValueError('Invalid/reserved feature name')
        if s.get('dtype') not in {'int64','float64'} or not s.get('description') or 'missing_value' not in s:raise ValueError(f'Incomplete definition: {name}')
        if type(s['missing_value']) not in (int,float) or not math.isfinite(s['missing_value']):raise ValueError('Invalid missing sentinel')
        if s['dtype']=='int64' and (type(s['missing_value']) is not int or not -(2**63)<=s['missing_value']<2**63):raise ValueError('Invalid integer sentinel')
        if 'valid_range' in s and (len(s['valid_range'])!=2 or not all(math.isfinite(x) for x in s['valid_range']) or s['valid_range'][0]>s['valid_range'][1]):raise ValueError('Invalid feature range')
        h=s.get('histogram')
        if h and (type(h.get('bins')) is not int or h['bins']<=0 or len(h.get('range',[]))!=2 or not all(math.isfinite(x) for x in h['range']) or h['range'][0]>=h['range'][1]):raise ValueError('Invalid histogram settings')
    return {'version':cfg['version'],'features':features}


def event_key(event):return (event['source_uid'],*(event['event_id'][k] for k in ('runId','eventId','eventIndex')))
def load_events(sample):
    sample=Path(sample);manifest=read_json(str(sample)+'.manifest.json')
    if manifest.get('version')!=1 or manifest.get('complete') is not True or manifest.get('sha256')!=sha(sample):raise ValueError('Incomplete or modified selected-hit sample')
    events=[];seen=set()
    for line in sample.read_text().splitlines():
        e=json.loads(line);key=event_key(e)
        if key in seen:raise ValueError(f'Duplicate event {key}')
        seen.add(key)
        if not isinstance(e['source_uid'],str) or not isinstance(e.get('group'),str) or not e['group'] or not isinstance(e.get('geometry'),str):raise ValueError('Invalid source/group/geometry')
        if any(type(x) is not int for x in key[1:]):raise ValueError('Event IDs must be integers')
        channels=set()
        for h in e['hits']:
            if type(h.get('station')) is not int or h['station'] not in range(1,6) or type(h.get('isVertical')) is not bool or type(h.get('layer_channel')) is not int or not 0<=h['layer_channel']<1536:raise ValueError('Invalid station/orientation/channel')
            if type(h.get('qdc')) not in (int,float) or not math.isfinite(h['qdc']) or h['qdc']<0:raise ValueError('Invalid QDC')
            c=(h['station'],h['isVertical'],h['layer_channel'])
            if c in channels:raise ValueError('Duplicate hit channel')
            channels.add(c)
            for k in ('x_cm','y_cm','z_cm','time_ns'):
                if k in h and (type(h[k]) not in (int,float) or not math.isfinite(h[k])):raise ValueError('Invalid optional hit coordinate/time')
        events.append(e)
    if not events or len(events)!=manifest['events']:raise ValueError('Empty sample or count mismatch')
    return events,manifest


def load_algorithm(path):
    spec=importlib.util.spec_from_file_location('feature_plugin_'+sha(path)[:12],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in ('cluster_hits','calculate_features'):
        if not callable(getattr(module,name,None)):raise ValueError(f'Algorithm needs {name}')
    return module


def validate_values(values,definitions):
    if set(values)!=set(definitions):raise ValueError(f'Feature schema mismatch: missing={set(definitions)-set(values)}, extra={set(values)-set(definitions)}')
    for name,s in definitions.items():
        v=values[name]
        if type(v) not in (int,float) or not math.isfinite(v):raise ValueError(f'{name}: finite numeric value required')
        if s['dtype']=='int64' and (type(v) is not int or not -(2**63)<=v<2**63):raise ValueError(f'{name}: int64 required')
        if v==s['missing_value']:continue
        if 'valid_range' in s and not s['valid_range'][0]<=v<=s['valid_range'][1]:raise ValueError(f'{name}: outside valid range')


def evaluate(module,hits,parameters,definitions):
    # Algorithm sees selected hits only, never labels, truth, identity or geometry metadata.
    selected=copy.deepcopy(hits);before=copy.deepcopy(selected)
    clusters=module.cluster_hits(selected,copy.deepcopy(parameters))
    values=module.calculate_features(selected,clusters,copy.deepcopy(parameters))
    if selected!=before:raise ValueError('Algorithm mutated selected hits')
    validate_values(values,definitions)
    validator=getattr(module,'validate_algorithm',None)
    if validator is not None:validator(selected,clusters,values)
    return values


def calculate(plan):
    events,_=load_events(plan['sample']);records=[]
    modules={name:load_algorithm(plan[name]['algorithm']) for name in ('baseline','candidate')}
    for event in events:
        row={k:event[k] for k in ('event_id','source_uid','group','geometry')}
        for name in modules:
            try:row[name]=evaluate(modules[name],event['hits'],plan[name]['parameters'],plan[name]['schema']['features'])
            except Exception as error:raise ValueError(f'{name}, event {event_key(event)}: {error}') from error
        records.append(row)
    out=Path(plan['paths']['calculate']);out.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import uproot
    temporary=out/'features.tmp.root'
    with uproot.recreate(temporary) as f:
        for name in modules:
            types={k:v['dtype'] for k,v in plan[name]['schema']['features'].items()}
            types.update({k:'int64' for k in ('runId','eventId','eventIndex')})
            types['source_uid']='string'
            f.mktree(name,types)
            arrays={k:np.array([r[name][k] for r in records],dtype=v['dtype']) for k,v in plan[name]['schema']['features'].items()}
            arrays.update({k:np.array([r['event_id'][k] for r in records],dtype='int64') for k in ('runId','eventId','eventIndex')})
            arrays['source_uid']=[r['source_uid'] for r in records]
            f[name].extend(arrays)
    os.replace(temporary,out/'features.root');write_json(out/'features.json',records)


def validate(plan):
    events,_=load_events(plan['sample']);out=Path(plan['paths']['calculate']);records=read_json(out/'features.json')
    if [event_key(e) for e in events]!=[event_key(e) for e in records]:raise ValueError('Output event identity/order differs')
    import uproot
    import numpy as np
    with uproot.open(out/'features.root') as root:
        for name in ('baseline','candidate'):
            for r in records:validate_values(r[name],plan[name]['schema']['features'])
            arrays=root[name].arrays(library='np')
            for feature in plan[name]['schema']['features']:
                np.testing.assert_array_equal(arrays[feature],[r[name][feature] for r in records])
            for k in ('runId','eventId','eventIndex'):np.testing.assert_array_equal(arrays[k],[r['event_id'][k] for r in records])
            np.testing.assert_array_equal(arrays['source_uid'],[r['source_uid'] for r in records])
    write_json(Path(plan['paths']['validate'])/'validation.json',{'passed':True,'events':len(records),'features_sha256':sha(out/'features.json'),'root_sha256':sha(out/'features.root'),'checks':['identity/order','schema/types/ranges','finite values','algorithm-specific checks during calculation','ROOT/JSON agreement']})


def histograms(plan):
    import numpy as np
    receipt=read_json(Path(plan['paths']['validate'])/'validation.json');f=Path(plan['paths']['calculate'])/'features.json'
    if receipt['features_sha256']!=sha(f):raise ValueError('Features changed after validation')
    rows=read_json(f);groups=sorted({(r['group'],r['geometry']) for r in rows});result={}
    for feature,definition in plan['candidate']['schema']['features'].items():
        h=definition.get('histogram')
        if not h:continue
        curves=[]
        for name in ('baseline','candidate'):
            if feature not in plan[name]['schema']['features']:continue
            missing=plan[name]['schema']['features'][feature]['missing_value']
            for group,geometry in groups:
                all_values=[r[name][feature] for r in rows if (r['group'],r['geometry'])==(group,geometry)]
                values=[v for v in all_values if v!=missing]
                counts,edges=np.histogram(values,bins=h['bins'],range=h['range'])
                total=int(counts.sum())
                curves.append({'algorithm':name,'group':group,'geometry':geometry,'counts':counts.tolist(),'edges':edges.tolist(),
                    'fractions':(counts/total if total else counts.astype(float)).tolist(),'selected':len(all_values),'missing':len(all_values)-len(values),
                    'underflow':sum(v<h['range'][0] for v in values),'overflow':sum(v>h['range'][1] for v in values)})
        result[feature]={'description':definition['description'],'curves':curves}
    if not result:raise ValueError('No histogram definitions')
    write_json(Path(plan['paths']['histogram'])/'histograms.json',result)


def report(plan):
    import base64
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hist=read_json(Path(plan['paths']['histogram'])/'histograms.json');out=Path(plan['paths']['plot']);out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(str(plan['sample'])+'.manifest.json')
    parts=['<!doctype html><html><meta charset="utf-8"><title>Feature comparison</title><style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}</style><body>',
           '<h1>Selected-hit feature comparison</h1>', '<p>'+html.escape(str(manifest['provenance']))+'</p>',
           '<p>Same events for baseline and candidate. Groups remain separated by source label and geometry. Unit-normalized in-range bin fractions; no luminosity/yield normalization. The rightmost bin includes its upper edge.</p>',
           '<p>Sample SHA-256: '+sha(plan['sample'])+'</p>',
           '<p>Candidate: '+html.escape(plan['candidate']['algorithm'])+'; parameters: '+html.escape(json.dumps(plan['candidate']['parameters'],sort_keys=True))+'</p>',
           '<p>Validation passed: identity, finite values, schema, ROOT/JSON agreement and enabled algorithm checks.</p>']
    for feature,item in hist.items():
        groups=sorted({(c['group'],c['geometry']) for c in item['curves']});fig,axes=plt.subplots(len(groups),1,figsize=(9,3*len(groups)),squeeze=False)
        for ax,(group,geometry) in zip(axes[:,0],groups):
            for c in item['curves']:
                if (c['group'],c['geometry'])==(group,geometry):ax.stairs(c['fractions'],c['edges'],label=c['algorithm'],linewidth=1.8)
            ax.set_title(group+' | '+geometry);ax.set_ylabel('Fraction per bin');ax.legend()
        axes[-1,0].set_xlabel(feature);fig.tight_layout();p=out/(feature+'.png');fig.savefig(p,dpi=110);plt.close(fig)
        parts+=['<h2>'+html.escape(feature)+'</h2><p>'+html.escape(item['description'])+'</p>', '<img alt="'+html.escape(feature)+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">',
                '<table><tr><th>Algorithm/group/geometry</th><th>Selected</th><th>Missing</th><th>Underflow</th><th>Overflow</th></tr>']
        for c in item['curves']:parts.append('<tr><td>'+html.escape('/'.join(c[k] for k in ('algorithm','group','geometry')))+'</td>'+''.join('<td>'+str(c[k])+'</td>' for k in ('selected','missing','underflow','overflow'))+'</tr>')
        parts.append('</table>')
    parts.append('</body></html>');(out/'report.html').write_text('\n'.join(parts))


def stage(planfile,name):
    plan=read_json(planfile)
    for path,expected in plan['source_hashes'].items():
        if sha(path)!=expected:raise ValueError(f'Input changed after planning: {path}')
    {'calculate':calculate,'validate':validate,'histogram':histograms,'plot':report}[name](plan)
    out=Path(plan['paths'][name])
    write_json(out/'complete.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'})

if __name__=='__main__':stage(sys.argv[1],sys.argv[2])
