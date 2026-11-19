"""Stream production sndData columns into source/geometry-separated histograms."""
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
import shutil
import tempfile
from common import ROOT, canonical, read_json, write_json, sha, verify_receipt, groups_for_row
from exposure import validate_coverage, normalize_counts


def distributions(plan):
    import numpy as np
    import uproot
    from engine import schema
    base=Path(plan['directory'])
    definitions=dict(schema(ROOT/'feature_dev/schemas/baseline.yaml')['features'])
    definitions={k:v for k,v in definitions.items() if 'histogram' in v}
    definitions.update({k:v for k,v in plan['plugin']['features'].items() if 'histogram' in v})
    if plan['dataset']=='mc-neutrino':
        definitions['mc_incoming_energy']=dict(dtype='float64',description='MC incoming neutrino energy [GeV]',missing_value=-999,histogram=dict(bins=30,range=[0,1000]))
    curves={};counts_by_source=Counter();group_sources=defaultdict(set);coverage=[]
    for index,job in enumerate(plan['jobs']):
        directory=Path(plan['source_directory'])/str(index);verify_receipt(directory)
        receipt=read_json(directory/'validation.json')
        row=job['record'];key=job['source_key'];count=0
        if not receipt.get('validation_passed') or receipt['feature_sha256']!=sha(directory/'features.root'):
            raise ValueError('Source validation receipt mismatch: '+key)
        with uproot.open(directory/'features.root') as f:
            context=json.loads(str(f['production_exposure_json']))
            if context['campaign_sha256']!=plan['campaign_sha256'] or context['audit_source_id']!=key:
                raise ValueError('Source exposure identity mismatch')
            tree=f['sndData'];available=set(tree.keys());columns={'pdgCode'}
            for feature in definitions:
                columns.update(c for c in (feature,plan['plugin']['name']+'__'+feature) if c in available)
            for arrays in tree.iterate(sorted(columns),step_size='32 MB',library='np'):
                groups=np.asarray(groups_for_row(plan['dataset'],row,arrays['pdgCode']))
                count+=len(groups)
                for group in sorted(set(groups)):
                    geometry=row['source_geometry_id'];group_sources[(group,geometry)].add(key)
                    mask=groups==group
                    for feature,definition in definitions.items():
                        for algorithm,column in [('production',feature),('plugin',plan['plugin']['name']+'__'+feature)]:
                            if column not in arrays:continue
                            h=definition['histogram'];values=arrays[column][mask]
                            good=values[values!=definition['missing_value']]
                            bins,edges=np.histogram(good,bins=h['bins'],range=h['range'])
                            curve_key=(feature,algorithm,group,geometry)
                            if curve_key not in curves:
                                curves[curve_key]=dict(counts=np.zeros(h['bins'],dtype=np.int64),edges=edges.tolist(),selected=0,missing=0,underflow=0,overflow=0)
                            curve=curves[curve_key];curve['counts']+=bins;curve['selected']+=len(values)
                            curve['missing']+=len(values)-len(good)
                            curve['underflow']+=int(np.sum(good<h['range'][0]));curve['overflow']+=int(np.sum(good>h['range'][1]))
        if count!=receipt['written_event_count']:raise ValueError('Coverage count mismatch: '+key)
        counts_by_source[key]=count
        coverage.append(dict(source_uid=key,original_source_uid=row['source_uid'],selected_entries=count,
                             processed_ranges=[[0,count]] if count else [],exposure=context,
                             validation_sha256=sha(directory/'validation.json'),root_sha256=sha(directory/'features.root')))
    normalizations={key:dict(mode='shape',group=key[0],geometry=key[1]) for key in group_sources}
    if plan['normalization']=='yield':
        audit=dict(plan['exposure_audit'],sample_sha256=plan['campaign_sha256'])
        records=validate_coverage(audit,plan['campaign_sha256'],counts_by_source,group_sources,{})
        roles={p['role'] for p in audit['populations'].values()}-{'reference_data'}
        if roles!=({'data'} if plan['dataset']=='real-data' else {'mc'}):raise ValueError('Wrong audit dataset role')
        normalizations={(r['group'],r['geometry']):r for r in records}
    result={}
    for (feature,algorithm,group,geometry),curve in sorted(curves.items()):
        curve['counts']=curve['counts'].tolist()
        curve.update(algorithm=algorithm,group=group,geometry=geometry)
        curve.update(normalize_counts(curve['counts'],normalizations[(group,geometry)],curve['underflow'],curve['overflow']))
        result.setdefault(feature,dict(description=definitions[feature]['description'],curves=[]))['curves'].append(curve)
    destination=base/'plots'
    if destination.exists():
        if (destination/'complete.json').exists():
            verify_receipt(destination);return
        destination.rmdir() # refuse unrecognized existing partial products
    with tempfile.TemporaryDirectory(prefix='.plots-',dir=base) as temporary:
        out=Path(temporary)
        write_json(out/'histograms.json',result)
        write_json(out/'coverage.json',dict(version=1,campaign_sha256=plan['campaign_sha256'],sources=coverage,
                   groups=[dict(group=g,geometry=z,sources=sorted(ids)) for (g,z),ids in sorted(group_sources.items())],
                   complete_selected_catalogue=True,sample=plan['sample']))
        write_json(out/'normalization.json',list(normalizations.values()))
        render(result,out,plan)
        write_json(out/'complete.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
        out.rename(destination)


def render(histograms,out,plan):
    import base64
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parts=['<!doctype html><meta charset="utf-8"><title>Production features</title>',
           '<h1>Production features: '+html.escape(plan['plugin']['name'])+'</h1>',
           '<p>Dataset: '+html.escape(plan['dataset'])+'; normalization: '+plan['normalization']+'. Geometry strata remain separate.</p>']
    for feature,item in histograms.items():
        keys=sorted({(c['group'],c['geometry']) for c in item['curves']})
        # One bounded figure per group/geometry, even for a large campaign.
        for index,key in enumerate(keys):
            fig,ax=plt.subplots(figsize=(8,4))
            selected=[c for c in item['curves'] if (c['group'],c['geometry'])==key]
            for curve in selected:
                ax.stairs(curve['normalized'],curve['edges'],label=curve['algorithm'])
            ax.set_xlabel(item['description']);ax.set_ylabel('Events per bin' if plan['normalization']=='yield' else 'Fraction per bin')
            ax.set_title(key[0]);ax.legend();fig.tight_layout()
            path=out/(feature+'-'+str(index)+'.png');fig.savefig(path,dpi=100);plt.close(fig)
            parts+=['<h2>'+html.escape(feature+' / '+key[0])+'</h2>', '<p>Geometry: '+html.escape(key[1])+'</p>',
                    '<pre>'+html.escape(json.dumps([dict(algorithm=c['algorithm'],normalization=c['normalization'],selected=c['selected'],missing=c['missing'],underflow=c['underflow'],overflow=c['overflow']) for c in selected]))+'</pre>',
                    '<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">']
    (out/'report.html').write_text('\n'.join(parts)+'\n')
