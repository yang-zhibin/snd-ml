"""ROOT production histograms and plots using the existing EM shower backend."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import csv
import json
import math
import re
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from common import ROOT as REPO, read_json, write_json, sha, verify_receipt, groups_for_row
from exposure import validate_coverage
from plot_settings import definitions, geometry_key, group_key, resolve_axis, safe_name

sys.path.insert(0, str(REPO/'evaluation_region_partitions'))
NEUTRINOS = {'CC_nue':12, 'CC_numu':14, 'NC_nue':112, 'NC_numu':114}


@contextmanager
def publication(destination):
    """Publish a complete directory, including recursive checksums, atomically."""
    destination=Path(destination)
    if destination.exists():
        destination.rmdir()  # only an empty Snakemake-created placeholder is allowed
    with tempfile.TemporaryDirectory(prefix='.'+destination.name+'-',dir=destination.parent) as tmp:
        out=Path(tmp)
        yield out
        write_json(out/'complete.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
        out.rename(destination)


def coverage(plan):
    """Validate full parent coverage before applying any additional plot selection."""
    import uproot
    counts=Counter(); memberships=defaultdict(set); sources=[]; strata={}; schemas=[]
    for index,job in enumerate(plan['jobs']):
        directory=Path(plan['source_directory'])/str(index)
        verify_receipt(directory)
        receipt=read_json(directory/'validation.json'); path=directory/'features.root'
        if not receipt.get('validation_passed') or receipt['feature_sha256']!=sha(path):
            raise ValueError('Source validation receipt mismatch: '+job['source_key'])
        row=job['record']; key=job['source_key']; geometry=row['source_geometry_id']
        with uproot.open(path) as f:
            context=json.loads(str(f['production_exposure_json']))
            if context['campaign_sha256']!=plan['campaign_sha256'] or context['audit_source_id']!=key:
                raise ValueError('Source exposure identity mismatch')
            schemas.append((str(path),set(f['sndData'].keys())))
            counts[key]=0
            observed=set()
            for arrays in f['sndData'].iterate(['pdgCode'],step_size='32 MB',library='np'):
                groups=groups_for_row(plan['dataset'],row,arrays['pdgCode'])
                counts[key]+=len(groups); observed.update(groups)
            for group in observed: memberships[(group,geometry)].add(key)
        if counts[key]!=receipt['written_event_count']:
            raise ValueError('Coverage count mismatch: '+key)
        groups=list(NEUTRINOS) if plan['dataset']=='mc-neutrino' else groups_for_row(plan['dataset'],row,[0])
        for group in groups:
            stratum=strata.setdefault((group,geometry),dict(group=group,geometry=geometry,
                                      geometry_key=geometry_key(geometry),paths=[],source_ids=[]))
            stratum['paths'].append(str(path));stratum['source_ids'].append(key)
        sources.append(dict(source_uid=key,selected_entries=counts[key],processed_ranges=[[0,counts[key]]] if counts[key] else [],
                            exposure=context,root_sha256=sha(path),validation_sha256=sha(directory/'validation.json')))
    features,_=specifications(plan)
    expressions=[str(v.get('expression',''))+' '+str(v.get('feature_cut','')) for v in features.values()]
    expressions+=list(map(str,plan['plot_settings']['histograms']['selections'].values()))
    expressions+=[str(v.get('feature_cut','')) for v in plan['plot_settings']['histograms'].get('features2d',{}).values()]
    tokens=set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*',' '.join(expressions)))
    known=set().union(*(cols for _,cols in schemas))
    required=(tokens & known) | {'pdgCode'}
    for name,cfg in features.items():
        expression=str(cfg['expression'])
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',expression): required.add(expression)
    for path,columns in schemas:
        missing=required-columns
        if missing: raise ValueError('Missing plot branches in '+path+': '+', '.join(sorted(missing)))
    norms={(g,z):dict(mode='shape') for g,z in strata}
    if plan['normalization']=='yield':
        if plan['sample']!='full' or plan['dataset']=='testbeam':
            raise ValueError('Capped/testbeam samples require shape normalization')
        audit=dict(plan['exposure_audit'],sample_sha256=plan['campaign_sha256'])
        records=validate_coverage(audit,plan['campaign_sha256'],counts,memberships,{})
        roles={p['role'] for p in audit['populations'].values()}-{'reference_data'}
        if roles!=({'data'} if plan['dataset']=='real-data' else {'mc'}):
            raise ValueError('Wrong audit dataset role')
        norms={(r['group'],r['geometry']):r for r in records}
        # Plot absent classes explicitly; never invent a nonzero exposure for them.
        for key in strata:
            if key not in norms:
                norms[key]=dict(mode='yield',scale=0.,empty_parent_group=True)
    return dict(sources=sources,strata=list(strata.values()),
                normalizations=[dict(group=g,geometry=z,**{k:v for k,v in r.items() if k not in ('group','geometry')}) for (g,z),r in norms.items()],
                campaign_sha256=plan['campaign_sha256'],sample=plan['sample'],complete_selected_catalogue=True)


def frame(plan,stratum,selection):
    import ROOT
    paths=ROOT.std.vector('string')()
    for path in stratum['paths']: paths.push_back(path)
    df=ROOT.RDataFrame('sndData',paths)
    if plan['dataset']=='mc-neutrino':
        df=df.Filter('abs(pdgCode) == '+str(NEUTRINOS[stratum['group']]))
    return df.Filter(str(selection))


def feature_frame(df,name,config):
    column='__plot_'+name
    if config.get('feature_cut'): df=df.Filter(str(config['feature_cut']))
    df=df.Define(column,'static_cast<double>('+str(config['expression'])+')')
    selected=df.Count()
    cut='std::isfinite('+column+')'
    if config.get('missing_value') is not None:
        cut+=' && '+column+' != '+str(float(config['missing_value']))
    return df.Filter(cut),column,selected


def pair_frame(df,name,pair,features):
    if pair.get('feature_cut'): df=df.Filter(str(pair['feature_cut']))
    xcfg=features[pair['x']];ycfg=features[pair['y']]
    # Both axis selections precede the denominator for missing-value accounting.
    for cfg in (xcfg,ycfg):
        if cfg.get('feature_cut'): df=df.Filter(str(cfg['feature_cut']))
    selected=df.Count()
    df,x,_=feature_frame(df,name+'_x',dict(xcfg,feature_cut=''))
    df,y,_=feature_frame(df,name+'_y',dict(ycfg,feature_cut=''))
    return df,x,y,selected


def specifications(plan):
    cfg=plan['plot_settings']['histograms']; features=definitions(plan)
    pairs=cfg.get('features2d',{})
    for name,pair in pairs.items():
        safe_name(name)
        if pair['x'] not in features or pair['y'] not in features:
            raise ValueError('2D pair must refer to configured features: '+name)
        # Avoid pathological allocations before submitting a campaign.
        if pair.get('x_bins',50)*pair.get('y_bins',50)>1000000:
            raise ValueError('2D histogram exceeds one million bins')
    return features,pairs


def resolve(plan):
    base=Path(plan['directory']); destination=base/'config'
    if (destination/'complete.json').exists(): verify_receipt(destination); return
    evidence=coverage(plan); features,pairs=specifications(plan)
    cfg=plan['plot_settings']['histograms']; extrema={}
    for stratum in evidence['strata']:
        for selection,expression in cfg['selections'].items():
            df=frame(plan,stratum,expression); actions=[]
            for name,feature in features.items():
                valid,col,_=feature_frame(df,name,feature)
                actions.append((('1d',name,selection,stratum['geometry_key'],'x'), valid.Count(),valid.Min(col),valid.Max(col)))
            for name,pair in pairs.items():
                valid,x,y,_=pair_frame(df,name,pair,features)
                for axis,col in [('x',x),('y',y)]:
                    actions.append((('2d',name,selection,stratum['geometry_key'],axis),valid.Count(),valid.Min(col),valid.Max(col)))
            # All actions on this graph are booked before the first GetValue.
            for key,count,low,high in actions:
                count=int(count.GetValue()); stats=extrema.setdefault(key,[0,None,None])
                if count:
                    lo=float(low.GetValue());hi=float(high.GetValue())
                    stats[1]=lo if stats[1] is None else min(stats[1],lo)
                    stats[2]=hi if stats[2] is None else max(stats[2],hi)
                    stats[0]+=count
    resolved={}
    for (kind,name,selection,geometry,axis),stats in extrema.items():
        key='/'.join((kind,name,selection,geometry))
        if kind=='1d':
            resolved[key]=dict(features[name],**resolve_axis(features[name],*stats))
        else:
            pair=pairs[name]; original=features[pair[axis]]
            axiscfg=dict(dtype=original.get('dtype'),bins=pair.get(axis+'_bins',50),range=pair.get(axis+'_range'),
                         logx=pair.get('log'+axis,False))
            limits=resolve_axis(axiscfg,*stats)
            entry=resolved.setdefault(key,dict(pair))
            entry.update({axis+'_min':limits['x_min'],axis+'_max':limits['x_max'],axis+'_bins':limits['bins'],
                          axis+'_title':pair.get(axis+'_title',original['axis_title']),axis+'_resolution':limits})
    with publication(destination) as out:
        import yaml
        for name,data in plan['plot_settings'].items():
            (out/(name+'.yaml')).write_text(yaml.safe_dump(data,sort_keys=False))
        (out/'resolved.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False))
        write_json(out/'resolved.json',resolved)
        write_json(out/'coverage.json',evidence)


def fill(plan):
    import ROOT
    from make_group_hist import book_histogram, finalize_histogram, write_metadata
    from make_group_hist2d import book_histogram2d, finalize_histogram2d
    base=Path(plan['directory']);verify_receipt(base/'config')
    if (base/'histograms/complete.json').exists():verify_receipt(base/'histograms');return
    resolved=read_json(base/'config/resolved.json');evidence=read_json(base/'config/coverage.json')
    features,pairs=specifications(plan);summary=[]
    with publication(base/'histograms') as out:
        for stratum in evidence['strata']:
            for selection,expression in plan['plot_settings']['histograms']['selections'].items():
                df=frame(plan,stratum,expression);pending=[]
                for name in features:
                    key='/'.join(('1d',name,selection,stratum['geometry_key']));cfg=resolved[key]
                    valid,column,selected=feature_frame(df,name,cfg)
                    pending.append((key,cfg,selected,valid.Count(),book_histogram(valid,name,cfg,column)))
                for name,pair in pairs.items():
                    key='/'.join(('2d',name,selection,stratum['geometry_key']));cfg=resolved[key]
                    valid,x,y,selected=pair_frame(df,name,pair,features)
                    pending.append((key,cfg,selected,valid.Count(),book_histogram2d(valid,name,cfg,x,y)))
                for key,cfg,selected,valid,result in pending:
                    count=int(valid.GetValue()); total=int(selected.GetValue())
                    is2d=key.startswith('2d/')
                    if is2d: hist,profile=finalize_histogram2d(result,cfg)
                    else: hist=finalize_histogram(result,key.split('/')[1],cfg)
                    path=out/key/(group_key(stratum['group'])+'.root');path.parent.mkdir(parents=True,exist_ok=True)
                    metadata=dict(group=stratum['group'],geometry=stratum['geometry'],selection=expression,
                                  selected_entries=total,valid_entries=count,missing_entries=total-count,
                                  raw_integral=float(hist.Integral()),source_ids=stratum['source_ids'],
                                  config=cfg,normalization=next(r for r in evidence['normalizations'] if
                                      r['group']==stratum['group'] and r['geometry']==stratum['geometry']))
                    f=ROOT.TFile(str(path),'RECREATE')
                    if f.IsZombie():raise ValueError('Cannot create histogram '+str(path))
                    hist.Write('hist2d' if is2d else 'hist')
                    if is2d:profile.Write('profile_x')
                    ROOT.TObjString(json.dumps(metadata,sort_keys=True)).Write('production_histogram_json')
                    write_metadata(f,dict(selected_entries=total,raw_integral=float(hist.Integral()),selection=expression))
                    f.Close();summary.append(dict(path=str(path.relative_to(out)),**metadata))
        write_json(out/'summary.json',summary)


def csv_summary(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def render(plan):
    """Use the same ROOT drawing functions as the user's EM shower workflow."""
    import ROOT
    import plot_comparison
    import plot_group_hist2d
    base=Path(plan['directory']);verify_receipt(base/'config');verify_receipt(base/'histograms')
    if (base/'plots/complete.json').exists():verify_receipt(base/'plots');return
    resolved=read_json(base/'config/resolved.json');records=read_json(base/'histograms/summary.json')
    grouped=defaultdict(list)
    for record in records:grouped[str(Path(record['path']).parent)].append(record)
    with publication(base/'plots') as out:
        for key,items in grouped.items():
            items.sort(key=lambda r:(list(NEUTRINOS).index(r['group']) if r['group'] in NEUTRINOS else 99,r['group']))
            destination=out/key;destination.mkdir(parents=True)
            pdf=destination/'plot.pdf';cfg=resolved[key];kind,name,selection,geometry=key.split('/')
            if kind=='2d':
                args=SimpleNamespace(hist_config=base/'config/pairs.yaml',pair=name,region=selection,
                    output=pdf,summary=destination/'summary.csv',hist=[(r['group'],str(base/'histograms'/r['path'])) for r in items],
                    normalization=plan['normalization'],scales={r['group']:r['normalization'].get('scale',0) for r in items},
                    exposures={r['group']:r['normalization'] for r in items},
                    styles=plan['plot_settings']['plotting'].get('plot',{}).get('particle_styles',{}),
                    labels=plan['plot_settings']['plotting'].get('plot',{}).get('particle_labels',{}),pair_config=cfg)
                plot_group_hist2d.plot(args)
                continue
            curves=[];rows=[]
            for index,record in enumerate(items):
                f=ROOT.TFile.Open(str(base/'histograms'/record['path']))
                raw=f.Get('hist').Clone('raw_'+str(index));raw.SetDirectory(0);f.Close()
                norm=record['normalization'];cfg=dict(cfg,plot_normalization='unit_area' if norm['mode']=='shape' else 'lumi')
                # Existing normalization function accepts a reference/source ratio.
                weight=norm.get('scale',0.)
                scaled,scale=plot_comparison.normalized_hist_for_plot(raw,'scaled_'+str(index),cfg,weight,1.)
                curves.append(dict(hist_scaled=scaled,scaled_integral=float(scaled.Integral()),group_name=record['group'],
                                   group_label=(plan['plot_settings']['plotting'].get('plot',{}).get('particle_labels',{}).get(record['group'],record['group'])+' (no valid events)') if not record['valid_entries'] else None,
                                   base_cut_key='none',extra_cut_key='none'))
                rows.append(dict(group=record['group'],geometry=record['geometry'],selection=record['selection'],
                    selected_entries=record['selected_entries'],valid_entries=record['valid_entries'],missing_entries=record['missing_entries'],
                    raw_integral=float(raw.Integral()),scaled_integral=float(scaled.Integral()),scale_factor=scale,
                    normalization=norm['mode'],exposure=json.dumps(norm,sort_keys=True),
                    underflow=float(raw.GetBinContent(0)),overflow=float(raw.GetBinContent(raw.GetNbinsX()+1))))
            canvas=ROOT.TCanvas('production_plot','',1000,800)
            eval_cfg=copy.deepcopy(plan['plot_settings']['plotting'])
            if plan['normalization']=='yield':
                eval_cfg.setdefault('plot',{})['normalization_label']='Audited luminosity yields (exposures in summary.csv)'
            eval_cfg.setdefault('plot',{}).update(region_preselection_label='Selection: '+items[0]['selection'],
                                                  region_labels={selection:selection})
            plot_comparison.draw_overlay(canvas,pdf,name,cfg,eval_cfg,'particles_in_region',selection,
                                         name+' / '+selection+' / '+geometry,1.,curves)
            canvas.Close();csv_summary(destination/'summary.csv',rows)
        write_json(out/'coverage.json',read_json(base/'config/coverage.json'))
        write_json(out/'normalization.json',read_json(base/'config/coverage.json')['normalizations'])


def distributions(plan):
    resolve(plan);fill(plan);render(plan)
