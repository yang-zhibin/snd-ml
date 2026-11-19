"""Bounded complete-DAG yield test; artificial inputs, never EOS production."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import ROOT, read_json, write_json
from selected_hits import SelectedHitWriter, sha
from datasets import prepare_sample
from test_exposure import source


def main():
    root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'feature_dev/output/normalization-20261119/integration'
    root.mkdir(parents=True,exist_ok=True)
    raw=root/'raw.jsonl';sample=root/'full.jsonl'
    if not sample.exists():
        writer=SelectedHitWriter(raw,{'sample_type':'MC_neutrino','kind':'artificial normalization test','max_events':0})
        for i in range(4):
            writer.add(dict(runId=1,eventId=i,eventIndex=i),[dict(station=1,isVertical=True,layer_channel=i,qdc=2)],'CC_nue' if i<2 else 'NC_nue','ve','fixture')
        writer.finish(4)
        prepare_sample('mc-neutrino',[raw],sample,'normalization-test-v1',0)
    config=root/'datasets.json'
    write_json(config,dict(version=1,datasets={'mc-neutrino':dict(samples={'full':str(sample)},required_groups=['CC_nue','NC_nue'])}))
    audit=dict(version=1,sample_sha256=sha(sample),populations={
        've':dict(role='mc',coverage_evidence='artificial complete parent fixture',sources=[source('ve',10,4),source('zero-selected',5,0)]),
        'target':dict(role='reference_data',coverage_evidence='artificial reference fixture',sources=[source('data',20,0,.1)])},
        groups=[dict(group=g,geometry='fixture',population='ve',target_population='target') for g in ['CC_nue','NC_nue']])
    audit_path=root/'exposure.json';write_json(audit_path,audit)
    def run(name,*extra,ok=True):
        command=[str(ROOT/'run_feature_dev.sh'),'--algorithm','gap_clusters','--dataset','mc-neutrino','--sample','full','--sample-root',str(root),'--dataset-config',str(config),'--output-root',str(root/'results'),'--run',name,*extra]
        result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (root/(name+'.log')).write_text(result.stdout)
        if (result.returncode==0)!=ok:raise AssertionError(result.stdout)
        if ok:return read_json(root/'results/mc-neutrino'/name/'manifest.json')
    run('missing-audit',ok=False)
    run('small-yield','--sample','small','--normalization','yield',ok=False)
    shape=run('shape','--normalization','shape')
    yields=run('yield','--normalization','yield','--exposure-manifest',str(audit_path))
    assert shape['stage_keys']['calculate']==yields['stage_keys']['calculate']
    assert shape['stage_keys']['histogram']!=yields['stage_keys']['histogram']
    hist=read_json(Path(yields['plan']['paths']['histogram'])/'histograms.json')
    for feature in hist.values():
        for c in feature['curves']:
            assert abs(c['normalization']['scale']-2/15)<1e-12
            for n,y,v in zip(c['counts'],c['normalized'],c['sumw2']):
                assert abs(y-n*2/15)<1e-12 and abs(v-n*(2/15)**2)<1e-12
    report=Path(yields['plan']['paths']['plot'])/'report.html'
    assert 'target_exposure_fb' in report.read_text() and 'data:image/png;base64' in report.read_text()
    repeat=run('yield','--normalization','yield','--exposure-manifest',str(audit_path))
    assert repeat['signature']==yields['signature']
    print('PASS: full shape/yield DAGs, shared calculation cache, MC scale and sumw2 for both algorithms, exposure report, identical rerun, missing-audit/small-yield refusal')
    print(report)

if __name__=='__main__':main()
