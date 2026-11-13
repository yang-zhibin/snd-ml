"""Exercise dataset routing on explicitly artificial exported hits, not physics data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import ROOT, write_json, read_json
from datasets import PHYSICS_DATASETS, prepare_sample
from test_datasets import export


def main():
    with tempfile.TemporaryDirectory(prefix='feature-datasets-') as tmp:
        root=Path(tmp);samples=root/'samples';samples.mkdir();output=root/'outputs'
        config={'version':1,'datasets':{}}
        for name,kind in zip(PHYSICS_DATASETS,['MC_neutrino','MC_testbeam','real_data']):
            source=samples/(name+'-export.jsonl');export(source,sample_type=kind,group=name,uid=name,count=3)
            prepared=samples/(name+'.jsonl');prepare_sample(name,[source],prepared,'test-v1',2)
            config['datasets'][name]={'samples':{'small':prepared.name,'full':prepared.name},'required_groups':[name]}
        settings=root/'datasets.json';write_json(settings,config)
        def run(dataset,runname='trial',*extra,ok=True):
            command=[sys.executable,str(ROOT/'feature_dev/cli.py'),'--dataset',dataset,'--dataset-config',str(settings),'--sample-root',str(samples),'--output-root',str(output),'--run',runname,*extra]
            p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if (p.returncode==0)!=ok:raise AssertionError(p.stdout)
            return p.stdout
        run('all','dry','--dry-run');assert not output.exists()
        run('mc-neutrino');assert not (output/'testbeam').exists() and not (output/'real-data').exists()
        first=read_json(output/'mc-neutrino/trial/manifest.json');feature=Path(first['plan']['paths']['calculate'])/'features.root';stamp=feature.stat().st_mtime_ns
        run('all');assert feature.stat().st_mtime_ns==stamp
        index=(output/'all/trial/index.html').read_text()
        assert all('../../'+name+'/trial/plot/report.html' in index for name in PHYSICS_DATASETS)
        assert not (output/'mc-muon').exists()
        for name in PHYSICS_DATASETS:
            manifest=read_json(output/name/'trial/manifest.json')
            assert manifest['plan']['dataset']['name']==name
            assert manifest['plan']['dataset']['groups'][0]['events']==2
        # An unrelated dataset config edit must not invalidate neutrino results.
        config['datasets']['testbeam']['description']='changed description'
        write_json(settings,config);run('mc-neutrino');assert feature.stat().st_mtime_ns==stamp
        assert 'choose a new' in run('testbeam',ok=False)
        run('mc-neutrino','full-test','--sample','full')
        run('synthetic','synthetic-test')
        assert (output/'synthetic/synthetic-test/plot/report.html').exists()
        # Missing all-input preflight must not leave partial runs.
        config['datasets']['real-data']['samples']['small']='missing.jsonl';write_json(settings,config)
        assert 'No prepared' in run('all','missing-test',ok=False)
        assert not (output/'mc-neutrino/missing-test').exists()
        print('PASS: dataset routing, dry-run isolation, all/index, independent reuse, unrelated-config isolation, immutable runs, full selection, synthetic namespace, optional muon exclusion, missing-input preflight (10 checks)')
if __name__=='__main__':main()
