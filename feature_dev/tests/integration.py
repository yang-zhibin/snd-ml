"""End-to-end acceptance checks; run in LCG with Snakemake available."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import ROOT, schema, write_json, read_json

def main():
    with tempfile.TemporaryDirectory(prefix='feature-dev-integration-') as temporary:
        root=Path(temporary);output=root/'outputs'
        def run(name,*extra,ok=True):
            p=subprocess.run([sys.executable,str(ROOT/'feature_dev/cli.py'),'--algorithm','gap_clusters','--output-root',str(output),'--run',name,*extra],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if (p.returncode==0)!=ok:raise AssertionError(p.stdout)
            return p.stdout
        run('dry','--dry-run');assert not output.exists()
        run('first')
        first=read_json(output/'first/manifest.json');calc=Path(first['plan']['paths']['calculate'])/'features.root';stamp=calc.stat().st_mtime_ns
        run('first');assert calc.stat().st_mtime_ns==stamp
        run('alias');assert read_json(output/'alias/manifest.json')['stage_keys']==first['stage_keys'];assert calc.stat().st_mtime_ns==stamp
        assert 'choose a new' in run('first','--parameters','{"max_missing_channels":0}',ok=False)
        d=schema(ROOT/'feature_dev/schemas/gap_clusters.yaml');d['features']['charge_fragmentation_v1']['histogram']['bins']=10
        custom=root/'schema.json';write_json(custom,d)
        run('new-bins','--schema',str(custom))
        changed=read_json(output/'new-bins/manifest.json')
        assert changed['stage_keys']['calculate']==first['stage_keys']['calculate']
        assert changed['stage_keys']['histogram']!=first['stage_keys']['histogram']
        assert calc.stat().st_mtime_ns==stamp
        run('zero-gap','--parameters','{"max_missing_channels":0}')
        zero=read_json(output/'zero-gap/manifest.json');assert zero['stage_keys']['calculate']!=first['stage_keys']['calculate']
        for record in read_json(Path(zero['plan']['paths']['calculate'])/'features.json'):
            assert all(record['candidate'][k]==v for k,v in record['baseline'].items())
        report=(output/'first/plot/report.html').read_text();assert 'data:image/png;base64,' in report and 'synthetic' in report
        features=Path(first['plan']['paths']['calculate'])/'features.json';features.write_text('[]')
        assert 'Cached output changed' in run('tampered',ok=False)
        print('PASS: dry-run isolation, complete DAG, identical rerun, cross-run reuse, immutable run ID, histogram-only reuse, parameter invalidation, gap-zero parity, standalone report, tamper detection (10 checks)')
if __name__=='__main__':main()
