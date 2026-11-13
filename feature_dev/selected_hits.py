"""ROOT-free selected-hit export shared by the production adapter and fixtures."""
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class SelectedHitWriter:
    def __init__(self,path,provenance):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.with_suffix(self.path.suffix+'.manifest.json').exists():
            raise FileExistsError('Export manifest already exists')
        self.stream=self.path.open('x');self.count=0;self.provenance=provenance
    def add(self,identity,hits,group,source_uid='',geometry='unknown'):
        clean=[]
        for h in hits:
            # These are the final selected hits supplied to production clustering.
            clean.append({k:h[k] for k in ('station','isVertical','layer_channel','qdc')})
            for old,new in [('x','x_cm'),('y','y_cm'),('z','z_cm'),('hitTimeNS','time_ns')]:
                if old in h:clean[-1][new]=h[old]
        row={'event_id':{k:int(identity[k]) for k in ('runId','eventId','eventIndex')},
             'source_uid':source_uid,'group':group,'geometry':geometry,'hits':clean}
        self.stream.write(json.dumps(row,allow_nan=False)+'\n');self.count+=1
    def finish(self,expected=None):
        self.stream.close()
        if expected is not None and expected!=self.count:raise ValueError('Export count mismatch')
        manifest={'version':1,'complete':True,'events':self.count,'sha256':sha(self.path),
                  'selection':'Final selected SciFi hits, after configured timing filters.',
                  'units':{'qdc':'detector QDC','position':'cm','time':'ns'},'provenance':self.provenance}
        self.path.with_suffix(self.path.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def merge_exports(output, inputs):
    """Combine validated exports, preserving source labels and event identities."""
    from engine import load_events, event_key
    events=[];provenance=[];seen=set()
    for source in inputs:
        rows,manifest=load_events(source)
        for row in rows:
            if event_key(row) in seen:raise ValueError('Duplicate event across input exports')
            seen.add(event_key(row));events.append(row)
        provenance.append({'path':str(source),'manifest':manifest})
    if not events:raise ValueError('No events to merge')
    writer=SelectedHitWriter(output,{'kind':'merged selected-hit exports','inputs':provenance})
    # Preserve optional standardized coordinates without translating them again.
    for row in events:
        writer.stream.write(json.dumps(row,allow_nan=False)+'\n');writer.count+=1
    writer.finish(len(events))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Merge selected-hit exports with duplicate-identity checks')
    parser.add_argument('output');parser.add_argument('inputs',nargs='+')
    args=parser.parse_args();merge_exports(args.output,args.inputs)
