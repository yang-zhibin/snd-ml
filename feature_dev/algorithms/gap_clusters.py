"""Example: join spatial runs separated by up to max_missing_channels.

A development hypothesis only. Missing channels contribute neither hits nor QDC.
"""
from scifi_plane_clustering import cluster_scifi_hits, _make_cluster, payload_from_analysis
from baseline import validate_algorithm


def cluster_hits(hits, parameters):
    unknown=set(parameters)-{'max_missing_channels'}
    if unknown:raise ValueError(f'Unknown parameters: {unknown}')
    gap=parameters.get('max_missing_channels',1)
    if type(gap) is not int or not 0 <= gap <= 3:
        raise ValueError('max_missing_channels must be an integer in 0..3')
    analysis=cluster_scifi_hits(hits)
    for (station,vertical),plane in analysis['planes'].items():
        runs=[]
        for cluster in plane['clusters']:
            items=list(zip(cluster['channels'],cluster['qdcs']))
            if runs and items[0][0]-runs[-1][-1][0]-1 <= gap:runs[-1].extend(items)
            else:runs.append(items)
        plane['clusters']=[_make_cluster(run,station,vertical,i) for i,run in enumerate(runs)]
    for name,vertical in [('vertical',True),('horizontal',False)]:
        clusters=[c for (s,v),p in analysis['planes'].items() if v==vertical for c in p['clusters']]
        ranked=sorted(clusters,key=lambda c:(-c['qdc'],-c['nhits'],c['station'],c['first_channel']))
        for rank,c in enumerate(ranked,1):c['orientation_rank']=rank
        analysis['orientations'][name]={'clusters':clusters,'ranked_clusters':ranked}
    return analysis


def calculate_features(hits, clusters, parameters):
    values=payload_from_analysis(clusters)['scalars']
    all_clusters=[c for view in clusters['orientations'].values() for c in view['clusters']]
    total=sum(c['qdc'] for c in all_clusters)
    leading=max((c['qdc'] for c in all_clusters),default=0)
    values['charge_fragmentation_v1']=1-leading/total if total else -999.0
    return values
