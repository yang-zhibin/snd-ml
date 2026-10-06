"""ROOT-free SciFi clustering and fixed-size shower-profile features."""
from __future__ import division
import math

STATIONS = (1, 2, 3, 4, 5)
ORIENTATIONS = ((True, "vertical"), (False, "horizontal"))
UNDEFINED_INT, UNDEFINED_FLOAT = -999, -999.0
SCHEMA_VERSION = "scifi_orientation_clusters_v1"

SLOT = (("exists","i"),("station","i"),("nhits","i"),("qdc","d"),
        ("qdc_per_hit","d"),("plane_hit_fraction","d"),("plane_qdc_fraction","d"),
        ("orientation_hit_fraction","d"),("orientation_qdc_fraction","d"),
        ("qdc_std","d"),("qdc_cv","d"),("qdc_roughness","d"))
RELATIONS = (("cluster2_to_cluster1_hit_ratio","d"),
 ("cluster2_to_cluster1_qdc_ratio","d"),("cluster2_to_cluster1_qdc_per_hit_ratio","d"),
 ("top2_station_delta","i"),("top2_station_separation","i"),("top2_same_station","i"),
 ("top2_qdc_downstream_asymmetry","d"),("top2_hit_downstream_asymmetry","d"))
OTHER = (("other_cluster_count","i"),("other_cluster_nhits","i"),
 ("other_cluster_qdc","d"),("other_cluster_qdc_per_hit","d"),
 ("other_cluster_hit_fraction","d"),("other_cluster_qdc_fraction","d"))
GENERAL = (("cluster_count","i"),("active_plane_count","i"),("active_station_count","i"),
 ("total_nhits","i"),("total_qdc","d"),("qdc_per_hit","d"),
 ("cluster_nhits_mean","d"),("cluster_nhits_std","d"),("cluster_nhits_cv","d"),
 ("cluster_total_qdc_mean","d"),("cluster_total_qdc_std","d"),("cluster_total_qdc_cv","d"),
 ("cluster_qdc_per_hit_mean","d"),("cluster_qdc_per_hit_std","d"),
 ("cluster_qdc_per_hit_cv","d"),("effective_hit_cluster_count","d"),
 ("effective_qdc_cluster_count","d"))
PROFILE = tuple(
 ("{}_frac{}".format(quantity, station), "d")
 for quantity in ("hit","qdc","cluster_count") for station in STATIONS
) + (("hit_mean_station","d"),("hit_std_station","d"),("qdc_mean_station","d"),
 ("qdc_std_station","d"),("cluster_mean_station","d"),("cluster_std_station","d"),
 ("max_hit_station","i"),("max_qdc_station","i"),("max_cluster_count_station","i"),
 ("hit_station_slope","d"),("qdc_station_slope","d"),
 ("hit_station_curvature","d"),("qdc_station_curvature","d"))
EVENT = (("scifi_vertical_hit_fraction","d"),("scifi_vertical_qdc_fraction","d"),
 ("scifi_vertical_cluster_fraction","d"),("scifi_orientation_hit_asymmetry","d"),
 ("scifi_orientation_qdc_asymmetry","d"),("scifi_orientation_cluster_asymmetry","d"),
 ("scifi_cluster_total_count","i"),("scifi_cluster_total_nhits","i"),
 ("scifi_cluster_total_qdc","d"),("scifi_cluster_qdc_per_hit","d"),
 ("scifi_other_clusters_count","i"),("scifi_other_clusters_nhits","i"),
 ("scifi_other_clusters_qdc","d"),("scifi_other_clusters_qdc_per_hit","d"),
 ("scifi_other_clusters_hit_fraction","d"),("scifi_other_clusters_qdc_fraction","d"))

def scalar_feature_dtypes():
    fields = []
    for _, name in ORIENTATIONS:
        prefix = "scifi_{}_".format(name)
        for slot in (1, 2):
            fields += [(prefix+"cluster{}_{}".format(slot, key), dtype) for key,dtype in SLOT]
        fields += [(prefix+key,dtype) for key,dtype in RELATIONS+OTHER+GENERAL+PROFILE]
    fields += list(EVENT)
    if len(fields) != len({name for name,_ in fields}):
        raise RuntimeError("duplicate SciFi cluster feature")
    return tuple(fields)

def _ratio(a, b):
    return float(a)/float(b) if b > 0 else 0.0


def direct_scifi_qdc_per_hit(hits):
    """Return selected SciFi QDC per hit without using cluster summaries."""
    total_qdc = 0.0
    total_nhits = 0
    for hit in hits:
        station = int(hit.get("station", 0))
        if not 1 <= station <= 5:
            continue
        try:
            qdc = max(float(hit.get("qdc", 0.0)), 0.0)
        except (TypeError, ValueError):
            qdc = 0.0
        total_qdc += qdc
        total_nhits += 1
    return _ratio(total_qdc, total_nhits)


def _stats(values):
    if not values:
        return UNDEFINED_FLOAT, UNDEFINED_FLOAT, UNDEFINED_FLOAT
    mean = float(sum(values))/len(values)
    std = math.sqrt(max(0.0, sum((float(x)-mean)**2 for x in values)/len(values)))
    return mean, std, _ratio(std, mean)

def _weighted(weights):
    total = float(sum(weights))
    if total <= 0:
        return UNDEFINED_FLOAT, UNDEFINED_FLOAT
    mean = sum(station*w for station,w in zip(STATIONS,weights))/total
    std = math.sqrt(max(0.0,sum(w*(station-mean)**2 for station,w in zip(STATIONS,weights))/total))
    return float(mean), std

def _fit(fractions, total):
    if total <= 0:
        return UNDEFINED_FLOAT, UNDEFINED_FLOAT
    x = (-2.0,-1.0,0.0,1.0,2.0)
    return (sum(a*b for a,b in zip(x,fractions))/10.0,
            sum((a*a-2.0)*b for a,b in zip(x,fractions))/14.0)

def _effective(values):
    total = float(sum(values))
    return 0.0 if total <= 0 else 1.0/sum((float(x)/total)**2 for x in values)

def _make_cluster(run, station, vertical, spatial_id):
    channels = tuple(x[0] for x in run)
    qdcs = tuple(x[1] for x in run)
    qdc = float(sum(qdcs)); mean = qdc/len(qdcs)
    std = math.sqrt(max(0.0,sum((x-mean)**2 for x in qdcs)/len(qdcs)))
    roughness = (sum(abs(qdcs[i+1]-qdcs[i]) for i in range(len(qdcs)-1))
                 /(len(qdcs)-1)/mean if len(qdcs)>1 and mean>0 else 0.0)
    return {"station":station,"is_vertical":vertical,"spatial_id":spatial_id,
            "first_channel":channels[0],"last_channel":channels[-1],
            "channels":channels,"qdcs":qdcs,"nhits":len(qdcs),"qdc":qdc,
            "qdc_per_hit":mean,"qdc_std":std,"qdc_cv":_ratio(std,mean),
            "qdc_roughness":roughness}

def cluster_scifi_hits(hits):
    grouped = {(s,v):{} for s in STATIONS for v,_ in ORIENTATIONS}
    for hit in hits:
        station=int(hit["station"]); vertical=bool(hit["isVertical"])
        channel=int(hit["layer_channel"]); qdc=float(hit["qdc"])
        if station not in STATIONS: raise ValueError("SciFi station outside 1..5: {}".format(station))
        if not 0 <= channel < 1536: raise ValueError("SciFi layer channel outside 0..1535: {}".format(channel))
        if not math.isfinite(qdc): raise ValueError("Non-finite SciFi QDC")
        if channel in grouped[(station,vertical)]:
            raise ValueError("Duplicate SciFi layer channel {} in station {}".format(channel,station))
        grouped[(station,vertical)][channel]=max(0.0,qdc)
    planes={}
    for station in STATIONS:
        for vertical,_ in ORIENTATIONS:
            items=sorted(grouped[(station,vertical)].items()); runs=[]
            for item in items:
                if not runs or item[0] != runs[-1][-1][0]+1: runs.append([])
                runs[-1].append(item)
            clusters=[_make_cluster(run,station,vertical,i) for i,run in enumerate(runs)]
            if sum(cluster["nhits"] for cluster in clusters) != len(items):
                raise AssertionError("plane hit closure")
            if not math.isclose(
                sum(cluster["qdc"] for cluster in clusters),
                sum(qdc for _,qdc in items), rel_tol=1e-12, abs_tol=1e-9
            ):
                raise AssertionError("plane QDC closure")
            planes[(station,vertical)]={"station":station,"is_vertical":vertical,
                "clusters":clusters,"total_nhits":len(items),
                "total_qdc":float(sum(q for _,q in items))}
    orientations={}
    for vertical,name in ORIENTATIONS:
        clusters=[c for s in STATIONS for c in planes[(s,vertical)]["clusters"]]
        ranked=sorted(clusters,key=lambda c:(-c["qdc"],-c["nhits"],c["station"],c["first_channel"]))
        for rank,c in enumerate(ranked,1): c["orientation_rank"]=rank
        orientations[name]={"clusters":clusters,"ranked_clusters":ranked}
    return {"planes":planes,"orientations":orientations}

def _slot(cluster, plane, total_hits, total_qdc):
    if cluster is None:
        return {"exists":0,"station":UNDEFINED_INT,"nhits":0,"qdc":0.0,
         "qdc_per_hit":0.0,"plane_hit_fraction":0.0,"plane_qdc_fraction":0.0,
         "orientation_hit_fraction":0.0,"orientation_qdc_fraction":0.0,
         "qdc_std":UNDEFINED_FLOAT,"qdc_cv":UNDEFINED_FLOAT,
         "qdc_roughness":UNDEFINED_FLOAT}
    return {"exists":1,"station":cluster["station"],"nhits":cluster["nhits"],
     "qdc":cluster["qdc"],"qdc_per_hit":cluster["qdc_per_hit"],
     "plane_hit_fraction":_ratio(cluster["nhits"],plane["total_nhits"]),
     "plane_qdc_fraction":_ratio(cluster["qdc"],plane["total_qdc"]),
     "orientation_hit_fraction":_ratio(cluster["nhits"],total_hits),
     "orientation_qdc_fraction":_ratio(cluster["qdc"],total_qdc),
     "qdc_std":cluster["qdc_std"],"qdc_cv":cluster["qdc_cv"],
     "qdc_roughness":cluster["qdc_roughness"]}

def _max_station(values):
    maximum=max(values)
    return STATIONS[values.index(maximum)] if maximum>0 else UNDEFINED_INT

def _orientation(analysis, vertical, name):
    ranked=analysis["orientations"][name]["ranked_clusters"]
    planes=[analysis["planes"][(s,vertical)] for s in STATIONS]
    sh=[p["total_nhits"] for p in planes]; sq=[p["total_qdc"] for p in planes]
    sc=[len(p["clusters"]) for p in planes]
    th=sum(sh); tq=float(sum(sq)); count=len(ranked); out={}
    for number in (1,2):
        cluster=ranked[number-1] if len(ranked)>=number else None
        plane=analysis["planes"][(cluster["station"],vertical)] if cluster else None
        out.update(("cluster{}_{}".format(number,k),v) for k,v in _slot(cluster,plane,th,tq).items())
    first=ranked[0] if ranked else None; second=ranked[1] if len(ranked)>1 else None
    out["cluster2_to_cluster1_hit_ratio"]=_ratio(second["nhits"] if second else 0,first["nhits"] if first else 0)
    out["cluster2_to_cluster1_qdc_ratio"]=_ratio(second["qdc"] if second else 0,first["qdc"] if first else 0)
    out["cluster2_to_cluster1_qdc_per_hit_ratio"]=_ratio(second["qdc_per_hit"] if second else 0,first["qdc_per_hit"] if first else 0)
    if not first or not second:
        delta=separation=UNDEFINED_INT; same=0; qa=ha=0.0
    else:
        delta=second["station"]-first["station"]; separation=abs(delta); same=int(delta==0)
        if same: qa=ha=0.0
        else:
            upstream,downstream=sorted((first,second),key=lambda c:c["station"])
            qa=_ratio(downstream["qdc"]-upstream["qdc"],downstream["qdc"]+upstream["qdc"])
            ha=_ratio(downstream["nhits"]-upstream["nhits"],downstream["nhits"]+upstream["nhits"])
    out.update({"top2_station_delta":delta,"top2_station_separation":separation,
     "top2_same_station":same,"top2_qdc_downstream_asymmetry":qa,
     "top2_hit_downstream_asymmetry":ha})
    others=ranked[2:]; oh=sum(c["nhits"] for c in others); oq=float(sum(c["qdc"] for c in others))
    out.update({"other_cluster_count":len(others),"other_cluster_nhits":oh,
     "other_cluster_qdc":oq,"other_cluster_qdc_per_hit":_ratio(oq,oh),
     "other_cluster_hit_fraction":_ratio(oh,th),"other_cluster_qdc_fraction":_ratio(oq,tq),
     "cluster_count":count,"active_plane_count":sum(x>0 for x in sh),
     "active_station_count":sum(x>0 for x in sh),"total_nhits":th,"total_qdc":tq,
     "qdc_per_hit":_ratio(tq,th),"effective_hit_cluster_count":_effective([c["nhits"] for c in ranked]),
     "effective_qdc_cluster_count":_effective([c["qdc"] for c in ranked])})
    for prefix,values in (("cluster_nhits",[c["nhits"] for c in ranked]),
                          ("cluster_total_qdc",[c["qdc"] for c in ranked]),
                          ("cluster_qdc_per_hit",[c["qdc_per_hit"] for c in ranked])):
        out[prefix+"_mean"],out[prefix+"_std"],out[prefix+"_cv"]=_stats(values)
    hf=[_ratio(x,th) for x in sh]; qf=[_ratio(x,tq) for x in sq]; cf=[_ratio(x,count) for x in sc]
    for station,a,b,c in zip(STATIONS,hf,qf,cf):
        out["hit_frac{}".format(station)]=a; out["qdc_frac{}".format(station)]=b
        out["cluster_count_frac{}".format(station)]=c
    out["hit_mean_station"],out["hit_std_station"]=_weighted(sh)
    out["qdc_mean_station"],out["qdc_std_station"]=_weighted(sq)
    out["cluster_mean_station"],out["cluster_std_station"]=_weighted(sc)
    out["max_hit_station"]=_max_station(sh); out["max_qdc_station"]=_max_station(sq)
    out["max_cluster_count_station"]=_max_station(sc)
    out["hit_station_slope"],out["hit_station_curvature"]=_fit(hf,th)
    out["qdc_station_slope"],out["qdc_station_curvature"]=_fit(qf,tq)
    if sum(sh)!=th or sum(sc)!=count: raise AssertionError("orientation closure")
    if not math.isclose(sum(sq),tq,rel_tol=1e-12,abs_tol=1e-9): raise AssertionError("orientation QDC closure")
    return out

def cluster_feature_payload(hits):
    analysis=cluster_scifi_hits(hits); scalars={}; summaries={}
    for vertical,name in ORIENTATIONS:
        summary=_orientation(analysis,vertical,name); summaries[name]=summary
        scalars.update(("scifi_{}_{}".format(name,k),v) for k,v in summary.items())
    v=summaries["vertical"]; h=summaries["horizontal"]
    count=v["cluster_count"]+h["cluster_count"]; hits=v["total_nhits"]+h["total_nhits"]
    qdc=v["total_qdc"]+h["total_qdc"]; oc=v["other_cluster_count"]+h["other_cluster_count"]
    oh=v["other_cluster_nhits"]+h["other_cluster_nhits"]; oq=v["other_cluster_qdc"]+h["other_cluster_qdc"]
    scalars.update({"scifi_vertical_hit_fraction":_ratio(v["total_nhits"],hits),
     "scifi_vertical_qdc_fraction":_ratio(v["total_qdc"],qdc),
     "scifi_vertical_cluster_fraction":_ratio(v["cluster_count"],count),
     "scifi_orientation_hit_asymmetry":_ratio(v["total_nhits"]-h["total_nhits"],hits),
     "scifi_orientation_qdc_asymmetry":_ratio(v["total_qdc"]-h["total_qdc"],qdc),
     "scifi_orientation_cluster_asymmetry":_ratio(v["cluster_count"]-h["cluster_count"],count),
     "scifi_cluster_total_count":count,"scifi_cluster_total_nhits":hits,
     "scifi_cluster_total_qdc":qdc,"scifi_cluster_qdc_per_hit":_ratio(qdc,hits),
     "scifi_other_clusters_count":oc,"scifi_other_clusters_nhits":oh,
     "scifi_other_clusters_qdc":oq,"scifi_other_clusters_qdc_per_hit":_ratio(oq,oh),
     "scifi_other_clusters_hit_fraction":_ratio(oh,hits),
     "scifi_other_clusters_qdc_fraction":_ratio(oq,qdc)})
    if set(scalars)!={name for name,_ in scalar_feature_dtypes()}: raise AssertionError("feature schema")
    clustered_hits=sum(c["nhits"] for x in analysis["orientations"].values() for c in x["clusters"])
    clustered_qdc=sum(c["qdc"] for x in analysis["orientations"].values() for c in x["clusters"])
    if clustered_hits!=hits: raise AssertionError("event hit closure")
    if not math.isclose(clustered_qdc,qdc,rel_tol=1e-12,abs_tol=1e-9): raise AssertionError("event QDC closure")
    return {"scalars":scalars,"analysis":analysis}
