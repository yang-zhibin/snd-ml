"""Example SciFi plugin. Field reference: feature_dev/README.md.

Run each function once per event; inputs are read-only.
"""
from scifi_plane_clustering import cluster_scifi_hits, _make_cluster, payload_from_analysis
from baseline import validate_algorithm


def cluster_hits(hits, parameters):
    """Merge channel runs across small gaps within each station and view.

    Parameters
    ----------
    hits : sequence of dict
        Selected SciFi hits for one event. Each hit contains:
        station (int, 1..5), isVertical (bool), layer_channel (int, 0..1535),
        qdc (float >= 0, detector QDC units).
        Optional: x_cm, y_cm, z_cm (cm), time_ns (ns).
    parameters : dict
        max_missing_channels : int, default 1
            Maximum gap to bridge, from 0 to 3 channels.

    Returns
    -------
    clusters : dict
        planes[(station, is_vertical)]['clusters'] gives a list of clusters.
        Each cluster has aligned channels/qdcs tuples, nhits and summed qdc.
        orientations['vertical' or 'horizontal'] provides clusters and
        ranked_clusters (descending QDC). These reference the same clusters.

    Examples
    --------
    >>> hits = [dict(station=1, isVertical=True, layer_channel=ch, qdc=q)
    ...         for ch, q in [(1, 2.0), (3, 3.0), (10, 5.0)]]
    >>> clusters = cluster_hits(hits, {"max_missing_channels": 1})
    >>> plane = clusters['planes'][(1, True)]
    >>> [(c['channels'], c['qdc']) for c in plane['clusters']]
    [((1, 3), 5.0), ((10,), 5.0)]
    """
    unknown = set(parameters) - {'max_missing_channels'}
    if unknown:
        raise ValueError(f'Unknown parameters: {unknown}')
    gap = parameters.get('max_missing_channels', 1)
    if type(gap) is not int or not 0 <= gap <= 3:
        raise ValueError('max_missing_channels must be an integer in 0..3')

    analysis = cluster_scifi_hits(hits)
    for (station, vertical), plane in analysis['planes'].items():
        runs = []
        for cluster in plane['clusters']:
            items = list(zip(cluster['channels'], cluster['qdcs']))
            if runs and items[0][0] - runs[-1][-1][0] - 1 <= gap:
                runs[-1].extend(items)
            else:
                runs.append(items)
        plane['clusters'] = [_make_cluster(run, station, vertical, i)
                             for i, run in enumerate(runs)]
    for name, vertical in [('vertical', True), ('horizontal', False)]:
        clusters = [c for (s, v), p in analysis['planes'].items()
                    if v == vertical for c in p['clusters']]
        ranked = sorted(clusters, key=lambda c: (-c['qdc'], -c['nhits'], c['station'], c['first_channel']))
        for rank, c in enumerate(ranked, 1):
            c['orientation_rank'] = rank
        analysis['orientations'][name] = {'clusters': clusters, 'ranked_clusters': ranked}
    return analysis


def calculate_features(hits, clusters, parameters):
    """Calculate baseline scalars and the charge fraction outside the largest cluster.

    Parameters
    ----------
    hits : sequence of dict
        The same event hits passed to cluster_hits.
    clusters : dict
        Result of cluster_hits for this event.
    parameters : dict
        The same settings passed to cluster_hits; no extra settings are used.

    Returns
    -------
    features : dict of str to int or float
        Baseline scalars plus charge_fragmentation_v1 (dimensionless):
        1 - largest_cluster_qdc / total_qdc. Returns -999.0 for zero total QDC.
        Output names and missing values must match schemas/gap_clusters.yaml.

    Examples
    --------
    >>> hits = [dict(station=1, isVertical=True, layer_channel=ch, qdc=q)
    ...         for ch, q in [(1, 2.0), (3, 3.0), (10, 5.0)]]
    >>> clusters = cluster_hits(hits, {})
    >>> calculate_features(hits, clusters, {})['charge_fragmentation_v1']
    0.5
    """
    # Keep the baseline features declared by the schema.
    values = payload_from_analysis(clusters)['scalars']
    all_clusters = [c for view in clusters['orientations'].values() for c in view['clusters']]

    # Count each cluster once; QDC / QDC is dimensionless.
    total = sum(c['qdc'] for c in all_clusters)
    leading = max((c['qdc'] for c in all_clusters), default=0)
    values['charge_fragmentation_v1'] = 1 - leading / total if total else -999.0

    return values
