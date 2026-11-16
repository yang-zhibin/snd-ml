"""Worked plugin: join spatial runs separated by up to max_missing_channels.

Called once per event with selected SciFi hits, after converter timing cuts.
No ROOT objects, event labels, truth, or file paths are passed to the algorithm.
This is a development hypothesis; missing channels add neither hits nor QDC.
"""
from scifi_plane_clustering import cluster_scifi_hits, _make_cluster, payload_from_analysis
from baseline import validate_algorithm


def cluster_hits(hits, parameters):
    # hits is a sequence of dictionaries; it may be empty. Treat it as read-only.
    # Required keys in EACH hit:
    #   station: int, 1..5 (station number, dimensionless)
    #   isVertical: bool, True=vertical plane, False=horizontal plane
    #   layer_channel: int, 0..1535, channel index within that station/view
    #   qdc: float >= 0, detector QDC units (not calibrated energy in GeV)
    # Optional keys: x_cm, y_cm, z_cm [cm], time_ns [ns]. Never assume present.
    # Detector exports use endpoint A from GetSiPMPosition for coordinates;
    # these are geometry positions, not reconstructed 3D interaction points.
    # time_ns is the converter-selected time, with its configured corrections /
    # time-zero convention; a present value can still be an invalid sentinel.
    # (station, isVertical, layer_channel) uniquely identifies an event's hit.
    # Example access: total_charge = sum(hit['qdc'] for hit in hits).
    # parameters is the CLI JSON dictionary: {"max_missing_channels": 1}.
    unknown = set(parameters) - {'max_missing_channels'}
    if unknown:
        raise ValueError(f'Unknown parameters: {unknown}')
    gap = parameters.get('max_missing_channels', 1)
    if type(gap) is not int or not 0 <= gap <= 3:
        raise ValueError('max_missing_channels must be an integer in 0..3')

    # Start with production contiguous-channel clusters. The returned dictionary:
    # analysis['planes'][(station, vertical)] -> one plane, including empty ones.
    # plane keys: station, is_vertical, clusters (list), total_nhits, total_qdc.
    # analysis['orientations']['vertical' or 'horizontal'] -> two cluster lists:
    #   clusters: station/channel order; ranked_clusters: descending QDC order.
    # Both lists reference the SAME cluster dictionaries; do not count both.
    analysis = cluster_scifi_hits(hits)
    for (station, vertical), plane in analysis['planes'].items():
        runs = []
        for cluster in plane['clusters']:
            # channels and qdcs are aligned tuples, one entry per actual hit.
            # Example: channels=(1, 3), qdcs=(2.0, 3.0) after bridging channel 2.
            # The absent channel has no tuple entry; nhits=2, qdc=5.0.
            items = list(zip(cluster['channels'], cluster['qdcs']))
            if runs and items[0][0] - runs[-1][-1][0] - 1 <= gap:
                runs[-1].extend(items)
            else:
                runs.append(items)
        # _make_cluster constructs each dictionary with these additional fields:
        # station, is_vertical; spatial_id (zero-based within this plane);
        # first_channel, last_channel (inclusive channel indices); nhits (count);
        # qdc (sum), qdc_per_hit (mean), qdc_std (population std), all in QDC;
        # qdc_cv (std/mean), qdc_roughness (mean adjacent absolute QDC difference
        # divided by mean QDC), both dimensionless, with helper-defined zero cases.
        # A channel span last-first+1 is NOT a width in cm and can exceed nhits.
        plane['clusters'] = [_make_cluster(run, station, vertical, i)
                             for i, run in enumerate(runs)]
    for name, vertical in [('vertical', True), ('horizontal', False)]:
        clusters = [c for (s, v), p in analysis['planes'].items()
                    if v == vertical for c in p['clusters']]
        ranked = sorted(clusters, key=lambda c: (-c['qdc'], -c['nhits'], c['station'], c['first_channel']))
        # orientation_rank is one-based across all stations of this orientation.
        for rank, c in enumerate(ranked, 1):
            c['orientation_rank'] = rank
        analysis['orientations'][name] = {'clusters': clusters, 'ranked_clusters': ranked}
    # Plane total_nhits / total_qdc remain valid: merging changes neither total.
    return analysis


def calculate_features(hits, clusters, parameters):
    # clusters is exactly the object returned by cluster_hits for these hits.
    # It is this example's plane/orientation dictionary, not a list of hits.
    # Keep the baseline scalar names declared by the schema's extends entry.
    values = payload_from_analysis(clusters)['scalars']
    all_clusters = [c for view in clusters['orientations'].values() for c in view['clusters']]

    # Worked feature: sum cluster charges, then find the largest cluster charge.
    # Both quantities have QDC units; their ratio (and fragmentation) is unitless.
    total = sum(c['qdc'] for c in all_clusters)
    leading = max((c['qdc'] for c in all_clusters), default=0)
    values['charge_fragmentation_v1'] = 1 - leading / total if total else -999.0

    # To use hit coordinates for a future cluster feature, join by all THREE
    # identity fields. Channel numbers alone repeat across stations and views:
    # hit_by_key = {(h['station'], h['isVertical'], h['layer_channel']): h for h in hits}
    # for c in all_clusters:
    #     members = [hit_by_key[(c['station'], c['is_vertical'], channel)]
    #                for channel in c['channels']]
    #     # Require every member to have z; do not silently replace missing z by 0.
    #     z_span_cm = (max(h['z_cm'] for h in members) - min(h['z_cm'] for h in members)
    #                  if all('z_cm' in h for h in members) else -999.0)
    #     # This is a geometry-coordinate span, often zero for a single plane.
    # Aggregate any new per-cluster quantity to an event scalar, add its exact
    # name/units/missing_value to schemas/gap_clusters.yaml, THEN return it here.
    # Return ordinary finite Python int/float values; -999.0 is the declared
    # missing value for fragmentation when total QDC is zero (also empty events).
    return values
