"""Adapter for the existing production SciFi algorithm."""
import math
from scifi_plane_clustering import cluster_scifi_hits, payload_from_analysis


def cluster_hits(hits, parameters):
    if parameters:
        raise ValueError('Baseline has no adjustable parameters')
    return cluster_scifi_hits(hits)


def calculate_features(hits, clusters, parameters):
    return payload_from_analysis(clusters)['scalars']


def validate_algorithm(hits, clusters, features):
    # Baseline partitions every selected hit, retaining its full nonnegative charge.
    if features['scifi_cluster_total_nhits'] != len(hits):raise ValueError('hit closure')
    if not math.isclose(features['scifi_cluster_total_qdc'], sum(h['qdc'] for h in hits), rel_tol=1e-12, abs_tol=1e-9):raise ValueError('charge closure')
