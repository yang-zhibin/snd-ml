from scifi_plane_clustering import payload_from_analysis
from gap_clusters import cluster_hits
from baseline import validate_algorithm


def view_density(positions_cm):
    """Conta coppie di hit entro 1 cm. Ogni coppia conta due volte."""
    return sum(
        abs(a - b) <= 1.0
        for i, a in enumerate(positions_cm)
        for j, b in enumerate(positions_cm)
        if i != j
    )


def cluster_density(cluster, hit_by_key):
    """Calcola la density di un cluster usando le posizioni in cm."""
    members = [
        hit_by_key[(cluster['station'], cluster['is_vertical'], ch)]
        for ch in cluster['channels']
    ]
    if cluster['is_vertical']:
        positions = [h.get('x_cm') for h in members]
    else:
        positions = [h.get('y_cm') for h in members]
    positions = [p for p in positions if p is not None]
    return view_density(positions)


def calculate_features(hits, clusters, parameters):
    values = payload_from_analysis(clusters)["scalars"]

    hit_by_key = {
        (h['station'], h['isVertical'], h['layer_channel']): h
        for h in hits
    }

    all_clusters = [
        c for view in clusters["orientations"].values()
        for c in view["clusters"]
    ]

    # --- leading_charge_fraction ---
    total_qdc = sum(c["qdc"] for c in all_clusters)
    leading_qdc = max((c["qdc"] for c in all_clusters), default=0.0)
    values["leading_charge_fraction"] = leading_qdc / total_qdc if total_qdc else -999.0

    # --- density features ---
    total_density = sum(cluster_density(c, hit_by_key) for c in all_clusters)

    for orientation in ["vertical", "horizontal"]:
        ranked = clusters["orientations"][orientation]["ranked_clusters"]
        for rank in [1, 2]:
            matching = [c for c in ranked if c["orientation_rank"] == rank]
            if not matching:
                values[f"scifi_{orientation}_cluster{rank}_density"] = -999.0
                values[f"scifi_{orientation}_cluster{rank}_density_fraction"] = -999.0
                values[f"scifi_{orientation}_cluster{rank}_station"] = -999  # ← int!
                continue
            c = matching[0]
            density = cluster_density(c, hit_by_key)
            fraction = density / total_density if total_density > 0 else -999.0
            values[f"scifi_{orientation}_cluster{rank}_density"] = density
            values[f"scifi_{orientation}_cluster{rank}_density_fraction"] = fraction
            values[f"scifi_{orientation}_cluster{rank}_station"] = c["station"]  # ← aggiunto qui

    return values