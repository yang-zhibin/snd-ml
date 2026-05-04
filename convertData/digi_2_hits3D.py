import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from analysis.analyses.snd_analysis_2024_0mu.sciFiTools import selectHits, getSumDensity


# ==========================================================
# Label mappings
# ==========================================================

particle_to_target = {
    12: 0, -12: 0,
    14: 1, -14: 1,
    16: 2, -16: 2,
    112: 3, -112: 3, 114: 3, -114: 3, 116: 3, -116: 3,
    130: 4, 310: 4,
    2112: 5,
    13: 6, -13: 6,
    0: 6,
}

particle_mapping = {
    12: "ve", -12: "ve",
    14: "vm", -14: "vm",
    16: "vt", -16: "vt",
    112: "NC", -112: "NC", 114: "NC", -114: "NC", 116: "NC", -116: "NC",
    130: "kaon", 310: "kaon",
    2112: "neutron",
    13: "muon", -13: "muon",
    0: "data",
}


# ==========================================================
# I/O and setup
# ==========================================================

def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules["Scifi"])
    lsOfGlobals.Add(snd_geo.modules["MuFilter"])
    return snd_geo


def open_root_file(file_path, tree_name="cbmsim", mode="read"):
    root_file = ROOT.TFile(file_path, mode)

    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {file_path}")

    tree = root_file.Get(tree_name)
    if not tree or not isinstance(tree, ROOT.TTree):
        raise RuntimeError(f"TTree '{tree_name}' not found in {file_path}")

    return root_file, tree


def save_dataset(obj, path):
    """
    Save nested object as compressed NPZ.

    Note: because events is a list of dictionaries containing arrays,
    loading needs allow_pickle=True.
    """
    np.savez_compressed(path, **obj)


def build_selection(args):
    """
    Build event selection.

    train/val:
        strict cuts for training-quality samples.

    test:
        looser cuts matching feature-file production, useful for later
        comparison and evaluation.
    """
    is_veto_sample = "veto_" in os.path.basename(args.out_path)

    if args.dataset_split in ("train", "val"):
        if is_veto_sample:
            selection = (
                "AvgSFChan == 1 && "
                "NoVetoHits == 0 && "
                "At_least_two_consecutive_SciFi_planes == 1 && "
                "SciFiContinuity == 1"
            )
        else:
            selection = (
                "AvgSFChan == 1 && "
                "NoVetoHits == 1 && "
                "At_least_two_consecutive_SciFi_planes == 1 && "
                "SciFiContinuity == 1"
            )

    elif args.dataset_split == "test":
        if is_veto_sample:
            selection = "AvgSFChan == 1 && NoVetoHits == 0"
        else:
            selection = "AvgSFChan == 1 && NoVetoHits == 1"

    else:
        raise ValueError(f"Unknown dataset_split: {args.dataset_split}")

    if "MC" not in args.type:
        selection += " && StableBeams==1 && IP1 == 1 && EventDeltat_m1_100 == 1"

    return selection


# ==========================================================
# Filtering
# ==========================================================

def filter_SciFiHits_qdc(SciFi_hits, mode="mean", verbose=False):
    """Filter SciFi hits by QDC threshold."""
    if not SciFi_hits:
        stats = {
            "n_before": 0,
            "thresholds": {
                "mean": None,
                "median": None,
                "half_maxQDC": None,
            },
            "counts_after": {
                "mean": 0,
                "median": 0,
                "half_maxQDC": 0,
            },
            "selected_mode": mode,
            "selected_threshold": None,
            "n_after_selected": 0,
        }
        if verbose:
            print("[SciFi QDC filter] no hits")
        return [], stats

    qdcs = np.array([float(h["qdc"]) for h in SciFi_hits], dtype=float)

    thresholds = {
        "mean": float(np.mean(qdcs)),
        "median": float(np.median(qdcs)),
        "half_maxQDC": 0.5 * float(np.max(qdcs)),
    }

    if mode not in thresholds:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {list(thresholds.keys())}")

    counts_after = {
        key: sum(1 for h in SciFi_hits if float(h["qdc"]) >= thr)
        for key, thr in thresholds.items()
    }

    if verbose:
        n_before = len(SciFi_hits)
        print(
            f"[SciFi QDC filter] before={n_before} | "
            f"mean_thr={thresholds['mean']:.4f}, after={counts_after['mean']} "
            f"({100.0 * counts_after['mean'] / n_before:.1f}%) | "
            f"median_thr={thresholds['median']:.4f}, after={counts_after['median']} "
            f"({100.0 * counts_after['median'] / n_before:.1f}%) | "
            f"half_maxQDC_thr={thresholds['half_maxQDC']:.4f}, after={counts_after['half_maxQDC']} "
            f"({100.0 * counts_after['half_maxQDC'] / n_before:.1f}%)"
        )

    selected_threshold = thresholds[mode]
    filtered_hits = [h for h in SciFi_hits if float(h["qdc"]) >= selected_threshold]

    stats = {
        "n_before": len(SciFi_hits),
        "thresholds": thresholds,
        "counts_after": counts_after,
        "selected_mode": mode,
        "selected_threshold": selected_threshold,
        "n_after_selected": len(filtered_hits),
    }

    return filtered_hits, stats


def filter_SciFiHits_time(
    SciFi_hits,
    lower_time_threshold=0.5,
    upper_time_threshold=2.3,
    bin_width=0.25,
):
    """Filter SciFi hits around MPV of hit time per station and orientation."""
    if not SciFi_hits:
        return [], {}

    groups = defaultdict(list)
    for h in SciFi_hits:
        groups[(h["station"], h["isVertical"])].append(h)

    filtered = []
    peak_by_group = {}
    max_bins = 2000

    for key, hits in groups.items():
        times = np.array([h["hitTimeCY"] for h in hits], dtype=float)
        tmin, tmax = float(times.min()), float(times.max())

        if tmax <= tmin:
            peak = tmin
        else:
            width = tmax - tmin
            nbins = int(np.ceil(width / bin_width))
            nbins = max(10, min(nbins, max_bins))
            hist, edges = np.histogram(times, bins=nbins, range=(tmin, tmax))
            i_max = int(np.argmax(hist))
            peak = 0.5 * (edges[i_max] + edges[i_max + 1])

        peak_by_group[key] = peak
        lo = peak - lower_time_threshold
        hi = peak + upper_time_threshold

        for h in hits:
            if lo <= h["hitTimeCY"] <= hi:
                filtered.append(h)

    return filtered, peak_by_group


# ==========================================================
# Hit-building helpers
# ==========================================================

def empty_hit_arrays(keys, dtypes):
    return {k: np.array([], dtype=dt) for k, dt in zip(keys, dtypes)}


def concatenate_hit_dicts(hit_dicts, keys, dtypes):
    non_empty = [h for h in hit_dicts if len(h[keys[0]]) > 0]
    if not non_empty:
        return empty_hit_arrays(keys, dtypes)

    return {
        k: np.concatenate([h[k] for h in non_empty]).astype(dt, copy=False)
        for k, dt in zip(keys, dtypes)
    }


def combine_station_hits_to_3d(vertical_hits, horizontal_hits, vertical_id_key, horizontal_id_key):
    """Combine one station's vertical and horizontal hits into crossed 3D hits."""
    if not vertical_hits or not horizontal_hits:
        return empty_hit_arrays(
            ["station", "x", "y", "z", "qdc", "vertical_id", "horizontal_id"],
            [np.int16, np.float32, np.float32, np.float32, np.float32, np.int32, np.int32],
        )

    station = vertical_hits[0]["station"]

    vx = np.array([h["x_mid"] for h in vertical_hits], dtype=np.float32)
    vz = np.array([h["z_mid"] for h in vertical_hits], dtype=np.float32)
    vq = np.array([h["qdc"] for h in vertical_hits], dtype=np.float32)
    vid = np.array([h[vertical_id_key] for h in vertical_hits], dtype=np.int32)

    hy = np.array([h["y_mid"] for h in horizontal_hits], dtype=np.float32)
    hz = np.array([h["z_mid"] for h in horizontal_hits], dtype=np.float32)
    hq = np.array([h["qdc"] for h in horizontal_hits], dtype=np.float32)
    hid = np.array([h[horizontal_id_key] for h in horizontal_hits], dtype=np.int32)

    nv = len(vertical_hits)
    nh = len(horizontal_hits)

    x = np.repeat(vx, nh)
    y = np.tile(hy, nv)
    z = 0.5 * (np.repeat(vz, nh) + np.tile(hz, nv))
    qdc = np.repeat(vq, nh) + np.tile(hq, nv)

    return {
        "station": np.full(nv * nh, station, dtype=np.int16),
        "x": x,
        "y": y,
        "z": z,
        "qdc": qdc,
        "vertical_id": np.repeat(vid, nh),
        "horizontal_id": np.tile(hid, nv),
    }


def build_centered_boxes_for_us_ds4(
    MuFilter_hits,
    voxel_size=(1.0, 1.0, 1.0),
    us_dim=(83.5, 6.0, 1.0),
    ds4_dim=(1.0, 63.5, 1.0),
):
    """Build voxelized 3D hits distributed inside US and DS4 bars."""
    dx, dy, dz = voxel_size

    selected_hits = []
    dims = []

    for h in MuFilter_hits:
        if h["detType"] == 2:
            selected_hits.append(h)
            dims.append(us_dim)
        elif h["detType"] == 3 and h["station"] == 4:
            selected_hits.append(h)
            dims.append(ds4_dim)

    if not selected_hits:
        return empty_hit_arrays(
            ["x", "y", "z", "qdc", "station", "detType", "barIndex", "source_hit"],
            [np.float32, np.float32, np.float32, np.float32, np.int16, np.int16, np.int32, np.int32],
        )

    x_all, y_all, z_all = [], [], []
    qdc_all, station_all, detType_all = [], [], []
    barIndex_all, source_hit_all = [], []

    for i, (h, dim) in enumerate(zip(selected_hits, dims)):
        sx, sy, sz = dim

        nx = max(1, int(round(sx / dx)))
        ny = max(1, int(round(sy / dy)))
        nz = max(1, int(round(sz / dz)))

        sx_eff = nx * dx
        sy_eff = ny * dy
        sz_eff = nz * dz

        x0 = h["x_mid"] - 0.5 * sx_eff
        y0 = h["y_mid"] - 0.5 * sy_eff
        z0 = h["z_mid"] - 0.5 * sz_eff

        xs = x0 + (np.arange(nx, dtype=np.float32) + 0.5) * dx
        ys = y0 + (np.arange(ny, dtype=np.float32) + 0.5) * dy
        zs = z0 + (np.arange(nz, dtype=np.float32) + 0.5) * dz

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        nvox = nx * ny * nz

        x_all.append(X.ravel())
        y_all.append(Y.ravel())
        z_all.append(Z.ravel())
        qdc_all.append(np.full(nvox, float(h["qdc"]) / nvox, dtype=np.float32))
        station_all.append(np.full(nvox, h["station"], dtype=np.int16))
        detType_all.append(np.full(nvox, h["detType"], dtype=np.int16))
        barIndex_all.append(np.full(nvox, h["barIndex"], dtype=np.int32))
        source_hit_all.append(np.full(nvox, i, dtype=np.int32))

    return {
        "x": np.concatenate(x_all),
        "y": np.concatenate(y_all),
        "z": np.concatenate(z_all),
        "qdc": np.concatenate(qdc_all),
        "station": np.concatenate(station_all),
        "detType": np.concatenate(detType_all),
        "barIndex": np.concatenate(barIndex_all),
        "source_hit": np.concatenate(source_hit_all),
    }


def build_all_3dHits(SciFi_3D_hits, DS_3D_hits, US_DS4_voxel_hits):
    keys = ["station", "x", "y", "z", "qdc", "detType"]
    dtypes = [np.int16, np.float32, np.float32, np.float32, np.float32, np.int16]

    scifi_block = {
        "station": SciFi_3D_hits["station"].astype(np.int16, copy=False),
        "x": SciFi_3D_hits["x"].astype(np.float32, copy=False),
        "y": SciFi_3D_hits["y"].astype(np.float32, copy=False),
        "z": SciFi_3D_hits["z"].astype(np.float32, copy=False),
        "qdc": SciFi_3D_hits["qdc"].astype(np.float32, copy=False),
        "detType": np.full(len(SciFi_3D_hits["x"]), 1, dtype=np.int16),
    }

    ds_block = {
        "station": DS_3D_hits["station"].astype(np.int16, copy=False),
        "x": DS_3D_hits["x"].astype(np.float32, copy=False),
        "y": DS_3D_hits["y"].astype(np.float32, copy=False),
        "z": DS_3D_hits["z"].astype(np.float32, copy=False),
        "qdc": DS_3D_hits["qdc"].astype(np.float32, copy=False),
        "detType": np.full(len(DS_3D_hits["x"]), 2, dtype=np.int16),
    }

    us_ds4_block = {
        "station": US_DS4_voxel_hits["station"].astype(np.int16, copy=False),
        "x": US_DS4_voxel_hits["x"].astype(np.float32, copy=False),
        "y": US_DS4_voxel_hits["y"].astype(np.float32, copy=False),
        "z": US_DS4_voxel_hits["z"].astype(np.float32, copy=False),
        "qdc": US_DS4_voxel_hits["qdc"].astype(np.float32, copy=False),
        "detType": np.full(len(US_DS4_voxel_hits["x"]), 3, dtype=np.int16),
    }

    return concatenate_hit_dicts([scifi_block, ds_block, us_ds4_block], keys, dtypes)


# ==========================================================
# Main hit processing
# ==========================================================

def process_hits_numpy(
    args,
    event,
    snd_geo,
    voxel_size=(1.0, 1.0, 1.0),
    us_dim=(83.5, 6.0, 1.0),
    ds4_dim=(1.0, 63.5, 1.0),
):
    Scifi = snd_geo.modules["Scifi"]
    MuFilter = snd_geo.modules["MuFilter"]
    A, B = ROOT.TVector3(), ROOT.TVector3()

    SciFi_hits = []

    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue

        detID = aHit.GetDetectorID()
        station = int(detID // 1000000)
        hitTimeCY = float(aHit.GetTime() / 6.25)
        qdc = float(aHit.GetSignal(0))

        mat = aHit.GetMat()
        sipm = aHit.GetSiPM()
        channel = aHit.GetSiPMChan()
        layer_channel = int(channel + sipm * 128 + mat * 4 * 128)

        Scifi.GetSiPMPosition(detID, A, B)
        Ax, Ay, Az = A.x(), A.y(), A.z()
        Bx, By, Bz = B.x(), B.y(), B.z()

        SciFi_hits.append({
            "station": station,
            "isVertical": bool(aHit.isVertical()),
            "layer_channel": layer_channel,
            "qdc": qdc,
            "hitTimeCY": hitTimeCY,
            "x_mid": 0.5 * (Ax + Bx),
            "y_mid": 0.5 * (Ay + By),
            "z_mid": 0.5 * (Az + Bz),
        })

    MuFilter_hits = []

    for aHit in event.Digi_MuFilterHits:
        if not aHit.isValid():
            continue

        detID = aHit.GetDetectorID()
        MuFilter.GetPosition(detID, A, B)

        detType = int(aHit.GetSystem())
        station = int((detID // 1000) % 10 + 1)
        hitTimeCY = float(aHit.GetTime() / 6.25)
        barIndex = int(detID % 100)

        qdc = 0.0
        for _, value in aHit.GetAllSignals():
            qdc += float(value)

        Ax, Ay, Az = A.x(), A.y(), A.z()
        Bx, By, Bz = B.x(), B.y(), B.z()

        MuFilter_hits.append({
            "detType": detType,
            "station": station,
            "isVertical": bool(aHit.isVertical()),
            "barIndex": barIndex,
            "qdc": qdc,
            "hitTimeCY": hitTimeCY,
            "x_mid": 0.5 * (Ax + Bx),
            "y_mid": 0.5 * (Ay + By),
            "z_mid": 0.5 * (Az + Bz),
        })

    SciFi_hits, time_stats = filter_SciFiHits_time(
        SciFi_hits,
        lower_time_threshold=args.scifi_time_lower,
        upper_time_threshold=args.scifi_time_upper,
        bin_width=args.scifi_time_bin_width,
    )

    SciFi_hits, qdc_stats = filter_SciFiHits_qdc(
        SciFi_hits,
        mode=args.scifi_qdc_mode,
        verbose=args.verbose_qdc_filter,
    )

    # SciFi 3D hits
    scifi_by_station = defaultdict(list)
    for h in SciFi_hits:
        scifi_by_station[h["station"]].append(h)

    scifi_station_arrays = []
    for station, hits in scifi_by_station.items():
        vertical_hits = [h for h in hits if h["isVertical"]]
        horizontal_hits = [h for h in hits if not h["isVertical"]]

        arr = combine_station_hits_to_3d(
            vertical_hits,
            horizontal_hits,
            vertical_id_key="layer_channel",
            horizontal_id_key="layer_channel",
        )
        if len(arr["x"]) > 0:
            scifi_station_arrays.append(arr)

    if scifi_station_arrays:
        SciFi_3D_hits = {
            "station": np.concatenate([a["station"] for a in scifi_station_arrays]),
            "x": np.concatenate([a["x"] for a in scifi_station_arrays]),
            "y": np.concatenate([a["y"] for a in scifi_station_arrays]),
            "z": np.concatenate([a["z"] for a in scifi_station_arrays]),
            "qdc": np.concatenate([a["qdc"] for a in scifi_station_arrays]),
            "vertical_layer_channel": np.concatenate([a["vertical_id"] for a in scifi_station_arrays]),
            "horizontal_layer_channel": np.concatenate([a["horizontal_id"] for a in scifi_station_arrays]),
        }
    else:
        SciFi_3D_hits = empty_hit_arrays(
            ["station", "x", "y", "z", "qdc", "vertical_layer_channel", "horizontal_layer_channel"],
            [np.int16, np.float32, np.float32, np.float32, np.float32, np.int32, np.int32],
        )

    # DS1/2/3 3D hits
    ds_hits = [h for h in MuFilter_hits if h["detType"] == 3 and h["station"] in (1, 2, 3)]
    ds_by_station = defaultdict(list)
    for h in ds_hits:
        ds_by_station[h["station"]].append(h)

    ds_station_arrays = []
    for station, hits in ds_by_station.items():
        vertical_hits = [h for h in hits if h["isVertical"]]
        horizontal_hits = [h for h in hits if not h["isVertical"]]

        arr = combine_station_hits_to_3d(
            vertical_hits,
            horizontal_hits,
            vertical_id_key="barIndex",
            horizontal_id_key="barIndex",
        )
        if len(arr["x"]) > 0:
            ds_station_arrays.append(arr)

    if ds_station_arrays:
        DS_3D_hits = {
            "station": np.concatenate([a["station"] for a in ds_station_arrays]),
            "x": np.concatenate([a["x"] for a in ds_station_arrays]),
            "y": np.concatenate([a["y"] for a in ds_station_arrays]),
            "z": np.concatenate([a["z"] for a in ds_station_arrays]),
            "qdc": np.concatenate([a["qdc"] for a in ds_station_arrays]),
            "vertical_barIndex": np.concatenate([a["vertical_id"] for a in ds_station_arrays]),
            "horizontal_barIndex": np.concatenate([a["horizontal_id"] for a in ds_station_arrays]),
        }
    else:
        DS_3D_hits = empty_hit_arrays(
            ["station", "x", "y", "z", "qdc", "vertical_barIndex", "horizontal_barIndex"],
            [np.int16, np.float32, np.float32, np.float32, np.float32, np.int32, np.int32],
        )

    US_DS4_voxel_hits = build_centered_boxes_for_us_ds4(
        MuFilter_hits,
        voxel_size=voxel_size,
        us_dim=us_dim,
        ds4_dim=ds4_dim,
    )

    all_3dHits = build_all_3dHits(
        SciFi_3D_hits,
        DS_3D_hits,
        US_DS4_voxel_hits,
    )

    return all_3dHits


# ==========================================================
# Event metadata
# ==========================================================
def extract_event_metadata(args, raw_tree, entry_number):
    event = {
        "runId": int(raw_tree.EventHeader.GetRunId()),
        "eventIndex": int(entry_number),
        "eventId": -999,
        "pdgCode": -999,
        "energy": -999.0,
        "isMC": -999,
    }

    if "MC" in args.type:
        event["isMC"] = 1
        try:
            event["eventId"] = int(raw_tree.EventHeader.GetEventNumber())
        except Exception:
            event["eventId"] = int(raw_tree.EventHeader.GetMCEntryNumber())

        if len(raw_tree.MCTrack) == 0:
            return event

        event_pdg0 = int(raw_tree.MCTrack[0].GetPdgCode())
        event_pdg1 = int(raw_tree.MCTrack[1].GetPdgCode()) if len(raw_tree.MCTrack) > 1 else 0

        event["energy"] = float(raw_tree.MCTrack[0].GetEnergy())

        neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
        if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
            event["pdgCode"] = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
        else:
            event["pdgCode"] = event_pdg0

    elif "real" in args.type:
        event["isMC"] = 0
        event["pdgCode"] = 0
        event["eventId"] = int(raw_tree.EventHeader.GetEventNumber())

    return event


# ==========================================================
# Optional plotting
# ==========================================================

def plot_all_3dHits(
    all_3dHits,
    plot_outpath,
    figsize=(27, 21),
    log_qdc=False,
    alpha=0.7,
    dpi=200,
    show=False,
):
    """Plot merged all_3dHits and save to file."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    x = np.asarray(all_3dHits.get("x", []), dtype=np.float32)
    y = np.asarray(all_3dHits.get("y", []), dtype=np.float32)
    z = np.asarray(all_3dHits.get("z", []), dtype=np.float32)
    qdc = np.asarray(all_3dHits.get("qdc", []), dtype=np.float32)
    detType = np.asarray(all_3dHits.get("detType", []), dtype=np.int16)

    if len(x) == 0:
        fig.suptitle("all_3dHits is empty")
        fig.savefig(plot_outpath, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return

    def _get_norm(values):
        if log_qdc:
            vals = np.clip(values, 1e-6, None)
            return vals, LogNorm(vmin=vals.min(), vmax=vals.max())
        return values, None

    scatters = []
    style_map = {
        1: {"label": "SciFi", "marker": "o", "size": 18, "cmap": "viridis", "alpha": alpha},
        2: {"label": "DS", "marker": "^", "size": 24, "cmap": "plasma", "alpha": alpha},
        3: {"label": "US/DS4", "marker": "s", "size": 10, "cmap": "magma", "alpha": alpha * 0.7},
    }

    for dt in [1, 2, 3]:
        mask = detType == dt
        if not np.any(mask):
            continue

        q_plot, norm = _get_norm(qdc[mask])
        style = style_map[dt]

        sc = ax.scatter(
            x[mask],
            y[mask],
            z[mask],
            c=q_plot,
            cmap=style["cmap"],
            norm=norm,
            s=style["size"],
            alpha=style["alpha"],
            marker=style["marker"],
            label=style["label"],
        )
        scatters.append(sc)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Merged 3D hits")
    ax.legend(loc="best")

    _set_axes_equal_3d_from_all_hits(ax, all_3dHits)

    if scatters:
        cbar = fig.colorbar(scatters[-1], ax=ax, pad=0.08, shrink=0.8)
        cbar.set_label("QDC")

    plt.tight_layout()
    fig.savefig(plot_outpath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def _set_axes_equal_3d_from_all_hits(ax, all_3dHits):
    x = np.asarray(all_3dHits.get("x", []), dtype=float)
    y = np.asarray(all_3dHits.get("y", []), dtype=float)
    z = np.asarray(all_3dHits.get("z", []), dtype=float)

    if len(x) == 0:
        return

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    half = 0.5 * max_range

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


# ==========================================================
# Main
# ==========================================================

def make_metadata(args, selection, n_selected):
    return {
        "selection": selection,
        "n_selected": int(n_selected),
        "type": args.type,
        "mode": args.mode,
        "dataset_split": args.dataset_split,
        "scifi_qdc_mode": args.scifi_qdc_mode,
        "scifi_time_lower": args.scifi_time_lower,
        "scifi_time_upper": args.scifi_time_upper,
        "scifi_time_bin_width": args.scifi_time_bin_width,
        "digi_path": args.digi_path,
        "preSelect_path": args.preSelect_path,
        "geo_path": args.geo_path,
    }


def main(args):
    print("start processing digi to hits3D")

    snd_geo = setup_geometry(args.geo_path)
    raw_data, raw_tree = open_root_file(args.digi_path)
    preSelect_data, preSelect_tree = open_root_file(args.preSelect_path, tree_name="cutFlowSummary")

    preSelect_tree.SetAlias("EventDeltat_m1_100", "EventDeltat_-1_100")

    selection = build_selection(args)

    print(f"Dataset split: {args.dataset_split}")
    print(f"Applying selection: {selection}")

    n_match = preSelect_tree.GetEntries(selection)
    print(f"Entries matching selection: {n_match}")

    if n_match == 0:
        output = {
            "events": [],
            "metadata": make_metadata(args, selection, 0),
        }
        save_dataset(output, args.out_path)
        print("No entries matched the selection condition, saved empty dataset")
        raw_data.Close()
        preSelect_data.Close()
        return 0

    elist_name = "elist"
    preSelect_tree.Draw(f">>{elist_name}", selection, "entrylist")
    elist = ROOT.gDirectory.Get(elist_name)

    if not elist or not isinstance(elist, ROOT.TEntryList):
        raise RuntimeError("Failed to create or retrieve TEntryList")

    preSelect_tree.SetEntryList(elist)

    events = []

    for i in range(elist.GetN()):
        if i % args.progress_every == 0:
            print(f"processed {i} / {elist.GetN()} events")

        entry_number = elist.GetEntry(i)
        raw_tree.GetEntry(entry_number)
        preSelect_tree.GetEntry(entry_number)

        event = extract_event_metadata(args, raw_tree, entry_number)
        all_3dHits = process_hits_numpy(args, raw_tree, snd_geo)

        event["all_3dHits"] = all_3dHits
        event["label"] = particle_to_target.get(event["pdgCode"], -1)
        event["particle_name"] = particle_mapping.get(event["pdgCode"], "unknown")
        events.append(event)

    output = {
        "events": events,
        "metadata": make_metadata(args, selection, len(events)),
    }

    save_dataset(output, args.out_path)

    raw_data.Close()
    preSelect_data.Close()

    print(f"finish processing digi to hits3D, saved {len(events)} events to {args.out_path}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-p", "--preSelectPath", dest="preSelect_path", help="pre selection data file path", required=True)
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default="RECREATE")
    parser.add_argument("-t", "--type", dest="type", help="data type, e.g. MC or real", required=True)

    parser.add_argument(
        "--dataset-split",
        dest="dataset_split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split. train/val use strict cuts; test uses looser feature-file cuts.",
    )

    parser.add_argument(
        "--scifi-qdc-mode",
        dest="scifi_qdc_mode",
        choices=["mean", "median", "half_maxQDC"],
        default="mean",
        help="QDC threshold mode for SciFi hit filtering.",
    )

    parser.add_argument("--scifi-time-lower", dest="scifi_time_lower", type=float, default=0.5)
    parser.add_argument("--scifi-time-upper", dest="scifi_time_upper", type=float, default=2.3)
    parser.add_argument("--scifi-time-bin-width", dest="scifi_time_bin_width", type=float, default=0.25)

    parser.add_argument(
        "--verbose-qdc-filter",
        dest="verbose_qdc_filter",
        action="store_true",
        help="Print QDC filter stats for every event. This can be very verbose.",
    )

    parser.add_argument(
        "--progress-every",
        dest="progress_every",
        type=int,
        default=10000,
        help="Print progress every N selected events.",
    )

    args = parser.parse_args()
    raise SystemExit(main(args))
