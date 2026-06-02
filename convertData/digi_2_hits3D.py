import ROOT
import array
import os
import time
from argparse import ArgumentParser
import SndlhcGeo
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from analysis.analyses.snd_analysis_2024_0mu.sciFiTools import selectHits

ROOT.TH1.AddDirectory(False)


TIMING_KEYS = [
    "entry_load",
    "metadata",
    "scifi_select",
    "scifi_build",
    "mufilter_build",
    "scifi_cross",
    "ds_cross",
    "us_ds4_voxel",
    "merge_hits",
    "root_fill",
    "root_write",
]


def add_timing(timings, key, elapsed):
    if timings is not None:
        timings[key] += elapsed


def print_timing_summary(timings, n_events):
    if not timings or n_events <= 0:
        return

    print("Timing summary:")
    total = sum(timings.values())
    for key in TIMING_KEYS:
        seconds = timings.get(key, 0.0)
        ms_per_event = 1000.0 * seconds / n_events
        print(f"  {key:16s}: {seconds:9.3f} s  ({ms_per_event:8.3f} ms/event)")
    print(f"  {'measured_total':16s}: {total:9.3f} s  ({1000.0 * total / n_events:8.3f} ms/event)")


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


def init_event_geometry(snd_geo, event_header):
    snd_geo.modules["Scifi"].InitEvent(event_header)
    snd_geo.modules["MuFilter"].InitEvent(event_header)


def open_root_file(file_path, tree_name="cbmsim", mode="read"):
    root_file = ROOT.TFile(file_path, mode)

    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {file_path}")

    tree = root_file.Get(tree_name)
    if not tree or not isinstance(tree, ROOT.TTree):
        raise RuntimeError(f"TTree '{tree_name}' not found in {file_path}")

    return root_file, tree


def is_muonDIS_sample(args):
    """Identify muonDIS jobs from the common path/type arguments."""
    fields = ("out_path", "type", "digi_path", "preSelect_path")
    return any("muondis" in str(getattr(args, field, "")).lower() for field in fields)


def non_negative_float(value):
    try:
        return max(float(value), 0.0)
    except Exception:
        return 0.0


def setup_event_deltat_alias(preSelect_tree):
    """Make EventDeltat_m1_100 usable for trees with different sanitized names."""
    alias_name = "EventDeltat_m1_100"
    if preSelect_tree.GetBranch(alias_name) or preSelect_tree.GetLeaf(alias_name):
        return alias_name

    for candidate in ("EventDeltat_1_100", "EventDeltat_-1_100"):
        if preSelect_tree.GetBranch(candidate) or preSelect_tree.GetLeaf(candidate):
            preSelect_tree.SetAlias(alias_name, candidate)
            return candidate

    return None


def build_selection(args, preSelect_tree):
    """Build the same broad event selection used by digi_2_features.py."""
    selection = "SciFiMinHits == 1"
    if "MC" not in args.type:
        event_deltat_branch = setup_event_deltat_alias(preSelect_tree)
        if not event_deltat_branch:
            raise RuntimeError("No EventDeltat cut branch found for real-data selection")

        data_selection = "StableBeams == 1 && IP1 == 1 && EventDeltat_m1_100 == 1"
        selection = selection + " && " + data_selection

    return selection


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


def combine_station_hits_to_3d(vertical_hits, horizontal_hits):
    """Combine one station's vertical and horizontal hits into crossed 3D hits."""
    if not vertical_hits or not horizontal_hits:
        return empty_hit_arrays(
            ["station", "x", "y", "z", "qdc"],
            [np.int16, np.float32, np.float32, np.float32, np.float32],
        )

    station = vertical_hits[0]["station"]

    vx = np.array([h["x_mid"] for h in vertical_hits], dtype=np.float32)
    vz = np.array([h["z_mid"] for h in vertical_hits], dtype=np.float32)
    vq = np.array([h["qdc"] for h in vertical_hits], dtype=np.float32)

    hy = np.array([h["y_mid"] for h in horizontal_hits], dtype=np.float32)
    hz = np.array([h["z_mid"] for h in horizontal_hits], dtype=np.float32)
    hq = np.array([h["qdc"] for h in horizontal_hits], dtype=np.float32)

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
    }


def make_centered_box_offsets(dim, voxel_size):
    """Build reusable voxel step grids for the centered-box formulation."""
    sx, sy, sz = dim
    dx, dy, dz = voxel_size

    nx = max(1, int(round(sx / dx)))
    ny = max(1, int(round(sy / dy)))
    nz = max(1, int(round(sz / dz)))

    half_x = 0.5 * nx * dx
    half_y = 0.5 * ny * dy
    half_z = 0.5 * nz * dz

    xs = (np.arange(nx, dtype=np.float32) + 0.5) * dx
    ys = (np.arange(ny, dtype=np.float32) + 0.5) * dy
    zs = (np.arange(nz, dtype=np.float32) + 0.5) * dz

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X.ravel(), Y.ravel(), Z.ravel(), half_x, half_y, half_z


def build_centered_boxes_for_us_ds4(
    MuFilter_hits,
    voxel_size=(1.0, 1.0, 1.0),
    us_dim=(83.5, 6.0, 1.0),
    ds4_dim=(1.0, 63.5, 1.0),
):
    """Build voxelized 3D hits distributed inside US and DS4 bars."""
    us_offsets = make_centered_box_offsets(us_dim, voxel_size)
    ds4_offsets = make_centered_box_offsets(ds4_dim, voxel_size)

    selected_hits = []
    offset_sets = []

    for h in MuFilter_hits:
        if h["detType"] == 2:
            selected_hits.append(h)
            offset_sets.append(us_offsets)
        elif h["detType"] == 3 and h["station"] == 4:
            selected_hits.append(h)
            offset_sets.append(ds4_offsets)

    if not selected_hits:
        return empty_hit_arrays(
            ["x", "y", "z", "qdc", "station"],
            [np.float32, np.float32, np.float32, np.float32, np.int16],
        )

    x_all, y_all, z_all = [], [], []
    qdc_all, station_all = [], []

    for i, (h, voxel_grid) in enumerate(zip(selected_hits, offset_sets)):
        step_x, step_y, step_z, half_x, half_y, half_z = voxel_grid
        nvox = len(step_x)

        x_all.append((h["x_mid"] - half_x) + step_x)
        y_all.append((h["y_mid"] - half_y) + step_y)
        z_all.append((h["z_mid"] - half_z) + step_z)
        qdc_all.append(np.full(nvox, float(h["qdc"]) / nvox, dtype=np.float32))
        station_all.append(np.full(nvox, h["station"], dtype=np.int16))

    return {
        "x": np.concatenate(x_all),
        "y": np.concatenate(y_all),
        "z": np.concatenate(z_all),
        "qdc": np.concatenate(qdc_all),
        "station": np.concatenate(station_all),
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
# ROOT output helpers
# ==========================================================

def create_output_root(path, mode, compression_level=9):
    dir_name = os.path.dirname(path)
    if dir_name and not path.startswith("root://"):
        os.makedirs(dir_name, exist_ok=True)

    out_file = ROOT.TFile(path, mode, "", compression_level)
    if not out_file or out_file.IsZombie():
        raise RuntimeError(f"Could not create ROOT file: {path}")
    out_file.SetCompressionLevel(compression_level)

    tree = ROOT.TTree("hit3D", "converted SND 3D hits tree")
    tree.SetDirectory(out_file)

    branch_vars = {
        "runId": array.array("i", [-999]),
        "eventId": array.array("i", [-999]),
        "eventIndex": array.array("i", [-999]),
        "pdgCode": array.array("i", [-999]),
        "isMC": array.array("i", [-999]),
        "label": array.array("i", [-999]),
        "energy": array.array("d", [-999.0]),
    }

    for name, value in branch_vars.items():
        dtype = "D" if value.typecode == "d" else "I"
        tree.Branch(name, value, f"{name}/{dtype}")

    vector_vars = {
        "hit_x": ROOT.std.vector("float")(),
        "hit_y": ROOT.std.vector("float")(),
        "hit_z": ROOT.std.vector("float")(),
        "hit_qdc": ROOT.std.vector("float")(),
        "hit_station": ROOT.std.vector("short")(),
        "hit_detType": ROOT.std.vector("short")(),
    }

    for name, vec in vector_vars.items():
        tree.Branch(name, vec)

    return out_file, tree, branch_vars, vector_vars


def reset_output_branches(branch_vars, vector_vars):
    for value in branch_vars.values():
        value[0] = -999.0 if value.typecode == "d" else -999

    for vec in vector_vars.values():
        vec.clear()


def fill_event_branches(event, branch_vars):
    branch_vars["runId"][0] = int(event.get("runId", -999))
    branch_vars["eventId"][0] = int(event.get("eventId", -999))
    branch_vars["eventIndex"][0] = int(event.get("eventIndex", -999))
    branch_vars["pdgCode"][0] = int(event.get("pdgCode", -999))
    branch_vars["isMC"][0] = int(event.get("isMC", -999))
    branch_vars["label"][0] = int(event.get("label", -999))
    branch_vars["energy"][0] = float(event.get("energy", -999.0))


def assign_vector(vec, values):
    vec.clear()
    vec.assign(values)


def fill_hit_vectors(all_3dHits, vector_vars):
    x = np.asarray(all_3dHits.get("x", []), dtype=np.float32)
    y = np.asarray(all_3dHits.get("y", []), dtype=np.float32)
    z = np.asarray(all_3dHits.get("z", []), dtype=np.float32)
    qdc = np.asarray(all_3dHits.get("qdc", []), dtype=np.float32)
    station = np.asarray(all_3dHits.get("station", []), dtype=np.int16)
    det_type = np.asarray(all_3dHits.get("detType", []), dtype=np.int16)

    lengths = {len(x), len(y), len(z), len(qdc), len(station), len(det_type)}
    if len(lengths) != 1:
        raise RuntimeError(f"Inconsistent hit3D array lengths: {sorted(lengths)}")

    assign_vector(vector_vars["hit_x"], x)
    assign_vector(vector_vars["hit_y"], y)
    assign_vector(vector_vars["hit_z"], z)
    assign_vector(vector_vars["hit_qdc"], qdc)
    assign_vector(vector_vars["hit_station"], station)
    assign_vector(vector_vars["hit_detType"], det_type)


def write_metadata_objects(out_file, metadata):
    out_file.cd()
    for key, value in metadata.items():
        ROOT.TNamed(f"metadata_{key}", str(value)).Write()


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
    timings=None,
):
    Scifi = snd_geo.modules["Scifi"]
    MuFilter = snd_geo.modules["MuFilter"]
    A, B = ROOT.TVector3(), ROOT.TVector3()

    t0 = time.perf_counter()
    if is_muonDIS_sample(args):
        selected_scifi_hits = [hit for hit in event.Digi_ScifiHits if hit.isValid()]
    else:
        selected_scifi_hits = selectHits(event, MC=("MC" in args.type))
    add_timing(timings, "scifi_select", time.perf_counter() - t0)

    t0 = time.perf_counter()
    SciFi_hits = []

    for aHit in selected_scifi_hits:
        if not aHit.isValid():
            continue

        detID = aHit.GetDetectorID()
        station = int(aHit.GetStation())
        qdc = non_negative_float(aHit.GetSignal(0))

        Scifi.GetSiPMPosition(detID, A, B)
        Ax, Ay, Az = A.x(), A.y(), A.z()
        Bx, By, Bz = B.x(), B.y(), B.z()

        SciFi_hits.append({
            "station": station,
            "isVertical": bool(aHit.isVertical()),
            "qdc": qdc,
            "x_mid": 0.5 * (Ax + Bx),
            "y_mid": 0.5 * (Ay + By),
            "z_mid": 0.5 * (Az + Bz),
        })
    add_timing(timings, "scifi_build", time.perf_counter() - t0)

    t0 = time.perf_counter()
    MuFilter_hits = []

    for aHit in event.Digi_MuFilterHits:
        if not aHit.isValid():
            continue

        detID = aHit.GetDetectorID()
        MuFilter.GetPosition(detID, A, B)

        detType = int(aHit.GetSystem())
        station = int((detID // 1000) % 10 + 1)

        qdc = 0.0
        for _, value in aHit.GetAllSignals():
            qdc += non_negative_float(value)

        Ax, Ay, Az = A.x(), A.y(), A.z()
        Bx, By, Bz = B.x(), B.y(), B.z()

        MuFilter_hits.append({
            "detType": detType,
            "station": station,
            "isVertical": bool(aHit.isVertical()),
            "qdc": qdc,
            "x_mid": 0.5 * (Ax + Bx),
            "y_mid": 0.5 * (Ay + By),
            "z_mid": 0.5 * (Az + Bz),
        })
    add_timing(timings, "mufilter_build", time.perf_counter() - t0)

    # SciFi 3D hits
    t0 = time.perf_counter()
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
        }
    else:
        SciFi_3D_hits = empty_hit_arrays(
            ["station", "x", "y", "z", "qdc"],
            [np.int16, np.float32, np.float32, np.float32, np.float32],
        )
    add_timing(timings, "scifi_cross", time.perf_counter() - t0)

    # DS1/2/3 3D hits
    t0 = time.perf_counter()
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
        }
    else:
        DS_3D_hits = empty_hit_arrays(
            ["station", "x", "y", "z", "qdc"],
            [np.int16, np.float32, np.float32, np.float32, np.float32],
        )
    add_timing(timings, "ds_cross", time.perf_counter() - t0)

    t0 = time.perf_counter()
    US_DS4_voxel_hits = build_centered_boxes_for_us_ds4(
        MuFilter_hits,
        voxel_size=voxel_size,
        us_dim=us_dim,
        ds4_dim=ds4_dim,
    )
    add_timing(timings, "us_ds4_voxel", time.perf_counter() - t0)

    t0 = time.perf_counter()
    all_3dHits = build_all_3dHits(
        SciFi_3D_hits,
        DS_3D_hits,
        US_DS4_voxel_hits,
    )
    add_timing(timings, "merge_hits", time.perf_counter() - t0)

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
        "digi_path": args.digi_path,
        "preSelect_path": args.preSelect_path,
        "geo_path": args.geo_path,
        "compression_level": args.compression_level,
    }


def main(args):
    print("start processing digi to hits3D")

    snd_geo = setup_geometry(args.geo_path)
    raw_data, raw_tree = open_root_file(args.digi_path)
    preSelect_data, preSelect_tree = open_root_file(args.preSelect_path, tree_name="cutFlowSummary")
    out_file, hit_tree, branch_vars, vector_vars = create_output_root(
        args.out_path,
        args.mode,
        compression_level=args.compression_level,
    )

    selection = build_selection(args, preSelect_tree)

    print("Using feature-aligned hit3D event selection")
    print(f"Applying selection: {selection}")

    n_match = preSelect_tree.GetEntries(selection)
    print(f"Entries matching selection: {n_match}")

    if n_match == 0:
        hit_tree.Write()
        write_metadata_objects(out_file, make_metadata(args, selection, 0))
        out_file.Close()
        raw_data.Close()
        preSelect_data.Close()
        print("No entries matched the selection condition, saved empty hit3D tree")
        return 0

    elist_name = "elist"
    preSelect_tree.Draw(f">>{elist_name}", selection, "entrylist")
    elist = ROOT.gDirectory.Get(elist_name)

    if not elist or not isinstance(elist, ROOT.TEntryList):
        raise RuntimeError("Failed to create or retrieve TEntryList")

    preSelect_tree.SetEntryList(elist)
    n_selected = int(elist.GetN())
    timings = {key: 0.0 for key in TIMING_KEYS}

    for i in tqdm(range(n_selected), desc="processing hit3D events", mininterval=30):
        t0 = time.perf_counter()
        cutflow_entry = elist.GetEntry(i)
        preSelect_tree.GetEntry(cutflow_entry)

        if hasattr(preSelect_tree, "entry"):
            raw_entry = int(preSelect_tree.entry)
        else:
            raw_entry = cutflow_entry

        raw_tree.GetEntry(raw_entry)
        init_event_geometry(snd_geo, raw_tree.EventHeader)
        add_timing(timings, "entry_load", time.perf_counter() - t0)

        t0 = time.perf_counter()
        event = extract_event_metadata(args, raw_tree, raw_entry)
        event["label"] = particle_to_target.get(event["pdgCode"], -1)
        add_timing(timings, "metadata", time.perf_counter() - t0)

        all_3dHits = process_hits_numpy(args, raw_tree, snd_geo, timings=timings)

        t0 = time.perf_counter()
        reset_output_branches(branch_vars, vector_vars)
        fill_event_branches(event, branch_vars)
        fill_hit_vectors(all_3dHits, vector_vars)
        hit_tree.Fill()
        add_timing(timings, "root_fill", time.perf_counter() - t0)

    t0 = time.perf_counter()
    hit_tree.Write()
    write_metadata_objects(out_file, make_metadata(args, selection, n_selected))
    add_timing(timings, "root_write", time.perf_counter() - t0)
    out_file.Close()
    raw_data.Close()
    preSelect_data.Close()

    print_timing_summary(timings, n_selected)
    print(f"finish processing digi to hits3D, saved {n_selected} events to {args.out_path}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-p", "--preSelectPath", dest="preSelect_path", help="pre selection data file path", required=True)
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output ROOT path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default="RECREATE")
    parser.add_argument("-t", "--type", dest="type", help="data type, e.g. MC or real", required=True)
    parser.add_argument(
        "--compression-level",
        dest="compression_level",
        type=int,
        default=9,
        help="ROOT compression level for the output file.",
    )

    args = parser.parse_args()
    if not 0 <= args.compression_level <= 9:
        parser.error("--compression-level must be between 0 and 9")
    raise SystemExit(main(args))
