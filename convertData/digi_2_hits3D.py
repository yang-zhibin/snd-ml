import ROOT
import array
import os
import time
import math
from argparse import ArgumentParser
import SndlhcGeo
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from analysis.analyses.snd_analysis_2024_0mu.sciFiTools import selectHits
from hit_time_selection import (
    apply_first_arrival_time0 as apply_muon_first_arrival_time0,
    is_muon_time_calibrated_sample,
    select_calibrated_scifi_hits,
)
from sample_selection import is_passing_muon_real_data, uses_external_cutflow

ROOT.TH1.AddDirectory(False)

NS_PER_CLOCK = 6.25
INVALID_TIME = -999.0
MC_HIT_TIME_WINDOW_NS = 25.0



TIMING_KEYS = [
    "entry_load",
    "metadata",
    "scifi_select",
    "scifi_build",
    "mufilter_build",
    "scifi_calibrated_select",
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


INDEX_UNKNOWN = 0
INDEX_SCIFI_CROSSED = 1
INDEX_US_VOXEL = 2
INDEX_DS_CROSSED = 3
INDEX_DS_SINGLE_ORIENTATION_VOXEL = 4
INDEX_SCIFI_VERTICAL_ONLY = 5
INDEX_SCIFI_HORIZONTAL_ONLY = 6
INDEX_DS_VERTICAL_ONLY = 7
INDEX_DS_HORIZONTAL_ONLY = 8


HIT_KEYS = [
    "station",
    "x",
    "y",
    "z",
    "qdc",
    "detType",
    "ix",
    "iy",
    "iz",
    "index_valid",
    "index_type",
    "v_channel",
    "h_channel",
    "v_qdc",
    "h_qdc",
    "time",
]
HIT_DTYPES = [
    np.int16,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.int16,
    np.int32,
    np.int32,
    np.int32,
    np.int16,
    np.int16,
    np.int32,
    np.int32,
    np.float32,
    np.float32,
    np.float32,
]

NATIVE_SCIFI_KEYS = [
    "station", "orientation", "channel", "det_id",
    "x", "y", "z", "qdc", "time_ns",
]
NATIVE_SCIFI_DTYPES = [
    np.int16, np.int16, np.int32, np.int64,
    np.float32, np.float32, np.float32, np.float32, np.float32,
]
NATIVE_SCIFI_SCHEMA = "scifi_native_hits_v1"

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


def is_mc_sample(args):
    return "MC" in str(args.type)


def needs_first_arrival_time0(args):
    return is_muon_time_calibrated_sample(args)


def applies_mc_25ns_hit_filter(args):
    fields = ("out_path", "type", "digi_path", "preSelect_path")
    text = " ".join(str(getattr(args, field, "")) for field in fields).lower()
    return is_mc_sample(args) and "mc_neutrino" not in text


def skip_scifi_minhits_selection(args):
    """Process all cutFlowSummary entries for muon MC samples."""
    sample_type = str(args.type).lower()
    return sample_type in ("mc_muon", "mc_muondis")


def has_veto_and_us_hits(event):
    has_veto = False
    has_us = False

    for hit in event.Digi_MuFilterHits:
        if not hit.isValid():
            continue

        system = int(hit.GetSystem())
        if system == 1:
            has_veto = True
        elif system == 2:
            has_us = True

        if has_veto and has_us:
            return True

    return False


def should_drop_real_has_veto_has_us(args, event):
    return (
        bool(getattr(args, "drop_real_has_veto_has_us", False))
        and "real" in str(args.type)
        and has_veto_and_us_hits(event)
    )


def non_negative_float(value):
    try:
        return max(float(value), 0.0)
    except Exception:
        return 0.0


def valid_time(value):
    try:
        value = float(value)
    except Exception:
        return False
    return math.isfinite(value) and value > -900.0


def scifi_time_ns(hit, scifi_det, args):
    if is_mc_sample(args):
        return float(hit.GetTime())

    raw_time_ns = float(hit.GetTime()) * NS_PER_CLOCK
    return float(scifi_det.GetCorrectedTime(hit.GetDetectorID(), raw_time_ns, 0))


def get_mufilter_all_times(hit):
    try:
        raw_times = hit.GetAllTimes()
    except TypeError:
        try:
            raw_times = hit.GetAllTimes(False)
        except Exception:
            return {}
    except Exception:
        return {}

    times = {}
    try:
        iterator = raw_times.items()
    except AttributeError:
        iterator = raw_times

    for item in iterator:
        try:
            channel, time_value = item
        except (TypeError, ValueError):
            try:
                channel, time_value = item[0], item[1]
            except Exception:
                continue
        times[int(channel)] = float(time_value)

    return times


def mufilter_channel_times_ns(hit, mufilter_det, args):
    det_id = int(hit.GetDetectorID())
    system = int(hit.GetSystem())
    raw_times = get_mufilter_all_times(hit)

    if not raw_times:
        try:
            raw_times = {0: hit.GetTime()}
        except Exception:
            raw_times = {}

    n_sipms = None
    if system == 3:
        try:
            n_sipms = int(mufilter_det.GetnSiPMs(det_id))
        except Exception:
            n_sipms = None

    times = []
    for channel, raw_time in raw_times.items():
        if not valid_time(raw_time):
            continue

        if is_mc_sample(args):
            time_ns = float(raw_time)
        else:
            time_ns = float(raw_time) * NS_PER_CLOCK
            if system == 3 and n_sipms:
                side = 0 if int(channel) < n_sipms else 1
                try:
                    time_ns = float(mufilter_det.GetCorrectedTime(det_id, side, time_ns, 0))
                except Exception:
                    pass

        times.append(time_ns)

    return times


def mufilter_hit_time_ns(hit, mufilter_det, args):
    times = mufilter_channel_times_ns(hit, mufilter_det, args)
    return min(times) if times else INVALID_TIME


def apply_first_arrival_time0(args, hit_collections, time_key="time"):
    return apply_muon_first_arrival_time0(
        args,
        hit_collections,
        time_key=time_key,
        invalid_time=INVALID_TIME,
    )


def keep_hit_in_mc_time_window(args, hit, time_key="time"):
    if not applies_mc_25ns_hit_filter(args):
        return True

    time_ns = hit.get(time_key, INVALID_TIME)
    if not valid_time(time_ns):
        return True

    return float(time_ns) <= MC_HIT_TIME_WINDOW_NS


def filter_late_mc_hits(args, hits, time_key="time"):
    return [hit for hit in hits if keep_hit_in_mc_time_window(args, hit, time_key)]


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
    selection = None if skip_scifi_minhits_selection(args) else "SciFiMinHits == 1"
    if "MC" not in args.type:
        event_deltat_branch = setup_event_deltat_alias(preSelect_tree)
        if not event_deltat_branch:
            raise RuntimeError("No EventDeltat cut branch found for real-data selection")

        data_selection = "StableBeams == 1 && IP1 == 1 && EventDeltat_m1_100 == 1"
        selection = selection + " && " + data_selection if selection else data_selection

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


def scifi_channel_info(hit):
    mat = int(hit.GetMat())
    sipm = int(hit.GetSiPM())
    sipm_channel = int(hit.GetSiPMChan())
    layer_channel = sipm_channel + sipm * 128 + mat * 4 * 128
    return mat, sipm, sipm_channel, layer_channel


def mufilter_channel_info(det_id):
    return int(det_id % 1000)


def empty_3d_hits():
    return empty_hit_arrays(HIT_KEYS, HIT_DTYPES)


def build_native_scifi_hits(scifi_hits):
    """Preserve every selected SciFi digi once, before Cartesian crossing."""
    if not scifi_hits:
        return empty_hit_arrays(NATIVE_SCIFI_KEYS, NATIVE_SCIFI_DTYPES)
    result = {
        "station": np.asarray([h["station"] for h in scifi_hits], dtype=np.int16),
        # The training contract defines 0=vertical and 1=horizontal.
        "orientation": np.asarray(
            [0 if h["isVertical"] else 1 for h in scifi_hits], dtype=np.int16
        ),
        "channel": np.asarray([h["channel"] for h in scifi_hits], dtype=np.int32),
        "det_id": np.asarray([h["det_id"] for h in scifi_hits], dtype=np.int64),
        "x": np.asarray([h["x_mid"] for h in scifi_hits], dtype=np.float32),
        "y": np.asarray([h["y_mid"] for h in scifi_hits], dtype=np.float32),
        "z": np.asarray([h["z_mid"] for h in scifi_hits], dtype=np.float32),
        "qdc": np.asarray([h["qdc"] for h in scifi_hits], dtype=np.float32),
        "time_ns": np.asarray([h["time"] for h in scifi_hits], dtype=np.float32),
    }
    lengths = {len(result[key]) for key in NATIVE_SCIFI_KEYS}
    if lengths != {len(scifi_hits)}:
        raise RuntimeError("Native SciFi hit vectors are not aligned")
    if np.any((result["station"] < 1) | (result["station"] > 5)):
        raise RuntimeError("Native SciFi station is outside [1,5]")
    if np.any(result["channel"] < 0) or np.any(result["det_id"] < 0):
        raise RuntimeError("Native SciFi channel/detector ID must be non-negative")
    for key in ("x", "y", "z", "qdc", "time_ns"):
        if not np.isfinite(result[key]).all():
            raise RuntimeError(f"Native SciFi {key} contains a non-finite value")
    if np.any(result["qdc"] < 0):
        raise RuntimeError("Native SciFi QDC must be non-negative")
    return result


def combine_station_hits_to_3d(vertical_hits, horizontal_hits, det_type, index_type):
    """Combine one station's vertical and horizontal hits into crossed 3D hits."""
    if not vertical_hits or not horizontal_hits:
        return empty_3d_hits()

    station = vertical_hits[0]["station"]

    vx = np.array([h["x_mid"] for h in vertical_hits], dtype=np.float32)
    vz = np.array([h["z_mid"] for h in vertical_hits], dtype=np.float32)
    vq = np.array([h["qdc"] for h in vertical_hits], dtype=np.float32)
    vc = np.array([h["channel"] for h in vertical_hits], dtype=np.int32)
    vt = np.array([h["time"] for h in vertical_hits], dtype=np.float32)

    hy = np.array([h["y_mid"] for h in horizontal_hits], dtype=np.float32)
    hz = np.array([h["z_mid"] for h in horizontal_hits], dtype=np.float32)
    hq = np.array([h["qdc"] for h in horizontal_hits], dtype=np.float32)
    hc = np.array([h["channel"] for h in horizontal_hits], dtype=np.int32)
    ht = np.array([h["time"] for h in horizontal_hits], dtype=np.float32)

    nv = len(vertical_hits)
    nh = len(horizontal_hits)
    n_crossed = nv * nh

    x = np.repeat(vx, nh)
    y = np.tile(hy, nv)
    z = 0.5 * (np.repeat(vz, nh) + np.tile(hz, nv))
    v_qdc = np.repeat(vq, nh)
    h_qdc = np.tile(hq, nv)
    qdc = v_qdc + h_qdc
    v_channel = np.repeat(vc, nh)
    h_channel = np.tile(hc, nv)
    hit_time = np.minimum(np.repeat(vt, nh), np.tile(ht, nv))

    return {
        "station": np.full(n_crossed, station, dtype=np.int16),
        "x": x,
        "y": y,
        "z": z,
        "qdc": qdc,
        "detType": np.full(n_crossed, det_type, dtype=np.int16),
        "ix": v_channel.astype(np.int32, copy=False),
        "iy": h_channel.astype(np.int32, copy=False),
        "iz": np.full(n_crossed, int(station) - 1, dtype=np.int32),
        "index_valid": np.ones(n_crossed, dtype=np.int16),
        "index_type": np.full(n_crossed, index_type, dtype=np.int16),
        "v_channel": v_channel.astype(np.int32, copy=False),
        "h_channel": h_channel.astype(np.int32, copy=False),
        "v_qdc": v_qdc.astype(np.float32, copy=False),
        "h_qdc": h_qdc.astype(np.float32, copy=False),
        "time": hit_time.astype(np.float32, copy=False),
    }


def build_one_orientation_hits(hits, det_type, index_type, known_axis):
    """Preserve hits with only one measured orientation as projection-like sparse hits."""
    if not hits:
        return empty_3d_hits()
    if known_axis not in {"x", "y"}:
        raise ValueError(f"known_axis must be 'x' or 'y', got {known_axis}")

    n_hits = len(hits)
    station = np.array([h["station"] for h in hits], dtype=np.int16)
    channel = np.array([h["channel"] for h in hits], dtype=np.int32)
    qdc = np.array([h["qdc"] for h in hits], dtype=np.float32)

    ix = channel.copy() if known_axis == "x" else np.full(n_hits, -1, dtype=np.int32)
    iy = channel.copy() if known_axis == "y" else np.full(n_hits, -1, dtype=np.int32)
    v_channel = channel.copy() if known_axis == "x" else np.full(n_hits, -1, dtype=np.int32)
    h_channel = channel.copy() if known_axis == "y" else np.full(n_hits, -1, dtype=np.int32)
    v_qdc = qdc.copy() if known_axis == "x" else np.zeros(n_hits, dtype=np.float32)
    h_qdc = qdc.copy() if known_axis == "y" else np.zeros(n_hits, dtype=np.float32)

    return {
        "station": station,
        "x": np.array([h["x_mid"] for h in hits], dtype=np.float32),
        "y": np.array([h["y_mid"] for h in hits], dtype=np.float32),
        "z": np.array([h["z_mid"] for h in hits], dtype=np.float32),
        "qdc": qdc,
        "detType": np.full(n_hits, det_type, dtype=np.int16),
        "ix": ix,
        "iy": iy,
        "iz": station.astype(np.int32) - 1,
        "index_valid": np.zeros(n_hits, dtype=np.int16),
        "index_type": np.full(n_hits, index_type, dtype=np.int16),
        "v_channel": v_channel,
        "h_channel": h_channel,
        "v_qdc": v_qdc,
        "h_qdc": h_qdc,
        "time": np.array([h["time"] for h in hits], dtype=np.float32),
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
    IX, IY, IZ = np.meshgrid(
        np.arange(nx, dtype=np.int32),
        np.arange(ny, dtype=np.int32),
        np.arange(nz, dtype=np.int32),
        indexing="ij",
    )
    return (
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
        IX.ravel(),
        IY.ravel(),
        IZ.ravel(),
        half_x,
        half_y,
        half_z,
    )


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
            offset_sets.append((us_offsets, INDEX_US_VOXEL))
        elif h["detType"] == 3 and h["station"] == 4:
            selected_hits.append(h)
            offset_sets.append((ds4_offsets, INDEX_DS_SINGLE_ORIENTATION_VOXEL))

    if not selected_hits:
        return empty_3d_hits()

    x_all, y_all, z_all = [], [], []
    qdc_all, station_all, det_type_all = [], [], []
    ix_all, iy_all, iz_all = [], [], []
    index_valid_all, index_type_all = [], []
    v_channel_all, h_channel_all = [], []
    v_qdc_all, h_qdc_all = [], []
    time_all = []

    for h, (voxel_grid, index_type) in zip(selected_hits, offset_sets):
        step_x, step_y, step_z, ix, iy, iz, half_x, half_y, half_z = voxel_grid
        nvox = len(step_x)
        channel = int(h.get("channel", -1))
        v_channel = channel if h.get("isVertical", False) else -1
        h_channel = -1 if h.get("isVertical", False) else channel
        v_qdc = float(h["qdc"]) if h.get("isVertical", False) else 0.0
        h_qdc = 0.0 if h.get("isVertical", False) else float(h["qdc"])

        x_all.append((h["x_mid"] - half_x) + step_x)
        y_all.append((h["y_mid"] - half_y) + step_y)
        z_all.append((h["z_mid"] - half_z) + step_z)
        qdc_all.append(np.full(nvox, float(h["qdc"]) / nvox, dtype=np.float32))
        station_all.append(np.full(nvox, h["station"], dtype=np.int16))
        det_type_all.append(np.full(nvox, h["detType"], dtype=np.int16))
        ix_all.append(ix.astype(np.int32, copy=False))
        iy_all.append(iy.astype(np.int32, copy=False))
        iz_all.append(iz.astype(np.int32, copy=False))
        index_valid_all.append(np.ones(nvox, dtype=np.int16))
        index_type_all.append(np.full(nvox, index_type, dtype=np.int16))
        v_channel_all.append(np.full(nvox, v_channel, dtype=np.int32))
        h_channel_all.append(np.full(nvox, h_channel, dtype=np.int32))
        v_qdc_all.append(np.full(nvox, v_qdc / nvox, dtype=np.float32))
        h_qdc_all.append(np.full(nvox, h_qdc / nvox, dtype=np.float32))
        time_all.append(np.full(nvox, float(h.get("time", INVALID_TIME)), dtype=np.float32))

    return {
        "station": np.concatenate(station_all),
        "x": np.concatenate(x_all),
        "y": np.concatenate(y_all),
        "z": np.concatenate(z_all),
        "qdc": np.concatenate(qdc_all),
        "detType": np.concatenate(det_type_all),
        "ix": np.concatenate(ix_all),
        "iy": np.concatenate(iy_all),
        "iz": np.concatenate(iz_all),
        "index_valid": np.concatenate(index_valid_all),
        "index_type": np.concatenate(index_type_all),
        "v_channel": np.concatenate(v_channel_all),
        "h_channel": np.concatenate(h_channel_all),
        "v_qdc": np.concatenate(v_qdc_all),
        "h_qdc": np.concatenate(h_qdc_all),
        "time": np.concatenate(time_all),
    }


def build_all_3dHits(SciFi_3D_hits, DS_3D_hits, US_DS4_voxel_hits):
    return concatenate_hit_dicts(
        [SciFi_3D_hits, DS_3D_hits, US_DS4_voxel_hits],
        HIT_KEYS,
        HIT_DTYPES,
    )


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
        "hitTime_time0_ns": array.array("d", [-999.0]),
        "hitTime_time0_applied": array.array("i", [-999]),
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
        "hit_ix": ROOT.std.vector("int")(),
        "hit_iy": ROOT.std.vector("int")(),
        "hit_iz": ROOT.std.vector("int")(),
        "hit_index_valid": ROOT.std.vector("short")(),
        "hit_index_type": ROOT.std.vector("short")(),
        "hit_v_channel": ROOT.std.vector("int")(),
        "hit_h_channel": ROOT.std.vector("int")(),
        "hit_v_qdc": ROOT.std.vector("float")(),
        "hit_h_qdc": ROOT.std.vector("float")(),
        "hit_time_ns": ROOT.std.vector("float")(),
        "scifi_native_station": ROOT.std.vector("short")(),
        "scifi_native_orientation": ROOT.std.vector("short")(),
        "scifi_native_channel": ROOT.std.vector("int")(),
        "scifi_native_det_id": ROOT.std.vector("long long")(),
        "scifi_native_x": ROOT.std.vector("float")(),
        "scifi_native_y": ROOT.std.vector("float")(),
        "scifi_native_z": ROOT.std.vector("float")(),
        "scifi_native_qdc": ROOT.std.vector("float")(),
        "scifi_native_time_ns": ROOT.std.vector("float")(),
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
    branch_vars["hitTime_time0_ns"][0] = float(event.get("hitTime_time0_ns", -999.0))
    branch_vars["hitTime_time0_applied"][0] = int(event.get("hitTime_time0_applied", -999))


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
    ix = np.asarray(all_3dHits.get("ix", []), dtype=np.int32)
    iy = np.asarray(all_3dHits.get("iy", []), dtype=np.int32)
    iz = np.asarray(all_3dHits.get("iz", []), dtype=np.int32)
    index_valid = np.asarray(all_3dHits.get("index_valid", []), dtype=np.int16)
    index_type = np.asarray(all_3dHits.get("index_type", []), dtype=np.int16)
    v_channel = np.asarray(all_3dHits.get("v_channel", []), dtype=np.int32)
    h_channel = np.asarray(all_3dHits.get("h_channel", []), dtype=np.int32)
    v_qdc = np.asarray(all_3dHits.get("v_qdc", []), dtype=np.float32)
    h_qdc = np.asarray(all_3dHits.get("h_qdc", []), dtype=np.float32)
    hit_time = np.asarray(all_3dHits.get("time", []), dtype=np.float32)

    lengths = {
        len(x),
        len(y),
        len(z),
        len(qdc),
        len(station),
        len(det_type),
        len(ix),
        len(iy),
        len(iz),
        len(index_valid),
        len(index_type),
        len(v_channel),
        len(h_channel),
        len(v_qdc),
        len(h_qdc),
        len(hit_time),
    }
    if len(lengths) != 1:
        raise RuntimeError(f"Inconsistent hit3D array lengths: {sorted(lengths)}")

    assign_vector(vector_vars["hit_x"], x)
    assign_vector(vector_vars["hit_y"], y)
    assign_vector(vector_vars["hit_z"], z)
    assign_vector(vector_vars["hit_qdc"], qdc)
    assign_vector(vector_vars["hit_station"], station)
    assign_vector(vector_vars["hit_detType"], det_type)
    assign_vector(vector_vars["hit_ix"], ix)
    assign_vector(vector_vars["hit_iy"], iy)
    assign_vector(vector_vars["hit_iz"], iz)
    assign_vector(vector_vars["hit_index_valid"], index_valid)
    assign_vector(vector_vars["hit_index_type"], index_type)
    assign_vector(vector_vars["hit_v_channel"], v_channel)
    assign_vector(vector_vars["hit_h_channel"], h_channel)
    assign_vector(vector_vars["hit_v_qdc"], v_qdc)
    assign_vector(vector_vars["hit_h_qdc"], h_qdc)
    assign_vector(vector_vars["hit_time_ns"], hit_time)


def fill_native_scifi_vectors(native_hits, vector_vars):
    values = {
        key: np.asarray(native_hits.get(key, []), dtype=dtype)
        for key, dtype in zip(NATIVE_SCIFI_KEYS, NATIVE_SCIFI_DTYPES)
    }
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"Inconsistent native SciFi array lengths: {sorted(lengths)}")
    branch_names = {
        "station": "scifi_native_station",
        "orientation": "scifi_native_orientation",
        "channel": "scifi_native_channel",
        "det_id": "scifi_native_det_id",
        "x": "scifi_native_x",
        "y": "scifi_native_y",
        "z": "scifi_native_z",
        "qdc": "scifi_native_qdc",
        "time_ns": "scifi_native_time_ns",
    }
    for key, branch in branch_names.items():
        assign_vector(vector_vars[branch], values[key])


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
    if needs_first_arrival_time0(args):
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
        mat, sipm, sipm_channel, layer_channel = scifi_channel_info(aHit)

        Scifi.GetSiPMPosition(detID, A, B)
        Ax, Ay, Az = A.x(), A.y(), A.z()
        Bx, By, Bz = B.x(), B.y(), B.z()

        SciFi_hits.append({
            "det_id": int(detID),
            "station": station,
            "isVertical": bool(aHit.isVertical()),
            "qdc": qdc,
            "time": scifi_time_ns(aHit, Scifi, args),
            "x_mid": 0.5 * (Ax + Bx),
            "y_mid": 0.5 * (Ay + By),
            "z_mid": 0.5 * (Az + Bz),
            "mat": mat,
            "sipm": sipm,
            "sipm_channel": sipm_channel,
            "channel": layer_channel,
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
        channel = mufilter_channel_info(detID)

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
            "time": mufilter_hit_time_ns(aHit, MuFilter, args),
            "x_mid": 0.5 * (Ax + Bx),
            "y_mid": 0.5 * (Ay + By),
            "z_mid": 0.5 * (Az + Bz),
            "channel": channel,
        })
    add_timing(timings, "mufilter_build", time.perf_counter() - t0)

    time0_ns, time0_applied = apply_first_arrival_time0(args, [SciFi_hits, MuFilter_hits])
    t0 = time.perf_counter()
    if needs_first_arrival_time0(args):
        SciFi_hits = select_calibrated_scifi_hits(
            SciFi_hits,
            time_key="time",
            ns_per_clock=NS_PER_CLOCK,
        )
    add_timing(timings, "scifi_calibrated_select", time.perf_counter() - t0)
    SciFi_hits = filter_late_mc_hits(args, SciFi_hits, "time")
    MuFilter_hits = filter_late_mc_hits(args, MuFilter_hits, "time")

    native_scifi_hits = build_native_scifi_hits(SciFi_hits)

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
            det_type=1,
            index_type=INDEX_SCIFI_CROSSED,
        )
        if len(arr["x"]) > 0:
            scifi_station_arrays.append(arr)
        scifi_station_arrays.append(
            build_one_orientation_hits(
                vertical_hits if not horizontal_hits else [],
                det_type=1,
                index_type=INDEX_SCIFI_VERTICAL_ONLY,
                known_axis="x",
            )
        )
        scifi_station_arrays.append(
            build_one_orientation_hits(
                horizontal_hits if not vertical_hits else [],
                det_type=1,
                index_type=INDEX_SCIFI_HORIZONTAL_ONLY,
                known_axis="y",
            )
        )

    if scifi_station_arrays:
        SciFi_3D_hits = concatenate_hit_dicts(scifi_station_arrays, HIT_KEYS, HIT_DTYPES)
    else:
        SciFi_3D_hits = empty_3d_hits()
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
            det_type=3,
            index_type=INDEX_DS_CROSSED,
        )
        if len(arr["x"]) > 0:
            ds_station_arrays.append(arr)
        ds_station_arrays.append(
            build_one_orientation_hits(
                vertical_hits if not horizontal_hits else [],
                det_type=3,
                index_type=INDEX_DS_VERTICAL_ONLY,
                known_axis="x",
            )
        )
        ds_station_arrays.append(
            build_one_orientation_hits(
                horizontal_hits if not vertical_hits else [],
                det_type=3,
                index_type=INDEX_DS_HORIZONTAL_ONLY,
                known_axis="y",
            )
        )

    if ds_station_arrays:
        DS_3D_hits = concatenate_hit_dicts(ds_station_arrays, HIT_KEYS, HIT_DTYPES)
    else:
        DS_3D_hits = empty_3d_hits()
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

    return all_3dHits, native_scifi_hits, time0_ns, time0_applied


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
        "selection_mode": (
            "passing_muon_direct"
            if is_passing_muon_real_data(args.type)
            else "external_cutflow"
        ),
        "input_tree": args.tree_name,
        "external_cutflow_used": int(not is_passing_muon_real_data(args.type)),
        "scifi_native_schema": NATIVE_SCIFI_SCHEMA,
        "scifi_native_orientation": "0_vertical_1_horizontal",
        "scifi_native_selection": "post_time_and_qdc_selection",
    }


def main(args):
    print("start processing digi to hits3D")

    direct_passing_muon = is_passing_muon_real_data(args.type)
    if uses_external_cutflow(args.type) and not args.preSelect_path:
        raise ValueError("--preSelectPath is required unless type is passing_muon_real_data")

    snd_geo = setup_geometry(args.geo_path)
    raw_data, raw_tree = open_root_file(args.digi_path, tree_name=args.tree_name)
    preSelect_data = None
    preSelect_tree = None
    if not direct_passing_muon:
        preSelect_data, preSelect_tree = open_root_file(
            args.preSelect_path, tree_name="cutFlowSummary"
        )
    else:
        print("Using direct passing-muon selection: processing reconstructed-tree entries")
    out_file, hit_tree, branch_vars, vector_vars = create_output_root(
        args.out_path,
        args.mode,
        compression_level=args.compression_level,
    )

    selection = None if direct_passing_muon else build_selection(args, preSelect_tree)

    print("Using feature-aligned hit3D event selection")
    if selection:
        print(f"Applying selection: {selection}")
        n_match = preSelect_tree.GetEntries(selection)
    elif direct_passing_muon:
        n_match = raw_tree.GetEntries()
    else:
        print("No selection applied")
        n_match = preSelect_tree.GetEntries()
    print(f"Entries matching selection: {n_match}")

    if n_match == 0:
        hit_tree.Write()
        write_metadata_objects(out_file, make_metadata(args, selection, 0))
        out_file.Close()
        raw_data.Close()
        if preSelect_data:
            preSelect_data.Close()
        print("No entries matched the selection condition, saved empty hit3D tree")
        return 0

    if selection:
        elist_name = "elist"
        preSelect_tree.Draw(f">>{elist_name}", selection, "entrylist")
        elist = ROOT.gDirectory.Get(elist_name)

        if not elist or not isinstance(elist, ROOT.TEntryList):
            raise RuntimeError("Failed to create or retrieve TEntryList")
    else:
        elist = ROOT.TEntryList("elist_all", "elist_all")
        entry_count = raw_tree.GetEntries() if direct_passing_muon else preSelect_tree.GetEntries()
        for j in range(entry_count):
            elist.Enter(j)

    if preSelect_tree:
        preSelect_tree.SetEntryList(elist)
    n_selected = int(elist.GetN())
    n_written = 0
    n_dropped_has_veto_has_us = 0
    timings = {key: 0.0 for key in TIMING_KEYS}

    for i in tqdm(range(n_selected), desc="processing hit3D events", mininterval=30):
        t0 = time.perf_counter()
        selected_entry = elist.GetEntry(i)
        if direct_passing_muon:
            raw_entry = selected_entry
        else:
            cutflow_entry = selected_entry
            preSelect_tree.GetEntry(cutflow_entry)
            if hasattr(preSelect_tree, "entry"):
                raw_entry = int(preSelect_tree.entry)
            else:
                raw_entry = cutflow_entry

        raw_tree.GetEntry(raw_entry)
        init_event_geometry(snd_geo, raw_tree.EventHeader)
        add_timing(timings, "entry_load", time.perf_counter() - t0)

        if should_drop_real_has_veto_has_us(args, raw_tree):
            n_dropped_has_veto_has_us += 1
            continue

        t0 = time.perf_counter()
        event = extract_event_metadata(args, raw_tree, raw_entry)
        event["label"] = particle_to_target.get(event["pdgCode"], -1)
        add_timing(timings, "metadata", time.perf_counter() - t0)

        all_3dHits, native_scifi_hits, time0_ns, time0_applied = process_hits_numpy(
            args,
            raw_tree,
            snd_geo,
            timings=timings,
        )
        event["hitTime_time0_ns"] = time0_ns
        event["hitTime_time0_applied"] = int(time0_applied)

        t0 = time.perf_counter()
        reset_output_branches(branch_vars, vector_vars)
        fill_event_branches(event, branch_vars)
        fill_hit_vectors(all_3dHits, vector_vars)
        fill_native_scifi_vectors(native_scifi_hits, vector_vars)
        hit_tree.Fill()
        n_written += 1
        add_timing(timings, "root_fill", time.perf_counter() - t0)

    t0 = time.perf_counter()
    hit_tree.Write()
    metadata = make_metadata(args, selection, n_written)
    metadata.update({
        "selection_mode": "passing_muon_direct" if direct_passing_muon else "external_cutflow",
        "input_tree": args.tree_name,
        "external_cutflow_used": int(not direct_passing_muon),
        "input_entries": int(raw_tree.GetEntries()),
        "dropped_has_veto_and_us": int(n_dropped_has_veto_has_us),
    })
    write_metadata_objects(out_file, metadata)
    add_timing(timings, "root_write", time.perf_counter() - t0)
    out_file.Close()
    raw_data.Close()
    if preSelect_data:
        preSelect_data.Close()

    print_timing_summary(timings, n_written)
    if args.drop_real_has_veto_has_us:
        print(
            "Dropped real-data events with both veto and US hits: "
            f"{n_dropped_has_veto_has_us}"
        )
    print(f"finish processing digi to hits3D, saved {n_written} events to {args.out_path}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-p", "--preSelectPath", dest="preSelect_path", help="pre selection data file path")
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output ROOT path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default="RECREATE")
    parser.add_argument("-t", "--type", dest="type", help="data type, e.g. MC or real", required=True)
    parser.add_argument("--tree-name", default="cbmsim", help="Input event-tree name (default: cbmsim)")
    parser.add_argument(
        "--compression-level",
        dest="compression_level",
        type=int,
        default=4,
        help="ROOT compression level for the output file.",
    )
    parser.add_argument(
        "--drop-real-has-veto-has-us",
        dest="drop_real_has_veto_has_us",
        action="store_true",
        help="For real data, skip events with both valid veto and upstream MuFilter hits.",
    )

    args = parser.parse_args()
    if not 0 <= args.compression_level <= 9:
        parser.error("--compression-level must be between 0 and 9")
    raise SystemExit(main(args))
