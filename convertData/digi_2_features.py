import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array
from collections import defaultdict
import math
import time
from analysis.analyses.snd_analysis_2024_0mu.sciFiTools import selectHits, getSumDensity
import numpy as np

from hit_time_selection import (
    apply_first_arrival_time0 as apply_muon_first_arrival_time0,
    is_muon_time_calibrated_sample,
    select_calibrated_scifi_hits,
)
from sample_selection import (
    EXTERNAL_CUTFLOW,
    SELECTION_MODES,
    resolve_selection_mode,
    uses_external_cutflow,
)
from reco_track_features import (
    FLAT_V1,
    RECO_TRACK_SCHEMAS,
    fill_reco_track_buffers,
    reset_reco_track_buffers,
    scalar_dtypes_for_schema,
    vector_dtypes_for_schema,
)
from scifi_plane_clustering import (
    SCHEMA_VERSION as SCIFI_CLUSTER_SCHEMA_VERSION,
    cluster_feature_payload,
    scalar_feature_dtypes as scifi_cluster_scalar_dtypes,
)

ROOT.TH1.AddDirectory(False)

NS_PER_CLOCK = 6.25
INVALID_TIME = -999.0
MC_HIT_TIME_WINDOW_NS = 25.0


def record_timing(timing, timing_counts, name, start_time):
    if timing is None or timing_counts is None:
        return

    timing[name] += time.perf_counter() - start_time
    timing_counts[name] += 1


def print_timing_summary(timing, timing_counts, total_elapsed, n_selected_entries, n_written_entries):
    print("Timing summary:")
    print(f"  total.job                                {total_elapsed:10.3f} s")
    print(f"  selected entries                         {n_selected_entries:10d}")
    print(f"  written entries                          {n_written_entries:10d}")

    if not timing:
        return

    name_width = max(len(name) for name in timing)
    header_name = "section"
    print(
        f"  {header_name:<{name_width}} "
        f"{'total [s]':>10} {'calls':>10} {'avg/call [ms]':>14} {'job [%]':>9}"
    )

    for name, elapsed in sorted(timing.items(), key=lambda item: item[1], reverse=True):
        calls = timing_counts.get(name, 0)
        avg_ms = 1000.0 * elapsed / calls if calls else 0.0
        percent = 100.0 * elapsed / total_elapsed if total_elapsed > 0 else 0.0
        print(
            f"  {name:<{name_width}} "
            f"{elapsed:10.3f} {calls:10d} {avg_ms:14.3f} {percent:9.2f}"
        )


def make_hit_time_debug(max_examples=5):
    return {
        "max_examples": max_examples,
        "events": 0,
        "filter_enabled_events": 0,
        "time0_applied_events": 0,
        "scifi_before": 0,
        "scifi_after": 0,
        "mufilter_before": 0,
        "mufilter_after": 0,
        "raw_min": None,
        "raw_max": None,
        "calibrated_min": None,
        "calibrated_max": None,
        "kept_min": None,
        "kept_max": None,
        "examples": [],
    }


def _update_range(debug, prefix, times):
    times = [float(value) for value in times if valid_time(value)]
    if not times:
        return

    low_key = f"{prefix}_min"
    high_key = f"{prefix}_max"
    low = min(times)
    high = max(times)
    debug[low_key] = low if debug[low_key] is None else min(debug[low_key], low)
    debug[high_key] = high if debug[high_key] is None else max(debug[high_key], high)


def _time_range(times):
    times = [float(value) for value in times if valid_time(value)]
    if not times:
        return None
    return (min(times), max(times))


def update_hit_time_debug(
    debug,
    args,
    event_index,
    event_id,
    time0_ns,
    time0_applied,
    raw_scifi_times,
    raw_mufilter_times,
    calibrated_scifi_times,
    calibrated_mufilter_times,
    kept_scifi_times,
    kept_mufilter_times,
):
    if debug is None:
        return

    debug["events"] += 1
    filter_enabled = applies_mc_25ns_hit_filter(args)
    if filter_enabled:
        debug["filter_enabled_events"] += 1
    if time0_applied:
        debug["time0_applied_events"] += 1

    debug["scifi_before"] += len(raw_scifi_times)
    debug["mufilter_before"] += len(raw_mufilter_times)
    debug["scifi_after"] += len(kept_scifi_times)
    debug["mufilter_after"] += len(kept_mufilter_times)

    _update_range(debug, "raw", raw_scifi_times + raw_mufilter_times)
    _update_range(debug, "calibrated", calibrated_scifi_times + calibrated_mufilter_times)
    _update_range(debug, "kept", kept_scifi_times + kept_mufilter_times)

    if len(debug["examples"]) >= debug["max_examples"]:
        return

    if filter_enabled or time0_applied:
        debug["examples"].append({
            "eventIndex": int(event_index),
            "eventId": int(event_id),
            "time0_ns": float(time0_ns),
            "time0_applied": int(time0_applied),
            "filter_enabled": int(filter_enabled),
            "scifi_before": len(raw_scifi_times),
            "scifi_after": len(kept_scifi_times),
            "mufilter_before": len(raw_mufilter_times),
            "mufilter_after": len(kept_mufilter_times),
            "raw_range_ns": _time_range(raw_scifi_times + raw_mufilter_times),
            "calibrated_range_ns": _time_range(calibrated_scifi_times + calibrated_mufilter_times),
            "kept_range_ns": _time_range(kept_scifi_times + kept_mufilter_times),
        })


def print_hit_time_debug_summary(debug):
    if not debug:
        return

    print("Hit-time debug summary:")
    print(f"  events processed                         {debug['events']:10d}")
    print(f"  25 ns MC filter enabled events           {debug['filter_enabled_events']:10d}")
    print(f"  first-arrival time0 applied events       {debug['time0_applied_events']:10d}")
    print(
        "  SciFi hits before/after/dropped          "
        f"{debug['scifi_before']:10d} {debug['scifi_after']:10d} "
        f"{debug['scifi_before'] - debug['scifi_after']:10d}"
    )
    print(
        "  MuFilter hits before/after/dropped       "
        f"{debug['mufilter_before']:10d} {debug['mufilter_after']:10d} "
        f"{debug['mufilter_before'] - debug['mufilter_after']:10d}"
    )
    print(f"  raw time range [ns]                      {debug['raw_min']} .. {debug['raw_max']}")
    print(f"  calibrated time range [ns]               {debug['calibrated_min']} .. {debug['calibrated_max']}")
    print(f"  kept time range [ns]                     {debug['kept_min']} .. {debug['kept_max']}")

    if debug["examples"]:
        print("  example events:")
        for example in debug["examples"]:
            print(f"    {example}")

def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo


def init_event_geometry(snd_geo, event_header):
    snd_geo.modules["Scifi"].InitEvent(event_header)
    snd_geo.modules["MuFilter"].InitEvent(event_header)


def open_root_file(file_path, tree_name='cbmsim', mode='read'):
    file = ROOT.TFile(file_path, mode)
    tree = file.Get(tree_name)
    if not tree or not isinstance(tree, ROOT.TTree):
        raise RuntimeError(f"TTree '{tree_name}' not found in {file_path}")
    return file, tree


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


def safe_fraction(num, den):
    return float(num) / float(den) if den else 0.0


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


def apply_first_arrival_time0(args, hit_collections, time_key="hitTimeNS"):
    return apply_muon_first_arrival_time0(
        args,
        hit_collections,
        time_key=time_key,
        invalid_time=INVALID_TIME,
    )


def keep_hit_in_mc_time_window(args, hit, time_key="hitTimeNS"):
    if not applies_mc_25ns_hit_filter(args):
        return True

    time_ns = hit.get(time_key, INVALID_TIME)
    if not valid_time(time_ns):
        return True

    return float(time_ns) <= MC_HIT_TIME_WINDOW_NS


def filter_late_mc_hits(args, hits, time_key="hitTimeNS"):
    return [hit for hit in hits if keep_hit_in_mc_time_window(args, hit, time_key)]


def weighted_mean_and_std(values, weights):
    total_weight = sum(weights)
    if total_weight <= 0:
        return -999.0, -999.0

    mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / total_weight
    if variance < 0:
        variance = 0.0
    return mean, math.sqrt(variance)


def std_or_sentinel(values):
    if len(values) < 2:
        return -999.0

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


# muonDIS truth plan:
#   1. Save primary MC tracks with mother_id == -1.
#   2. Count all raw DIS secondaries with mother_id == 0 as nSecondaryRaw.
#   3. Starting from valid SciFi digi hits, use Digi_ScifiHits2MCPoints to find
#      linked ScifiPoint objects, then their MCTrack IDs. Save a secondary track
#      only if it has mother_id == 0 and generated at least one SciFi digi hit.
#      A set of track IDs avoids saving the same secondary once per hit.
#   4. For each saved secondary, save one daughter MCTrack start position as a
#      proxy for where that secondary first interacted/scattered in material.
def make_track_vector_branches():
    return {
        "primary_pdg": ROOT.std.vector("int")(),
        "primary_energy": ROOT.std.vector("double")(),
        "primary_startX": ROOT.std.vector("double")(),
        "primary_startY": ROOT.std.vector("double")(),
        "primary_startZ": ROOT.std.vector("double")(),
        "primary_px": ROOT.std.vector("double")(),
        "primary_py": ROOT.std.vector("double")(),
        "primary_pz": ROOT.std.vector("double")(),

        "secondary_pdg": ROOT.std.vector("int")(),
        "secondary_energy": ROOT.std.vector("double")(),
        "secondary_startX": ROOT.std.vector("double")(),
        "secondary_startY": ROOT.std.vector("double")(),
        "secondary_startZ": ROOT.std.vector("double")(),
        "secondary_px": ROOT.std.vector("double")(),
        "secondary_py": ROOT.std.vector("double")(),
        "secondary_pz": ROOT.std.vector("double")(),

        "secondary_interactionChildTrackID": ROOT.std.vector("int")(),
        "secondary_interactionChildPdg": ROOT.std.vector("int")(),
        "secondary_interactionX": ROOT.std.vector("double")(),
        "secondary_interactionY": ROOT.std.vector("double")(),
        "secondary_interactionZ": ROOT.std.vector("double")(),
    }


def make_hit_time_vector_branches():
    return {
        "hitTime_raw_ns": ROOT.std.vector("double")(),
        "hitTime_raw_detType": ROOT.std.vector("int")(),
    }


def branch_vector_vars(tree, vector_branches):
    for name, vec in vector_branches.items():
        tree.Branch(name, vec)


def reset_vector_branches(vector_branches):
    for vec in vector_branches.values():
        vec.clear()


def fill_scifi_cluster_features(scifi_hits, branch_vars, event_identity=None):
    """Fill the fixed two-cluster-per-orientation feature schema."""
    try:
        payload = cluster_feature_payload(scifi_hits)
    except (KeyError, TypeError, ValueError) as error:
        identity = event_identity or {}
        raise ValueError(
            "SciFi clustering failed for run={runId} event={eventId} "
            "eventIndex={eventIndex}: {error}".format(error=error, **{
                "runId": identity.get("runId", "unknown"),
                "eventId": identity.get("eventId", "unknown"),
                "eventIndex": identity.get("eventIndex", "unknown"),
            })
        )

    for name, value in payload["scalars"].items():
        branch_vars[name][0] = value
    return payload["analysis"]


def fill_track_info(prefix, track, vector_branches):
    vector_branches[f"{prefix}_pdg"].push_back(int(track.GetPdgCode()))
    vector_branches[f"{prefix}_energy"].push_back(float(track.GetEnergy()))
    vector_branches[f"{prefix}_startX"].push_back(float(track.GetStartX()))
    vector_branches[f"{prefix}_startY"].push_back(float(track.GetStartY()))
    vector_branches[f"{prefix}_startZ"].push_back(float(track.GetStartZ()))
    vector_branches[f"{prefix}_px"].push_back(float(track.GetPx()))
    vector_branches[f"{prefix}_py"].push_back(float(track.GetPy()))
    vector_branches[f"{prefix}_pz"].push_back(float(track.GetPz()))


def fill_raw_hit_time_vectors(SciFi_hits, MuFilter_hits, vector_branches):
    for hit in SciFi_hits:
        vector_branches["hitTime_raw_ns"].push_back(float(hit.get("hitTimeNS", INVALID_TIME)))
        vector_branches["hitTime_raw_detType"].push_back(int(hit.get("detType", 0)))

    for hit in MuFilter_hits:
        vector_branches["hitTime_raw_ns"].push_back(float(hit.get("hitTimeNS", INVALID_TIME)))
        vector_branches["hitTime_raw_detType"].push_back(int(hit.get("detType", -1)))


def fill_secondary_interaction_info(child_track_id, child_track, vector_branches):
    if child_track is None:
        vector_branches["secondary_interactionChildTrackID"].push_back(-999)
        vector_branches["secondary_interactionChildPdg"].push_back(-999)
        vector_branches["secondary_interactionX"].push_back(-999.0)
        vector_branches["secondary_interactionY"].push_back(-999.0)
        vector_branches["secondary_interactionZ"].push_back(-999.0)
        return

    vector_branches["secondary_interactionChildTrackID"].push_back(int(child_track_id))
    vector_branches["secondary_interactionChildPdg"].push_back(int(child_track.GetPdgCode()))
    vector_branches["secondary_interactionX"].push_back(float(child_track.GetStartX()))
    vector_branches["secondary_interactionY"].push_back(float(child_track.GetStartY()))
    vector_branches["secondary_interactionZ"].push_back(float(child_track.GetStartZ()))


def collect_first_child_by_mother(mc_tracks, mother_track_ids):
    first_child_by_mother = {}
    mother_track_ids = set(mother_track_ids)

    for child_id in range(int(mc_tracks.GetEntriesFast())):
        child = mc_tracks.At(child_id)
        if child is None:
            continue

        mother_id = int(child.GetMotherId())
        if mother_id in mother_track_ids and mother_id not in first_child_by_mother:
            first_child_by_mother[mother_id] = (child_id, child)

    return first_child_by_mother


def get_scifi_hit2mc(event):
    for name in ("Digi_ScifiHits2MCPoints", "Scifi_2_mcpoints", "Scifi_2_MCPoints"):
        if not hasattr(event, name):
            continue
        links_array = getattr(event, name)
        if links_array and links_array.GetEntriesFast() > 0:
            return links_array[0]
    return None


def iter_linked_mcpoint_indices(hit2mc, key):
    try:
        links = hit2mc.wList(int(key))
    except Exception:
        return

    for item in links:
        try:
            yield int(item[0])
        except TypeError:
            yield int(item)


def collect_scifi_visible_secondary_track_ids(event):
    if not all(hasattr(event, name) for name in ("Digi_ScifiHits", "ScifiPoint", "MCTrack")):
        return set()

    hit2mc = get_scifi_hit2mc(event)
    if hit2mc is None:
        return set()

    secondary_track_ids = set()
    scifi_points = event.ScifiPoint
    mc_tracks = event.MCTrack

    for hit_index, hit in enumerate(event.Digi_ScifiHits):
        if not hit.isValid():
            continue

        point_ids = list(iter_linked_mcpoint_indices(hit2mc, hit.GetDetectorID()))
        if not point_ids:
            point_ids = list(iter_linked_mcpoint_indices(hit2mc, hit_index))

        for point_id in point_ids:
            if point_id < 0 or point_id >= scifi_points.GetEntriesFast():
                continue
            point = scifi_points.At(point_id)
            if point is None:
                continue

            track_id = int(point.GetTrackID())
            if track_id < 0 or track_id >= mc_tracks.GetEntriesFast():
                continue
            track = mc_tracks.At(track_id)
            if track is None:
                continue
            if int(track.GetMotherId()) == 0:
                secondary_track_ids.add(track_id)

    return secondary_track_ids


def process_muonDIS_tracks(event, vector_branches):
    if not hasattr(event, "MCTrack"):
        return 0, 0, 0

    mc_tracks = event.MCTrack
    visible_secondary_ids = collect_scifi_visible_secondary_track_ids(event)
    first_child_by_mother = collect_first_child_by_mother(mc_tracks, visible_secondary_ids)
    n_primary = 0
    n_secondary_raw = 0
    n_secondary = 0

    for track_id in range(int(mc_tracks.GetEntriesFast())):
        track = mc_tracks.At(track_id)
        if track is None:
            continue

        mother_id = int(track.GetMotherId())
        if mother_id == -1:
            fill_track_info("primary", track, vector_branches)
            n_primary += 1
        elif mother_id == 0:
            n_secondary_raw += 1
            if track_id in visible_secondary_ids:
                fill_track_info("secondary", track, vector_branches)
                child_id, child_track = first_child_by_mother.get(track_id, (-999, None))
                fill_secondary_interaction_info(child_id, child_track, vector_branches)
                n_secondary += 1

    return n_primary, n_secondary_raw, n_secondary


def create_output_file(path, mode):
    """Create and return a new ROOT file and a new tree for output."""
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    out_file = ROOT.TFile(path, mode)
    new_tree = ROOT.TTree('sndData', 'converted SND hits tree')
    new_tree.SetDirectory(out_file)
    return out_file, new_tree


def process_count_and_qdc(SciFi_hits, MuFilter_hits, branch_vars):
    # ---------- helpers ----------
    def qdc_value(x):
        # x can be float/int or dict-like (e.g. GetAllSignals())
        if isinstance(x, dict):
            return sum(non_negative_float(value) for value in x.values())
        return non_negative_float(x)

    # ---------- init counters ----------
    scifi_counts = [0] * 5
    veto_counts = [0] * 3
    us_counts = [0] * 5
    ds_counts = [0] * 4

    scifi_qdc = [0.0] * 5
    veto_qdc = [0.0] * 3
    us_qdc = [0.0] * 5
    ds_qdc = [0.0] * 4

    # for SciFi event-ID logic (need both views info)
    scifi_hor = [0] * 5
    scifi_ver = [0] * 5

    # ---------- fill SciFi ----------
    for h in SciFi_hits:
        st = int(h.get("station", 0))
        if 1 <= st <= 5:
            i = st - 1
            scifi_counts[i] += 1
            scifi_qdc[i] += qdc_value(h.get("qdc", 0.0))
            if h.get("isVertical", False):
                scifi_ver[i] += 1
            else:
                scifi_hor[i] += 1

    # ---------- fill MuFilter ----------
    for h in MuFilter_hits:
        det = int(h.get("detType", -1))   # 1=veto, 2=US, 3=DS
        st = int(h.get("station", 0))     # expected 1-based
        q = qdc_value(h.get("qdc", 0.0))

        if det == 1 and 1 <= st <= 3:
            veto_counts[st - 1] += 1
            veto_qdc[st - 1] += q
        elif det == 2 and 1 <= st <= 5:
            us_counts[st - 1] += 1
            us_qdc[st - 1] += q
        elif det == 3 and 1 <= st <= 4:
            ds_counts[st - 1] += 1
            ds_qdc[st - 1] += q

    # ---------- write count branches ----------
    for i in range(3):
        branch_vars[f"count_veto{i+1}"][0] = veto_counts[i]
    branch_vars["count_veto"][0] = sum(veto_counts)

    for i in range(5):
        branch_vars[f"count_scifi{i+1}"][0] = scifi_counts[i]
    branch_vars["count_scifi"][0] = sum(scifi_counts)

    for i in range(5):
        branch_vars[f"count_us{i+1}"][0] = us_counts[i]
    branch_vars["count_us"][0] = sum(us_counts)

    for i in range(4):
        branch_vars[f"count_ds{i+1}"][0] = ds_counts[i]
    branch_vars["count_ds"][0] = sum(ds_counts)

    # ---------- write qdc branches ----------
    for i in range(3):
        branch_vars[f"qdc_veto{i+1}"][0] = veto_qdc[i]
    branch_vars["qdc_veto"][0] = sum(veto_qdc)

    for i in range(5):
        branch_vars[f"qdc_scifi{i+1}"][0] = scifi_qdc[i]
    branch_vars["qdc_scifi"][0] = sum(scifi_qdc)

    for i in range(5):
        branch_vars[f"qdc_us{i+1}"][0] = us_qdc[i]
    branch_vars["qdc_us"][0] = sum(us_qdc)

    for i in range(4):
        branch_vars[f"qdc_ds{i+1}"][0] = ds_qdc[i]
    branch_vars["qdc_ds"][0] = sum(ds_qdc)
    

 
def process_vetoHitTime(MuFilter_hits, branch_vars):
    veto_hits = [h for h in MuFilter_hits if h["detType"] == 1]

    per_station = {}
    for s in (1, 2, 3):
        times = [h["hitTimeCY"] for h in veto_hits if h["station"] == s]
        per_station[s] = {
            "earliest": min(times) if times else -1,
            "latest": max(times) if times else -1,
        }

    all_times = [h["hitTimeCY"] for h in veto_hits]
    overall_earliest = min(all_times) if all_times else -1
    overall_latest = max(all_times) if all_times else -1

    branch_vars["vetoHitTime_earlist"][0] = overall_earliest
    branch_vars["vetoHitTime_latest"][0] = overall_latest

    for s in (1, 2, 3):
        branch_vars[f"vetoHitTime_earlist_veto{s}"][0] = per_station[s]["earliest"]
        branch_vars[f"vetoHitTime_latest_veto{s}"][0] = per_station[s]["latest"]
    
    

def process_avgPos(SciFi_hits, MuFilter_hits, branch_vars):
    """
    Compute average hit positions for:
      - veto_{1-2}_y
      - veto_3_x
      - scifi_{1-5}_{x,y}
      - us_{1-5}_y
      - ds_{1-4}_{x,y}

    Convention:
      - horizontal planes -> average y
      - vertical planes   -> average x
    """
    sums = defaultdict(float)
    counts = defaultdict(int)

    # combine both hit collections
    all_hits = SciFi_hits + MuFilter_hits

    for hit in all_hits:
        detType = hit["detType"]
        station = hit["station"]
        isVertical = hit["isVertical"]

        if detType == 1:  # Veto
            if station in [1, 2] and not isVertical:
                key = f"avg_veto{station}_y"
                sums[key] += hit["y"]
                counts[key] += 1

                sums["avg_veto_y"] += hit["y"]
                counts["avg_veto_y"] += 1

            elif station == 3 and isVertical:
                key = "avg_veto3_x"
                sums[key] += hit["x"]
                counts[key] += 1

                sums["avg_veto_x"] += hit["x"]
                counts["avg_veto_x"] += 1

        elif detType == 0:  # SciFi
            if 1 <= station <= 5:
                if isVertical:
                    key = f"avg_scifi{station}_x"
                    sums[key] += hit["x"]
                    counts[key] += 1

                    sums["avg_scifi_x"] += hit["x"]
                    counts["avg_scifi_x"] += 1
                else:
                    key = f"avg_scifi{station}_y"
                    sums[key] += hit["y"]
                    counts[key] += 1

                    sums["avg_scifi_y"] += hit["y"]
                    counts["avg_scifi_y"] += 1

        elif detType == 2:  # Upstream
            if 1 <= station <= 5 and not isVertical:
                key = f"avg_us{station}_y"
                sums[key] += hit["y"]
                counts[key] += 1

                sums["avg_us_y"] += hit["y"]
                counts["avg_us_y"] += 1

        elif detType == 3:  # Downstream
            if 1 <= station <= 4:
                if isVertical:
                    key = f"avg_ds{station}_x"
                    sums[key] += hit["x"]
                    counts[key] += 1

                    sums["avg_ds_x"] += hit["x"]
                    counts["avg_ds_x"] += 1
                else:
                    key = f"avg_ds{station}_y"
                    sums[key] += hit["y"]
                    counts[key] += 1

                    sums["avg_ds_y"] += hit["y"]
                    counts["avg_ds_y"] += 1

    # Fill all requested branch_vars safely
    for key in branch_vars:
        if key.startswith("avg_"):
            branch_vars[key][0] = sums[key] / counts[key] if counts[key] > 0 else -999.


def process_qdcAvgPos(SciFi_hits, MuFilter_hits, branch_vars):
    """
    Compute QDC-weighted average hit positions using the same detector/view
    convention as process_avgPos.
    """
    weighted_sums = defaultdict(float)
    weight_sums = defaultdict(float)

    def add_weighted_position(key, position, qdc):
        weight = non_negative_float(qdc)
        if weight <= 0:
            return
        weighted_sums[key] += float(position) * weight
        weight_sums[key] += weight

    # combine both hit collections
    all_hits = SciFi_hits + MuFilter_hits

    for hit in all_hits:
        detType = hit["detType"]
        station = hit["station"]
        isVertical = hit["isVertical"]
        qdc = hit.get("qdc", 0.0)

        if detType == 1:  # Veto
            if station in [1, 2] and not isVertical:
                add_weighted_position(f"qdcAvg_veto{station}_y", hit["y"], qdc)
                add_weighted_position("qdcAvg_veto_y", hit["y"], qdc)

            elif station == 3 and isVertical:
                add_weighted_position("qdcAvg_veto3_x", hit["x"], qdc)
                add_weighted_position("qdcAvg_veto_x", hit["x"], qdc)

        elif detType == 0:  # SciFi
            if 1 <= station <= 5:
                if isVertical:
                    add_weighted_position(f"qdcAvg_scifi{station}_x", hit["x"], qdc)
                    add_weighted_position("qdcAvg_scifi_x", hit["x"], qdc)
                else:
                    add_weighted_position(f"qdcAvg_scifi{station}_y", hit["y"], qdc)
                    add_weighted_position("qdcAvg_scifi_y", hit["y"], qdc)

        elif detType == 2:  # Upstream
            if 1 <= station <= 5 and not isVertical:
                add_weighted_position(f"qdcAvg_us{station}_y", hit["y"], qdc)
                add_weighted_position("qdcAvg_us_y", hit["y"], qdc)

        elif detType == 3:  # Downstream
            if 1 <= station <= 4:
                if isVertical:
                    add_weighted_position(f"qdcAvg_ds{station}_x", hit["x"], qdc)
                    add_weighted_position("qdcAvg_ds_x", hit["x"], qdc)
                else:
                    add_weighted_position(f"qdcAvg_ds{station}_y", hit["y"], qdc)
                    add_weighted_position("qdcAvg_ds_y", hit["y"], qdc)

    # Fill all requested branch_vars safely
    for key in branch_vars:
        if key.startswith("qdcAvg_"):
            branch_vars[key][0] = (
                weighted_sums[key] / weight_sums[key]
                if weight_sums[key] > 0
                else -999.
            )

    

def build_scifi_hit_dict(hit, Scifi, args):
    detID = hit.GetDetectorID()
    station = hit.GetStation()
    hit_time_ns = scifi_time_ns(hit, Scifi, args)
    clock_cycle = hit_time_ns / NS_PER_CLOCK if valid_time(hit_time_ns) else INVALID_TIME
    qdc = non_negative_float(hit.GetSignal(0))
    mat = hit.GetMat()
    sipm = hit.GetSiPM()
    channel = hit.GetSiPMChan()
    layer_channel = channel + sipm * 128 + mat * 4 * 128

    A, B = ROOT.TVector3(), ROOT.TVector3()
    Scifi.GetSiPMPosition(detID, A, B)

    return {
        "detType": 0,
        "station": station,
        "isVertical": hit.isVertical(),
        "layer_channel": layer_channel,
        "qdc": qdc,
        "x": A.x(),
        "y": A.y(),
        "z": A.z(),
        "hitTimeNS": hit_time_ns,
        "hitTimeCY": clock_cycle,
    }


def filter_SciFiHits(SciFi_hits, lower_time_threshold=0.5, upper_time_threshold=2.3, bin_width=0.25):
    """
    Legacy/debug-only local filter. Production features use SND selectHits.

    Filter SciFi hits around MPV of hit time (per station + orientation).

    Inputs:
      SciFi_hits: list of dicts, each with at least:
        - station (int)
        - isVertical (bool)
        - hitTimeCY (float)  # in clock cycles
      lower_time_threshold, upper_time_threshold: in clock cycles

    Returns:
      filtered_hits, peak_by_group
    """
    if not SciFi_hits:
        return [], {}

    groups = defaultdict(list)
    for h in SciFi_hits:
        groups[(h["station"], h["isVertical"])].append(h)

    filtered = []
    peak_by_group = {}

    MAX_BINS = 2000
    for key, hits in groups.items():
        times = np.array([h["hitTimeCY"] for h in hits], dtype=float)

        # Robust MPV estimate via histogram mode
        tmin, tmax = float(times.min()), float(times.max())
        if tmax <= tmin:
            peak = tmin
        else:
            width = tmax - tmin
            nbins = int(np.ceil(width / bin_width))
            nbins = max(10, min(nbins, MAX_BINS))
            
            hist, edges = np.histogram(times, bins=nbins, range=(tmin, tmax))
            i_max = int(np.argmax(hist))
            peak = 0.5 * (edges[i_max] + edges[i_max + 1])

        peak_by_group[key] = peak

        lo = peak - lower_time_threshold
        hi = peak + upper_time_threshold
        for h in hits:
            t = h["hitTimeCY"]
            if lo <= t <= hi:
                filtered.append(h)

    return filtered, peak_by_group


def fill_mycode_density(branch_vars):
    station_x = [float(branch_vars[f"density_scifi{i}_x"][0]) for i in range(1, 6)]
    station_y = [float(branch_vars[f"density_scifi{i}_y"][0]) for i in range(1, 6)]

    # Match getSumDensity station ranking: choose by total density x+y
    station_sum = [station_x[i] + station_y[i] for i in range(5)]
    station_min = [min(station_x[i], station_y[i]) for i in range(5)]


    best_idx = 0
    best_sum = station_sum[0]

    second_idx = -1
    second_sum = 0.0

    for i in range(5):
        s = station_sum[i]

        # Match getSumDensity: strict '>' for best, so first max wins ties
        if s > best_sum:
            second_idx = best_idx
            second_sum = best_sum
            best_idx = i
            best_sum = s
        # Match getSumDensity: second-best must be strictly below best and strictly above current second
        elif s < best_sum and s > second_sum:
            second_idx = i
            second_sum = s

    second_density = station_min[second_idx] if second_idx >= 0 else 0.0
    branch_vars["density_mycode_scifi"][0] = station_min[best_idx]
    branch_vars["density_mycodescifi_second"][0] = second_density
    branch_vars["density_mycode_scifi_second"][0] = second_density
    branch_vars["density_mycode_scifi_hor"][0] = station_y[best_idx]
    branch_vars["density_mycode_scifi_ver"][0] = station_x[best_idx]

def hitWeightDensity(SciFi_hits, branch_vars):
    # Python equivalent of sndSciFiTools.cxx hitWeightComputation (width = 1 cm)
    def sum_hit_weights_1d(positions, width=1.0):
        if not positions:
            return 0.0

        pos = sorted(float(p) for p in positions)
        n = len(pos)
        left = 0
        right = 0
        total = 0.0

        for i in range(n):
            x = pos[i]
            while right < n and pos[right] <= x + width:
                right += 1
            while left < n and pos[left] < x - width:
                left += 1

            neighbors = (right - left) - 1  # exclude self
            if neighbors < 0:
                neighbors = 0
            total += neighbors

        return total

    # collect positions per station and orientation
    # vertical -> x density, horizontal -> y density
    st_x = {i: [] for i in range(1, 6)}
    st_y = {i: [] for i in range(1, 6)}

    for h in SciFi_hits:
        st = int(h.get("station", 0))
        if st < 1 or st > 5:
            continue

        if h.get("isVertical", False):
            st_x[st].append(h.get("x", 0.0))
        else:
            st_y[st].append(h.get("y", 0.0))

    total_x = 0.0
    total_y = 0.0

    for st in range(1, 6):
        dx = sum_hit_weights_1d(st_x[st], width=1.0)
        dy = sum_hit_weights_1d(st_y[st], width=1.0)
        d = dx + dy

        branch_vars[f"density_scifi{st}_x"][0] = dx
        branch_vars[f"density_scifi{st}_y"][0] = dy
        branch_vars[f"density_scifi{st}"][0] = d

        total_x += dx
        total_y += dy

    branch_vars["density_scifi_x"][0] = total_x
    branch_vars["density_scifi_y"][0] = total_y
    branch_vars["density_scifi"][0] = total_x + total_y

    return branch_vars


def process_scifi_topology_features(SciFi_hits, branch_vars):
    stations = [1, 2, 3, 4, 5]
    counts = [0] * 5
    qdcs = [0.0] * 5
    x_counts = [0] * 5
    y_counts = [0] * 5
    station_x_positions = [[] for _ in stations]
    station_y_positions = [[] for _ in stations]
    all_x_positions = []
    all_y_positions = []

    for hit in SciFi_hits:
        station = int(hit.get("station", 0))
        if station < 1 or station > 5:
            continue

        idx = station - 1
        qdc = non_negative_float(hit.get("qdc", 0.0))
        counts[idx] += 1
        qdcs[idx] += qdc

        if hit.get("isVertical", False):
            x_counts[idx] += 1
            x = float(hit.get("x", 0.0))
            station_x_positions[idx].append(x)
            all_x_positions.append(x)
        else:
            y_counts[idx] += 1
            y = float(hit.get("y", 0.0))
            station_y_positions[idx].append(y)
            all_y_positions.append(y)

    total_count = sum(counts)
    total_qdc = sum(qdcs)
    active_stations = [station for station, count in zip(stations, counts) if count > 0]

    branch_vars["scifi_first_station"][0] = min(active_stations) if active_stations else -999
    branch_vars["scifi_last_station"][0] = max(active_stations) if active_stations else -999
    branch_vars["scifi_n_active_stations"][0] = len(active_stations)
    branch_vars["scifi_n_active_stations_xy"][0] = sum(
        1 for x_count, y_count in zip(x_counts, y_counts) if x_count > 0 and y_count > 0
    )

    branch_vars["scifi_max_count_station"][0] = (
        max(range(5), key=lambda idx: counts[idx]) + 1 if total_count > 0 else -999
    )
    branch_vars["scifi_max_qdc_station"][0] = (
        max(range(5), key=lambda idx: qdcs[idx]) + 1 if total_qdc > 0 else -999
    )
    branch_vars["scifi_count_peak_fraction"][0] = safe_fraction(max(counts), total_count)
    branch_vars["scifi_qdc_peak_fraction"][0] = safe_fraction(max(qdcs), total_qdc)

    for idx, station in enumerate(stations):
        # Keep the two SciFi views explicit.  ``count_scifiN`` remains the
        # backwards-compatible X+Y total; the view counts make it possible to
        # apply per-plane multiplicity requirements without approximating them
        # with the station total.
        branch_vars[f"count_scifi{station}_x"][0] = x_counts[idx]
        branch_vars[f"count_scifi{station}_y"][0] = y_counts[idx]
        branch_vars[f"scifi_count_frac{station}"][0] = safe_fraction(counts[idx], total_count)
        branch_vars[f"scifi_qdc_frac{station}"][0] = safe_fraction(qdcs[idx], total_qdc)
        branch_vars[f"scifi{station}_std_x"][0] = std_or_sentinel(station_x_positions[idx])
        branch_vars[f"scifi{station}_std_y"][0] = std_or_sentinel(station_y_positions[idx])

    count_mean, count_std = weighted_mean_and_std(stations, counts)
    qdc_mean, qdc_std = weighted_mean_and_std(stations, qdcs)
    branch_vars["scifi_count_mean_station"][0] = count_mean
    branch_vars["scifi_count_std_station"][0] = count_std
    branch_vars["scifi_qdc_mean_station"][0] = qdc_mean
    branch_vars["scifi_qdc_std_station"][0] = qdc_std
    branch_vars["scifi_std_x"][0] = std_or_sentinel(all_x_positions)
    branch_vars["scifi_std_y"][0] = std_or_sentinel(all_y_positions)


def fill_time_summary(prefix, times, branch_vars):
    times = [float(value) for value in times if valid_time(value)]
    if not times:
        return

    time_min = min(times)
    time_max = max(times)
    branch_vars[f"{prefix}_min_ns"][0] = time_min
    branch_vars[f"{prefix}_max_ns"][0] = time_max
    branch_vars[f"{prefix}_mean_ns"][0] = sum(times) / len(times)
    branch_vars[f"{prefix}_std_ns"][0] = std_or_sentinel(times)
    branch_vars[f"{prefix}_range_ns"][0] = time_max - time_min


def process_hit_time_features(SciFi_hits, MuFilter_hits, branch_vars, time0_ns, time0_applied):
    scifi_times = [hit.get("hitTimeNS", INVALID_TIME) for hit in SciFi_hits]
    veto_times = [
        hit.get("hitTimeNS", INVALID_TIME)
        for hit in MuFilter_hits
        if int(hit.get("detType", -1)) == 1
    ]
    us_times = [
        hit.get("hitTimeNS", INVALID_TIME)
        for hit in MuFilter_hits
        if int(hit.get("detType", -1)) == 2
    ]
    ds_times = [
        hit.get("hitTimeNS", INVALID_TIME)
        for hit in MuFilter_hits
        if int(hit.get("detType", -1)) == 3
    ]

    branch_vars["hitTime_time0_ns"][0] = time0_ns
    branch_vars["hitTime_time0_applied"][0] = int(time0_applied)

    fill_time_summary("hitTime", scifi_times + veto_times + us_times + ds_times, branch_vars)
    fill_time_summary("hitTime_scifi", scifi_times, branch_vars)
    fill_time_summary("hitTime_veto", veto_times, branch_vars)
    fill_time_summary("hitTime_us", us_times, branch_vars)
    fill_time_summary("hitTime_ds", ds_times, branch_vars)


def process_hits(
    args,
    event,
    snd_geo,
    branch_vars,
    timing=None,
    timing_counts=None,
    hit_time_debug=None,
    hit_time_vector_branches=None,
):
    """Process all hits in the event and update hits array and averages."""
    process_start = time.perf_counter()
    eventId = branch_vars["eventId"][0]
    MC = args.type
    Scifi = snd_geo.modules['Scifi']
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    step_start = time.perf_counter()
    if needs_first_arrival_time0(args):
        selected_scifi_hits = [hit for hit in event.Digi_ScifiHits if hit.isValid()]
    else:
        selected_scifi_hits = selectHits(event, MC=("MC" in args.type))
    record_timing(timing, timing_counts, "process_hits.select_scifi", step_start)

    step_start = time.perf_counter()
    SciFi_hit_entries = []
    for aHit in selected_scifi_hits:
        if not aHit.isValid():
            continue
        SciFi_hit_entries.append({
            "raw": aHit,
            "hit": build_scifi_hit_dict(aHit, Scifi, args),
        })
    SciFi_hits = [entry["hit"] for entry in SciFi_hit_entries]
    record_timing(timing, timing_counts, "process_hits.build_scifi_dicts", step_start)

    # Process MuFilter hits
    step_start = time.perf_counter()
    MuFilter_hits = []
    for aHit in event.Digi_MuFilterHits:
        if not aHit.isValid():
            continue

        detID = aHit.GetDetectorID()      
        MuFilter.GetPosition(detID, A, B)
        detType = aHit.GetSystem()
        station = (detID // 1000) % 10
        hit_time_ns = mufilter_hit_time_ns(aHit, MuFilter, args)
        clock_cycle = aHit.GetTime() / NS_PER_CLOCK
        barIndex = detID%100
        
        
        qdc = 0.0
        for key, value in aHit.GetAllSignals():
            qdc += non_negative_float(value)
        
        
        MuFilter_hits.append({
            "detType": detType,
            "station": station+1,
            "isVertical": aHit.isVertical(),
            "qdc":qdc,
            "barIndex":barIndex,
            "hitTimeNS": hit_time_ns,
            "hitTimeCY": clock_cycle,
            "x":A.x(),
            "y":A.y(),
            "z":A.z(),
        })
    record_timing(timing, timing_counts, "process_hits.build_mufilter_dicts", step_start)

    raw_scifi_times = [hit.get("hitTimeNS", INVALID_TIME) for hit in SciFi_hits]
    raw_mufilter_times = [hit.get("hitTimeNS", INVALID_TIME) for hit in MuFilter_hits]
    if hit_time_vector_branches is not None:
        fill_raw_hit_time_vectors(SciFi_hits, MuFilter_hits, hit_time_vector_branches)

    step_start = time.perf_counter()
    time0_ns, time0_applied = apply_first_arrival_time0(args, [SciFi_hits, MuFilter_hits])
    if time0_applied:
        for hit in SciFi_hits + MuFilter_hits:
            time_ns = hit.get("hitTimeNS", INVALID_TIME)
            if valid_time(time_ns):
                hit["hitTimeCY"] = float(time_ns) / NS_PER_CLOCK
    record_timing(timing, timing_counts, "process_hits.time0", step_start)

    calibrated_scifi_times = [hit.get("hitTimeNS", INVALID_TIME) for hit in SciFi_hits]
    calibrated_mufilter_times = [hit.get("hitTimeNS", INVALID_TIME) for hit in MuFilter_hits]

    step_start = time.perf_counter()
    if needs_first_arrival_time0(args):
        SciFi_hit_entries = select_calibrated_scifi_hits(
            SciFi_hit_entries,
            hit_getter=lambda entry: entry["hit"],
            time_key="hitTimeNS",
            ns_per_clock=NS_PER_CLOCK,
        )
    record_timing(timing, timing_counts, "process_hits.select_calibrated_scifi", step_start)

    step_start = time.perf_counter()
    SciFi_hit_entries = [
        entry
        for entry in SciFi_hit_entries
        if keep_hit_in_mc_time_window(args, entry["hit"], "hitTimeNS")
    ]
    SciFi_hits = [entry["hit"] for entry in SciFi_hit_entries]
    MuFilter_hits = filter_late_mc_hits(args, MuFilter_hits, "hitTimeNS")
    record_timing(timing, timing_counts, "process_hits.filter_late_mc_hits", step_start)

    kept_scifi_times = [hit.get("hitTimeNS", INVALID_TIME) for hit in SciFi_hits]
    kept_mufilter_times = [hit.get("hitTimeNS", INVALID_TIME) for hit in MuFilter_hits]
    update_hit_time_debug(
        hit_time_debug,
        args,
        branch_vars["eventIndex"][0],
        branch_vars["eventId"][0],
        time0_ns,
        time0_applied,
        raw_scifi_times,
        raw_mufilter_times,
        calibrated_scifi_times,
        calibrated_mufilter_times,
        kept_scifi_times,
        kept_mufilter_times,
    )

    step_start = time.perf_counter()
    selected_scifi_hits_for_density = [entry["raw"] for entry in SciFi_hit_entries]
    dens, dens2, dver, dhor = getSumDensity(selected_scifi_hits_for_density, return_2ndhighest=True, return_hv=True)
    branch_vars["density_sndsw_scifi"][0] = dens
    branch_vars["density_sndsw_scifi_second"][0] = dens2
    branch_vars["density_sndsw_scifi_ver"][0] = dver
    branch_vars["density_sndsw_scifi_hor"][0] = dhor
    record_timing(timing, timing_counts, "process_hits.sndsw_density", step_start)
    
    step_start = time.perf_counter()
    hitWeightDensity(SciFi_hits, branch_vars)
    record_timing(timing, timing_counts, "process_hits.hitWeightDensity", step_start)
    
    step_start = time.perf_counter()
    process_count_and_qdc(SciFi_hits, MuFilter_hits, branch_vars)
    record_timing(timing, timing_counts, "process_hits.count_qdc", step_start)
    
    step_start = time.perf_counter()
    process_avgPos(SciFi_hits, MuFilter_hits, branch_vars)
    record_timing(timing, timing_counts, "process_hits.avg_pos", step_start)

    step_start = time.perf_counter()
    process_qdcAvgPos(SciFi_hits, MuFilter_hits, branch_vars)
    record_timing(timing, timing_counts, "process_hits.qdc_avg_pos", step_start)

    step_start = time.perf_counter()
    process_scifi_topology_features(SciFi_hits, branch_vars)
    record_timing(timing, timing_counts, "process_hits.scifi_topology", step_start)

    step_start = time.perf_counter()
    fill_scifi_cluster_features(
        SciFi_hits,
        branch_vars,
        {
            "runId": branch_vars["runId"][0],
            "eventId": branch_vars["eventId"][0],
            "eventIndex": branch_vars["eventIndex"][0],
        },
    )
    record_timing(timing, timing_counts, "process_hits.scifi_clusters", step_start)

    step_start = time.perf_counter()
    process_hit_time_features(SciFi_hits, MuFilter_hits, branch_vars, time0_ns, time0_applied)
    record_timing(timing, timing_counts, "process_hits.hit_time_features", step_start)
    
    step_start = time.perf_counter()
    fill_mycode_density(branch_vars)
    record_timing(timing, timing_counts, "process_hits.mycode_density", step_start)
    
    step_start = time.perf_counter()
    process_vetoHitTime(MuFilter_hits, branch_vars)
    record_timing(timing, timing_counts, "process_hits.veto_time", step_start)

    record_timing(timing, timing_counts, "process_hits.total", process_start)

    return 





def main(args):
    reco_track_schema = getattr(args, "reco_track_schema", FLAT_V1)
    if not args.extract_reco_tracks and reco_track_schema != FLAT_V1:
        raise ValueError(
            "--reco-track-schema only applies when --extract-reco-tracks is enabled"
        )
    print("start processing digi to features")
    job_start = time.perf_counter()
    timing = defaultdict(float)
    timing_counts = defaultdict(int)
    hit_time_debug = make_hit_time_debug()
    
    step_start = time.perf_counter()
    snd_geo = setup_geometry(args.geo_path )
    record_timing(timing, timing_counts, "setup.geometry", step_start)

    step_start = time.perf_counter()
    selection_mode = resolve_selection_mode(
        args.type, getattr(args, "selection_mode", None)
    )
    direct_selection = selection_mode != EXTERNAL_CUTFLOW
    if uses_external_cutflow(args.type, selection_mode) and not args.preSelect_path:
        raise ValueError("--preSelectPath is required for external_cutflow mode")

    raw_data, raw_tree = open_root_file(args.digi_path, tree_name=args.tree_name)
    preSelect_data = None
    preSelect_tree = None
    event_deltat_branch = None
    if not direct_selection:
        preSelect_data, preSelect_tree = open_root_file(
            args.preSelect_path, tree_name='cutFlowSummary'
        )
        event_deltat_branch = setup_event_deltat_alias(preSelect_tree)
    else:
        print(f"Using {selection_mode}: processing input-tree entries directly")
    reco_track_branch_available = bool(raw_tree.GetBranch("Reco_MuonTracks"))
    if args.extract_reco_tracks and not reco_track_branch_available:
        raise RuntimeError(
            "--extract-reco-tracks was requested, but input tree "
            f"{args.tree_name!r} has no Reco_MuonTracks branch"
        )
    record_timing(timing, timing_counts, "setup.open_files", step_start)
    
    step_start = time.perf_counter()
    out_file, new_tree = create_output_file(args.out_path, args.mode)
    record_timing(timing, timing_counts, "setup.output_file", step_start)
    
    elist_name = "elist"


    step_start = time.perf_counter()
    selection = None if direct_selection or skip_scifi_minhits_selection(args) else "SciFiMinHits == 1"
    if not direct_selection and "MC" not in args.type:
        if not event_deltat_branch:
            raise RuntimeError("No EventDeltat cut branch found for real-data selection")
        data_selection = "StableBeams == 1 && IP1 == 1 && EventDeltat_m1_100 == 1"

        if selection:
            selection = selection + " && " + data_selection
        else:
            selection = data_selection
            
    if selection:
        print(f"Applying selection: {selection}")
        n_match = preSelect_tree.GetEntries(selection)
    elif direct_selection:
        n_match = raw_tree.GetEntries()
    else:
        print("No selection applied")
        n_match = preSelect_tree.GetEntries()

    print(f"Entries matching selection: {n_match}")

    if n_match == 0:
        elist = ROOT.TEntryList("elist_empty", "elist_empty")
    elif selection:
        preSelect_tree.Draw(f">>{elist_name}", selection, "entrylist")
        elist = ROOT.gDirectory.Get(elist_name)

        if not elist or not isinstance(elist, ROOT.TEntryList):
            raise RuntimeError("Failed to create or retrieve TEntryList")

        preSelect_tree.SetEntryList(elist)
    else:
        elist = ROOT.TEntryList("elist_all", "elist_all")
        entry_count = raw_tree.GetEntries() if direct_selection else preSelect_tree.GetEntries()
        for j in range(entry_count):
            elist.Enter(j)
    record_timing(timing, timing_counts, "setup.selection", step_start)

    step_start = time.perf_counter()
    branches = [
        ("runId", 'i'), ("eventId", 'i'), ("pdgCode", 'i'), ("isMC", 'i'), ("eventIndex", 'i'),
        ("nPrimary", 'i'), ("nSecondaryRaw", 'i'), ("nSecondary", 'i'),
        ("px", 'f'), ("py", 'f'), ("pz", 'f'),  # Floats
        ("mc_incoming_energy", 'd'),
        ("x", 'f'), ("y", 'f'), ("z", 'f'),    # Floats
        
        ("count_veto1", 'i'), ("count_veto2", 'i'), ("count_veto3", 'i'),  ("count_veto", 'i'), 
        ("count_scifi1", 'i'), ("count_scifi2", 'i'), ("count_scifi3", 'i'),("count_scifi4", 'i'), ("count_scifi5", 'i'), ("count_scifi", 'i'),
        ("count_scifi1_x", 'i'), ("count_scifi1_y", 'i'),
        ("count_scifi2_x", 'i'), ("count_scifi2_y", 'i'),
        ("count_scifi3_x", 'i'), ("count_scifi3_y", 'i'),
        ("count_scifi4_x", 'i'), ("count_scifi4_y", 'i'),
        ("count_scifi5_x", 'i'), ("count_scifi5_y", 'i'),
        ("count_us1", 'i'), ("count_us2", 'i'), ("count_us3", 'i'),("count_us4", 'i'), ("count_us5", 'i'), ("count_us", 'i'),
        ("count_ds1", 'i'), ("count_ds2", 'i'), ("count_ds3", 'i'), ("count_ds4", 'i'),("count_ds", 'i'),
        
        ("qdc_veto1", 'd'), ("qdc_veto2", 'd'), ("qdc_veto3", 'd'),  ("qdc_veto", 'd'), 
        ("qdc_scifi1", 'd'), ("qdc_scifi2", 'd'), ("qdc_scifi3", 'd'),("qdc_scifi4", 'd'), ("qdc_scifi5", 'd'), ("qdc_scifi", 'd'),
        ("qdc_us1", 'd'), ("qdc_us2", 'd'), ("qdc_us3", 'd'),("qdc_us4", 'd'), ("qdc_us5", 'd'), ("qdc_us", 'd'),
        ("qdc_ds1", 'd'), ("qdc_ds2", 'd'), ("qdc_ds3", 'd'), ("qdc_ds4", 'd'),("qdc_ds", 'd'),
        
        ("avg_veto1_y", 'd'), ("avg_veto2_y", 'd'), ("avg_veto3_x", 'd'), ("avg_veto_x", 'd'), ("avg_veto_y", 'd'),
        ("avg_scifi1_x", 'd'), ("avg_scifi1_y", 'd'),
        ("avg_scifi2_x", 'd'), ("avg_scifi2_y", 'd'),
        ("avg_scifi3_x", 'd'), ("avg_scifi3_y", 'd'),
        ("avg_scifi4_x", 'd'), ("avg_scifi4_y", 'd'),
        ("avg_scifi5_x", 'd'), ("avg_scifi5_y", 'd'), ("avg_scifi_y", 'd'), ("avg_scifi_x", 'd'),
        ("avg_us1_y", 'd'), ("avg_us2_y", 'd'), ("avg_us3_y", 'd'), ("avg_us4_y", 'd'), ("avg_us5_y", 'd'), ("avg_us_y", 'd'),
        ("avg_ds1_x", 'd'), ("avg_ds1_y", 'd'),
        ("avg_ds2_x", 'd'), ("avg_ds2_y", 'd'),
        ("avg_ds3_x", 'd'), ("avg_ds3_y", 'd'),
        ("avg_ds4_x", 'd'), ("avg_ds4_y", 'd'), ("avg_ds_x", 'd'), ("avg_ds_y", 'd'),

        ("qdcAvg_veto1_y", 'd'), ("qdcAvg_veto2_y", 'd'), ("qdcAvg_veto3_x", 'd'), ("qdcAvg_veto_x", 'd'), ("qdcAvg_veto_y", 'd'),
        ("qdcAvg_scifi1_x", 'd'), ("qdcAvg_scifi1_y", 'd'),
        ("qdcAvg_scifi2_x", 'd'), ("qdcAvg_scifi2_y", 'd'),
        ("qdcAvg_scifi3_x", 'd'), ("qdcAvg_scifi3_y", 'd'),
        ("qdcAvg_scifi4_x", 'd'), ("qdcAvg_scifi4_y", 'd'),
        ("qdcAvg_scifi5_x", 'd'), ("qdcAvg_scifi5_y", 'd'), ("qdcAvg_scifi_y", 'd'), ("qdcAvg_scifi_x", 'd'),
        ("qdcAvg_us1_y", 'd'), ("qdcAvg_us2_y", 'd'), ("qdcAvg_us3_y", 'd'), ("qdcAvg_us4_y", 'd'), ("qdcAvg_us5_y", 'd'), ("qdcAvg_us_y", 'd'),
        ("qdcAvg_ds1_x", 'd'), ("qdcAvg_ds1_y", 'd'),
        ("qdcAvg_ds2_x", 'd'), ("qdcAvg_ds2_y", 'd'),
        ("qdcAvg_ds3_x", 'd'), ("qdcAvg_ds3_y", 'd'),
        ("qdcAvg_ds4_x", 'd'), ("qdcAvg_ds4_y", 'd'), ("qdcAvg_ds_x", 'd'), ("qdcAvg_ds_y", 'd'),
        
        # Hit density sums per plane
        ("density_scifi1", 'd'), ("density_scifi2", 'd'), ("density_scifi3", 'd'), ("density_scifi4", 'd'), ("density_scifi5", 'd'), ("density_scifi", 'd'),
        ("density_scifi1_x", 'd'), ("density_scifi2_x", 'd'), ("density_scifi3_x", 'd'), ("density_scifi4_x", 'd'), ("density_scifi5_x", 'd'), ("density_scifi_x", 'd'),
        ("density_scifi1_y", 'd'), ("density_scifi2_y", 'd'), ("density_scifi3_y", 'd'), ("density_scifi4_y", 'd'), ("density_scifi5_y", 'd'), ("density_scifi_y", 'd'),
        
        ("density_mycode_scifi", 'd'), ("density_mycodescifi_second", 'd'), ("density_mycode_scifi_second", 'd'), ("density_mycode_scifi_hor", 'd'), ("density_mycode_scifi_ver", 'd'), 
        
        ("density_sndsw_scifi", 'd'), ("density_sndsw_scifi_second", 'd'), ("density_sndsw_scifi_hor", 'd'), ("density_sndsw_scifi_ver", 'd'), 

        ("scifi_first_station", 'i'), ("scifi_last_station", 'i'),
        ("scifi_n_active_stations", 'i'), ("scifi_n_active_stations_xy", 'i'),
        ("scifi_max_count_station", 'i'), ("scifi_max_qdc_station", 'i'),

        ("scifi_count_peak_fraction", 'd'), ("scifi_qdc_peak_fraction", 'd'),
        ("scifi_count_frac1", 'd'), ("scifi_count_frac2", 'd'), ("scifi_count_frac3", 'd'), ("scifi_count_frac4", 'd'), ("scifi_count_frac5", 'd'),
        ("scifi_qdc_frac1", 'd'), ("scifi_qdc_frac2", 'd'), ("scifi_qdc_frac3", 'd'), ("scifi_qdc_frac4", 'd'), ("scifi_qdc_frac5", 'd'),
        ("scifi_count_mean_station", 'd'), ("scifi_count_std_station", 'd'),
        ("scifi_qdc_mean_station", 'd'), ("scifi_qdc_std_station", 'd'),
        ("scifi_std_x", 'd'), ("scifi_std_y", 'd'),
        ("scifi1_std_x", 'd'), ("scifi1_std_y", 'd'),
        ("scifi2_std_x", 'd'), ("scifi2_std_y", 'd'),
        ("scifi3_std_x", 'd'), ("scifi3_std_y", 'd'),
        ("scifi4_std_x", 'd'), ("scifi4_std_y", 'd'),
        ("scifi5_std_x", 'd'), ("scifi5_std_y", 'd'),

        ("hitTime_time0_ns", 'd'), ("hitTime_time0_applied", 'i'),
        ("hitTime_min_ns", 'd'), ("hitTime_max_ns", 'd'), ("hitTime_mean_ns", 'd'), ("hitTime_std_ns", 'd'), ("hitTime_range_ns", 'd'),
        ("hitTime_scifi_min_ns", 'd'), ("hitTime_scifi_max_ns", 'd'), ("hitTime_scifi_mean_ns", 'd'), ("hitTime_scifi_std_ns", 'd'), ("hitTime_scifi_range_ns", 'd'),
        ("hitTime_veto_min_ns", 'd'), ("hitTime_veto_max_ns", 'd'), ("hitTime_veto_mean_ns", 'd'), ("hitTime_veto_std_ns", 'd'), ("hitTime_veto_range_ns", 'd'),
        ("hitTime_us_min_ns", 'd'), ("hitTime_us_max_ns", 'd'), ("hitTime_us_mean_ns", 'd'), ("hitTime_us_std_ns", 'd'), ("hitTime_us_range_ns", 'd'),
        ("hitTime_ds_min_ns", 'd'), ("hitTime_ds_max_ns", 'd'), ("hitTime_ds_mean_ns", 'd'), ("hitTime_ds_std_ns", 'd'), ("hitTime_ds_range_ns", 'd'),
        

        ("vetoHitTime_earlist", 'd'), ("vetoHitTime_latest", 'd'),
        ("vetoHitTime_earlist_veto1", 'd'), ("vetoHitTime_latest_veto1", 'd'),
        ("vetoHitTime_earlist_veto2", 'd'), ("vetoHitTime_latest_veto2", 'd'),
        ("vetoHitTime_earlist_veto3", 'd'), ("vetoHitTime_latest_veto3", 'd'),
    ]
    branches.extend(scifi_cluster_scalar_dtypes())
    branch_names = [name for name, _ in branches]
    if len(branch_names) != len(set(branch_names)):
        raise RuntimeError("Duplicate scalar branch name in feature schema")

    # Dictionary to hold branch variables
    branch_vars = {}

    # Create branches dynamically
    for name, dtype in branches:
        branch_vars[name] = array.array(dtype, [-999])  # Initialize the array
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")

    track_vector_branches = make_track_vector_branches()
    branch_vector_vars(new_tree, track_vector_branches)
    hit_time_vector_branches = make_hit_time_vector_branches()
    branch_vector_vars(new_tree, hit_time_vector_branches)
    reco_track_scalar_branches = {}
    reco_track_vector_branches = {}
    if args.extract_reco_tracks:
        reco_track_scalar_dtypes = scalar_dtypes_for_schema(reco_track_schema)
        reco_track_vector_dtypes = vector_dtypes_for_schema(reco_track_schema)
        for name, dtype in reco_track_scalar_dtypes.items():
            initial = 0 if dtype == "i" else math.nan
            reco_track_scalar_branches[name] = array.array(dtype, [initial])
            root_dtype = "I" if dtype == "i" else "D"
            new_tree.Branch(
                name,
                reco_track_scalar_branches[name],
                f"{name}/{root_dtype}",
            )
        reco_track_vector_branches = {
            name: ROOT.std.vector(dtype)()
            for name, dtype in reco_track_vector_dtypes.items()
        }
        branch_vector_vars(new_tree, reco_track_vector_branches)
    record_timing(timing, timing_counts, "setup.branches", step_start)

    
    n_selected_entries = elist.GetN()
    n_written_entries = 0
    kept_cutflow_entries = ROOT.TEntryList("elist_kept", "elist_kept")
    direct_cutflow = None
    direct_cutflow_entry = None
    direct_cutflow_marker = None
    if direct_selection:
        marker_name = "PassingMuonReco"
        marker_title = "Passing-muon direct selection"
        direct_cutflow = ROOT.TTree("cutFlowSummary", marker_title)
        direct_cutflow_entry = array.array('q', [0])
        direct_cutflow_marker = array.array('i', [1])
        direct_cutflow.Branch("entry", direct_cutflow_entry, "entry/L")
        direct_cutflow.Branch(marker_name, direct_cutflow_marker, f"{marker_name}/I")
    n_dropped_has_veto_has_us = 0
    for i in range(n_selected_entries):
        step_start = time.perf_counter()
        reset_vector_branches(track_vector_branches)
        reset_vector_branches(hit_time_vector_branches)
        if args.extract_reco_tracks:
            reset_reco_track_buffers(
                reco_track_scalar_branches,
                reco_track_vector_branches,
                reco_track_branch_available,
            )
        if i % 10000 == 0:
            progress = 100.0 * i / n_selected_entries if n_selected_entries else 100.0
            print(f"processed {i}/{n_selected_entries} events ({progress:.2f}%)")
        # Reset all branch variables before filling them
        for key in branch_vars:
            branch_vars[key][0] = -999
        record_timing(timing, timing_counts, "event.reset_branches", step_start)

        step_start = time.perf_counter()
        selected_entry = elist.GetEntry(i)
        if direct_selection:
            cutflow_entry = None
            raw_entry = selected_entry
        else:
            cutflow_entry = selected_entry
            preSelect_tree.GetEntry(cutflow_entry)
            if hasattr(preSelect_tree, "entry"):
                raw_entry = int(preSelect_tree.entry)
            else:
                raw_entry = cutflow_entry
        record_timing(timing, timing_counts, "event.read_preselect", step_start)

        step_start = time.perf_counter()
        raw_tree.GetEntry(raw_entry)
        record_timing(timing, timing_counts, "event.read_raw", step_start)

        step_start = time.perf_counter()
        init_event_geometry(snd_geo, raw_tree.EventHeader)
        record_timing(timing, timing_counts, "event.init_geometry", step_start)

        step_start = time.perf_counter()
        drop_event = should_drop_real_has_veto_has_us(args, raw_tree)
        record_timing(timing, timing_counts, "event.drop_real_has_veto_has_us", step_start)
        if drop_event:
            n_dropped_has_veto_has_us += 1
            continue

        if args.extract_reco_tracks:
            step_start = time.perf_counter()
            fill_reco_track_buffers(
                raw_tree.Reco_MuonTracks,
                reco_track_scalar_branches,
                reco_track_vector_branches,
                schema=reco_track_schema,
            )
            record_timing(
                timing, timing_counts, "event.reco_track_features", step_start
            )

        step_start = time.perf_counter()
        if direct_selection:
            direct_cutflow_entry[0] = raw_entry
            direct_cutflow_marker[0] = 1
            direct_cutflow.Fill()
        else:
            kept_cutflow_entries.Enter(cutflow_entry)
        
        branch_vars["eventIndex"][0] = raw_entry
        branch_vars["runId"][0] = raw_tree.EventHeader.GetRunId()
        record_timing(timing, timing_counts, "event.metadata_base", step_start)
        
        
        if ('MC' in  args.type):
            step_start = time.perf_counter()
            branch_vars["isMC"][0] = 1
            try:
                branch_vars["eventId"][0] = raw_tree.EventHeader.GetEventNumber()
            except Exception:
                branch_vars["eventId"][0] = raw_tree.EventHeader.GetMCEntryNumber()
                
            mc_tracks = raw_tree.MCTrack
            n_mc_tracks = int(mc_tracks.GetEntriesFast())
            neutrino_pdgCode = [12, -12, 14, -14, 16, -16]

            if n_mc_tracks >= 1:
                track0 = mc_tracks[0]
                event_pdg0 = track0.GetPdgCode()
                branch_vars["pdgCode"][0] = event_pdg0
                branch_vars["px"][0] = track0.GetPx()
                branch_vars["py"][0] = track0.GetPy()
                branch_vars["pz"][0] = track0.GetPz()
                branch_vars["mc_incoming_energy"][0] = track0.GetEnergy()

                if n_mc_tracks >= 2:
                    track1 = mc_tracks[1]
                    event_pdg1 = track1.GetPdgCode()

                    if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
                        branch_vars["pdgCode"][0] = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100

                    branch_vars["x"][0] = track1.GetStartX()
                    branch_vars["y"][0] = track1.GetStartY()
                    branch_vars["z"][0] = track1.GetStartZ()
            record_timing(timing, timing_counts, "event.mc_metadata", step_start)

            if is_muonDIS_sample(args):
                step_start = time.perf_counter()
                n_primary, n_secondary_raw, n_secondary = process_muonDIS_tracks(raw_tree, track_vector_branches)
                branch_vars["nPrimary"][0] = n_primary
                branch_vars["nSecondaryRaw"][0] = n_secondary_raw
                branch_vars["nSecondary"][0] = n_secondary
                record_timing(timing, timing_counts, "event.muonDIS_truth", step_start)
            
            
        
        elif('real' in  args.type):
            step_start = time.perf_counter()
            branch_vars["isMC"][0] = 0
            branch_vars["pdgCode"][0] = 0
            branch_vars["eventId"][0] = raw_tree.EventHeader.GetEventNumber()
            record_timing(timing, timing_counts, "event.real_metadata", step_start)

        process_hits(
            args,
            raw_tree,
            snd_geo,
            branch_vars,
            timing,
            timing_counts,
            hit_time_debug,
            hit_time_vector_branches,
        )
        #if i>2:
        #    break
        step_start = time.perf_counter()
        new_tree.Fill()
        n_written_entries += 1
        record_timing(timing, timing_counts, "event.output_fill", step_start)
    # Finalize the output file
    step_start = time.perf_counter()
    new_tree.Write()
    record_timing(timing, timing_counts, "finalize.write_sndData", step_start)

    step_start = time.perf_counter()
    if direct_selection:
        direct_cutflow.Write()
    else:
        if kept_cutflow_entries.GetN() > 0:
            preSelect_tree.SetEntryList(kept_cutflow_entries)
            cutflow_selected = preSelect_tree.CopyTree("")
        else:
            preSelect_tree.SetEntryList(0)
            cutflow_selected = preSelect_tree.CopyTree("0")
        cutflow_selected.SetName("cutFlowSummary")
        cutflow_selected.Write()
    ROOT.TNamed(
        "metadata_selection_mode",
        selection_mode,
    ).Write()
    ROOT.TNamed("metadata_input_tree", str(args.tree_name)).Write()
    ROOT.TNamed("metadata_external_cutflow_used", "0" if direct_selection else "1").Write()
    ROOT.TNamed("metadata_scifi_cluster_schema", SCIFI_CLUSTER_SCHEMA_VERSION).Write()
    ROOT.TNamed(
        "metadata_reco_track_features",
        reco_track_schema if args.extract_reco_tracks else "disabled",
    ).Write()
    ROOT.TNamed(
        "metadata_reco_track_source_branch",
        "Reco_MuonTracks" if args.extract_reco_tracks else "",
    ).Write()
    record_timing(timing, timing_counts, "finalize.write_cutFlowSummary", step_start)

    step_start = time.perf_counter()
    out_file.Close()
    record_timing(timing, timing_counts, "finalize.close_file", step_start)
    if args.drop_real_has_veto_has_us:
        print(
            "Dropped real-data events with both veto and US hits: "
            f"{n_dropped_has_veto_has_us}"
        )
    print_timing_summary(
        timing,
        timing_counts,
        time.perf_counter() - job_start,
        n_selected_entries,
        n_written_entries,
    )
    print_hit_time_debug_summary(hit_time_debug)
    print("finish processing digi to feature")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--preSelectPath", dest="preSelect_path", help="pre selection data file path")
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)
    parser.add_argument("--tree-name", default="cbmsim", help="Input event-tree name (default: cbmsim)")
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default=None,
        help=(
            "Selection source. Omit to preserve historical behavior: "
            "passing_muon_real_data uses passing_muon_direct and other types "
            "use external_cutflow."
        ),
    )
    parser.add_argument(
        "--drop-real-has-veto-has-us",
        dest="drop_real_has_veto_has_us",
        action="store_true",
        help="For real data, skip events with both valid veto and upstream MuFilter hits.",
    )
    parser.add_argument(
        "--extract-reco-tracks",
        action="store_true",
        help=(
            "Convert Reco_MuonTracks into flat per-track vectors, per-type counts, "
            "and deterministic best-track scalar features."
        ),
    )
    parser.add_argument(
        "--reco-track-schema",
        choices=RECO_TRACK_SCHEMAS,
        default=FLAT_V1,
        help=(
            "Reconstructed-track output schema used with --extract-reco-tracks. "
            "flat_v1 is the unchanged default; flat_v2 adds fitted measurement "
            "points, detector IDs, and measured-plane summaries."
        ),
    )

    args = parser.parse_args()

    main(args)
    
# python digi_2_features.py -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/preSelect_MC_neutrino_volTarget_100fb-1_0.root -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_20240126_digCPP.root -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root -o /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/vetoTagged_feature_MC_neutrino_volTarget_100fb-1_0.root -t MC_neutrino

# python digi_2_features.py -p /eos/experiment/sndlhc/users/zhibin/MC_muon/down/scoring_1.8_Bfield_4xstat/preSelect_MC_muon_down_scoring_1.8_Bfield_4xstat_3.root -d /eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield_4xstat/sndLHC.Ntuple-TGeant4-160urad_magfield_2022TCL6_muons_rock_2e8pr_Trks.root -g /eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield_4xstat/geofile_full.Ntuple-TGeant4.root -o /eos/experiment/sndlhc/users/zhibin/MC_muon/down/scoring_1.8_Bfield_4xstat/vetoTagged_feature_MC_muon_down_scoring_1.8_Bfield_4xstat_3.root -t MC_muon
# python digi_2_features.py -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/2024_vm/10/nueAnalysisFilter_MC_neutrino_2024_vm_10.root -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/nu14/volume_volTarget/10/sndLHC.Genie-TGeant4_dig.root -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/nu14/volume_volTarget/10/geofile_full.Genie-TGeant4.root -o /eos/experiment/sndlhc/users/zhibin/MC_neutrino/2024_vm/10/feature_MC_neutrino_2024_vm_10.root -t MC_ve
