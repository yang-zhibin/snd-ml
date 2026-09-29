#!/usr/bin/env python3
"""Synthetic tests for detector-ID channel-sequence remapping."""

from __future__ import annotations

import math

from detid_channel_shift import (
    ChannelAddress,
    ChannelLayout,
    PlaneKey,
    TrackLine,
    build_all_plane_activity_line,
    build_five_station_activity_line,
    build_two_station_activity_line,
    choose_anchor,
    choose_target_edge,
    choose_translation_offset,
    choose_transform_axis,
    map_detector_ids,
    summarize_mapping,
)


KEY = PlaneKey("Scifi", 0, 1, 1)
DS_KEY = PlaneKey("MuFilter", 3, 0, 1)


def layout():
    # Deliberately non-monotonic detector IDs and two physical gaps.
    centres = [0.0, 1.0, 2.0, 4.0, 5.0, 8.0, 9.0]
    det_ids = [50, 10, 70, 20, 90, 30, 60]
    return ChannelLayout.ordered(KEY, [
        ChannelAddress(det_id, KEY, -1, centre, centre - 0.4, centre + 0.4, (index,))
        for index, (det_id, centre) in enumerate(zip(det_ids, centres))
    ])


def uniform_ds_layout():
    return ChannelLayout.ordered(DS_KEY, [
        ChannelAddress(
            30060 + index, DS_KEY, -1, -58.0 + 1.01 * index,
            -58.5 + 1.01 * index, -57.5 + 1.01 * index, (index,),
        )
        for index in range(60)
    ])


def test_geometry_order_not_detid_order():
    value = layout()
    assert [item.det_id for item in value.channels] == [50, 10, 70, 20, 90, 30, 60]
    assert value.by_det_id[20].ordinal == 3


def test_low_edge_sparse_mapping():
    value = layout()
    shift = choose_anchor(value, [70, 20, 60], "low", -2.1)
    assert shift.anchor_det_id == 70
    assert shift.offset == -2
    mapped = map_detector_ids(value, [70, 20, 60], shift)
    assert [item.destination_det_id for item in mapped] == [50, 10, 90]
    # Source ordinal separation 6-2=4 is retained at the destination.
    assert mapped[2].destination_ordinal - mapped[0].destination_ordinal == 4


def test_high_edge_and_overflow():
    value = layout()
    shift = choose_anchor(value, [20, 90], "high", 4.2)
    assert shift.anchor_det_id == 90
    assert shift.offset == 2
    mapped = map_detector_ids(value, [20, 90, 30], shift)
    assert mapped[0].destination_det_id == 30
    assert mapped[1].destination_det_id == 60
    assert mapped[2].status == "outside_high"
    assert summarize_mapping(mapped)["count_closure"]


def test_gap_changes_realized_shift():
    value = layout()
    shift = choose_anchor(value, [70], "low", -2.0)
    mapped = map_detector_ids(value, [70, 20, 90], shift)
    shifts = [item.realized_shift_cm for item in mapped]
    assert shifts == [-2.0, -3.0, -3.0]
    assert math.isclose(summarize_mapping(mapped)["realized_shift_spread_cm"], 1.0)


def test_deterministic_anchor_tie():
    value = layout()
    # Channels at 1 and 2 are equally close after this requested shift.
    shift = choose_anchor(value, [10, 70], "low", -1.5)
    assert shift.anchor_det_id == 10


def test_translation_offset_approximates_requested_distance_without_forcing_edge():
    value = uniform_ds_layout()
    source_ids = [30100, 30102, 30103]
    shift = choose_translation_offset(value, source_ids, -28.51530289516636)
    assert shift.selection_policy == "closest_physical_translation"
    assert shift.offset == -28
    assert shift.reference_destination_det_id != value.channels[0].det_id
    assert abs(shift.translation_error_cm) <= shift.quantization_tolerance_cm
    mapped = map_detector_ids(value, source_ids, shift)
    assert all(item.status == "mapped" for item in mapped)
    assert all(math.isclose(item.realized_shift_cm, -28.28) for item in mapped)


def test_translation_offset_keeps_one_plane_offset_and_drops_without_clamping():
    value = uniform_ds_layout()
    shift = choose_translation_offset(value, [30065, 30100], -28.51530289516636)
    assert shift.offset == -28
    mapped = map_detector_ids(value, [30065, 30100], shift)
    assert mapped[0].status == "outside_low"
    assert mapped[1].status == "mapped"
    assert shift.mapped_hits == 1
    assert shift.dropped_hits == 1


def test_translation_offset_handles_gaps_with_constant_sequence_offset():
    value = layout()
    shift = choose_translation_offset(value, [20, 90, 30], -3.2)
    mapped = map_detector_ids(value, [20, 90, 30], shift)
    assert shift.offset < 0
    retained = [item for item in mapped if item.destination_det_id is not None]
    assert all(
        item.destination_ordinal - item.source_ordinal == shift.offset
        for item in retained
    )


def test_subchannel_translation_selects_zero_offset():
    value = uniform_ds_layout()
    shift = choose_translation_offset(value, [30065, 30100], -0.0143)
    assert shift.offset == 0
    assert math.isclose(shift.realized_shift_mean_cm, 0.0)
    assert abs(shift.translation_error_cm) <= shift.quantization_tolerance_cm


def test_layout_translation_drops_sparse_edge_hit_instead_of_reducing_shift():
    key = PlaneKey("MuFilter", 1, 1, 0)
    value = ChannelLayout.ordered(key, [
        ChannelAddress(
            11000 + index, key, -1, 6.0 * index,
            6.0 * index - 3.0, 6.0 * index + 3.0, (index,),
        )
        for index in range(7)
    ])
    shift = choose_translation_offset(value, [11001], 33.67)
    assert shift.offset == 6
    assert shift.mapped_hits == 0
    assert shift.dropped_hits == 1
    mapped = map_detector_ids(value, [11001], shift)
    assert mapped[0].status == "outside_high"


def test_missing_anchor_and_no_wrap():
    value = layout()
    try:
        choose_anchor(value, [], "low", -1.0)
    except ValueError as error:
        assert "No anchor" in str(error)
    else:
        raise AssertionError("Missing anchor candidates were accepted")
    shift = choose_anchor(value, [20], "low", -4.0)
    mapped = map_detector_ids(value, [50, 10, 20], shift)
    assert [item.status for item in mapped] == ["outside_low", "outside_low", "mapped"]


def test_duplicate_source_rejected():
    value = layout()
    shift = choose_anchor(value, [20], "low", -4.0)
    try:
        map_detector_ids(value, [20, 20], shift)
    except ValueError as error:
        assert "Duplicate source" in str(error)
    else:
        raise AssertionError("Duplicate source detector IDs were accepted")


def test_axis_policy_x_low_and_y_high():
    x_choice = choose_transform_axis(0.02, -0.005, 1, "track_pz", 0.0, 1.2)
    assert x_choice.selected_axis == "x"
    assert x_choice.selected_view == 1
    assert x_choice.entry_side == "low"
    assert math.isclose(x_choice.axis_dominance_ratio, 4.0)
    y_choice = choose_transform_axis(0.002, -0.01, 1, "track_pz", 0.0, 1.2)
    assert y_choice.selected_axis == "y"
    assert y_choice.selected_view == 0
    assert y_choice.entry_side == "high"


def test_axis_policy_reverses_side_with_pz():
    downstream = choose_transform_axis(0.02, 0.001, 1, "track_pz")
    upstream = choose_transform_axis(0.02, 0.001, -1, "track_pz")
    assert downstream.entry_side == "low"
    assert upstream.entry_side == "high"
    assert downstream.travel_slope == -upstream.travel_slope


def test_axis_policy_rejects_ambiguous_and_small_slopes():
    cases = [
        (0.01, -0.01, 0.0, 1.0, "equal absolute slopes"),
        (0.011, 0.01, 0.0, 1.2, "dominance"),
        (0.001, 0.0001, 0.002, 1.2, "not above"),
    ]
    for sx, sy, minimum, ratio, message in cases:
        try:
            choose_transform_axis(sx, sy, 1, "assumed_positive_z", minimum, ratio)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"Invalid slopes {sx}/{sy} were accepted")


def test_track_line_projection():
    line = TrackLine(100.0, 2.0, -3.0, 0.1, -0.2)
    assert math.isclose(line.coordinate_at("x", 110.0), 3.0)
    assert math.isclose(line.coordinate_at("y", 110.0), -5.0)


def test_two_station_activity_line_uses_unweighted_station_means():
    coordinates = {
        (1, 1): [-42.0, -38.0],
        (1, 0): [28.0, 32.0, 36.0],
        (2, 1): [-34.0, -30.0],
        (2, 0): [24.0, 28.0],
        (3, 1): [10.0],
        (3, 0): [10.0],
    }
    z_values = {
        (1, 1): 300.0, (1, 0): 301.0,
        (2, 1): 320.0, (2, 0): 321.0,
        (3, 1): 340.0, (3, 0): 341.0,
    }
    result = build_two_station_activity_line(coordinates, z_values, 1)
    assert result.first.station == 1 and result.second.station == 2
    assert result.first.n_x_hits == 2 and result.first.n_y_hits == 3
    assert math.isclose(result.first.x_cm, -40.0)
    assert math.isclose(result.first.y_cm, 32.0)
    assert math.isclose(result.line.slope_xz, 0.4)
    assert math.isclose(result.line.slope_yz, -0.3)
    assert math.isclose(result.line.coordinate_at("x", 300.0), -40.0)
    assert math.isclose(result.line.coordinate_at("y", 301.0), 32.0)


def test_two_station_activity_line_respects_travel_order():
    coordinates = {
        (1, 1): [1.0], (1, 0): [2.0],
        (4, 1): [4.0], (4, 0): [5.0],
        (5, 1): [6.0], (5, 0): [8.0],
    }
    z_values = {
        (1, 1): 100.0, (1, 0): 101.0,
        (4, 1): 400.0, (4, 0): 401.0,
        (5, 1): 500.0, (5, 0): 501.0,
    }
    result = build_two_station_activity_line(coordinates, z_values, -1)
    assert result.direction == "negative_z"
    assert result.first.station == 5 and result.second.station == 4


def test_two_station_activity_line_requires_both_views_in_two_stations():
    coordinates = {
        (1, 1): [1.0], (1, 0): [2.0],
        (2, 1): [3.0],
    }
    z_values = {(1, 1): 100.0, (1, 0): 101.0, (2, 1): 200.0}
    try:
        build_two_station_activity_line(coordinates, z_values, 1)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("Incomplete two-station activity input was accepted")


def test_all_plane_activity_line_fits_every_plane_average():
    coordinates = {
        (1, 1): [-1.0, 1.0],
        (2, 1): [99.0, 101.0],
        (3, 1): [2.0],
        (1, 0): [5.0],
        (2, 0): [3.0],
        (3, 0): [1.0],
    }
    z_values = {
        (1, 1): 0.0, (2, 1): 1.0, (3, 1): 2.0,
        (1, 0): 0.0, (2, 0): 1.0, (3, 0): 2.0,
    }
    result = build_all_plane_activity_line(coordinates, z_values, 1)
    assert [point.station for point in result.x_points] == [1, 2, 3]
    assert [point.n_hits for point in result.x_points] == [2, 2, 1]
    # The middle-plane outlier changes the intercept but not the OLS slope.
    # A first-two-plane construction would instead have slope 100.
    assert math.isclose(result.line.slope_xz, 1.0)
    assert math.isclose(result.line.slope_yz, -2.0)
    assert math.isclose(result.line.reference_z_cm, 1.0)
    assert math.isclose(result.line.reference_x_cm, 34.0)
    assert math.isclose(result.line.reference_y_cm, 3.0)


def test_all_plane_activity_line_allows_different_active_planes_by_projection():
    coordinates = {
        (1, 1): [1.0], (3, 1): [5.0], (5, 1): [9.0],
        (2, 0): [8.0], (4, 0): [4.0],
    }
    z_values = {
        (1, 1): 10.0, (3, 1): 30.0, (5, 1): 50.0,
        (2, 0): 20.0, (4, 0): 40.0,
    }
    result = build_all_plane_activity_line(coordinates, z_values, -1)
    assert result.direction == "negative_z"
    assert [point.station for point in result.x_points] == [5, 3, 1]
    assert [point.station for point in result.y_points] == [4, 2]
    assert math.isclose(result.line.slope_xz, 0.2)
    assert math.isclose(result.line.slope_yz, -0.2)


def test_all_plane_activity_line_requires_two_planes_in_each_projection():
    coordinates = {
        (1, 1): [1.0], (2, 1): [2.0],
        (1, 0): [3.0],
    }
    z_values = {(1, 1): 10.0, (2, 1): 20.0, (1, 0): 11.0}
    try:
        build_all_plane_activity_line(coordinates, z_values, 1)
    except ValueError as error:
        assert "at least two active y-projection planes" in str(error)
    else:
        raise AssertionError("One-plane activity projection was accepted")


def test_five_station_activity_line_requires_complete_xy_coverage():
    coordinates = {
        (station, view): [10.0 * view + station]
        for station in range(1, 6) for view in (0, 1)
    }
    z_values = {
        (station, view): 100.0 * station + view
        for station in range(1, 6) for view in (0, 1)
    }
    result = build_five_station_activity_line(coordinates, z_values, 1)
    assert len(result.x_points) == 5
    assert len(result.y_points) == 5
    incomplete = dict(coordinates)
    del incomplete[(4, 0)]
    try:
        build_five_station_activity_line(incomplete, z_values, 1)
    except ValueError as error:
        assert "station 4 view 0" in str(error)
    else:
        raise AssertionError("Incomplete five-station activity input was accepted")


def test_five_station_activity_line_equal_weights_station_means():
    coordinates = {}
    z_values = {}
    for station in range(1, 6):
        z_values[(station, 1)] = float(station)
        z_values[(station, 0)] = float(station)
        coordinates[(station, 1)] = [2.0 * station] * (100 if station == 3 else 1)
        coordinates[(station, 0)] = [-3.0 * station]
    result = build_five_station_activity_line(coordinates, z_values, 1)
    assert math.isclose(result.line.slope_xz, 2.0)
    assert math.isclose(result.line.slope_yz, -3.0)


def test_near_low_edge_overrides_opposite_slope_side():
    decision = choose_target_edge(-46.0, -48.0, 48.0, "high", 5.0)
    assert decision.near_edge_override
    assert decision.selection_policy == "nearest_edge_within_threshold"
    assert decision.closest_side == "low"
    assert decision.target_side == "low"


def test_near_high_edge_overrides_opposite_slope_side():
    decision = choose_target_edge(44.0, -48.0, 48.0, "low", 5.0)
    assert decision.near_edge_override
    assert decision.target_side == "high"
    assert math.isclose(decision.closest_edge_distance_cm, 4.0)


def test_near_edge_threshold_is_inclusive():
    decision = choose_target_edge(-43.0, -48.0, 48.0, "high", 5.0)
    assert decision.near_edge_override
    assert decision.target_side == "low"


def test_beyond_near_edge_threshold_keeps_slope_side():
    decision = choose_target_edge(-42.999, -48.0, 48.0, "high", 5.0)
    assert not decision.near_edge_override
    assert decision.selection_policy == "slope_entry_side"
    assert decision.closest_side == "low"
    assert decision.target_side == "high"


def test_near_edge_decision_does_not_select_an_axis():
    axis = choose_transform_axis(0.02, -0.005, 1, "track_pz")
    edge = choose_target_edge(47.0, -48.0, 48.0, axis.entry_side, 5.0)
    assert axis.selected_axis == "x"
    assert edge.target_side == "high"


def test_near_edge_policy_rejects_invalid_inputs():
    cases = [
        (0.0, 1.0, -1.0, "low", 5.0, "below high"),
        (0.0, -1.0, 1.0, "middle", 5.0, "Unsupported"),
        (0.0, -1.0, 1.0, "low", -0.1, "non-negative"),
    ]
    for track, low, high, side, threshold, message in cases:
        try:
            choose_target_edge(track, low, high, side, threshold)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("Invalid v4 edge-decision input was accepted")


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in sorted(tests, key=lambda value: value.__name__):
        test()
    print(f"detid channel-shift synthetic tests: OK ({len(tests)} tests)")


if __name__ == "__main__":
    main()
