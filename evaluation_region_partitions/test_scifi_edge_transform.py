#!/usr/bin/env python3
"""Synthetic tests for the SciFi edge-transform arithmetic."""

from __future__ import annotations

import math

from scifi_edge_transform import (
    ActiveInterval,
    HitCoordinate,
    PlaneGeometry,
    PlaneKey,
    build_plane_transforms,
    choose_locked_side,
    classify_transformed_coordinate,
    fit_linear_projection,
    transform_hits,
    validate_transform,
)


def plane(station, view, low=0.0, high=10.0, z=None):
    # A synthetic central gap from 4 to 6 cm.
    return PlaneGeometry(
        PlaneKey(station, view), float(z if z is not None else station * 10),
        low, high,
        (
            ActiveInterval(low, 4.0, 0, 0, 0),
            ActiveInterval(6.0, high, 0, 0, 1),
        ),
    )


def test_linear_fit():
    fit = fit_linear_projection([(0, 1), (10, 3), (20, 5)], "x")
    assert math.isclose(fit.slope, 0.2)
    assert math.isclose(fit.intercept, 1.0)
    assert math.isclose(fit.at(15), 4.0)
    assert fit.max_abs_residual_cm < 1e-12
    try:
        fit_linear_projection([(1, 2), (1, 3)], "y")
    except ValueError as error:
        assert "distinct z" in str(error)
    else:
        raise AssertionError("Degenerate projection was accepted")


def test_locked_side_and_plane_dependent_shift():
    planes = [plane(1, 1, z=10), plane(2, 1, z=20), plane(3, 1, z=30)]
    # The last plane is locally nearer the high side, but the event is nearer
    # the low side in total and therefore remains locked low.
    positions = {
        planes[0].key: 1.0,
        planes[1].key: 2.0,
        planes[2].key: 9.0,
    }
    assert choose_locked_side(planes, positions) == "low"
    transforms = build_plane_transforms(planes, positions, inward_margin_cm=0.25)
    assert {item.locked_side for item in transforms.values()} == {"low"}
    assert transforms[planes[2].key].plane_local_closest_side == "high"
    assert math.isclose(transforms[planes[0].key].delta_cm, -0.75)
    assert math.isclose(transforms[planes[2].key].delta_cm, -8.75)


def test_high_edge_and_residual_closure():
    geometry = plane(1, 0)
    transforms = build_plane_transforms(
        [geometry], {geometry.key: 8.0}, inward_margin_cm=0.5,
    )
    transform = transforms[geometry.key]
    assert transform.locked_side == "high"
    assert math.isclose(transform.target_cm, 9.5)
    assert math.isclose(transform.delta_cm, 1.5)
    hits = [
        HitCoordinate(geometry.key, 7.5, 1),
        HitCoordinate(geometry.key, 8.0, 2),
        HitCoordinate(geometry.key, 8.5, 3),
    ]
    converted = transform_hits(hits, {geometry.key: geometry}, transforms)
    audit = validate_transform(hits, converted, transforms)
    assert audit["passed"], audit["failures"]
    assert audit["n_original_hits"] == audit["n_transformed_hits"] == 3
    assert audit["max_residual_difference_cm"] < 1e-12


def test_active_gap_outside_classification():
    geometry = plane(1, 1)
    assert classify_transformed_coordinate(2.0, geometry)[0] == "inside_active"
    assert classify_transformed_coordinate(5.0, geometry) == ("inside_gap", None)
    assert classify_transformed_coordinate(-0.1, geometry) == ("outside_geometry", None)
    assert classify_transformed_coordinate(10.1, geometry) == ("outside_geometry", None)


def test_unlocked_mode_and_margin_validation():
    planes = [plane(1, 0), plane(2, 0)]
    positions = {planes[0].key: 1.0, planes[1].key: 9.0}
    transforms = build_plane_transforms(
        planes, positions, lock_side_per_orientation=False,
    )
    assert transforms[planes[0].key].locked_side == "low"
    assert transforms[planes[1].key].locked_side == "high"
    try:
        build_plane_transforms(planes, positions, inward_margin_cm=11.0)
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("Invalid inward margin was accepted")


def main():
    tests = [
        test_linear_fit,
        test_locked_side_and_plane_dependent_shift,
        test_high_edge_and_residual_closure,
        test_active_gap_outside_classification,
        test_unlocked_mode_and_margin_validation,
    ]
    for test in tests:
        test()
    print(f"SciFi edge-transform synthetic tests: OK ({len(tests)} tests)")


if __name__ == "__main__":
    main()
