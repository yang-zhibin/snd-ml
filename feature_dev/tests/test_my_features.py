"""Tests for cluster density features."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algorithms'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'convertData'))

from my_features import view_density


def test_view_density_basic():
    """Example from SCIFI_DENSITY.md: positions [0.0, 0.4, 1.2] -> 4."""
    assert view_density([0.0, 0.4, 1.2]) == 4


def test_view_density_empty():
    """Empty list -> 0."""
    assert view_density([]) == 0


def test_view_density_single():
    """Single hit -> 0, no pairs possible."""
    assert view_density([0.5]) == 0


def test_view_density_outside_window():
    """Two hits more than 1 cm apart -> 0."""
    assert view_density([0.0, 2.0]) == 0


def test_view_density_exactly_1cm():
    """Two hits exactly 1 cm apart -> 2 (inclusive boundary)."""
    assert view_density([0.0, 1.0]) == 2


def test_view_density_all_close():
    """Three hits all within 1 cm -> N*(N-1) = 6."""
    assert view_density([0.0, 0.3, 0.6]) == 6


if __name__ == "__main__":
    # Esegui tutti i test manualmente
    tests = [
        test_view_density_basic,
        test_view_density_empty,
        test_view_density_single,
        test_view_density_outside_window,
        test_view_density_exactly_1cm,
        test_view_density_all_close,
    ]
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
