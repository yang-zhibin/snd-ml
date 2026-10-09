#!/usr/bin/env python3
"""Focused unit tests for the frozen 2024 real-data selection policy."""

from __future__ import annotations

import unittest

from real_data_2024_selection import (
    deterministic_sample_bucket,
    event_header_identity,
    has_valid_veto_hit,
    keep_veto_10pct,
)


class FakeHit:
    def __init__(self, system: int, valid: bool = True):
        self.system = system
        self.valid = valid

    def isValid(self):
        return self.valid

    def GetSystem(self):
        return self.system


class FakeHeader:
    def GetRunId(self):
        return 8285

    def GetEventNumber(self):
        return 123456


class RealDataSelectionPolicyTest(unittest.TestCase):
    def test_valid_veto_hit_definition(self):
        self.assertFalse(has_valid_veto_hit([]))
        self.assertFalse(has_valid_veto_hit([FakeHit(1, valid=False), FakeHit(2)]))
        self.assertTrue(has_valid_veto_hit([FakeHit(2), FakeHit(1)]))

    def test_event_header_identity(self):
        self.assertEqual(event_header_identity(FakeHeader(), 17), (8285, 123456, 17))

    def test_sampling_fixed_vectors(self):
        vectors = {
            (8285, 123456, 17): 8,
            (8285, 123457, 18): 9,
            (9000, 1, 0): 0,
            (9000, 2, 1): 0,
            (0, 0, 0): 8,
        }
        for identity, expected in vectors.items():
            self.assertEqual(deterministic_sample_bucket(*identity), expected)
            self.assertEqual(keep_veto_10pct(*identity), expected == 0)

    def test_sampling_denominator_validation(self):
        with self.assertRaisesRegex(ValueError, "denominator"):
            deterministic_sample_bucket(1, 2, 3, denominator=0)


if __name__ == "__main__":
    unittest.main()
