#!/usr/bin/env python3
"""Focused tests for the veto-tagged transform-comparison population contract."""

from __future__ import annotations

import ROOT

from make_muon_dis_transform_comparison_hist_shard import (
    COUNTER_BINS,
    COUNTER_LABELS,
    classify_population,
    counter_label,
    validate_counter_closure,
)
from plot_muon_dis_transform_comparison import histogram_path, safe_ratio


def test_population_classification():
    assert classify_population(0, False) == "signal_region"
    assert classify_population(0, True) == "signal_region"
    assert classify_population(1, True) == "passing_muon"
    assert classify_population(5, False) == "muon_dis"
    try:
        classify_population(-1, False)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative veto count was accepted")


def make_counter():
    counter = ROOT.TH1D("test_counts", "", len(COUNTER_LABELS), 0.5, len(COUNTER_LABELS) + 0.5)
    for index, label in enumerate(COUNTER_LABELS, 1):
        counter.GetXaxis().SetBinLabel(index, label)
    values = {
        "all RealDataDQ": 100,
        "no-Veto signal region": 10,
        "passing-muon candidate": 30,
        "passing-muon eligible": 25,
        "passing-muon matched": 23,
        "passing-muon ineligible": 5,
        "passing-muon eligible missing": 2,
        "Muon-DIS candidate": 60,
        "Muon-DIS eligible": 50,
        "Muon-DIS matched": 47,
        "Muon-DIS ineligible": 10,
        "Muon-DIS eligible missing": 3,
    }
    for label, value in values.items():
        counter.SetBinContent(COUNTER_BINS[label], value)
    return counter


def test_counter_closure():
    counter = make_counter()
    validate_counter_closure(counter)
    counter.SetBinContent(COUNTER_BINS["Muon-DIS matched"], 46)
    try:
        validate_counter_closure(counter)
    except RuntimeError as error:
        assert "match-count closure" in str(error)
    else:
        raise AssertionError("Broken match closure was accepted")


def test_histogram_paths():
    assert histogram_path("simple", "passing_muon", "count_veto", "original") == (
        "methods/simple/populations/passing_muon/features/count_veto/original"
    )
    assert histogram_path("reco_scifi", "muon_dis", "count_veto", "signal_region") == (
        "reference/signal_region/features/count_veto/original"
    )
    assert counter_label("passing_muon", "matched") == "passing-muon matched"


def test_safe_ratio():
    denominator = ROOT.TH1D("denominator", "", 2, 0, 2)
    numerator = ROOT.TH1D("numerator", "", 2, 0, 2)
    denominator.SetBinContent(1, 4)
    numerator.SetBinContent(1, 2)
    numerator.SetBinContent(2, 3)
    ratio = safe_ratio(numerator, denominator, "ratio")
    assert ratio.GetBinContent(1) == 0.5
    assert ratio.GetBinContent(2) == 0.0


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in sorted(tests, key=lambda value: value.__name__):
        test()
    print(f"transform-comparison tests: OK ({len(tests)} tests)")


if __name__ == "__main__":
    main()
