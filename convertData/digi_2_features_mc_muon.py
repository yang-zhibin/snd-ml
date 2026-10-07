#!/usr/bin/env python3
"""Run the feature converter with legacy MC-muon header compatibility."""

from __future__ import annotations

from argparse import ArgumentParser

import digi_2_features as converter


def install_legacy_header_compatibility():
    """Skip detector InitEvent only when MC stores a FairEventHeader."""
    original = converter.init_event_geometry
    legacy_header = None

    def compatible_init(snd_geo, event_header):
        nonlocal legacy_header
        if legacy_header is None:
            try:
                class_name = str(event_header.ClassName())
            except Exception:
                class_name = type(event_header).__name__
            legacy_header = class_name == "FairEventHeader"
            if legacy_header:
                print(
                    "Legacy MC_muon FairEventHeader detected: skipping detector "
                    "InitEvent (requires SNDLHCEventHeader)"
                )
        if not legacy_header:
            original(snd_geo, event_header)

    converter.init_event_geometry = compatible_init


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-p", "--preSelectPath", dest="preSelect_path")
    parser.add_argument("-d", "--digiPath", dest="digi_path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", default="RECREATE")
    parser.add_argument("-t", "--type", dest="type", required=True)
    parser.add_argument("--tree-name", default="cbmsim")
    parser.add_argument("--source-domain-id", type=int, default=0)
    parser.add_argument("--source-pdg-code", type=int)
    parser.add_argument("--source-energy-gev", type=float)
    parser.add_argument("--source-uid", default="")
    parser.add_argument("--source-geometry-id", default="")
    parser.add_argument("--source-geometry-checksum", default="")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--selected-entry-start", type=int, default=None)
    parser.add_argument("--selected-entry-stop", type=int, default=None)
    parser.add_argument(
        "--selection-mode", choices=converter.SELECTION_MODES, default=None
    )
    parser.add_argument("--drop-real-has-veto-has-us", action="store_true")
    parser.add_argument("--extract-reco-tracks", action="store_true")
    args = parser.parse_args()
    if str(args.type).strip().lower() != "mc_muon":
        parser.error("this compatibility wrapper is restricted to MC_muon")
    return args


if __name__ == "__main__":
    install_legacy_header_compatibility()
    converter.main(parse_args())
