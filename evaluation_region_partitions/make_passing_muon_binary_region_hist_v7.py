#!/usr/bin/env python3
"""Build the V7 detector-only Run-8329/pooled-MC comparison histograms.

V7 intentionally preserves the validated V6 detector histogram definition.
Its flat-v2 reconstructed-track diagnostics are built separately by
``make_passing_muon_reco_diagnostics_v7.py`` because those tracks exist only
for the Run-8329 PassingMuonReco sample, not for the configured MC samples.
"""

from make_passing_muon_binary_region_hist_v5 import main


if __name__ == "__main__":
    main()
