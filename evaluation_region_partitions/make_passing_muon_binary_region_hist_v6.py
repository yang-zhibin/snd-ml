#!/usr/bin/env python3
"""Build V6 with the validated V5 detector-only histogram engine."""

# V6 intentionally shares the generic event-loop and ROOT schema with V5.
# Its behavior is selected entirely by the V6 configuration: two nested
# coverage working points, all 15 pooled-MC inputs, and station-count features.
from make_passing_muon_binary_region_hist_v5 import main


if __name__ == "__main__":
    main()
