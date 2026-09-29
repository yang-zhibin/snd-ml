# Data-Driven 3D-CNN Dataset Test Report

Tested on 2026-09-15. This is a focused implementation and real-input pilot
test; the complete 18,628-source production was not launched.

## Result

The preparation workflow now passes its automated tests, Snakemake parsing,
real feature/Hit3D identity checks, V7 selection-count comparisons, catalog
construction, five-fold assignment, and density-index closure.

Four implementation bugs were found and corrected:

1. ROOT implicit multithreading could reorder independent feature and Hit3D
   snapshots. The splitter is now single-threaded and performs a complete
   post-write event-identity and original-entry check.
2. The run-8329 sampling pilot incorrectly expected run 8329 in the normal
   2024 skim audit. It now reads the completed V7 run-8329 and pooled-MC
   histogram metadata.
3. MC-muon production labels were taken from the generic
   `production_profile` field. They now use the configured `partition`
   column, retaining `scoring_1.8` and `scoring_at449m`.
4. The runtime density sampler scanned every dataset file for every event
   draw. It now builds cumulative event pools once and selects a file in
   logarithmic time. Density indices are stored as 32-bit entry numbers.

## Checks performed

- Source discovery: 33,581 metadata rows; 18,628 have both named feature and
  Hit3D paths present. This is intentionally a fast existence check, not a
  ROOT-tree integrity scan.
- All four currently Hit3D-ready MC-muon sources reproduce the V7 parent and
  loose-pass counts exactly.
- Real data source `src_ce22bbfdbab7f68e8fec` (run 8285): 9,004 input,
  1,155 passing-muon, 7,849 muon-DIS, with zero identity mismatches.
- MC source `src_12f4fa7a3415006089a3`: 243,193 input, 24,130 V7 parent,
  935 passing-muon, 23,195 muon-DIS.
- Real density-index closure:
  `2900 + 2785 + 841 + 1165 + 158 = 7849`.
- MC density-index closure:
  `22098 + 525 + 184 + 299 + 89 = 23195`.
- A real ten-source pilot produced 16 logical dataset rows and five
  group-exclusive folds. Available classes were CC electron neutrino, CC muon
  neutrino, NC, neutral kaon, neutron, passing-muon, and muon-DIS.
  CC tau neutrino remains optional because no tau-neutrino source is configured.
- Automated tests include selection equivalence, bin boundaries, malformed
  inputs, fold leakage, luminosity deduplication, natural validation,
  deterministic uniform training sampling, and validator failure injection.
- Dry-runs succeeded for source discovery and the V7 density-pilot target.

## Sampling warning

Uniform density-bin training is technically implemented as requested, but the
V7 pilot shows strong repetition for rare MC bins. The relative sampling
multiplier reaches about 75 for the MC `density_ge5000` bin. This is a physics
and overfitting risk rather than a software failure. Before final training,
consider capping repetition, merging sparse bins, or using the planned
soft-balanced alternative. Validation remains completely natural.

Some catalogued files exist but contain zero tree entries. They safely produce
no logical dataset row, but submitting them wastes processing time. A full
integrity pre-scan can remove them later if that cost becomes important.
