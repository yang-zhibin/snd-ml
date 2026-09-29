# Run-8329 passing-muon detector-selection study V7

V7 keeps the V6 detector-only Run-8329 versus pooled-MC comparison unchanged
and adds a separate reconstructed-track diagnostic result. The track result is
separate because the configured MC feature files do not contain reconstructed
tracks. Missing MC tracks are never represented by zeros.

## Input population

The track input is the `flat_v2` PassingMuonReco production for run 8329. For
each original source file, all track-feature chunks are concatenated and
intersected by exact `(runId, eventId)` keys with that source's real-data DQ
feature file. The filter snapshots every input branch. A manifest validator
then checks the source coverage, event counts, DQ acceptance, schema name, and
all required `flat_v2` scalar and vector branches before histogramming.

ROOT writes a branchless tree when an RDataFrame snapshot selection contains
zero events. Such a file is accepted only when its entry count and DQ audit are
both exactly zero, and it is omitted from the histogram `TChain`. Every
non-empty input must still contain the complete `flat_v2` schema.

The existing V2 real-data DQ feature files are immutable reference products in
this workflow. They are passed as filter parameters rather than Snakemake
inputs, preventing v7 from trying to rebuild protected historical DQ files when
converter source timestamps change. A missing or unreadable reference still
causes the filtering job to fail immediately when ROOT opens it.

The detector candidate and complement use exactly the V6 `coverage_loose` and
`coverage_tight` expressions. Track quantities are diagnostics only; they do
not change either detector selection.

## Reconstructed-track plots

The reconstructed-track report is split by reconstruction method. For each
detector selection, the Simple PDF has distinct SciFi and downstream sections;
the Hough PDF has distinct SciFi, downstream, and Combined-Hough sections.
Each section contains the following best-track distributions:

- first and last fitted-point `z`;
- first and last fitted-point `(x,y)`;
- signed endpoint `delta z` and 3D endpoint separation;
- associated measurement count and unique measured-plane count;
- `chi2/NDF`, NDF, `dx/dz`, `dy/dz`, and the 2D slope distribution;
- measurements versus measured planes;
- `chi2/NDF` versus measurements;
- first fitted-point `z` versus measurements.

Each track-property histogram is conditional on `<track type>_has_best == 1`.
Each method PDF starts with event-level track counts computed only from that
method's configured track types. The output also records `has_any`,
`has_converged`, and `has_best` counts and
fractions separately for detector candidates and topology failures. This
prevents absent or failed tracks from becoming underflow or zero-valued physics
entries.

`n_measurements` means associated GenFit measurements. These correspond to
hits for Hough tracks but can correspond to clusters for Simple tracks.
`n_unique_planes` counts detector planes represented by associated
measurements; it is not the number of detector planes geometrically crossed by
the fitted line.

## Outputs

The workflow writes:

- the unchanged detector/MC histogram ROOT file and two detector PDFs;
- `coverage_loose_counts.pdf` and `coverage_tight_counts.pdf`, containing raw
  event-count and detector-family N-minus-one diagnostics;
- `coverage_loose_stacked_counts.pdf` and `coverage_tight_stacked_counts.pdf`,
  containing all detector-study features as raw coverage-pass/fail stacks;
- `reco_track_diagnostics_flat_v2.root`;
- `coverage_loose_reco_simple.pdf` and `coverage_loose_reco_hough.pdf`;
- `coverage_tight_reco_simple.pdf` and `coverage_tight_reco_hough.pdf`;
- the validated flat-v2 DQ-intersection manifest and one audit CSV per source.

Per-source audit CSVs are stored under the local v7 evaluation directory,
outside `snakemake/metadata/updated/`. That metadata directory is reserved for
the repository's generic metadata-update rule and cannot safely contain
wildcard outputs from this workflow.

The count-enabled detector histogram uses the separately named
`mc_muon_up_metadata_v7_counts.csv` and `mc_muon_up_validation_v7_counts.csv`
reference products. The earlier protected V7 validation products are retained
unchanged.

All published products use the
`passing_muon_run8329_detector_selection_study_v7` output directory. V6 paths
and files are not modified.

The count reports do not normalize histograms to unit area. Their detailed
pages remove one detector family's cuts while retaining every other cut, then
stack the events that pass and fail the removed requirements. Consequently,
the stack closes to the inclusive N-minus-one parent population bin by bin.
This avoids using a Veto requirement to pre-sculpt the Veto distribution being
tested. The reports include station counts, detector totals, active Veto-plane
count, and Veto measurements beyond the first measurement in each active
plane. No Gaussian or Poisson fit is imposed by this diagnostic stage.

The separate `*_stacked_counts.pdf` reports reuse every feature, binning, and
feature-validity requirement from the detector comparison PDFs. For each 1D
feature, coverage-pass and coverage-fail data are stacked and their inclusive
sum is overlaid; the lower panel shows their bin-by-bin fractions. The 2D
features are shown as pass, fail, and inclusive raw-count panels. These reports
use linear count scales and do not apply a reconstructed-track requirement.

## Run

From `snakemake/`:

```bash
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_detector_selection_study_v7
```

The aggregate intentionally includes the flat-v2 conversion and DQ filtering
when those products do not already exist.

To produce only the two raw-count PDFs and their detector histogram dependency:

```bash
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_binary_region_study_count_plots_v7
```

To produce the two all-feature coverage pass/fail stacked-count PDFs from the
existing detector histogram:

```bash
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_coverage_stacked_count_plots_v7
```

## Data-only loose event displays

The optional data-only event-display target selects 19 Run-8329 categories:
the `coverage_loose` candidate, its failures with total Veto count below three,
equal to three, or above three, and five sets of matching failure subsets. The
additional sets require `count_scifi > 100` or `density_sndsw_scifi` greater
than 100, 500, 1000, or 5000. Each additional criterion has its own output
folder containing the three Veto-partition PDFs. The selector reserves 50
events per category by deterministic round-robin source sampling and renders
up to 25 successful SNDSW event pages per category. The SciFi subsets
intentionally overlap their base failure categories, while identities remain
unique within each category. Tracks are disabled; the display shows detector
geometry and QDC-coloured hits. A category with no displayable events receives
one audited `No events found` page. Page-to-event summary CSVs are merged only
after all 19 categories have a sequential event or placeholder page set.

This target is intentionally excluded from the main V7 aggregate:

```bash
snakemake --profile .config/snakemake/htcondor --jobs 20 \
  passing_muon_run8329_loose_event_displays_data_v7
```

The MC-only companion target uses the same 19 categories for the pooled
upward-muon MC productions and writes to `event_display_mc_loose_coverage/`.
It applies the configured muon truth/activity baseline before the detector
selection and preserves the feature-tree `eventIndex` mapping back to each
production's original DIGI and geometry files:

```bash
snakemake --profile .config/snakemake/htcondor --jobs 20 \
  passing_muon_up_mc_loose_event_displays_v7
```

## SciFi-density event-display category distributions

The full-population distribution workflow corresponds to the four cumulative
SciFi-density failure subsets used by the loose event displays:

- `scifi_density_gt100`;
- `scifi_density_gt500`;
- `scifi_density_gt1000`;
- `scifi_density_gt5000`.

For each threshold, the Run-8329 and pooled upward-muon MC population is
selected with the same `coverage_loose` failure and MC baseline expressions as
the event-display workflow. The Veto `<3`, `=3`, and `>3` categories are united
into one population, with an explicit count-closure check against the three
subgroups. The PDFs use the same 46 ordered features, binning, feature-validity
cuts, overflow handling, and configured log axes as `coverage_loose.pdf`.
Each PDF overlays independently normalized Run-8329 and pooled-MC shapes; its
summary page retains the absolute selected counts and per-production MC counts.

The histograms use every event in the input feature trees. They do not use the
source-balanced event-display reserve manifest or depend on successful SNDSW
rendering. The output files are:

```text
histograms/loose_event_display_scifi_density_categories.root
event_display_category_distributions/scifi_density_gt100.pdf
event_display_category_distributions/scifi_density_gt500.pdf
event_display_category_distributions/scifi_density_gt1000.pdf
event_display_category_distributions/scifi_density_gt5000.pdf
```

Run the isolated workflow from `snakemake/` with:

```bash
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_loose_event_display_scifi_density_distributions_v7
```

This target is excluded from the main V7 aggregate and from both event-display
rendering targets, so it cannot modify or launch those existing products.
