# Run-8329 and pooled-MC detector-selection study V6

V6 is a configuration-driven extension of the validated V5 detector study.
It reads the existing Run-8329 RealDataDQ partitions and all 15 existing
upward-muon MC feature files. It does not regenerate detector features and it
does not reconstruct MC tracks.

The four MC productions are checked independently and then added without
luminosity weights. Consequently, pooled MC is a diagnostic shape comparison,
not a predicted event rate.

## Coverage definitions

Both working points require all three Veto stations, X and Y activity in all
five SciFi stations, all five US stations, and all four DS stations.

`coverage_tight` preserves the V5 multiplicity definition:

- fewer than 10 combined X+Y hits in every SciFi station;
- fewer than 2 hits in every Veto station;
- fewer than 2 hits in every US station.

`coverage_loose` requires:

- fewer than 20 combined X+Y hits in every SciFi station;
- at most 2 hits in every Veto station and at most 4 Veto hits in total;
- at most 2 hits in every US station and at most 6 US hits in total.

The histogram builder checks for both data and every MC production that no
event can pass tight while failing loose.

## Features and PDFs

The 17 station counts (`count_veto1..3`, `count_scifi1..5`, `count_us1..5`,
and `count_ds1..4`) are normal configured features. They use the same booking,
overflow folding, normalization, metadata, and plotting code as the existing
V5 features.

V6 produces only two reports:

- `coverage_loose.pdf`;
- `coverage_tight.pdf`.

Every feature page contains Run-8329 candidate, Run-8329 topology failure,
pooled-MC candidate, and pooled-MC topology failure. The ratio panel compares
MC with data separately for the candidate and topology-failure populations.
There are no production-specific MC PDFs and no separate MC-summary PDF.

## Workflow

The complete DAG contains five jobs:

1. validate and standardize metadata for the 15 existing MC files;
2. build one V5-compatible combined histogram file;
3. render the loose PDF;
4. render the tight PDF;
5. verify the three outputs through the aggregate target.

Inspect the DAG from the repository root:

```bash
snakemake passing_muon_run8329_binary_region_study_v6 \
  --snakefile snakemake/snakefile --dry-run
```

Submit from `snakemake/`:

```bash
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_binary_region_study_v6
```

Final outputs:

- `/eos/experiment/sndlhc/users/zhibin/snd-ml/evaluation_region_partitions/passing_muon_run8329_detector_selection_study_v6/histograms/detector_selection_combined.root`;
- `evaluation_region_partitions/passing_muon_run8329_detector_selection_study_v6/coverage_loose.pdf`;
- `evaluation_region_partitions/passing_muon_run8329_detector_selection_study_v6/coverage_tight.pdf`.
