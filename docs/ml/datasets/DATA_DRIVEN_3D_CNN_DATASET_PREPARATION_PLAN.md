# Data-Driven 3D-Hit CNN: Dataset Preparation Plan

## 1. Freeze the eight physics-class definitions

The CNN classes will be:

1. CC muon neutrino
2. CC electron neutrino
3. CC tau neutrino
4. NC neutrino
5. Neutral kaon
6. Neutron
7. Passing muon
8. Muon-DIS

Store finer information as metadata:

- Neutrinos: flavor, CC/NC interaction, true energy, and production.
- Kaons and neutrons: particle type and generated-energy bin.
- Passing muon and muon-DIS: data/MC domain, run, source file, and selection version.

First confirm that usable Hit3D events exist for every intended class, especially CC tau neutrinos. A class without sufficient events may need to remain reserved until its sample is available.

## 2. Freeze the data passing-muon and muon-DIS definitions

Use the V7 `coverage_loose` detector selection within a common muon-enriched data population:

```text
Muon-enriched parent data
          |
   V7 coverage_loose
       /        \
     pass       fail
      |           |
passing-muon   muon-DIS-like
   proxy          proxy
```

- Passing events are the data representation of passing muons.
- Failing events are the data representation of muon-DIS-like events.
- Apply the identical V7 expression to every data run.
- Keep the two categories mutually exclusive.
- Record them as detector-defined proxy labels rather than particle-truth labels.
- Use reconstructed-track information for validation, not as an unrecorded modification of the detector selection.

The parent muon-enriched requirement must also be fixed. Without it, the loose-selection failure population would contain many non-muon backgrounds.

Relevant existing references:

- `document/PASSING_MUON_RUN8329_DETECTOR_SELECTION_V7.md`
- `snakemake/metadata/configs/region_evaluation/passing_muon_binary_region_study_v7.yaml`
- `evaluation_region_partitions/make_passing_muon_binary_region_hist_v5.py`

## 3. Test and choose the muon-DIS density sampling

Perform a focused pilot study before the complete dataset audit. This requires only a minimal scan of the muon-DIS density distribution, not yet the full class-by-class population audit.

Use:

- Data events failing `coverage_loose` within the muon-enriched parent population.
- Available MC muon-DIS events.
- `density_sndsw_scifi` from the aligned feature data.
- The currently available Hit3D events.

Compare three candidate strategies:

1. **Natural sampling:** preserve the observed SciFi-density distribution.
2. **Uniform density-bin sampling:** give each fixed density interval approximately equal training exposure.
3. **Soft-balanced density sampling:** increase rare-density exposure while capping oversampling or event weights.

Use the same fixed numerical density intervals for data and MC. Choose their boundaries from the pilot distributions and then freeze them. Treat invalid density values as a separate category.

Test each strategy with the same preliminary source-grouped split, the same small baseline CNN or controlled proxy study, multiple random seeds, and an untouched validation sample with its natural density distribution.

Compare:

- Overall muon-DIS performance.
- Performance versus SciFi density.
- Data-versus-MC stability.
- Variation between source groups or folds.
- Rare-event repetition.
- Effective training sample size.

Choose the strategy that gives stable density-dependent performance without allowing a few rare events to dominate.

Freeze the resulting sampling specification:

```text
density variable
density-bin boundaries
target probability per bin
maximum oversampling
maximum training weight
data/MC sampling rule
invalid-density treatment
```

## 4. Audit the available event populations using the chosen scheme

Once the density bins and sampling logic are frozen, perform the complete audit.

For neutrinos, report:

```text
CC numu, CC nue, CC nutau, or NC
x production
x source file
x true-energy range
```

For neutral hadrons, report:

```text
kaon or neutron
x generated-energy bin
x source file
```

For muon backgrounds, report:

```text
passing muon or muon-DIS
x data or MC
x run/production
x source file
x frozen SciFi-density bin
```

The audit should distinguish:

- Raw event count.
- Events with valid and aligned Hit3D/features.
- Number of independent source groups.
- Events the sampler would draw per epoch.
- Expected repetition or undersampling.
- File-level integrated or equivalent luminosity.

## 5. Decide the training balance

Use the audit to decide:

- Relative exposure of the eight physics classes.
- Data/MC mixture for passing muons.
- Data/MC mixture for muon-DIS.
- Coverage of kaon and neutron energy bins.
- Whether any class or density bin requires capped oversampling.

Do not require equal data/MC representation automatically. Guarantee regular exposure to both domains without forcing extreme repetition of the smaller sample. Retain all events for evaluation even if training draws a balanced subset during each epoch.

A possible sampling hierarchy is:

```text
choose physics class
-> choose domain where applicable
-> choose density or energy bin
-> choose source group
-> choose event
```

The exact probabilities will be decided after the audit.

## 6. Define training and physics weights

Keep two independent weighting systems.

### Training weights

Training weights control optimization and reflect the selected balance across:

- Eight physics classes.
- Data/MC domains.
- Muon-DIS density bins.
- Kaon and neutron energy bins, if necessary.

### Physics weights

Physics weights are used only for final yields. `lumi_per_file` describes the whole source file:

- Store it on each event row for traceability.
- Sum it once per unique source file.
- Never sum the repeated value over all events.
- Never use it for class balancing.
- Data events normally receive physics weight 1.
- MC normalization uses the file's equivalent luminosity and the target luminosity.

Training undersampling or oversampling must not change the physical luminosity.

## 7. Construct and freeze five folds

Assign complete source groups to folds:

- Group data by run and source file.
- Group MC by production and source file.
- Never divide one source file across folds.
- Preserve coverage of all eight classes.
- Preserve data/MC and SciFi-density coverage for the two muon classes.
- Preserve kaon and neutron energy-bin coverage.
- Keep any future transformed event in its original event's fold.

The fold assignment must be created once and frozen. For each model, calculate sampling parameters only from its four training folds. Keep the held-out fold naturally distributed and never balance it artificially for evaluation.

## 8. Publish the training dataset definition

The final products of this stage are:

1. Eight-class label definition.
2. Frozen V7 passing-muon/muon-DIS proxy definition.
3. Muon-DIS density-sampling study and selected strategy.
4. Complete population and luminosity audit.
5. Chosen class and data/MC balance.
6. Frozen five-fold event manifest.
7. Separate training-weight and physics-weight definitions.

The agreed order is:

```text
Define eight classes
-> freeze V7 passing-muon/muon-DIS proxies
-> test and choose muon-DIS density sampling
-> audit all populations using the frozen density bins
-> choose class and data/MC balance
-> build five source-grouped folds
-> publish the training manifest
```
