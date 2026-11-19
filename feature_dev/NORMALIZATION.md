# Plot normalization

This guide describes `run_shower_features.sh`.
**Luminosity belongs to a source file or parent population, not to each event.**
The plotter calculates an event weight from that exposure.

## 1. Choose shape or yield

| Goal | Mode | Information needed |
| --- | --- | --- |
| Compare distribution shapes, including testbeam | `--normalization shape` | Histogram counts only |
| Compare MC predictions with real-data event yields | `--normalization yield` | Complete source coverage, MC luminosity and target data luminosity |

`auto` chooses shapes for small samples and testbeam. For full MC or real data,
it requires a luminosity audit. Capped samples cannot use yield normalization.
Use explicit `shape` to inspect a full sample without an audit.

## 2. What information is needed, and where is it saved?

All luminosities below are integrated luminosities in **fb⁻¹**.
Paths are relative to `<output-root>/<dataset>/<run>/`.

| Information | Where to find it | How it is used |
| --- | --- | --- |
| Source identity and original catalogue row | `plan.json` → `jobs`; input catalogues listed in [datasets.yaml](datasets.yaml) | Identify each parent file once |
| Catalogue luminosity per file | Catalogue `source_lumi_per_file` (fallback `lumi_per_file`); copied to each feature ROOT file | Retain original exposure information for validation |
| Sampling probability and exposure type | Catalogue/plan; ROOT `production_exposure_json` | Determine whether a physical exposure is known |
| Saved selected-event count and validation | `sources/<index>/validation.json` | Check that processing covered the selected stream |
| Campaign source counts, identities and geometries | `config/coverage.json` | Build/check the audit against actual outputs |
| **Audited** source luminosities, sampling and group mapping | Your `--exposure-manifest audit.json`; embedded as `exposure_audit` in `plan.json` and ROOT `production_exposure_json` | Authoritative inputs to yield calculation |
| Final source/target exposure and weight per group/geometry | `plots/normalization.json` (`source_exposure_fb`, `target_exposure_fb`, `scale`) | Inspect the weights actually used |
| Raw/weighted bins and component metadata | `histograms/1d/.../<class>.root` or `histograms/2d/.../<class>.root` | Validate the resulting histogram |

The catalogue values and audited values are stored separately. The plotter uses
**the validated audit** for yields; embedding a catalogue luminosity in ROOT does
not establish complete campaign coverage.

### Metadata inside features.root

Each `sources/<index>/features.root` contains:

| ROOT object | Content |
| --- | --- |
| `sndData` | Standard and plugin features, one entry per saved event |
| `source_exposure` | One metadata row for this source file |
| `production_exposure_json` | Source identity, exposure type, campaign hash, validity and supplied audit |

The `source_exposure` branches are `source_uid`, `audit_source_id`,
`luminosity_fb`, `sampling_probability`, `effective_luminosity_fb` and
`exposure_valid`. The effective value is `luminosity_fb × sampling_probability`
when valid. Small/unknown exposure has `exposure_valid=false` and an invalid
(NaN) effective luminosity. A catalogue luminosity may still be retained.

There is **no per-event luminosity branch** and no campaign `normalization` tree
in these production source files. Do not multiply file luminosity by the number
of events, or sum it again for every event. Campaign weights are saved separately
in `plots/normalization.json`.

## 3. Calculate the normalization

### Shapes: testbeam or distribution comparisons

For each class and feature, combine raw counts across geometries, then divide
by their in-range total:

```text
shape_bin = bin_count / sum_of_bin_counts
```

Example: counts `[20, 30, 50]` become `[0.2, 0.3, 0.5]`.
There is no bin-width scaling. Missing/nonfinite feature values are excluded.
The total is evaluated after the configured flow handling: by default 1D
overflow is folded into the last bin; 2D flows remain outside the integral.

### MC yields

For each audited parent population:

```text
L_MC   = sum of luminosities of its unique parent MC files
L_data = sum of (data-file luminosity × sampling probability)
w_MC   = L_data / L_MC
bin_yield = bin_count × w_MC
```

Example: MC files represent `10 + 15 = 25 fb⁻¹`, and target data represent
`5 fb⁻¹`. Each MC event has weight `5/25 = 0.2`; 100 events predict 20 events.

- Count each parent file once, including files with zero selected events.
- CC and NC from the same flavour use the same parent MC exposure. Do not
  estimate separate luminosities from their selected-event counts.
- Electron-neutrino and muon-neutrino populations have separate MC denominators.
- Additional plot cuts change event counts, not the parent luminosity.
- MC subsampling is currently rejected for yield mode.

### Real-data yields

Data events have weight **1** at their effective recorded exposure.
For a uniform random 10% sample of `50 fb⁻¹`:

```text
L_data = 50 × 0.1 = 5 fb⁻¹
w_data = 1
```

Normalize MC to that same `5 fb⁻¹`. Do not also multiply the data by 10.
This sampling correction applies to a known random sampling procedure, not an
event-selection efficiency or the first N events of a file.

### Combining geometries and errors

Apply each component's audited weight before adding its bins:

```text
combined_bin = sum(component_bin_count × component_weight)
variance     = sum(component_bin_count × component_weight²)
error        = sqrt(variance)
```

The output has one combined plot. Geometry and weights remain in its metadata.
Do not apply another luminosity scale to an already weighted histogram.
Audit mappings must describe the intended parent populations and target data
exposures; combining plots does not establish that mapping automatically.

ROOT histogram files store raw `hist`/`hist2d` and, for yields,
`hist_yield`/`hist2d_yield`. `production_histogram_json` records the components.
CSV yield summaries leave the single `scale_factor` blank because component
weights can differ. Statistical errors exclude luminosity uncertainty; shape
errors treat the observed normalization denominator as fixed. The 2D profile
page remains a raw-event binned mean, not a yield-weighted profile.

## 4. Prepare the audit and run

1. Establish complete source coverage from the parent catalogue and successful
   source validation records, including empty files. A full shape run can provide
   `config/coverage.json`; a small run cannot establish full coverage.
2. Obtain verified MC and target-data luminosities and sampling probabilities.
   Current real-data `event_count_only` catalogue rows do not provide physical
   luminosity; selected-event counts cannot supply it.
3. Write [the audit JSON](NORMALIZATION_AUDIT.md), using the campaign hash,
   source IDs and group/geometry names from the campaign records.
4. Run the full campaign with the audit and a configured HTCondor profile:

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample full --normalization yield \
  --exposure-manifest /path/to/audit.json \
  --profile /path/to/htcondor-profile --run full-yield01
```

Use `--dataset real-data` with its own data audit for a data yield run.
The validator checks identities, counts, processed ranges and group membership.
The audit producer must establish physical luminosity and complete parent/data
coverage; the software does not infer these from EOS automatically.

## 5. Read the saved values

Run in the documented [LCG analysis environment](../docs/project/ENVIRONMENT_AND_SNAKEMAKE.md):

```python
import json
from pathlib import Path
import uproot

run = Path("/path/to/output/mc-neutrino/full-yield01")
with uproot.open(run / "sources/0/features.root") as root:
    source = root["source_exposure"].arrays(library="np")
    context = json.loads(str(root["production_exposure_json"]))
    print(source["luminosity_fb"], source["exposure_valid"])
    print(context["coverage_status"])  # requires_campaign_validation

weights = json.loads((run / "plots/normalization.json").read_text())
for row in weights:
    print(row["group"], row["geometry"], row.get("scale"))
    # Shape mode has no universal event weight; its scale depends on the feature.
```

Implementation: [source metadata](../feature_production/worker.py),
[audit validation](exposure.py), [histograms and weights](../feature_production/distributions.py).
The older `run_feature_dev.sh` has a different ROOT layout; its retained reference
is [standalone normalization](NORMALIZATION_STANDALONE.md).
