# Shape and yield normalization

The development launcher supports `--normalization auto|shape|yield`.

| Input | `auto` | `yield` |
| --- | --- | --- |
| Small/capped sample | In-range bin fractions | Rejected |
| Testbeam or synthetic | In-range bin fractions | Rejected |
| Prepared full MC or collision data | Requires an exposure audit | Requires an exposure audit |

Use shapes for the existing automatic small-sample workflow:

```bash
./run_feature_dev.sh --algorithm gap_clusters --dataset mc-neutrino \
  --sample small --run trial01
```

For an **uncapped prepared sample with audited coverage**:

```bash
./run_feature_dev.sh --algorithm gap_clusters --dataset mc-neutrino \
  --sample full --sample-root /path/to/prepared-samples \
  --normalization yield --exposure-manifest /path/to/exposure.json \
  --run full-yield01
```

The sample location comes from `datasets.yaml` (or `--dataset-config`).
Use `--normalization shape` to inspect a prepared full sample without an audit.
Run each yield dataset separately. Automatic full preparation is still blocked:
this change does not implement full-campaign plugin sharding or remove the
256 MiB input guard. Never relabel a capped sample as a complete campaign.

## Calculation

For each group and geometry:

- **MC:** event weight = target data luminosity / complete parent MC luminosity.
  CC and NC from the same flavour reference the same parent population.
  Electron and muon neutrino populations have separate denominators.
- **Data:** event weight = 1. For uniform random sampling with probability `p`,
  represented exposure = `p × parent luminosity`. Do not also apply `1/p`.
- **Testbeam/shapes:** divide by the in-range count; no bin-width scaling.

Histograms retain raw `counts`, legacy shape `fractions`, `normalized` bins,
`sumw2`, raw and normalized flows, and the normalization/exposure record.
For constant weight `w`, bin variance is `n × w²`. Shape `sumw2` treats the
observed normalization denominator as fixed; it is not a covariance estimate.
Missing values are excluded. The last bin includes its upper edge; flow bins
are saved separately. Baseline and candidate use the same exposure weight.
Changing only normalization reuses calculated features and rebuilds histograms.
The standalone HTML report displays the exposures and labels yields in events
per bin. Luminosity uncertainty is not included in statistical errors.

## Exposure audit format (version 1)

This minimal **illustrative** MC audit has one nonempty source and one empty
source. Replace every value with a verified campaign measurement. Never copy
these values as physics inputs.

```json
{
  "version": 1,
  "sample_sha256": "SHA256_OF_PREPARED_JSONL",
  "populations": {
    "ve": {
      "role": "mc",
      "coverage_evidence": "path/to/complete-parent-catalogue-and-validation-receipts",
      "sources": [
        {"source_uid": "ve-file-1", "luminosity_fb": 15,
         "sampling_probability": 1, "selected_entries": 4,
         "processed_ranges": [[0, 2], [2, 4]]},
        {"source_uid": "ve-file-empty", "luminosity_fb": 15,
         "sampling_probability": 1, "selected_entries": 0,
         "processed_ranges": []}
      ]
    },
    "veto_data": {
      "role": "reference_data",
      "coverage_evidence": "path/to/audited-target-data-coverage",
      "sources": [
        {"source_uid": "data-file-1", "luminosity_fb": 20,
         "sampling_probability": 0.1, "sampling_method": "uniform_random",
         "selected_entries": 2, "processed_ranges": [[0, 2]]}
      ]
    }
  },
  "groups": [
    {"group": "CC_nue", "geometry": "YOUR_GEOMETRY",
     "population": "ve", "target_population": "veto_data"},
    {"group": "NC_nue", "geometry": "YOUR_GEOMETRY",
     "population": "ve", "target_population": "veto_data"}
  ]
}
```

`luminosity_fb` means integrated luminosity in **fb⁻¹**. Source luminosity is
counted once, including sources with zero selected events. `processed_ranges`
are half-open ranges in the selected-event stream of that parent source; they
must partition `[0, selected_entries)` exactly. They are not raw ROOT entry
ranges. For a real-data sample, use role `data`, its actual source IDs and
counts, and omit `target_population`. MC subsampling is currently rejected.

Prepare the audit from the full parent catalogue and successful per-source
receipts, including empty outputs, before joining the selected-hit products.
Record the complete selected counts and shard ranges, check source identities,
and obtain the final sample digest with `sha256sum sample.jsonl`. An omitted
failed source must never be disguised as a smaller completed population.

The validator checks duplicate sources, range gaps/overlaps, positive exposures,
known sampling, nested cap metadata, matching group membership, sample digest,
and exact selected-event counts per source. The audit is copied into the run
manifest and contributes to the histogram cache key.

**Validation boundary:** `coverage_evidence` is an audit reference, not an
independently executed catalogue check. External reference-data event counts
and completeness of the declared parent population must be established by the
audit producer. This release does not automatically generate trustworthy
exposure audits from EOS. Current real-data development catalogues have
`event_count_only`/zero active exposure; those are not sufficient for yields.
Do not invent their luminosity from selected-event counts.

## Existing distribution workflow

`evaluation_region_partitions/normalization.py` supplies the positive-exposure
ratio used by this launcher and the existing production distribution script.
The production script's existing `common_exposure_19fb` mode remains unchanged;
it does not consume this new audit or run arbitrary development plugins.
Connecting the audited per-source plugin campaign to production distribution
rules remains a separate implementation step.

## Validation

`feature_dev/tests/test_exposure.py` covers the accounting and refusal rules.
`feature_dev/tests/normalization_integration.py` runs tiny artificial full-sample
shape/yield DAGs and checks shared calculation caches, baseline/candidate
weights, sumw2, report metadata, repeat reuse and early refusal cases.
These fixtures validate software behavior, not physical luminosities or EOS
campaign coverage.

## Metadata inside features.root

New runs publish **`<run>/features.root`**, a link to `export/features.root`.
Copy this file with symlinks dereferenced to obtain a standalone ROOT product.
It contains the unchanged `baseline` and `candidate` event trees plus:

| Object | Content |
| --- | --- |
| `source_exposure` (TTree) | One row per unique source, including zero-selected sources and reference-data sources |
| `normalization` (TTree) | One row per group/geometry, with source/target exposures and event weight |
| `exposure_audit_json` (TObjString) | Complete validated audit, serialized as canonical JSON; `null` when unavailable |
| `exposure_audit_sha256` (TObjString) | SHA-256 of the exact UTF-8 string stored above |
| `feature_metadata_json` (TObjString) | Metadata version 1, audit version, units, sample SHA, calculated ROOT SHA and coverage status |

`source_exposure` branches:

- `source_uid`, `population`, `role`: source identity and its parent population.
- `in_sample`: false for external `reference_data` sources; true for sample
  sources, including those with zero selected events.
- `luminosity_fb`: parent-file exposure, in **fb⁻¹**.
- `sampling_probability`, `effective_luminosity_fb`: probability and its product
  with parent exposure. MC sampling must be 1.
- `selected_entries`: selected-event count for that source. For reference data,
  this describes its external audited stream, not entries in the feature trees.
- `exposure_valid`: whether these exposure values are known from an audit.

`normalization` branches: `group`, `geometry`, `population`, `target_population`,
`mode`, `source_exposure_fb`, `target_exposure_fb`, `event_weight`,
`exposure_valid`, and `weight_valid`. Exposures use **fb⁻¹**; weights are
**dimensionless**. A data population targets itself. MC references the named
data population. Do not sum repeated population exposures across CC/NC rows.

Without an audit (small samples, testbeam, or explicitly requested shapes),
source IDs and observed counts remain available, but luminosities/probabilities
are **NaN with `exposure_valid=false`**. Shape records have
`weight_valid=false` and NaN `event_weight`: the per-feature in-range shape
normalization is recorded in histogram JSON, not as a universal event weight.
No per-event luminosity branch is added. Join event `source_uid` to the source
metadata; never sum file luminosity once per event.

Example with uproot:

```python
import json
import uproot

with uproot.open("/path/to/run/features.root") as root:
    sources = root["source_exposure"].arrays(library="np")
    normalizations = root["normalization"].arrays(library="np")
    audit = json.loads(str(root["exposure_audit_json"]))
    print(sources["source_uid"], sources["luminosity_fb"])
    print(normalizations["group"], normalizations["event_weight"])
    # Check exposure_valid / weight_valid before using numeric values.
```

The export stage checks every metadata field and embedded audit against the run
plan, and compares every copied feature branch and event identity to the
validated calculation file. `export/validation.json` records the checks and file
hashes; the stage receipt detects later file tampering on reuse.

`calculate/features.root` remains an **internal calculation cache without these
metadata objects**. Changing exposure regenerates the published copy, without
recalculating features or modifying a prior run's published file. Use a new run
name after changing inputs. Existing outputs are not patched retrospectively.
The exposure audit's original coverage limitations still apply: embedding it
makes the file self-contained, but does not independently establish EOS coverage.
