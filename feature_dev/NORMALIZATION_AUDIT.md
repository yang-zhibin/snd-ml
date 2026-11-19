# Production luminosity audit reference

Start with [the normalization guide](NORMALIZATION.md) for the calculation.
Pass this version-1 JSON using `--exposure-manifest /path/to/audit.json`.

## Required fields

| Field | Meaning / source |
| --- | --- |
| `campaign_sha256` | Campaign identity from `plan.json` or `config/coverage.json`; use the full campaign, not a capped run |
| `populations` | Named parent MC/data populations, each with its complete list of sources |
| `role` | `mc`, `data`, or external `reference_data` |
| `coverage_evidence` | Reference to verified catalogue/coverage evidence; the validator requires it but does not independently inspect it |
| `source_uid` | For processed sources, exact `family:source_uid` identity from coverage or ROOT `audit_source_id`; external reference-data IDs must also be unique |
| `luminosity_fb` | Verified parent-file luminosity in fb⁻¹ |
| `sampling_probability` | 1 for complete data and MC; known probability for randomly sampled data |
| `sampling_method` | Must be `uniform_random` when probability is below 1; MC sampling below 1 is rejected |
| `selected_entries` | Complete selected-stream count for this source, before additional plot cuts |
| `processed_ranges` | Half-open ranges that cover `[0, selected_entries)` exactly; `[]` for an empty source |
| `groups` | Observed group/geometry mappings to parent populations; MC also names its target data population |

A selected-stream range is not a raw ROOT entry range. For an unsharded source
with 100 selected entries, use `[[0, 100]]`. Include sources with zero selected
events; do not silently drop failed or unprocessed files.

## Illustrative example

This example represents one geometry and electron-neutrino CC/NC groups.
Replace every identity, count, luminosity and evidence reference with verified
values. Add the other populated group/geometry mappings and parent populations
required by your campaign.

```json
{
  "version": 1,
  "campaign_sha256": "FULL_CAMPAIGN_SHA256",
  "populations": {
    "ve": {
      "role": "mc",
      "coverage_evidence": "/path/to/verified-mc-coverage",
      "sources": [
        {"source_uid": "family:mc-file-1", "luminosity_fb": 15,
         "sampling_probability": 1, "selected_entries": 4,
         "processed_ranges": [[0, 4]]},
        {"source_uid": "family:mc-file-empty", "luminosity_fb": 15,
         "sampling_probability": 1, "selected_entries": 0,
         "processed_ranges": []}
      ]
    },
    "target_data": {
      "role": "reference_data",
      "coverage_evidence": "/path/to/verified-data-coverage",
      "sources": [
        {"source_uid": "external:data-file-1", "luminosity_fb": 20,
         "sampling_probability": 0.1, "sampling_method": "uniform_random",
         "selected_entries": 2, "processed_ranges": [[0, 2]]}
      ]
    }
  },
  "groups": [
    {"group": "CC_nue", "geometry": "EXACT_GEOMETRY_ID",
     "population": "ve", "target_population": "target_data"},
    {"group": "NC_nue", "geometry": "EXACT_GEOMETRY_ID",
     "population": "ve", "target_population": "target_data"}
  ]
}
```

Here the MC denominator is `15 + 15 = 30 fb⁻¹`, including the empty source.
The target is `20 × 0.1 = 2 fb⁻¹`. Both CC_nue and NC_nue have weight `2/30`.
The parent exposure is shared; do not add it again for each class or list the
same source under another population.

For real-data processing use role `data`, actual processed source IDs and group
names, and omit `target_population`. Its event weight is 1.

The external reference-data selected counts describe its own audited stream;
they are not entries in the MC feature files. Their correctness, and completeness
of the declared populations, must be established outside this validator.

The older standalone launcher uses `sample_sha256` of its prepared JSONL instead
of `campaign_sha256`; see [its reference](NORMALIZATION_STANDALONE.md).
