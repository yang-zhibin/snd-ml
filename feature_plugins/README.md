# Feature plugin interface

Each plugin is a directory containing `features.py` and `schema.yaml`.
The supplied example is [silvia/features.py](silvia/features.py), adapted from
Silvia's commit `d8cd5d08b4da94a0b453749573ba10150da68f72`.
Use `--plugin silvia` or `--plugin /absolute/path/to/your_plugin` with
[the production launcher](../feature_production/README.md).

## Functions

```python
def cluster_hits(hits, parameters):
    """Return your cluster representation for one event."""
    ...


def calculate_features(hits, clusters, parameters):
    """Return {schema_feature_name: scalar_value} for one event."""
    ...
```

If `cluster_hits` is omitted, use `calculate_features(hits, parameters)`.
An optional `validate_algorithm(hits, clusters, values)` checks each event after calculation.
`parameters` is the JSON object supplied with `--parameters '{"max_missing_channels": 1}'`.
Hit dictionaries and parameters are isolated copies; modifying hits is rejected.
No ROOT objects, truth labels or output-tree buffers are passed to the plugin.

## `hits`: list of dictionaries

| Key | Meaning | Unit / values |
| --- | --- | --- |
| `station` | SciFi station | integer 1–5 |
| `isVertical` | fibre orientation | boolean |
| `layer_channel` | channel index within the plane | integer |
| `qdc` | calibrated charge used by the converter | QDC units |
| `x_cm`, `y_cm`, `z_cm` | detector position, when available | cm |
| `time_ns` | hit time, when available | ns |

These are the final SciFi hits selected by the production converter. Empty hit
lists are valid. Optional coordinates must be checked with `hit.get('x_cm')`;
missing coordinates are not zero position.

## `clusters`: algorithm-defined object

The example imports the existing gap-clustering function. Its result has
`orientations['vertical'|'horizontal']['clusters'|'ranked_clusters']`.
Each cluster has `station`, `is_vertical`, `channels` (member channel indices),
`qdc` (summed charge), and `orientation_rank` (1 is leading within that view).
Recover members using `(station, isVertical, layer_channel)` as in the example;
a channel number alone is not a unique hit key across stations and views.
Other plugins may define their own cluster representation.

## Output schema

Declare every exported feature explicitly; production schemas do not inherit
baseline branches. Example:

```yaml
version: example-v1
features:
  leading_charge_fraction:
    dtype: float64
    description: Leading cluster QDC divided by total event cluster QDC.
    unit: dimensionless
    missing_value: -999.0
    valid_range: [0, 1]
    histogram: {bins: 25, range: [0, 1]}
```

Supported dtypes are `float64` and `int64`. Return the declared missing sentinel
for an undefined quantity. Missing keys, nonfinite values, wrong types, unknown
extra outputs and invalid ranges fail with event context. Recognized legacy
baseline outputs are discarded, so Silvia's existing function can still return
its baseline dictionary. Only explicitly declared outputs become branches.

Branch names are `<plugin_directory_name>__<feature_name>`. Existing production
branches are never overwritten, including similarly named cluster stations.
The schema, settings and source hashes are saved in every output ROOT file.
Local Python helpers inside the plugin folder are fingerprinted automatically;
external dependencies require explicit provenance support before use.

The adapted Silvia example keeps the existing formula on valid coordinates:
density counts ordered pairs within 1 cm (each distinct pair twice). Missing
coordinates yield the sentinel, not a false zero; an unknown event density also
makes its density fractions unknown. See the
[SciFi density report](../feature_dev/SCIFI_DENSITY.md) for the underlying logic.
