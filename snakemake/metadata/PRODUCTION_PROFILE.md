# Production Profile

`snakemake/metadata/update_path_file.py` writes a `production_profile` column
into updated metadata.

Current profile values:

- `mc_default`: all MC samples.
- `real_data_default`: real-data runs outside the 2024 skim-run list.
- `real_data_2024_skim_run`: real-data runs in the 2024 skim-run list used by
  `get_subset_metadata.py`.

Production uses this policy:

```text
if data_type != real_data:
    do not pass --drop-real-has-veto-has-us
elif production_profile == real_data_2024_skim_run:
    do not pass --drop-real-has-veto-has-us
else:
    pass --drop-real-has-veto-has-us
```

The flag is passed to both `digi_2_features.py` and `digi_2_hits3D.py`, so their
output event sequences remain aligned. When enabled, real-data events with at
least one valid veto MuFilter hit and at least one valid upstream MuFilter hit
are skipped during production.
