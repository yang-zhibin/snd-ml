import argparse
import os
from pathlib import Path

import pandas as pd


SOURCE_PATH_COLUMNS = {"raw_path", "geo_path"}
DIRECT_OUTPUT_COLUMNS = {"digi_path"}
NON_PRODUCT_PATH_COLUMNS = {
    "output_base_path",
    "original_digi_path",
}


def _is_missing(value):
    if pd.isna(value):
        return True
    return str(value).strip() == "" or str(value).strip().lower() == "nan"


def _suffix_for_column(column_name):
    name = column_name[:-5] if column_name.endswith("_path") else column_name
    if name.endswith("pt_hit"):
        return ".pt.gz"
    if name.endswith("npz_hit"):
        return ".npz"
    if name.endswith("pkl_hit"):
        return ".pkl.gz"
    if name.startswith("preCutEff") or name.startswith("matrix_"):
        return ".csv"
    return ".root"


def _path_product_name(column_name):
    if not column_name.endswith("_path"):
        raise ValueError(f"Expected a *_path column, got {column_name}")
    return column_name[:-5]


def _format_partition(value):
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def _eventbuilder_filename(product, tag, data_type, source_subfolder, partition, suffix):
    safe_subfolder = str(source_subfolder).replace("/", "_")
    safe_partition = _format_partition(partition).replace("/", "_")
    return f"{product}_{tag}_{data_type}_{safe_subfolder}_{safe_partition}{suffix}"


def _build_output_base(output_root, dataset_dir, source_subfolder, partition):
    return (
        f"{output_root.rstrip('/')}/"
        f"{dataset_dir.strip('/')}/"
        f"{str(source_subfolder).strip('/')}/"
        f"{_format_partition(partition).strip('/')}"
    )


def _validate_required_columns(df, required_columns):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_nonempty(df, column):
    bad_rows = df[df[column].apply(_is_missing)]
    if len(bad_rows) == 0:
        return
    examples = bad_rows[["subfolder", "partition", column]].head(10).to_dict("records")
    raise ValueError(f"Column {column} has missing values. Examples: {examples}")


def make_eventbuilder_metadata(
    input_csv,
    output_csv,
    output_root,
    dataset_dir,
    tag,
    subfolder_suffix,
):
    df = pd.read_csv(input_csv)
    _validate_required_columns(
        df,
        [
            "data_type",
            "subfolder",
            "partition",
            "n_event",
            "digi_path",
            "geo_path",
            "raw_path",
        ],
    )
    _validate_nonempty(df, "raw_path")
    _validate_nonempty(df, "geo_path")

    source_subfolder = df["subfolder"].astype(str)
    df["source_metadata_csv"] = os.path.basename(input_csv)
    df["source_subfolder"] = source_subfolder
    df["original_digi_path"] = df["digi_path"]
    df["subfolder"] = source_subfolder + subfolder_suffix
    df["production_profile"] = f"mc_eventbuilder_{tag.lower()}"

    df["output_base_path"] = [
        _build_output_base(output_root, dataset_dir, src_subfolder, partition)
        for src_subfolder, partition in zip(source_subfolder, df["partition"])
    ]

    path_columns = [
        column
        for column in df.columns
        if column.endswith("_path")
        and column not in SOURCE_PATH_COLUMNS
        and column not in DIRECT_OUTPUT_COLUMNS
        and column not in NON_PRODUCT_PATH_COLUMNS
    ]

    for column in path_columns:
        product = _path_product_name(column)
        suffix = _suffix_for_column(column)
        df[column] = [
            _eventbuilder_filename(
                product,
                tag,
                data_type,
                src_subfolder,
                partition,
                suffix,
            )
            for data_type, src_subfolder, partition in zip(
                df["data_type"],
                source_subfolder,
                df["partition"],
            )
        ]

    if "newDigi_path" not in df.columns:
        raise ValueError("Input metadata must contain newDigi_path")

    df["digi_path"] = df["output_base_path"].str.rstrip("/") + "/" + df["newDigi_path"].astype(str).str.lstrip("/")

    for column in path_columns:
        duplicated = df[column].duplicated(keep=False)
        if duplicated.any():
            examples = df.loc[duplicated, ["subfolder", "partition", column]].head(10).to_dict("records")
            raise ValueError(f"Generated duplicate values in {column}. Examples: {examples}")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote EventBuilder metadata: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create an EventBuilder MC-neutrino metadata variant."
    )
    parser.add_argument("--input", required=True, help="Input updated metadata CSV")
    parser.add_argument("--output", required=True, help="Output EventBuilder metadata CSV")
    parser.add_argument(
        "--output-root",
        default="/eos/experiment/sndlhc/users/zhibin",
        help="EOS/user root under which EventBuilder outputs are written",
    )
    parser.add_argument(
        "--dataset-dir",
        default="MC_neutrino_EventBuilder",
        help="Dataset directory below output-root",
    )
    parser.add_argument(
        "--tag",
        default="EventBuilder",
        help="Filename tag inserted into generated product names",
    )
    parser.add_argument(
        "--subfolder-suffix",
        default="_EventBuilder",
        help="Suffix added to the metadata subfolder value",
    )
    args = parser.parse_args()

    make_eventbuilder_metadata(
        input_csv=args.input,
        output_csv=args.output,
        output_root=args.output_root,
        dataset_dir=args.dataset_dir,
        tag=args.tag,
        subfolder_suffix=args.subfolder_suffix,
    )


if __name__ == "__main__":
    main()
