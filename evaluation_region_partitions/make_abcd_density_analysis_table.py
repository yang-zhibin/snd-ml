#!/usr/bin/env python3
"""Materialize the validated Region V1 data and neutrino-MC density table."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import ROOT
import yaml


ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


BASE_COLUMNS = [
    "table_version",
    "binning_version",
    "partition_version",
    "region_version",
    "region",
    "region_label",
    "row_type",
    "root_bin_index",
    "bin_low",
    "bin_high",
    "overflow_folded_into_last_bin",
    "data_exposure",
    "data_raw_events",
]

SAMPLE_COLUMNS = [
    "raw_mc_events",
    "sum_weights",
    "sum_weights_squared",
    "effective_entries",
    "source_mc_luminosity",
    "normalization_weight",
]

DERIVED_COLUMNS = [
    "neutrino_background_sum_weights",
    "neutrino_background_sum_weights_squared",
    "neutrino_background_effective_entries",
    "standard_neutrino_sum_weights",
    "standard_neutrino_sum_weights_squared",
    "standard_neutrino_effective_entries",
    "cc_nue_fraction_of_standard_neutrino_mc",
    "cc_nue_signal_fraction_of_data",
    "source_validation_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--mc-config", required=True)
    parser.add_argument("--product-directory", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def read_summary(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one summary row in {path}")
    return rows[0]


def close(left: float, right: float, validation: dict) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=float(validation["relative_tolerance"]),
        abs_tol=float(validation["absolute_tolerance"]),
    )


def require_close(
    observed: float, expected: float, description: str, validation: dict
) -> None:
    if not close(observed, expected, validation):
        raise ValueError(
            f"{description}: observed {observed:.17g}, expected {expected:.17g}"
        )


def require_integer(value: float, description: str, validation: dict) -> int:
    rounded = round(value)
    require_close(value, float(rounded), description, validation)
    return int(rounded)


def histogram_values(histogram, bins: int) -> list[float]:
    return [float(histogram.GetBinContent(index)) for index in range(bins + 1)]


def validate_histogram(
    histogram,
    name: str,
    bins: int,
    minimum: float,
    maximum: float,
    validation: dict,
) -> None:
    if not histogram:
        raise KeyError(f"Missing histogram {name!r}")
    if int(histogram.GetNbinsX()) != bins:
        raise ValueError(f"{name} has {histogram.GetNbinsX()} bins, expected {bins}")
    require_close(histogram.GetXaxis().GetXmin(), minimum, f"{name} minimum", validation)
    require_close(histogram.GetXaxis().GetXmax(), maximum, f"{name} maximum", validation)
    if bool(validation.get("require_zero_overflow", True)):
        require_close(
            float(histogram.GetBinContent(bins + 1)),
            0.0,
            f"{name} overflow after folding",
            validation,
        )


def effective_entries(sum_weights: float, sum_weights_squared: float) -> float:
    if sum_weights_squared <= 0.0:
        return 0.0
    return sum_weights * sum_weights / sum_weights_squared


def fmt(value: float) -> str:
    return f"{value:.17g}"


def load_sources(
    product_directory: Path,
    data_config: dict,
    mc_config: dict,
    table_config: dict,
) -> tuple[dict, dict]:
    validation = table_config["validation"]
    data_hist_name = str(data_config["histogram"]["name"])
    mc_hist_config = mc_config["histogram"]
    bins = int(data_config["histogram"]["bins"])
    minimum = float(data_config["histogram"]["min"])
    maximum = float(data_config["histogram"]["max"])
    data_sources: dict[str, dict] = {}
    mc_sources: dict[str, dict[str, dict]] = {}

    for region, region_config in data_config["regions"].items():
        data_root_path = product_directory / "hists" / f"{region}__real_data.root"
        data_summary_path = product_directory / "summaries" / f"{region}__real_data.csv"
        summary = read_summary(data_summary_path)
        if bool(validation.get("require_source_status_pass", True)) and summary["status"] != "PASS":
            raise ValueError(f"Source summary is not PASS: {data_summary_path}")
        root_file = ROOT.TFile.Open(str(data_root_path))
        if not root_file or root_file.IsZombie():
            raise OSError(f"Could not open {data_root_path}")
        histogram = root_file.Get(data_hist_name)
        validate_histogram(histogram, data_hist_name, bins, minimum, maximum, validation)
        values = histogram_values(histogram, bins)
        if bool(validation.get("require_integer_raw_counts", True)):
            values = [
                float(require_integer(value, f"{region} data bin {index}", validation))
                for index, value in enumerate(values)
            ]
        total = math.fsum(values)
        require_close(
            total,
            float(summary["histogram_integral_with_flow"]),
            f"{region} data summary integral",
            validation,
        )
        require_close(
            total,
            float(region_config["expected_n_events"]),
            f"{region} configured data events",
            validation,
        )
        exposure = float(summary["catalogue_exposure"])
        require_close(
            exposure,
            float(region_config["expected_exposure"]),
            f"{region} data exposure",
            validation,
        )
        data_sources[region] = {
            "values": values,
            "total": total,
            "exposure": exposure,
        }
        root_file.Close()

        mc_sources[region] = {}
        for sample, sample_config in table_config["samples"].items():
            mc_root_path = product_directory / "hists" / f"{region}__{sample}.root"
            mc_summary_path = product_directory / "summaries" / f"{region}__{sample}.csv"
            mc_summary = read_summary(mc_summary_path)
            if bool(validation.get("require_source_status_pass", True)) and mc_summary["status"] != "PASS":
                raise ValueError(f"Source summary is not PASS: {mc_summary_path}")
            if mc_summary["sample_role"] != sample_config["role"]:
                raise ValueError(f"Role mismatch for {region} {sample}")
            mc_file = ROOT.TFile.Open(str(mc_root_path))
            if not mc_file or mc_file.IsZombie():
                raise OSError(f"Could not open {mc_root_path}")
            raw = mc_file.Get(str(mc_hist_config["raw_name"]))
            weighted = mc_file.Get(str(mc_hist_config["weighted_name"]))
            sumw2 = mc_file.Get(str(mc_hist_config["sumw2_name"]))
            for histogram, name in (
                (raw, str(mc_hist_config["raw_name"])),
                (weighted, str(mc_hist_config["weighted_name"])),
                (sumw2, str(mc_hist_config["sumw2_name"])),
            ):
                validate_histogram(histogram, name, bins, minimum, maximum, validation)
            raw_values = histogram_values(raw, bins)
            weighted_values = histogram_values(weighted, bins)
            sumw2_values = histogram_values(sumw2, bins)
            weight = float(mc_summary["normalization_weight"])
            for index, (raw_value, weighted_value, sumw2_value) in enumerate(
                zip(raw_values, weighted_values, sumw2_values)
            ):
                if bool(validation.get("require_integer_raw_counts", True)):
                    raw_values[index] = float(
                        require_integer(raw_value, f"{region} {sample} raw bin {index}", validation)
                    )
                require_close(
                    weighted_value,
                    raw_values[index] * weight,
                    f"{region} {sample} weighted bin {index}",
                    validation,
                )
                require_close(
                    sumw2_value,
                    raw_values[index] * weight * weight,
                    f"{region} {sample} sumw2 bin {index}",
                    validation,
                )
            raw_total = math.fsum(raw_values)
            weighted_total = math.fsum(weighted_values)
            sumw2_total = math.fsum(sumw2_values)
            require_close(raw_total, float(mc_summary["raw_integral_with_flow"]), f"{region} {sample} raw total", validation)
            require_close(weighted_total, float(mc_summary["sum_weights"]), f"{region} {sample} weighted total", validation)
            require_close(sumw2_total, float(mc_summary["sum_weights_squared"]), f"{region} {sample} sumw2 total", validation)
            require_close(float(mc_summary["target_data_exposure"]), exposure, f"{region} {sample} target exposure", validation)
            mc_sources[region][sample] = {
                "raw": raw_values,
                "weighted": weighted_values,
                "sumw2": sumw2_values,
                "raw_total": raw_total,
                "weighted_total": weighted_total,
                "sumw2_total": sumw2_total,
                "source_mc_luminosity": float(mc_summary["source_mc_luminosity"]),
                "weight": weight,
            }
            mc_file.Close()

    return data_sources, mc_sources


def validate_configs(data_config: dict, mc_config: dict, table_config: dict) -> None:
    if table_config["binning_version"] != data_config["version"]:
        raise ValueError("Table binning version does not match the data histogram version")
    if mc_config["binning_version"] != data_config["version"]:
        raise ValueError("MC and data histogram binning versions differ")
    if list(data_config["regions"]) != list(mc_config["regions"]):
        raise ValueError("Data and MC region order/configuration differ")
    if set(table_config["samples"]) != set(mc_config["samples"]):
        raise ValueError("Table and MC sample sets differ")
    for key in ("bins", "min", "max", "bin_width", "underflow", "overflow"):
        if data_config["histogram"][key] != mc_config["histogram"][key]:
            raise ValueError(f"Data and MC histogram setting {key!r} differs")
    if data_config["histogram"]["underflow"] != "preserve":
        raise ValueError("The table requires preserved underflow")
    if data_config["histogram"]["overflow"] != "fold_last":
        raise ValueError("The table requires overflow folded into the last bin")


def table_columns(table_config: dict) -> list[str]:
    columns = list(BASE_COLUMNS)
    for sample_config in table_config["samples"].values():
        prefix = sample_config["column_prefix"]
        columns.extend(f"{prefix}_{suffix}" for suffix in SAMPLE_COLUMNS)
    columns.extend(DERIVED_COLUMNS)
    return columns


def make_row(
    table_config: dict,
    data_config: dict,
    region: str,
    region_label: str,
    row_type: str,
    root_bin_index: int | None,
    bin_low: float | None,
    bin_high: float | None,
    data_exposure: float,
    data_value: float,
    sample_values: dict[str, dict[str, float]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "table_version": table_config["version"],
        "binning_version": table_config["binning_version"],
        "partition_version": data_config["catalogue"]["partition_version"],
        "region_version": data_config["catalogue"]["region_version"],
        "region": region,
        "region_label": region_label,
        "row_type": row_type,
        "root_bin_index": "" if root_bin_index is None else root_bin_index,
        "bin_low": "" if bin_low is None else fmt(bin_low),
        "bin_high": "" if bin_high is None else fmt(bin_high),
        "overflow_folded_into_last_bin": True,
        "data_exposure": fmt(data_exposure),
        "data_raw_events": int(round(data_value)),
    }
    background_sumw = 0.0
    background_sumw2 = 0.0
    standard_sumw = 0.0
    standard_sumw2 = 0.0
    for sample, sample_config in table_config["samples"].items():
        prefix = sample_config["column_prefix"]
        values = sample_values[sample]
        sumw = values["weighted"]
        sumw2 = values["sumw2"]
        row[f"{prefix}_raw_mc_events"] = int(round(values["raw"]))
        row[f"{prefix}_sum_weights"] = fmt(sumw)
        row[f"{prefix}_sum_weights_squared"] = fmt(sumw2)
        row[f"{prefix}_effective_entries"] = fmt(effective_entries(sumw, sumw2))
        row[f"{prefix}_source_mc_luminosity"] = fmt(values["source_mc_luminosity"])
        row[f"{prefix}_normalization_weight"] = fmt(values["weight"])
        standard_sumw += sumw
        standard_sumw2 += sumw2
        if sample_config["role"] == "neutrino_background":
            background_sumw += sumw
            background_sumw2 += sumw2
    signal_sumw = sample_values["CC_nue"]["weighted"]
    row.update(
        {
            "neutrino_background_sum_weights": fmt(background_sumw),
            "neutrino_background_sum_weights_squared": fmt(background_sumw2),
            "neutrino_background_effective_entries": fmt(
                effective_entries(background_sumw, background_sumw2)
            ),
            "standard_neutrino_sum_weights": fmt(standard_sumw),
            "standard_neutrino_sum_weights_squared": fmt(standard_sumw2),
            "standard_neutrino_effective_entries": fmt(
                effective_entries(standard_sumw, standard_sumw2)
            ),
            "cc_nue_fraction_of_standard_neutrino_mc": (
                fmt(signal_sumw / standard_sumw) if standard_sumw > 0 else ""
            ),
            "cc_nue_signal_fraction_of_data": (
                fmt(signal_sumw / data_value) if data_value > 0 else ""
            ),
            "source_validation_status": "PASS",
        }
    )
    return row


def build_rows(
    table_config: dict,
    data_config: dict,
    data_sources: dict,
    mc_sources: dict,
) -> list[dict[str, object]]:
    histogram = data_config["histogram"]
    bins = int(histogram["bins"])
    minimum = float(histogram["min"])
    width = float(histogram["bin_width"])
    rows: list[dict[str, object]] = []
    for region, region_config in data_config["regions"].items():
        data = data_sources[region]
        samples = mc_sources[region]
        indices = [0] + list(range(1, bins + 1))
        for index in indices:
            if index == 0:
                row_type = "underflow"
                low = None
                high = minimum
            else:
                row_type = "bin"
                low = minimum + (index - 1) * width
                high = minimum + index * width
            sample_values = {
                sample: {
                    "raw": values["raw"][index],
                    "weighted": values["weighted"][index],
                    "sumw2": values["sumw2"][index],
                    "source_mc_luminosity": values["source_mc_luminosity"],
                    "weight": values["weight"],
                }
                for sample, values in samples.items()
            }
            rows.append(
                make_row(
                    table_config,
                    data_config,
                    region,
                    region_config["label"],
                    row_type,
                    index,
                    low,
                    high,
                    data["exposure"],
                    data["values"][index],
                    sample_values,
                )
            )
        inclusive_values = {
            sample: {
                "raw": values["raw_total"],
                "weighted": values["weighted_total"],
                "sumw2": values["sumw2_total"],
                "source_mc_luminosity": values["source_mc_luminosity"],
                "weight": values["weight"],
            }
            for sample, values in samples.items()
        }
        rows.append(
            make_row(
                table_config,
                data_config,
                region,
                region_config["label"],
                "inclusive",
                None,
                None,
                None,
                data["exposure"],
                data["total"],
                inclusive_values,
            )
        )
    return rows


def validate_table_reconstruction(
    rows: list[dict[str, object]],
    table_config: dict,
    data_config: dict,
    data_sources: dict,
    mc_sources: dict,
) -> None:
    validation = table_config["validation"]
    for region in data_config["regions"]:
        regular = [
            row
            for row in rows
            if row["region"] == region and row["row_type"] != "inclusive"
        ]
        inclusive = next(
            row
            for row in rows
            if row["region"] == region and row["row_type"] == "inclusive"
        )
        require_close(
            math.fsum(float(row["data_raw_events"]) for row in regular),
            data_sources[region]["total"],
            f"{region} table data reconstruction",
            validation,
        )
        require_close(
            float(inclusive["data_raw_events"]),
            data_sources[region]["total"],
            f"{region} inclusive data reconstruction",
            validation,
        )
        for sample, sample_config in table_config["samples"].items():
            prefix = sample_config["column_prefix"]
            source = mc_sources[region][sample]
            for suffix, expected in (
                ("raw_mc_events", source["raw_total"]),
                ("sum_weights", source["weighted_total"]),
                ("sum_weights_squared", source["sumw2_total"]),
            ):
                column = f"{prefix}_{suffix}"
                require_close(
                    math.fsum(float(row[column]) for row in regular),
                    expected,
                    f"{region} {sample} table {suffix} reconstruction",
                    validation,
                )
                require_close(
                    float(inclusive[column]),
                    expected,
                    f"{region} {sample} inclusive {suffix} reconstruction",
                    validation,
                )


def atomic_csv(path: str | Path, columns: list[str], rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(args: argparse.Namespace) -> None:
    table_config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    mc_config = load_yaml(args.mc_config)
    validate_configs(data_config, mc_config, table_config)
    data_sources, mc_sources = load_sources(
        Path(args.product_directory), data_config, mc_config, table_config
    )
    rows = build_rows(table_config, data_config, data_sources, mc_sources)
    expected_rows = len(data_config["regions"]) * (
        int(data_config["histogram"]["bins"]) + 2
    )
    if len(rows) != expected_rows:
        raise ValueError(f"Produced {len(rows)} rows, expected {expected_rows}")
    if bool(table_config["validation"].get("require_exact_table_reconstruction", True)):
        validate_table_reconstruction(
            rows, table_config, data_config, data_sources, mc_sources
        )
    atomic_csv(args.output, table_columns(table_config), rows)
    print(
        f"Wrote {len(rows)} validated rows for {len(data_config['regions'])} "
        f"regions to {args.output} (status=PASS)"
    )


if __name__ == "__main__":
    build(parse_args())
