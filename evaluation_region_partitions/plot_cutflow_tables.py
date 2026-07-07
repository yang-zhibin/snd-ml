#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import pandas as pd
import yaml


PARTICLE_ORDER = [
    "real_data",
    "CC_nue",
    "CC_numu",
    "NC_nue",
    "NC_numu",
    "muonDIS",
    "muon",
    "CC_nue_EventBuilder",
    "CC_numu_EventBuilder",
    "NC_nue_EventBuilder",
    "NC_numu_EventBuilder",
]
CATEGORY_ORDER = {
    "veto": ["no_veto", "has_veto"],
    "us": ["has_us", "no_us"],
}
EVENTBUILDER_SUFFIX = "_EventBuilder"
COUNT_COLUMNS = ["initial_count", "cumulative_count", "relative_count"]
FIDUCIAL_MERGED_REGION_MAP = {
    "signal_no_veto_has_us_fiducial_inside": "merged_01_signal_sb2_no_veto_has_us",
    "sideband_no_veto_has_us_fiducial_outside": "merged_01_signal_sb2_no_veto_has_us",
    "sideband_no_veto_no_us_fiducial_inside": "merged_02_sb1_sb3_no_veto_no_us",
    "sideband_no_veto_no_us_fiducial_outside": "merged_02_sb1_sb3_no_veto_no_us",
    "sideband_has_veto_has_us_fiducial_inside": "merged_03_sb4_sb6_has_veto_has_us",
    "sideband_has_veto_has_us_fiducial_outside": "merged_03_sb4_sb6_has_veto_has_us",
    "sideband_has_veto_no_us_fiducial_inside": "merged_04_sb5_sb7_has_veto_no_us",
    "sideband_has_veto_no_us_fiducial_outside": "merged_04_sb5_sb7_has_veto_no_us",
}
FIDUCIAL_MERGED_REGION_LABELS = {
    "merged_01_signal_sb2_no_veto_has_us": "Signal/SB2: no veto, US",
    "merged_02_sb1_sb3_no_veto_no_us": "SB1/SB3: no veto, no US",
    "merged_03_sb4_sb6_has_veto_has_us": "SB4/SB6: veto, US",
    "merged_04_sb5_sb7_has_veto_no_us": "SB5/SB7: veto, no US",
}


def format_count(value: object) -> str:
    count = float(value)
    if not math.isfinite(count):
        return "0"
    if count == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(count))))
    if abs(count) >= 1.0e4:
        mantissa = count / (10**exponent)
        return rf"${mantissa:.2f}\times10^{{{exponent}}}$"
    return f"{count:.3g}"


def format_efficiency(value: object) -> str:
    efficiency = 100.0 * float(value)
    if not math.isfinite(efficiency):
        return "0"
    if efficiency == 0:
        return "0"
    if 0 < abs(efficiency) < 0.01:
        exponent = int(math.floor(math.log10(abs(efficiency))))
        mantissa = efficiency / (10**exponent)
        return rf"${mantissa:.2f}\times10^{{{exponent}}}$"
    return f"{efficiency:.3g}"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def reference_lumi(metadata: pd.DataFrame, eval_cfg: dict) -> float:
    ref = eval_cfg["normalization"]["reference"]
    selected = metadata.loc[
        (metadata["region"].astype(str) == str(ref["region"]))
        & (metadata["particle_group"].astype(str) == str(ref["particle_group"]))
    ]
    lumi = pd.to_numeric(selected.get("lumi_per_partition"), errors="coerce").fillna(0.0).sum()
    lumi = float(lumi)
    if lumi <= 0 and eval_cfg["normalization"].get("fail_on_missing_reference", True):
        raise RuntimeError(
            "Reference luminosity is zero or missing for "
            f"region={ref['region']}, particle_group={ref['particle_group']}"
        )
    return lumi


def region_label(value: object, region_labels: dict[str, str] | None = None) -> str:
    value = str(value)
    if value == "ALL":
        return value
    if region_labels:
        return str(region_labels.get(value, value))
    return value


def title_text(
    group: tuple[object, ...],
    has_table_name: bool,
    region_labels: dict[str, str] | None = None,
) -> str:
    if has_table_name:
        table_name, study, aggregation_mode, region, particle_group = [
            str(item) for item in group
        ]
    else:
        table_name = "cutflow_efficiency"
        study, aggregation_mode, region, particle_group = [str(item) for item in group]
    pieces = [
        table_name.replace("_", " ").title(),
        study.replace("_", " "),
        aggregation_mode.replace("_", " "),
        region_label(region, region_labels),
        particle_group,
    ]
    return " | ".join(piece for piece in pieces if piece)


def rows_for_group(group_rows: pd.DataFrame) -> list[list[str]]:
    group_rows = group_rows.sort_values("cut_index")
    first = group_rows.iloc[0]
    rows = [
        [
            "1",
            "Total",
            format_count(first.get("display_initial_count", first["initial_count"])),
            "100.00",
        ]
    ]
    for step, (_, row) in enumerate(group_rows.iterrows(), start=2):
        rows.append(
            [
                str(step),
                "\n".join(textwrap.wrap(str(row["cut_label"]), width=26)),
                format_count(row.get("display_relative_count", row["relative_count"])),
                format_efficiency(row["relative_efficiency"]),
            ]
        )
    return rows


def draw_table_page(fig, table_rows: list[list[str]], title: str) -> None:
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    left = 0.035
    right = 0.965
    top = 0.915
    header_y = 0.84
    row_start = 0.715
    base_row_height = 0.055
    row_heights = [
        base_row_height * max(1, row[1].count("\n") + 1)
        for row in table_rows
    ]
    bottom = max(0.08, row_start - sum(row_heights) - 0.025)

    x_step = left + 0.025
    x_selection = left + 0.14
    x_events = right - 0.215
    x_eff = right - 0.025

    ax.text(
        left,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=12,
        family="serif",
    )

    ax.hlines(top, left, right, color="black", linewidth=1.8)
    ax.text(x_step, header_y, "Step", ha="left", va="center", fontsize=16, weight="bold", family="serif")
    ax.text(x_selection, header_y, "Selection", ha="left", va="center", fontsize=16, weight="bold", family="serif")
    ax.text((x_events + x_eff) / 2, header_y + 0.035, "Efficiency", ha="center", va="center", fontsize=16, weight="bold", family="serif")
    ax.text(x_events, header_y - 0.035, "Events", ha="right", va="center", fontsize=15, family="serif")
    ax.text(x_eff, header_y - 0.035, "Eff. (%)", ha="right", va="center", fontsize=15, family="serif")
    ax.hlines(header_y - 0.07, left, right, color="black", linewidth=0.8)

    y = row_start
    for row, row_height in zip(table_rows, row_heights):
        ax.text(x_step, y, row[0], ha="left", va="center", fontsize=15, family="serif")
        ax.text(
            x_selection,
            y,
            row[1],
            ha="left",
            va="center",
            fontsize=15,
            family="serif",
            linespacing=1.15,
        )
        ax.text(x_events, y, row[2], ha="right", va="center", fontsize=15, family="serif")
        ax.text(x_eff, y, row[3], ha="right", va="center", fontsize=15, family="serif")
        y -= row_height

    ax.hlines(bottom, left, right, color="black", linewidth=1.8)


def particle_sort_key(particle_group: object) -> tuple[int, str]:
    particle_group = str(particle_group)
    if particle_group in PARTICLE_ORDER:
        return PARTICLE_ORDER.index(particle_group), particle_group
    return len(PARTICLE_ORDER), particle_group


def is_eventbuilder_particle(particle_group: object) -> bool:
    return str(particle_group).endswith(EVENTBUILDER_SUFFIX)


def particle_family_label(is_eventbuilder: bool) -> str:
    return "EventBuilder" if is_eventbuilder else "Standard"


def particle_label(
    value: object,
    particle_labels: dict[str, str] | None = None,
) -> str:
    value = str(value)
    if particle_labels and value in particle_labels:
        label = str(particle_labels[value])
        if is_eventbuilder_particle(value):
            label = label.replace("EventBuilder ", "").replace(" EventBuilder", "")
            if not label.startswith("MC "):
                label = f"MC {label}"
        return label
    return value.replace(EVENTBUILDER_SUFFIX, "").replace("real_data", "real data").replace("_", " ")


def format_lumi(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        return ""
    return rf"$L_{{ref}} = {float(value):.4g}\ \mathrm{{fb}}^{{-1}}$"


def count_header(particle_group: object) -> str:
    return "Events" if str(particle_group) == "real_data" else "Yield"


def particle_columns_rows(group_rows: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    particles = sorted(
        {str(value) for value in group_rows["particle_group"].dropna().unique()},
        key=particle_sort_key,
    )
    rows_by_particle = {
        str(particle): particle_rows.sort_values("cut_index").reset_index(drop=True)
        for particle, particle_rows in group_rows.groupby("particle_group", sort=False)
    }
    if not particles:
        return [], []

    reference_rows = rows_by_particle[particles[0]]
    table_rows = [["1", "Total"]]
    for particle in particles:
        particle_rows = rows_by_particle.get(particle, pd.DataFrame())
        if particle_rows.empty:
            table_rows[0].extend(["", ""])
            continue
        first = particle_rows.iloc[0]
        table_rows[0].extend(
            [
                format_count(first.get("display_initial_count", first["initial_count"])),
                "100.00",
            ]
        )

    for step, (_, reference_row) in enumerate(reference_rows.iterrows(), start=2):
        row_values = [
            str(step),
            "\n".join(textwrap.wrap(str(reference_row["cut_label"]), width=22)),
        ]
        cut_name = str(reference_row["cut_name"])
        for particle in particles:
            particle_rows = rows_by_particle.get(particle, pd.DataFrame())
            selected = particle_rows.loc[
                particle_rows["cut_name"].astype(str) == cut_name
            ]
            if selected.empty:
                row_values.extend(["", ""])
                continue
            row = selected.iloc[0]
            row_values.extend(
                [
                    format_count(row.get("display_relative_count", row["relative_count"])),
                    format_efficiency(row["relative_efficiency"]),
                ]
            )
        table_rows.append(row_values)

    return particles, table_rows


def draw_particle_columns_table(
    ax,
    group_rows: pd.DataFrame,
    title: str,
    top: float,
    bottom: float,
    left: float = 0.025,
    right: float = 0.985,
    particle_labels: dict[str, str] | None = None,
    reference_lumi_value: float | None = None,
) -> None:
    particles, table_rows = particle_columns_rows(group_rows)
    if not particles or not table_rows:
        ax.text(left, top, f"{title}: no rows", ha="left", va="top", fontsize=10, family="serif")
        return

    n_particles = len(particles)
    rule_top = top - 0.034
    header_y = rule_top - 0.048
    subheader_y = header_y - 0.046
    header_rule_y = subheader_y - 0.036
    font_size = max(7.0, min(10.0, 9.6 - 0.2 * max(n_particles - 4, 0)))
    header_font = max(7.4, font_size - 0.25)

    line_counts = [max(1, row[1].count("\n") + 1) for row in table_rows]
    row_units = [1.15 + 0.9 * (line_count - 1) for line_count in line_counts]
    available_height = header_rule_y - bottom - 0.012
    unit_height = available_height / max(sum(row_units), 1.0)
    min_row_height = 0.054
    if min_row_height * len(row_units) <= available_height:
        row_heights = [
            max(min_row_height, min(0.12, unit_height * row_unit))
            for row_unit in row_units
        ]
    else:
        row_heights = [
            max(0.0, min(0.12, unit_height * row_unit))
            for row_unit in row_units
        ]

    step_width = 0.05
    selection_width = 0.22 if n_particles >= 6 else 0.28
    numeric_width = max(
        0.04,
        (right - left - step_width - selection_width) / max(2 * n_particles, 1),
    )
    x_step = left + 0.006
    x_selection = left + step_width + 0.008
    numeric_left = left + step_width + selection_width
    table_right = numeric_left + 2 * n_particles * numeric_width

    ax.text(left, top, title, ha="left", va="top", fontsize=12, family="serif")
    lumi_text = format_lumi(reference_lumi_value)
    if lumi_text:
        ax.text(right, top, lumi_text, ha="right", va="top", fontsize=10, family="serif")
    ax.hlines(rule_top, left, right, color="black", linewidth=1.5)
    ax.text(x_step, header_y, "Step", ha="left", va="center", fontsize=header_font, weight="bold", family="serif")
    ax.text(x_selection, header_y, "Selection", ha="left", va="center", fontsize=header_font, weight="bold", family="serif")

    vertical_rules = [left, left + step_width, numeric_left, table_right]
    row_rule_positions = [header_rule_y]
    for idx, particle in enumerate(particles):
        pair_left = numeric_left + idx * 2 * numeric_width
        pair_right = pair_left + 2 * numeric_width
        vertical_rules.append(pair_right)
        ax.text(
            (pair_left + pair_right) / 2,
            header_y + 0.005,
            particle_label(particle, particle_labels),
            ha="center",
            va="center",
            fontsize=header_font,
            weight="bold",
            family="serif",
            linespacing=0.9,
        )
        ax.text(pair_left + numeric_width * 0.9, subheader_y, count_header(particle), ha="right", va="center", fontsize=header_font, family="serif")
        ax.text(pair_left + numeric_width * 1.9, subheader_y, "Eff. (%)", ha="right", va="center", fontsize=header_font, family="serif")

    ax.hlines(header_rule_y, left, table_right, color="black", linewidth=0.8)

    y_top = header_rule_y
    for row, effective_row_height in zip(table_rows, row_heights):
        y_bottom = y_top - effective_row_height
        y = (y_top + y_bottom) / 2.0
        ax.text(x_step, y, row[0], ha="left", va="center", fontsize=font_size, family="serif")
        ax.text(x_selection, y, row[1], ha="left", va="center", fontsize=font_size, family="serif", linespacing=1.1)
        values = row[2:]
        for idx in range(n_particles):
            pair_left = numeric_left + idx * 2 * numeric_width
            ax.text(pair_left + numeric_width * 0.86, y, values[2 * idx], ha="right", va="center", fontsize=font_size, family="serif")
            ax.text(pair_left + numeric_width * 1.86, y, values[2 * idx + 1], ha="right", va="center", fontsize=font_size, family="serif")
        row_rule_positions.append(y_bottom)
        y_top = y_bottom

    bottom_rule_y = y_top
    for row_rule_y in row_rule_positions[1:-1]:
        ax.hlines(row_rule_y, left, table_right, color="0.82", linewidth=0.45)
    for x_rule in sorted(set(round(value, 6) for value in vertical_rules)):
        linewidth = 0.75 if x_rule in {round(left, 6), round(table_right, 6)} else 0.45
        color = "black" if x_rule in {round(left, 6), round(table_right, 6)} else "0.78"
        ax.vlines(x_rule, header_rule_y, bottom_rule_y, color=color, linewidth=linewidth)
    ax.hlines(bottom_rule_y, left, table_right, color="black", linewidth=1.5)


def draw_particle_page(
    fig,
    group_rows: pd.DataFrame,
    title: str,
    particle_labels: dict[str, str] | None = None,
    reference_lumi_value: float | None = None,
) -> None:
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw_particle_columns_table(
        ax,
        group_rows,
        title,
        top=0.965,
        bottom=0.08,
        particle_labels=particle_labels,
        reference_lumi_value=reference_lumi_value,
    )


def draw_category_pair_page(
    fig,
    group_rows: pd.DataFrame,
    title: str,
    categories: list[str],
    particle_labels: dict[str, str] | None = None,
    reference_lumi_value: float | None = None,
) -> None:
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.025, 0.985, title, ha="left", va="top", fontsize=12, family="serif")
    lumi_text = format_lumi(reference_lumi_value)
    if lumi_text:
        ax.text(0.985, 0.985, lumi_text, ha="right", va="top", fontsize=10, family="serif")
    for category, top, bottom in zip(categories, [0.92, 0.49], [0.54, 0.08]):
        category_rows = group_rows.loc[group_rows["region"].astype(str) == category]
        draw_particle_columns_table(
            ax,
            category_rows,
            category.replace("_", " ").title(),
            top=top,
            bottom=bottom,
            particle_labels=particle_labels,
            reference_lumi_value=None,
        )


def table_title(
    group: object,
    columns: list[str],
    region_labels: dict[str, str] | None = None,
) -> str:
    if not isinstance(group, tuple):
        group = (group,)
    pieces = []
    for column, value in zip(columns, group):
        value = str(value)
        if column == "table_name":
            value = value.replace("_", " ").title()
        elif column in {"study", "aggregation_mode"}:
            value = value.replace("_", " ")
        elif column == "region":
            value = region_label(value, region_labels)
        pieces.append(value)
    return " | ".join(piece for piece in pieces if piece)


def figure_size_for_group(group_rows: pd.DataFrame, stacked: bool = False) -> tuple[float, float]:
    n_particles = int(group_rows["particle_group"].nunique())
    width = max(13.0, 5.2 + 2.45 * n_particles)
    height = 13.0 if stacked else 7.2
    return width, height


def plot_single_particle_tables(
    data: pd.DataFrame,
    output_pdf_path: Path,
    region_labels: dict[str, str] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    group_columns = ["study", "aggregation_mode", "region", "particle_group"]
    has_table_name = "table_name" in data.columns
    if has_table_name:
        group_columns = ["table_name"] + group_columns
    with PdfPages(output_pdf_path) as pdf:
        for page, (group, group_rows) in enumerate(data.groupby(group_columns, sort=True), start=1):
            fig = plt.figure(figsize=(10.5, 6.4))
            draw_table_page(
                fig,
                rows_for_group(group_rows),
                title_text(group, has_table_name, region_labels),
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def split_particle_families(group_rows: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    families = []
    eventbuilder_mask = group_rows["particle_group"].map(is_eventbuilder_particle)
    real_data_mask = group_rows["particle_group"].astype(str) == "real_data"
    standard_rows = group_rows.loc[~eventbuilder_mask].copy()
    eventbuilder_rows = group_rows.loc[eventbuilder_mask | real_data_mask].copy()
    if not standard_rows.empty:
        families.append((particle_family_label(False), standard_rows))
    if eventbuilder_mask.any():
        families.append((particle_family_label(True), eventbuilder_rows))
    return families


def plot_particle_column_tables(
    data: pd.DataFrame,
    output_pdf_path: Path,
    region_labels: dict[str, str] | None = None,
    particle_labels: dict[str, str] | None = None,
    reference_lumi_value: float | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    has_table_name = "table_name" in data.columns
    table_names = (
        data["table_name"].astype(str)
        if has_table_name
        else pd.Series(["cutflow_efficiency"] * len(data), index=data.index)
    )
    standard_data = data.loc[~table_names.isin(["veto", "us"])].copy()
    veto_data = data.loc[table_names == "veto"].copy()
    us_data = data.loc[table_names == "us"].copy()

    page = 0
    with PdfPages(output_pdf_path) as pdf:
        if not standard_data.empty:
            group_columns = ["study", "aggregation_mode", "region"]
            if has_table_name:
                group_columns = ["table_name"] + group_columns
            for group, group_rows in standard_data.groupby(group_columns, sort=True):
                base_title = table_title(group, group_columns, region_labels)
                for family_label, family_rows in split_particle_families(group_rows):
                    page += 1
                    fig = plt.figure(figsize=figure_size_for_group(family_rows))
                    draw_particle_page(
                        fig,
                        family_rows,
                        f"{base_title} | {family_label}",
                        particle_labels=particle_labels,
                        reference_lumi_value=reference_lumi_value,
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

        for table_name, category_data, categories in [
            ("veto", veto_data, CATEGORY_ORDER["veto"]),
            ("us", us_data, CATEGORY_ORDER["us"]),
        ]:
            if category_data.empty:
                continue
            group_columns = ["study", "aggregation_mode"]
            if has_table_name:
                group_columns = ["table_name"] + group_columns
            for group, group_rows in category_data.groupby(group_columns, sort=True):
                base_title = table_title(group, group_columns, region_labels)
                if not has_table_name:
                    base_title = f"{table_name.title()} | {base_title}"
                for family_label, family_rows in split_particle_families(group_rows):
                    page += 1
                    fig = plt.figure(figsize=figure_size_for_group(family_rows, stacked=True))
                    draw_category_pair_page(
                        fig,
                        family_rows,
                        f"{base_title} | {family_label}",
                        categories,
                        particle_labels=particle_labels,
                        reference_lumi_value=reference_lumi_value,
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)


def read_inputs(input_csv: str | None, input_csvs: list[str]) -> pd.DataFrame:
    frames = []
    if input_csv:
        frames.append(pd.read_csv(input_csv))

    for item in input_csvs:
        if "=" not in item:
            raise ValueError(f"Expected --input-csv value like name=path, got {item!r}")
        table_name, path = item.split("=", 1)
        frame = pd.read_csv(path)
        frame.insert(0, "table_name", table_name)
        frames.append(frame)

    if not frames:
        raise ValueError("Provide --input or at least one --input-csv name=path")
    return pd.concat(frames, ignore_index=True)


def veto_category(region: object) -> str:
    region = str(region)
    if "_no_veto_" in region:
        return "no_veto"
    if "_has_veto_" in region:
        return "has_veto"
    return "unknown_veto"


def us_category(region: object) -> str:
    region = str(region)
    if "_has_us_" in region or region.endswith("_has_us"):
        return "has_us"
    if "_no_us_" in region or region.endswith("_no_us"):
        return "no_us"
    return "unknown_us"


def row_key(row: pd.Series, region_value: str | None = None) -> tuple[str, str, str, str, str]:
    return (
        str(row["study"]),
        str(row["particle_group"]),
        str(row["cut_index"]),
        str(row["cut_name"]),
        str(row["region"] if region_value is None else region_value),
    )


def apply_display_counts_from_lookup(
    data: pd.DataFrame,
    mask: pd.Series,
    lookup: dict[tuple[str, str, str, str, str], dict[str, float]],
) -> None:
    for index, row in data.loc[mask].iterrows():
        values = lookup.get(row_key(row))
        if not values:
            continue
        for column in COUNT_COLUMNS:
            data.at[index, f"display_{column}"] = values[column]


def aggregate_normalized_counts(
    source_rows: pd.DataFrame,
    category_column: str,
) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
    lookup = {}
    if source_rows.empty:
        return lookup

    group_columns = ["study", "particle_group", "cut_index", "cut_name", category_column]
    for group_key, group_rows in source_rows.groupby(group_columns, sort=False):
        study, particle_group, cut_index, cut_name, category = [str(value) for value in group_key]
        lookup[(study, particle_group, cut_index, cut_name, category)] = {
            column: float(group_rows[f"display_{column}"].sum())
            for column in COUNT_COLUMNS
        }
    return lookup


def normalize_display_counts(
    data: pd.DataFrame,
    eval_cfg: dict | None,
    metadata_csv: str | None,
) -> tuple[pd.DataFrame, float | None]:
    data = data.copy()
    for column in COUNT_COLUMNS + ["lumi"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    for column in COUNT_COLUMNS:
        data[f"display_{column}"] = data[column]

    if not eval_cfg or not metadata_csv:
        return data, None

    metadata = pd.read_csv(metadata_csv)
    reference_lumi_value = reference_lumi(metadata, eval_cfg)
    if reference_lumi_value <= 0:
        return data, None

    lumi = pd.to_numeric(data.get("lumi"), errors="coerce").fillna(0.0)
    scale = pd.Series(1.0, index=data.index)
    valid_lumi = lumi > 0
    scale.loc[valid_lumi] = reference_lumi_value / lumi.loc[valid_lumi]
    for column in COUNT_COLUMNS:
        data[f"display_{column}"] = data[column] * scale

    if "table_name" not in data.columns:
        return data, reference_lumi_value

    table_name = data["table_name"].astype(str)
    regional_mask = (
        (table_name == "regional")
        & data["aggregation_mode"].astype(str).isin(
            ["per_region_particle", "fiducial_merged_region_particle"]
        )
    )
    regional_rows = data.loc[regional_mask].copy()
    if regional_rows.empty:
        return data, reference_lumi_value

    overall_source = regional_rows.copy()
    overall_source["_overall_region"] = "ALL"
    overall_lookup = aggregate_normalized_counts(overall_source, "_overall_region")
    overall_mask = (
        (table_name == "overall")
        & (data["aggregation_mode"].astype(str) == "overall_particle")
    )
    apply_display_counts_from_lookup(data, overall_mask, overall_lookup)

    veto_source = regional_rows.copy()
    veto_source["_veto_category"] = veto_source["region"].map(veto_category)
    veto_lookup = aggregate_normalized_counts(veto_source, "_veto_category")
    veto_mask = (
        (table_name == "veto")
        & (data["aggregation_mode"].astype(str) == "veto_category_particle")
    )
    apply_display_counts_from_lookup(data, veto_mask, veto_lookup)

    us_source = regional_rows.copy()
    us_source["_us_category"] = us_source["region"].map(us_category)
    us_lookup = aggregate_normalized_counts(us_source, "_us_category")
    us_mask = (
        (table_name == "us")
        & (data["aggregation_mode"].astype(str) == "us_category_particle")
    )
    apply_display_counts_from_lookup(data, us_mask, us_lookup)

    return data, reference_lumi_value


def ratio(numerator: object, denominator: object) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def binomial_error(numerator: object, denominator: object) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    efficiency = ratio(numerator, denominator)
    return math.sqrt(max(efficiency * (1.0 - efficiency), 0.0) / denominator)


def merge_fiducial_inside_outside_regions(data: pd.DataFrame) -> pd.DataFrame:
    if "table_name" not in data.columns:
        return data

    data = data.copy()
    table_name = data["table_name"].astype(str)
    merge_mask = (
        (table_name == "regional")
        & (data["study"].astype(str) == "fiducial_cut_comparison")
        & (data["aggregation_mode"].astype(str) == "per_region_particle")
        & (data["region"].astype(str).isin(FIDUCIAL_MERGED_REGION_MAP))
    )
    if not bool(merge_mask.any()):
        return data

    passthrough = data.loc[~merge_mask].copy()
    source = data.loc[merge_mask].copy()
    source["_merged_region"] = source["region"].astype(str).map(FIDUCIAL_MERGED_REGION_MAP)
    source["region"] = source["_merged_region"]

    for column in COUNT_COLUMNS + [
        "lumi",
        "n_partitions",
        "n_added_files",
        "n_skipped_files",
    ]:
        if column in source.columns:
            source[column] = pd.to_numeric(source[column], errors="coerce").fillna(0.0)
    for column in COUNT_COLUMNS:
        display_column = f"display_{column}"
        if display_column in source.columns:
            source[display_column] = pd.to_numeric(
                source[display_column],
                errors="coerce",
            ).fillna(0.0)

    group_columns = [
        "table_name",
        "study",
        "aggregation_mode",
        "region",
        "particle_group",
        "cut_index",
        "cut_name",
    ]
    sum_columns = [
        column
        for column in [
            *COUNT_COLUMNS,
            *(f"display_{column}" for column in COUNT_COLUMNS),
            "lumi",
            "n_partitions",
            "n_added_files",
            "n_skipped_files",
        ]
        if column in source.columns
    ]
    first_columns = [
        column
        for column in source.columns
        if column not in set(group_columns + sum_columns + ["_merged_region"])
    ]
    aggregations = {column: "sum" for column in sum_columns}
    aggregations.update({column: "first" for column in first_columns})

    merged = source.groupby(group_columns, sort=False, as_index=False).agg(aggregations)

    if {"cumulative_count", "initial_count"}.issubset(merged.columns):
        merged["cumulative_efficiency"] = [
            ratio(row["cumulative_count"], row["initial_count"])
            for _, row in merged.iterrows()
        ]
        merged["cumulative_efficiency_error"] = [
            binomial_error(row["cumulative_count"], row["initial_count"])
            for _, row in merged.iterrows()
        ]
    if {"relative_count", "initial_count"}.issubset(merged.columns):
        merged["relative_efficiency"] = [
            ratio(row["relative_count"], row["initial_count"])
            for _, row in merged.iterrows()
        ]
        merged["relative_efficiency_error"] = [
            binomial_error(row["relative_count"], row["initial_count"])
            for _, row in merged.iterrows()
        ]

    merged = merged.reindex(columns=data.columns)
    return pd.concat([passthrough, merged], ignore_index=True)


def plot_tables(
    input_csv: str | None,
    input_csvs: list[str],
    output_pdf: str,
    layout: str,
    eval_config: str | None = None,
    metadata_csv: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    data = read_inputs(input_csv, input_csvs)
    if data.empty:
        raise RuntimeError("No rows found in input CSVs")
    region_labels = {}
    particle_labels = {}
    eval_cfg = None
    if eval_config:
        eval_cfg = load_yaml(eval_config)
        region_labels = eval_cfg.get("plot", {}).get("region_labels", {}) or {}
        particle_labels = eval_cfg.get("plot", {}).get("particle_labels", {}) or {}
        if metadata_csv is None:
            metadata_value = eval_cfg.get("metadata", {}).get("partition_csv")
            if metadata_value:
                metadata_csv = str(resolve_path(metadata_value, Path(eval_config).resolve().parents[4]))
    data, reference_lumi_value = normalize_display_counts(data, eval_cfg, metadata_csv)
    data = merge_fiducial_inside_outside_regions(data)
    region_labels = {**FIDUCIAL_MERGED_REGION_LABELS, **region_labels}

    output_pdf_path = Path(output_pdf)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if layout == "single-particle":
        plot_single_particle_tables(data, output_pdf_path, region_labels)
    elif layout == "particle-columns":
        plot_particle_column_tables(
            data,
            output_pdf_path,
            region_labels,
            particle_labels,
            reference_lumi_value,
        )
    else:
        raise ValueError(f"Unknown table layout: {layout}")

    print(f"Wrote cutflow table PDF: {output_pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LaTeX-style cutflow tables from CSV.")
    parser.add_argument("--input", default=None, help="Cutflow summary CSV")
    parser.add_argument(
        "--input-csv",
        action="append",
        default=[],
        help="Named cutflow CSV in the form table_name=path. Can be repeated.",
    )
    parser.add_argument("--output-pdf", required=True, help="Output multi-page PDF")
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Optional evaluation options YAML for plot labels",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional region partition metadata CSV for luminosity normalization",
    )
    parser.add_argument(
        "--layout",
        choices=["single-particle", "particle-columns"],
        default="particle-columns",
        help="Table layout. particle-columns compares particle groups side-by-side.",
    )
    args = parser.parse_args()

    plot_tables(
        input_csv=args.input,
        input_csvs=args.input_csv,
        output_pdf=args.output_pdf,
        layout=args.layout,
        eval_config=args.eval_config,
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()
