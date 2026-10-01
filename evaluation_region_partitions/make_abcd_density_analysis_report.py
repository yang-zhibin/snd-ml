#!/usr/bin/env python3
"""Render the validated Region V1 Step 3 analysis CSV as a LaTeX report."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import yaml


SAMPLES = [
    ("CC_nue", "cc_nue", r"$\mathrm{CC}\,\nu_e$", "target signal"),
    ("CC_numu", "cc_numu", r"$\mathrm{CC}\,\nu_\mu$", "neutrino background"),
    ("NC_nue", "nc_nue", r"$\mathrm{NC}\,\nu_e$", "neutrino background"),
    ("NC_numu", "nc_numu", r"$\mathrm{NC}\,\nu_\mu$", "neutrino background"),
]

REQUIRED_BASE_COLUMNS = {
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
    "neutrino_background_sum_weights",
    "neutrino_background_sum_weights_squared",
    "neutrino_background_effective_entries",
    "standard_neutrino_sum_weights",
    "standard_neutrino_sum_weights_squared",
    "standard_neutrino_effective_entries",
    "cc_nue_fraction_of_standard_neutrino_mc",
    "cc_nue_signal_fraction_of_data",
    "source_validation_status",
}

SAMPLE_SUFFIXES = {
    "raw_mc_events",
    "sum_weights",
    "sum_weights_squared",
    "effective_entries",
    "source_mc_luminosity",
    "normalization_weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-tex", required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_table(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    required = set(REQUIRED_BASE_COLUMNS)
    for _, prefix, _, _ in SAMPLES:
        required.update(f"{prefix}_{suffix}" for suffix in SAMPLE_SUFFIXES)
    missing = required - set(columns)
    if missing:
        raise ValueError("Input table is missing columns: " + ", ".join(sorted(missing)))
    if not rows:
        raise ValueError("Input table is empty")
    if any(row["source_validation_status"] != "PASS" for row in rows):
        raise ValueError("Input table contains a non-PASS source validation status")
    return columns, rows


def tex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def integer(value: str) -> str:
    return f"{int(round(float(value))):,}"


def number(value: str, significant: int = 5) -> str:
    numeric = float(value)
    if numeric == 0:
        return "0"
    if abs(numeric) >= 10000 or abs(numeric) < 0.001:
        return f"{numeric:.{significant - 1}e}"
    return f"{numeric:.{significant}g}"


def yield_with_error(row: dict[str, str], prefix: str) -> str:
    yield_value = float(row[f"{prefix}_sum_weights"])
    error = math.sqrt(float(row[f"{prefix}_sum_weights_squared"]))
    return f"{number(str(yield_value))} $\\pm$ {number(str(error), 3)}"


def density_interval(row: dict[str, str]) -> str:
    if row["row_type"] == "underflow":
        return rf"$x < {int(round(float(row['bin_high']))):,}$"
    low = int(round(float(row["bin_low"])))
    high = int(round(float(row["bin_high"])))
    return f"{low:,}--{high:,}"


def inclusive_summary(inclusive: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{landscape}",
        r"\section{Inclusive region summary}",
        r"\begin{longtable}{lrrrrrr}",
        r"\caption{Inclusive data and luminosity-normalized standard-neutrino MC.}\label{tab:inclusive}\\",
        r"\toprule",
        r"Region & $L_{\mathrm{data}}$ & $N_{\mathrm{data}}$ & $S_{\mathrm{CC}\nu_e}$ & $B_{\nu}$ & $S+B$ & $S/N_{\mathrm{data}}$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Region & $L_{\mathrm{data}}$ & $N_{\mathrm{data}}$ & $S_{\mathrm{CC}\nu_e}$ & $B_{\nu}$ & $S+B$ & $S/N_{\mathrm{data}}$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in inclusive:
        fraction = 100.0 * float(row["cc_nue_signal_fraction_of_data"])
        lines.append(
            " & ".join(
                [
                    tex_escape(row["region_label"]),
                    number(row["data_exposure"], 7),
                    integer(row["data_raw_events"]),
                    number(row["cc_nue_sum_weights"]),
                    number(row["neutrino_background_sum_weights"]),
                    number(row["standard_neutrino_sum_weights"]),
                    f"{fraction:.4g}\\%",
                ]
            )
            + r" \\" 
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\begin{minipage}{\linewidth}",
            r"\small\textbf{Definitions used in Table~\ref{tab:inclusive}.}",
            r"\textbf{Region} is the short label for one of the eight mutually exclusive Region V1 selections. "
            r"$L_{\mathrm{data}}$ is the real-data exposure represented by that region. "
            r"$N_{\mathrm{data}}$ is the inclusive unweighted data count. "
            r"$S_{\mathrm{CC}\nu_e}$ is the luminosity-normalized target-signal MC yield. "
            r"$B_\nu$ is the sum of the normalized $\mathrm{CC}\nu_\mu$, $\mathrm{NC}\nu_e$, and $\mathrm{NC}\nu_\mu$ background yields. "
            r"$S+B$ is $S_{\mathrm{CC}\nu_e}+B_\nu$. "
            r"$S/N_{\mathrm{data}}$ is the nominal $\mathrm{CC}\nu_e$ signal-MC yield divided by the observed data count, reported as a percentage. "
            r"The MC totals are diagnostic expectations and are not subtracted from data.",
            r"\end{minipage}",
            r"\end{landscape}",
        ]
    )
    return "\n".join(lines)


def inclusive_mc_audit(inclusive: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{landscape}",
        r"\section{Inclusive MC normalization and statistical audit}",
        r"\scriptsize",
        r"\begin{longtable}{llrrrrrr}",
        r"\caption{Per-sample raw entries, normalization, weighted yields, and MC statistical information.}\label{tab:mc-audit}\\",
        r"\toprule",
        r"Region & Sample & $N_{\mathrm{MC}}$ & $L_{\mathrm{MC}}$ & $w$ & $\sum w$ & $\sum w^2$ & $N_{\mathrm{eff}}$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Region & Sample & $N_{\mathrm{MC}}$ & $L_{\mathrm{MC}}$ & $w$ & $\sum w$ & $\sum w^2$ & $N_{\mathrm{eff}}$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in inclusive:
        for _, prefix, display, _ in SAMPLES:
            lines.append(
                " & ".join(
                    [
                        tex_escape(row["region_label"]),
                        display,
                        integer(row[f"{prefix}_raw_mc_events"]),
                        number(row[f"{prefix}_source_mc_luminosity"]),
                        number(row[f"{prefix}_normalization_weight"]),
                        number(row[f"{prefix}_sum_weights"]),
                        number(row[f"{prefix}_sum_weights_squared"]),
                        number(row[f"{prefix}_effective_entries"]),
                    ]
                )
                + r" \\" 
            )
        lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\begin{minipage}{\linewidth}",
            r"\small\textbf{Definitions used in Table~\ref{tab:mc-audit}.}",
            r"\textbf{Sample} identifies the target signal $\mathrm{CC}\nu_e$ or one of the three standard-neutrino backgrounds. "
            r"$N_{\mathrm{MC}}$ is the unweighted selected MC entry count. "
            r"$L_{\mathrm{MC}}$ is the complete production's equivalent source luminosity. "
            r"The constant event normalization is $w=L_{\mathrm{data}}/L_{\mathrm{MC}}$. "
            r"The expected yield is $\sum w$ and its MC statistical variance is $\sum w^2$. "
            r"The effective number of MC entries is $N_{\mathrm{eff}}=(\sum w)^2/(\sum w^2)$, or zero when $\sum w^2=0$.",
            r"\end{minipage}",
            r"\end{landscape}",
        ]
    )
    return "\n".join(lines)


def binned_tables(rows: list[dict[str, str]], region_order: list[str]) -> str:
    sections: list[str] = []
    for region_index, region in enumerate(region_order):
        region_rows = [
            row
            for row in rows
            if row["region"] == region and row["row_type"] != "inclusive"
        ]
        label = region_rows[0]["region_label"]
        sections.extend(
            [
                r"\begin{landscape}",
                *([r"\section{Per-bin density tables}"] if region_index == 0 else []),
                rf"\subsection{{{tex_escape(label)}: {tex_escape(region)}}}",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3pt}",
                r"\begin{longtable}{lrrrrrrr}",
                rf"\caption{{Materialized density-bin yields for {tex_escape(label)}. MC entries are luminosity-normalized; uncertainties shown are $\sqrt{{\sum w^2}}$.}}\\",
                r"\toprule",
                r"Density interval & Data & $\mathrm{CC}\nu_e$ & $\mathrm{CC}\nu_\mu$ & $\mathrm{NC}\nu_e$ & $\mathrm{NC}\nu_\mu$ & $B_\nu$ & $S+B$ \\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"Density interval & Data & $\mathrm{CC}\nu_e$ & $\mathrm{CC}\nu_\mu$ & $\mathrm{NC}\nu_e$ & $\mathrm{NC}\nu_\mu$ & $B_\nu$ & $S+B$ \\",
                r"\midrule",
                r"\endhead",
            ]
        )
        for row in region_rows:
            background = float(row["neutrino_background_sum_weights"])
            background_error = math.sqrt(
                float(row["neutrino_background_sum_weights_squared"])
            )
            total = float(row["standard_neutrino_sum_weights"])
            total_error = math.sqrt(float(row["standard_neutrino_sum_weights_squared"]))
            sections.append(
                " & ".join(
                    [
                        density_interval(row),
                        integer(row["data_raw_events"]),
                        yield_with_error(row, "cc_nue"),
                        yield_with_error(row, "cc_numu"),
                        yield_with_error(row, "nc_nue"),
                        yield_with_error(row, "nc_numu"),
                        f"{number(str(background))} $\\pm$ {number(str(background_error), 3)}",
                        f"{number(str(total))} $\\pm$ {number(str(total_error), 3)}",
                    ]
                )
                + r" \\" 
            )
        sections.extend([r"\bottomrule", r"\end{longtable}"])
        if region_index == 0:
            sections.extend(
                [
                    r"\begin{minipage}{\linewidth}",
                    r"\small\textbf{Definitions first used in the per-bin tables.}",
                    r"\textbf{Density interval} is a lower-inclusive, upper-exclusive bin of \texttt{density\_sndsw\_scifi}; the separate underflow row contains $x<0$. "
                    r"\textbf{Data} is the raw integer event count in the bin. "
                    r"Each sample column displays the normalized yield $Y=\sum w$ followed by its MC-only statistical uncertainty $\sqrt{\sum w^2}$. "
                    r"$B_\nu$ and $S+B$ have the meanings defined below Table~\ref{tab:inclusive}; their variances are sums of the independent component $\sum w^2$ values. "
                    r"Overflow is already folded into the final interval $39{,}500$--$40{,}000$.",
                    r"\end{minipage}",
                ]
            )
        sections.append(r"\end{landscape}")
    return "\n".join(sections)


def definitions(input_table: str, rows: list[dict[str, str]]) -> str:
    first = rows[0]
    sample_lines = []
    for key, _, display, role in SAMPLES:
        sample_lines.append(rf"\item {display} (\texttt{{{tex_escape(key)}}}): {role}.")
    return "\n".join(
        [
            r"\section{Complete data dictionary and calculation methods}",
            r"\sloppy",
            r"\subsection{Scope}",
            r"This report is a human-readable rendering of the validated Step~3 input table. It does not contain an ABCD prediction, transfer factor, $\kappa$, closure test, merged binning, hypothesis test, or likelihood fit.",
            rf"The authoritative machine-readable source is \path{{{input_table}}}.",
            rf"Table version: \texttt{{{tex_escape(first['table_version'])}}}; binning version: \texttt{{{tex_escape(first['binning_version'])}}}; partition/region versions: \texttt{{{tex_escape(first['partition_version'])}}}/\texttt{{{tex_escape(first['region_version'])}}}.",
            r"\subsection{Samples}",
            r"\begin{itemize}",
            *sample_lines,
            r"\end{itemize}",
            r"The combined neutrino-background quantity $B_\nu$ is the sum of the three background samples. The standard-neutrino total $S+B$ additionally includes the target $\mathrm{CC}\,\nu_e$ sample. These MC totals are diagnostic expectations; they are not subtracted from data in the data-driven ABCD estimate.",
            r"\subsection{Binning and rows}",
            r"The observable $x$ is \texttt{density\_sndsw\_scifi}. The frozen binning has 80 uniform bins on $[0,40000)$ with width 500. Underflow is retained as a separate row. ROOT overflow was folded into the final regular bin $[39500,40000)$ when the source histograms were produced; therefore there is no separate overflow row. Each region also has one inclusive CSV row equal to underflow plus all 80 regular bins.",
            r"\texttt{row\_type} is \texttt{underflow}, \texttt{bin}, or \texttt{inclusive}. \texttt{root\_bin\_index} is 0 for underflow and 1--80 for regular bins; it is blank for an inclusive row. \texttt{bin\_low} and \texttt{bin\_high} are the lower-inclusive and upper-exclusive regular-bin edges and are blank where an edge is not applicable. \texttt{overflow\_folded\_into\_last\_bin} records the flow policy.",
            r"\subsection{Data quantities}",
            r"$L_{\mathrm{data}}$ (\texttt{data\_exposure}) is the real-data exposure represented by the region. No-veto regions use 19.780680927948 exposure units; veto-active regions use 1.960319252235. $N_{\mathrm{data}}$ (\texttt{data\_raw\_events}) is the unweighted integer count used as a Poisson observation in later inference.",
            r"\subsection{MC normalization and statistical quantities}",
            r"For sample $s$ in region $R$, \texttt{<prefix>\_raw\_mc\_events} is the unweighted selected MC entry count $N_{s,R}$. \texttt{<prefix>\_source\_mc\_luminosity} is the complete production's equivalent source luminosity $L_{s}^{\mathrm{MC}}$, not the sum of only non-empty output partitions.",
            r"No event-level physics-weight branch exists in this production. The constant normalization is",
            r"\[w_{s,R}=\frac{L_R^{\mathrm{data}}}{L_s^{\mathrm{MC}}}.\]",
            r"It is stored as \texttt{<prefix>\_normalization\_weight}. Electron-flavour samples use $L_s^{\mathrm{MC}}=14640$; muon-flavour samples use $L_s^{\mathrm{MC}}=13350$.",
            r"The normalized yield and MC statistical variance are",
            r"\[Y_{s,R,i}=\sum_{e\in(s,R,i)}w_{s,R},\qquad V_{s,R,i}=\sum_{e\in(s,R,i)}w_{s,R}^{2}.\]",
            r"They are stored as \texttt{<prefix>\_sum\_weights} and \texttt{<prefix>\_sum\_weights\_squared}. The PDF displays $Y\pm\sqrt{V}$; this is MC statistical uncertainty only, not a total systematic uncertainty.",
            r"The effective MC entries are",
            r"\[N_{\mathrm{eff}}=\frac{(\sum w)^2}{\sum w^2},\]",
            r"stored as \texttt{<prefix>\_effective\_entries}; it is zero when $\sum w^2=0$. The prefixes are \texttt{cc\_nue}, \texttt{cc\_numu}, \texttt{nc\_nue}, and \texttt{nc\_numu}.",
            r"\subsection{Combined quantities and fractions}",
            r"\texttt{neutrino\_background\_sum\_weights} is the sum of the three background yields. \texttt{neutrino\_background\_sum\_weights\_squared} is the sum of their independent MC variances, and \texttt{neutrino\_background\_effective\_entries} is computed from those combined sums.",
            r"\texttt{standard\_neutrino\_sum\_weights}, \texttt{standard\_neutrino\_sum\_weights\_squared}, and \texttt{standard\_neutrino\_effective\_entries} are the analogous quantities after including $\mathrm{CC}\,\nu_e$.",
            r"\texttt{cc\_nue\_fraction\_of\_standard\_neutrino\_mc} is $Y_{\mathrm{CC}\nu_e}/(S+B)$. \texttt{cc\_nue\_signal\_fraction\_of\_data} is $Y_{\mathrm{CC}\nu_e}/N_{\mathrm{data}}$. Fractions are blank when their denominator is zero.",
            r"\subsection{Provenance and validation fields}",
            r"\texttt{table\_version}, \texttt{binning\_version}, \texttt{partition\_version}, and \texttt{region\_version} identify the contracts used to produce a row. \texttt{region} is the full machine-readable region name and \texttt{region\_label} is its short display label. \texttt{source\_validation\_status=PASS} means the source ROOT and summary products passed binning, exposure, integer-count, normalization, $\sum w$, $\sum w^2$, and exact table-reconstruction checks before the CSV was written.",
        ]
    )


def document(config: dict, input_table: str, rows: list[dict[str, str]]) -> str:
    table_versions = {row["table_version"] for row in rows}
    if table_versions != {str(config["table_version"])}:
        raise ValueError(f"Unexpected table versions: {table_versions}")
    region_order = list(dict.fromkeys(row["region"] for row in rows))
    inclusive = [row for row in rows if row["row_type"] == "inclusive"]
    counts = Counter(row["row_type"] for row in rows)
    expected_rows = len(region_order) * 82
    if len(rows) != expected_rows or counts != {"underflow": 8, "bin": 640, "inclusive": 8}:
        raise ValueError(f"Unexpected production table structure: {len(rows)} rows, {counts}")
    preamble = "\n".join(
        [
            r"\documentclass[10pt,a4paper]{article}",
            r"\usepackage[margin=1.5cm]{geometry}",
            r"\usepackage{booktabs,longtable,array,pdflscape}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[hidelinks]{hyperref}",
            r"\usepackage{xcolor}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{5pt}",
            r"\begin{document}",
            rf"\title{{{tex_escape(config['title'])}}}",
            r"\author{SND@LHC analysis workflow}",
            r"\date{}",
            r"\maketitle",
            r"\begin{center}\fcolorbox{black}{gray!10}{\parbox{0.92\textwidth}{\textbf{Step 3 materialization only.} This document contains validated input counts and normalized MC diagnostics. It contains no ABCD closure result or background prediction.}}\end{center}",
            r"\tableofcontents",
            r"\clearpage",
        ]
    )
    return "\n\n".join(
        [
            preamble,
            inclusive_summary(inclusive),
            inclusive_mc_audit(inclusive),
            binned_tables(rows, region_order),
            definitions(input_table, rows),
            r"\end{document}",
        ]
    ) + "\n"


def atomic_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    _, rows = load_table(args.input_table)
    content = document(config, str(Path(args.input_table).resolve()), rows)
    atomic_text(args.output_tex, content)
    print(f"Wrote LaTeX report with {len(rows)} materialized rows to {args.output_tex}")


if __name__ == "__main__":
    main()
