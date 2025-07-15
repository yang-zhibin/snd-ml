import pathlib
import re
import pandas as pd
import matplotlib.pyplot as plt


def plot_events_vs_energy(
    csv_paths,
    *,
    aggregate=True,
    show=True,
    save_dir=None,
    logy=False, 
):
    """
    Make one bar plot per `data_type`, with bars spanning [E_low, E_high].

    Parameters
    ----------
    csv_paths : list[str or pathlib.Path]
        One or more CSV files to merge.
    aggregate : bool, default True
        If True, rows that share identical (E_low, E_high) within a given
        `data_type` are summed into a single bar.  Set False to keep every
        row separate (bars may overlap if duplicates exist).
    show : bool, default True
        Whether to display each figure interactively.
    save_dir : str or pathlib.Path, optional
        Folder in which to save each plot as “{data_type}.png”.
    """
    # ------------------------------------------------------------------
    # 1) Load all CSVs into a single DataFrame
    # ------------------------------------------------------------------
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 2) Split the `subfolder` column into useful pieces
    #    Pattern:  {model}/{particle}_{E_low}_{E_high}
    # ------------------------------------------------------------------
    pat = re.compile(r"""
        (?P<model>[^/]+)/                    # mc_model_type
        (?P<particle>[^_]+)_                 # particle
        (?P<E_low>\d+\.?\d*)_                # E_low
        (?P<E_high>\d+\.?\d*)                # E_high
        (?:_.*)?                             # optional suffix
    """, re.VERBOSE)


    def _parse(row):
        m = pat.fullmatch(row["subfolder"])
        if m is None:
            raise ValueError(f"Unparsable subfolder: {row['subfolder']}")
        gd = m.groupdict()
        return pd.Series(
            {
                "mc_model": gd["model"],
                "particle": gd["particle"],
                "E_low": float(gd["E_low"]),
                "E_high": float(gd["E_high"]),
            }
        )

    df = df.join(df.apply(_parse, axis=1))

    # ------------------------------------------------------------------
    # 3) One plot per data_type
    # ------------------------------------------------------------------
    for dtype, g in df.groupby("data_type"):
        if aggregate:
            plot_df = (
                g.groupby(["E_low", "E_high"], as_index=False)["n_event"].sum()
            )
        else:
            plot_df = g[["E_low", "E_high", "n_event"]].copy()

        # Ensure bars are ordered left→right
        plot_df.sort_values("E_low", inplace=True)

        plt.figure()
        for _, row in plot_df.iterrows():
            width = row.E_high - row.E_low
            plt.bar(
                row.E_low,
                row.n_event,
                width=width,
                align="edge",
                edgecolor="k",
                linewidth=0.5,
            )

        plt.title(f"{dtype}: n_event per energy bin")
        plt.xlabel("Energy (GeV)")
        plt.ylabel("events")
        if logy:                      
            plt.yscale("log")    
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        if save_dir:
            save_dir = pathlib.Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / f"{dtype}_n_event_vs_energy.png"
            plt.savefig(out, dpi=150)

        if show:
            plt.show()
        else:
            plt.close()
            
            
def main():
    metadata_csv_list = [
        "MC_kaon_FTFP_BERT_metadata.csv",
        # "MC_kaon_QGSP_BERT_HP_PEN_metadata.csv",
        "MC_neutron_FTFP_BERT_metadata.csv",
        # "MC_neutron_QGSP_BERT_HP_PEN_metadata.csv",
    ]
    root_dir = "/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated"
    save_dir = "./plots"

    csv_paths = [f"{root_dir}/{fname}" for fname in metadata_csv_list]

    plot_events_vs_energy(
        csv_paths=csv_paths,
        aggregate=True,
        show=False,
        save_dir=save_dir,
        logy=True, 
    )

    

if __name__ == "__main__": 
    main()


