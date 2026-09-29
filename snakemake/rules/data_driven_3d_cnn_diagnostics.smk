DD3D_DIAGNOSTICS_DIR = f"{DD3D_META}/diagnostics"
DD3D_EVENT_DISPLAY_DIR = f"{DD3D_DIAGNOSTICS_DIR}/3d_event_displays"
DD3D_EVENT_DISPLAY_FILES = [
    f"{DD3D_EVENT_DISPLAY_DIR}/{name}"
    for name in [
        "training_events_3d_cv0.png",
        "training_events_3d_cv0.pdf",
        "selected_events.csv",
        "CC_numu_examples.png",
        "CC_nue_examples.png",
        "NC_examples.png",
        "neutral_kaon_examples.png",
        "neutron_examples.png",
        "passing_muon_examples.png",
        "muon_DIS_examples.png",
    ]
]
DD3D_DIAGNOSTIC_FILES = [
    f"{DD3D_DIAGNOSTICS_DIR}/{name}"
    for name in [
        "dataset_report.md",
        "population_table.csv",
        "population_table.md",
        "fold_balance_metrics.csv",
        "population_by_class_and_fold.png",
        "class_fraction_by_fold.png",
        "fold_balance_ratio.png",
        "domain_composition.png",
        "source_groups_by_fold.png",
        "luminosity_by_class_and_fold.png",
        "muondis_density_table.csv",
        "muondis_density_table.md",
        "muondis_density_by_fold.csv",
        "muondis_cv_sampling_multipliers.csv",
        "muondis_density_counts.png",
        "muondis_natural_vs_uniform.png",
        "muondis_sampling_multiplier.png",
        "muondis_density_by_fold.png",
    ]
] + DD3D_EVENT_DISPLAY_FILES


rule build_data_driven_3d_cnn_diagnostics_v1:
    input:
        config=DD3D_CONFIG,
        validation=DD3D_VALIDATION,
        folded_catalog=DD3D_FOLDED_CATALOG,
        fold_summary=DD3D_FOLD_SUMMARY,
        density_manifest=DD3D_DENSITY_MANIFEST,
        fold_views=DD3D_FOLD_VIEWS,
        report_script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_data_driven_3d_cnn/"
            "build_dataset_report.py"
        ),
        population_script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_data_driven_3d_cnn/"
            "plot_population_and_folds.py"
        ),
        density_script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_data_driven_3d_cnn/"
            "plot_muondis_density_sampling.py"
        ),
        common_script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_data_driven_3d_cnn/"
            "plotting_common.py"
        ),
        event_display_script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_data_driven_3d_cnn/"
            "plot_training_event_examples.py"
        ),
        fold_dataset_script=(
            f"{PERSONAL_WORK_SPACE}/train_3d/src/train_3d/data/"
            "data_driven_fold_dataset.py"
        ),
        event_reader_script=(
            f"{PERSONAL_WORK_SPACE}/train_3d/src/train_3d/data/"
            "scifi_hit_event_dataset.py"
        ),
    output:
        DD3D_DIAGNOSTIC_FILES
    resources:
        runtime=10 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/matplotlib-${{USER}}"
        PYTHONDONTWRITEBYTECODE=1 python3 {input.event_display_script} \
          --fold-views {input.fold_views} \
          --output-dir {DD3D_EVENT_DISPLAY_DIR} \
          --cv-fold 0 \
          --seed 42 \
          --examples-per-class 3 \
          --reader-backend pyroot
        PYTHONDONTWRITEBYTECODE=1 python3 {input.report_script} \
          --config {input.config} \
          --validation-report {input.validation} \
          --folded-catalog {input.folded_catalog} \
          --density-index-manifest {input.density_manifest} \
          --output-dir {DD3D_DIAGNOSTICS_DIR}
        """


rule data_driven_3d_cnn_diagnostics_v1:
    input:
        DD3D_DIAGNOSTIC_FILES
