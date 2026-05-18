
rule generate_raw_metadata:
    input:
        script = f'{PERSONAL_WORK_SPACE}/snakemake/metadata/generate_path_file.py'
    output:
        raw_metadata = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/raw/{{metadata_csv}}"
    params:
        script = f'{PERSONAL_WORK_SPACE}/snakemake/metadata/generate_path_file.py'
    threads:1
    resources:
        runtime=5*60*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        """
        echo "Generating raw metadata"
        #avoid unbound variable error (only occur when using snakemake)
        set +u 
        source {env_script_lcg}
        python {params.script} -o {output.raw_metadata} -f
        """


rule update_metadata:
    input:
        raw_metadata = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/raw/{{metadata_csv}}",
        script = f'{PERSONAL_WORK_SPACE}/snakemake/metadata/update_path_file.py',
    output:
        updated_metadata = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/{{metadata_csv}}"
    wildcard_constraints:
        metadata_csv=r"(?!.*_subset\.csv$)(?!.*_skim_runs_metadata.*\.csv$).*\.csv$" #aviod ambiguous with generate subset metadata
    threads:1
    resources:
        runtime=45*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        """
        echo "Updating metadata"
        #avoid unbound variable error (only occur when using snakemake)
        set +u 
        source {env_script_lcg}
        python {input.script} -i {input.raw_metadata} -o {output.updated_metadata}
        """

rule skim_real_data_2024:
    input:
        metadata=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/real_data_2024_metadata.csv",
        script=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/get_subset_metadata.py",
    output:
        full=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/real_data_2024_skim_runs_metadata.csv",
        subset=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/real_data_2024_skim_runs_metadata_subset.csv"
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        """
        echo "Creating 2024 skimmed real-data metadata"
        set +u
        source {env_script_lcg}
        python {input.script} \
            real-data \
            --input {input.metadata} \
            --output-full {output.full} \
            --output-subset {output.subset}
        """


rule subset_neutral_bkg:
    input:
        metadata=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/{{particle}}_FTFP_BERT_metadata.csv",
        script=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/get_subset_metadata.py",
    output:
        subset=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/{{particle}}_FTFP_BERT_metadata_subset.csv"
    wildcard_constraints:
        particle="MC_kaon|MC_neutron"
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        """
        echo "Creating neutral background subset for {wildcards.particle}"
        set +u
        source {env_script_lcg}
        python {input.script} \
            neutral-bkg \
            --input {input.metadata} \
            --output {output.subset}
        """


rule generate_train_metadata:
    input:
        script = f'{PERSONAL_WORK_SPACE}/snakemake/metadata/generate_train_dataset.py',
    output:
        train_metadata = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/{{train_split}}_{{split_dataset}}.csv"
    params:
        train_set_list = " ".join(train_metadata_csv_list),
        metadata_dir = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated",
        split_dataset = split_dataset
    threads:1
    wildcard_constraints:
        split_dataset = r"neutral_hadron|neutrino|all"
    resources:
        runtime=45*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        """
        echo "generateing trianing set metadata"
        #avoid unbound variable error (only occur when using snakemake)
        set +u 
        source {env_script_lcg}
        python {input.script} \
            -o {output.train_metadata} \
            -m {params.train_set_list} \
            -i {params.metadata_dir} \
            --split-dataset {params.split_dataset}
        """
