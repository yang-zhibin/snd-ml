
rule process_nueAnalysis_eff:
    input: 
        script = f"{PERSONAL_WORK_SPACE}/evaluation/nueAnalysis_eff.py"
    params:
        partition="{partition}",
        nue_eos_dir = f"{EOS_Work_SPACE}/nueAnalysis/",
        metadata_dir = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
    output:
        nueAnalysis_out_path=f"{{EOS_Work_SPACE}}/nueAnalysis/eff_{{partition}}.root"
    wildcard_constraints:
        partition=".+"
    threads:1
    resources:
        runtime=2*60*60,
        mem_mb=8000,
        disk_mb=8000,
        nvidia_gpu=0

    shell:
        r"""
        echo "running nueAnalysis eff calculation"

        # Avoid unbound variable error (only occurs when using Snakemake)
        set +u

        # Set up environment
        source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc11-opt
        export EOSSHIP=root://eosuser.cern.ch/

        # Create a unique temporary directory
        tmp_dir=$(mktemp -d)

        echo "tmp_dir: ${{tmp_dir}}"

        # Run evaluation script
        python {input.script} \
            -p {params.partition} \
            -o "${{tmp_dir}}" \
            -d {params.metadata_dir}


        # Copy results to EOS
        xrdcp -rf "${{tmp_dir}}"/* "{params.nue_eos_dir}/" || {{ echo "xrdcp failed"; exit 1; }}

        # Cleanup
        rm -rf "${{tmp_dir}}"

        """



rule process_nueAnalysis_hist:
    input: 
        script = f"{PERSONAL_WORK_SPACE}/evaluation/nueAnalysis_hist.py"
    params:
        partition="{partition}",
        nue_eos_dir = f"{EOS_Work_SPACE}/nueAnalysis/",
        metadata_dir = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
    output:
        nueAnalysis_out_path=f"{{EOS_Work_SPACE}}/nueAnalysis/hist_{{partition}}.root"
    wildcard_constraints:
        partition=".+"
    threads:1
    resources:
        runtime=2*60*60,
        mem_mb=8000,
        disk_mb=8000,
        nvidia_gpu=0

    shell:
        r"""
        echo "running nueAnalysis hist calculation"

        # Avoid unbound variable error (only occurs when using Snakemake)
        set +u

        # Set up environment
        source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc11-opt
        export EOSSHIP=root://eosuser.cern.ch/

        # Create a unique temporary directory
        tmp_dir=$(mktemp -d)

        echo "tmp_dir: ${{tmp_dir}}"

        # Run evaluation script
        python {input.script} \
            -p {params.partition} \
            -o "${{tmp_dir}}" \
            -d {params.metadata_dir}


        # Copy results to EOS
        xrdcp -rf "${{tmp_dir}}"/* "{params.nue_eos_dir}/" || {{ echo "xrdcp failed"; exit 1; }}

        # Cleanup
        rm -rf "${{tmp_dir}}"

        """



rule process_nueAnalysis_digi:
    input: 
        script = f"{PERSONAL_WORK_SPACE}/evaluation/nueAnalysis_digi.py"
    params:
        partition="{partition}",
        nue_eos_dir = f"{EOS_Work_SPACE}/nueAnalysis/",
        metadata_dir = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
    output:
        nueAnalysis_out_path=f"{{EOS_Work_SPACE}}/nueAnalysis/digi_{{partition}}.root"
    wildcard_constraints:
        partition=".+"
    threads:1
    resources:
        runtime=2*60*60,
        mem_mb=8000,
        disk_mb=8000,
        nvidia_gpu=0

    shell:
        r"""
        echo "running nueAnalysis digi calculation"

        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u  # Avoid unbound variable errors

        echo "Source SNDSW environment script"
        source {env_script_sndsw_nue}
        
        export EOSSHIP=root://eosuser.cern.ch/

        # Create a unique temporary directory
        tmp_dir=$(mktemp -d)

        echo "tmp_dir: ${{tmp_dir}}"

        # Run evaluation script
        python {input.script} \
            -p {params.partition} \
            -o "${{tmp_dir}}" \
            -d {params.metadata_dir}


        # Copy results to EOS
        xrdcp -rf "${{tmp_dir}}"/* "{params.nue_eos_dir}/" || {{ echo "xrdcp failed"; exit 1; }}

        # Cleanup
        rm -rf "${{tmp_dir}}"
        """




        

rule plot_nueAnalysis_hist:
    output:
        nueAnalysis_hist_path = f"{PERSONAL_WORK_SPACE}/evaluation/nueAnalysis/hist/HAD_scale_factor_{{HAD_scale_factor}}__{{base_cut_key}}__{{extra_cut_key}}/{{hist_name}}__{{base_cut_key}}__{{extra_cut_key}}.done"
    params:
        script = f"{PERSONAL_WORK_SPACE}/evaluation/plot_hist_nueAnalysis.py",
        outdir=lambda wc, output: os.path.dirname(output.nueAnalysis_hist_path),
        base_cut=lambda wc: base_cut_map[wc.base_cut_key],
        extra_cut=lambda wc: extra_cut_map[wc.extra_cut_key],
    wildcard_constraints:
        hist_name="|".join(hist_names),
        base_cut_key="|".join(base_cut_map.keys()),
        extra_cut_key="|".join(extra_cut_map.keys()),
        HAD_scale_factor=r"[0-9.]+"
    threads:1
    resources:
        runtime=30*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0,
    shell:
        r'''
        echo "plotting nueAnalysis hist"

        set +u

        source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc11-opt
        export EOSSHIP=root://eosuser.cern.ch/

        # Create temporary working directory
        TMPDIR=$(mktemp -d)
        echo "Using TMPDIR=$TMPDIR"

        # Run script locally
        python {params.script} \
            --feature "{wildcards.hist_name}" \
            --base_cut "{params.base_cut}" \
            --extra_cut "{params.extra_cut}" \
            --had-scale-factor {wildcards.HAD_scale_factor} \
            --outdir "$TMPDIR"


        # Copy PDFs to EOS
        PDF=$(ls "$TMPDIR"/*.pdf)
        xrdcp -f "$PDF" "{params.outdir}/"

        # Cleanup
        rm -rf "$TMPDIR"

        # Mark completion
        touch "{output.nueAnalysis_hist_path}"
    '''
rule find_HAD_scale_factor:
    output:
        nueAnalysis_hist_path = f"{PERSONAL_WORK_SPACE}/evaluation/nueAnalysis/hist/find_HAD_scale_factor_with_differnt_cuts__{{background_model}}__{{scale_factor_target}}__{{scale_application}}__{{extra_cut_key}}/{{base_cut_key}}.done",
    params:
        script = f"{PERSONAL_WORK_SPACE}/evaluation/find_had_scale_factor.py",
        outdir=lambda wc, output: os.path.dirname(output.nueAnalysis_hist_path),
        base_cut=lambda wc: base_cut_map[wc.base_cut_key],
        extra_cut=lambda wc: extra_cut_map[wc.extra_cut_key],
        extra_cut_key=lambda wc: wc.extra_cut_key,
        background_model=lambda wc: wc.background_model,
        scale_factor_target=lambda wc: wc.scale_factor_target,
        normalise_flag=lambda wc: "--normalise" if wc.scale_application == "apply_scale" else "",
    wildcard_constraints:
        background_model="neutral_hadrons|muonDIS",
        scale_factor_target="fixed|data",
        scale_application="apply_scale|no_apply_scale",
        base_cut_key="|".join(base_cut_map.keys()),
        extra_cut_key="|".join(extra_cut_map.keys()),
    threads:1
    resources:
        runtime=30*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0,
    shell:
        r'''
        echo "finding background scale factor"

        set +u

        source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc11-opt
        export EOSSHIP=root://eosuser.cern.ch/

        TMPDIR=$(mktemp -d)
        echo "Using TMPDIR=$TMPDIR"

        python {params.script} \
            --base_cut "{params.base_cut}" \
            --extra_cut "{params.extra_cut}" \
            --extra-cut-key "{params.extra_cut_key}" \
            --background-model "{params.background_model}" \
            --scale-factor-target "{params.scale_factor_target}" \
            {params.normalise_flag} \
            --outdir "$TMPDIR"


        PDF=$(ls "$TMPDIR"/*.pdf)
        xrdcp -f "$PDF" "{params.outdir}/"

        rm -rf "$TMPDIR"

        touch "{output.nueAnalysis_hist_path}"
        '''


rule compare_hadron_muondis:
    output:
        comparison_path = f"{PERSONAL_WORK_SPACE}/evaluation/nueAnalysis/hist/compare_hadron_muondis__{{reference}}__{{muondis_secondary}}__{{base_cut_key}}__{{extra_cut_key}}/{{hist_name}}__{{base_cut_key}}__{{extra_cut_key}}.done"
    params:
        script = f"{PERSONAL_WORK_SPACE}/evaluation/compare_hadron_muondis.py",
        outdir=lambda wc, output: os.path.dirname(output.comparison_path),
        base_cut=lambda wc: base_cut_map[wc.base_cut_key],
        extra_cut=lambda wc: extra_cut_map[wc.extra_cut_key],
        extra_cut_key=lambda wc: wc.extra_cut_key,
    wildcard_constraints:
        hist_name="|".join(compare_hist_names),
        reference="kaon|neutron|data",
        muondis_secondary="all|neutral_kaon|charged_kaon|kaon_all|neutron|antineutron|neutron_all|gamma|proton|pion_charged",
        base_cut_key="|".join(base_cut_map.keys()),
        extra_cut_key="|".join(extra_cut_map.keys()),
    threads:1
    resources:
        runtime=30*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0,
    shell:
        r'''
        echo "comparing neutral-hadron MC with muonDIS"

        set +u

        source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc11-opt
        export EOSSHIP=root://eosuser.cern.ch/

        TMPDIR=$(mktemp -d)
        echo "Using TMPDIR=$TMPDIR"

        python {params.script} \
            --reference "{wildcards.reference}" \
            --muondis-secondary "{wildcards.muondis_secondary}" \
            --feature "{wildcards.hist_name}" \
            --base_cut "{params.base_cut}" \
            --extra_cut "{params.extra_cut}" \
            --extra-cut-key "{params.extra_cut_key}" \
            --outdir "$TMPDIR"

        OUTPUTS=$(find "$TMPDIR" -maxdepth 1 -type f \( -name "*.pdf" -o -name "*.txt" \))
        mkdir -p "{params.outdir}"
        for output_file in $OUTPUTS; do
            xrdcp -f "$output_file" "{params.outdir}/"
        done

        rm -rf "$TMPDIR"

        touch "{output.comparison_path}"
        '''


rule plot_saved_event_displays:
    input:
        script=f"{PERSONAL_WORK_SPACE}/evaluation/plot_saved_events.py",
        eff_files=expand(f"{EOS_Work_SPACE}/nueAnalysis/eff_{{partition}}.root", partition=nueAnalysis_partitions),
    output:
        done=saved_event_display_target,
    params:
        input_dir=f"{EOS_Work_SPACE}/nueAnalysis",
        outdir=lambda wc, output: os.path.dirname(output.done),
        metadata_dir=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated",
        max_events=20,
        output_mode="vector_pdf",
    threads: 1
    resources:
        runtime=60*60,
        mem_mb=4000,
        disk_mb=4000,
        nvidia_gpu=0,
    shell:
        r'''
        echo "plotting saved event displays"
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u  # Avoid unbound variable errors

        source {env_script_sndsw_nue}
        export EOSSHIP=root://eosuser.cern.ch/

        mkdir -p "{params.outdir}"

        python {input.script} \
            "{params.input_dir}" \
            --outdir "{params.outdir}" \
            --metadata-dir "{params.metadata_dir}" \
            --max-events {params.max_events} \
            --max-events-real-data all \
            --backend sndsw \
            --output-mode {params.output_mode}

        touch "{output.done}"
        '''


rule nueAnalysis_step_cut_hist:
    output:
        nueAnalysis_hist_path = f"{PERSONAL_WORK_SPACE}/evaluation/nueAnalysis/hist/HAD_scale_factor_{{HAD_scale_factor}}__step_cut/{{hist_name}}__{{base_cut_key}}__{{extra_cut_key}}.done"
    params:
        script = f"{PERSONAL_WORK_SPACE}/evaluation/plot_hist_nueAnalysis.py",
        outdir=lambda wc, output: os.path.dirname(output.nueAnalysis_hist_path),
        base_cut=lambda wc: base_cut_map[wc.base_cut_key],
        extra_cut=lambda wc: extra_cut_map[wc.extra_cut_key],
    wildcard_constraints:
        hist_name="|".join(hist_names),
        base_cut_key="|".join(base_cut_map.keys()),
        extra_cut_key="|".join(extra_cut_map.keys()),
        HAD_scale_factor=r"[0-9.]+"
    threads:1
    resources:
        runtime=30*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0,
    shell:
        r'''
        echo "plotting nueAnalysis hist"

        set +u

        source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc11-opt
        export EOSSHIP=root://eosuser.cern.ch/

        # Create temporary working directory
        TMPDIR=$(mktemp -d)
        echo "Using TMPDIR=$TMPDIR"

        # Run script locally
        python {params.script} \
            --feature "{wildcards.hist_name}" \
            --base_cut "{params.base_cut}" \
            --extra_cut "{params.extra_cut}" \
            --had-scale-factor {wildcards.HAD_scale_factor} \
            --outdir "$TMPDIR"


        # Copy PDFs to EOS
        PDF=$(ls "$TMPDIR"/*.pdf)
        xrdcp -f "$PDF" "{params.outdir}/"

        # Cleanup
        rm -rf "$TMPDIR"

        # Mark completion
        touch "{output.nueAnalysis_hist_path}"
    '''
