
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

        # Local completion marker for Snakemake
        touch "{output.nueAnalysis_out_path}"
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

        # Local completion marker for Snakemake
        touch "{output.nueAnalysis_out_path}"
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
        nueAnalysis_hist_path = f"{PERSONAL_WORK_SPACE}/evaluation/nueAnalysis/hist/find_HAD_scale_factor_with_differnt_cuts__{{extra_cut_key}}/{{base_cut_key}}.done",
    params:
        script = f"{PERSONAL_WORK_SPACE}/evaluation/find_had_scale_factor.py",
        outdir=lambda wc, output: os.path.dirname(output.nueAnalysis_hist_path),
        base_cut=lambda wc: base_cut_map[wc.base_cut_key],
        extra_cut=lambda wc: extra_cut_map[wc.extra_cut_key],
    wildcard_constraints:
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
        echo "finding neutral hadron scale factor"

        set +u

        source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc11-opt
        export EOSSHIP=root://eosuser.cern.ch/

        TMPDIR=$(mktemp -d)
        echo "Using TMPDIR=$TMPDIR"

        python {params.script} \
            --base_cut "{params.base_cut}" \
            --extra_cut "{params.extra_cut}" \
            --outdir "$TMPDIR"


        PDF=$(ls "$TMPDIR"/*.pdf)
        xrdcp -f "$PDF" "{params.outdir}/"

        rm -rf "$TMPDIR"

        touch "{output.nueAnalysis_hist_path}"
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