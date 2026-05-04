
def load_metadata():
    if train_mode:
        metadata_prefix = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/training/{train_split}"
        metadata_files = glob.glob(f"{metadata_prefix}*.csv")
        if not metadata_files:
            print(f"Warning: No CSV files found with prefix {metadata_prefix}")
            return None
        df = pd.concat((pd.read_csv(file) for file in metadata_files), ignore_index=True)
        return df

    else:
        metadata_rootpath = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
        metadata_files = [os.path.join(metadata_rootpath, file_name) for file_name in (raw_metadata_csv_list + subset_metadata_csv_list)]
        metadata_files = [file for file in metadata_files if os.path.isfile(file)]
        if not metadata_files:  
            print("Warning: No metadata files found.")
            return None
        elif len(metadata_files) == 1:  
            return pd.read_csv(metadata_files[0])
        else:
            df = pd.concat((pd.read_csv(file) for file in metadata_files), ignore_index=True)
            return df

METADATA = load_metadata()

def collect_metadata_targets(target_name, after_model=False):
    if METADATA is None:
        return []

    # -------------------------
    # MODEL-RELATED TARGETS
    # -------------------------
    if after_model:
        process_model_set = {process_model} if isinstance(process_model, str) else set(process_model)
        target_paths = []
        for m in process_model_set:
            if target_name == "model_output":
                col = f"prediction_{m}_output_path"
            else:
                raise ValueError(f"Unknown model target: {target_name}")

            if col in METADATA.columns:
                paths = (
                    METADATA["output_base_path"].astype(str).str.rstrip("/") + "/" +
                    METADATA[col].astype(str).str.lstrip("/")
                )
                target_paths.extend(paths.tolist())

        return (set([p for p in target_paths if str(p).lower() != "nan"]))

    # -------------------------
    # NORMAL TARGETS
    # -------------------------
    def combine_paths(base, rel):
        valid = base.notna() & rel.notna()
        return (
            base[valid].astype(str).str.rstrip("/") + "/" +
            rel[valid].astype(str).str.lstrip("/")
        ).tolist()
    target_paths = []

    if target_name in METADATA.columns:
        target_paths.extend(combine_paths(METADATA["output_base_path"], METADATA[target_name]))

    if process_veto:
        col = f"veto_{target_name}"
        if col in METADATA.columns:
            target_paths.extend(combine_paths(METADATA["output_base_path"], METADATA[col]))


    return (set(target_paths))


def get_metadata_value(ref_name, ref_value, target_name, after_model=False):
    """
    Look up `target_name` in METADATA using (`ref_name`, `ref_value`).

    If after_model=True, infer the real metadata column name from filenames like:
        model_xxx_output...
        eval_xxx_output...
        matrix_xxx_output...

    Example:
        get_metadata_value("eval", "/path/eval_mytag_output.root", "model_output", after_model=True)
    """

    if METADATA is None:
        return ""

    original_ref_name = ref_name

    # Handle post-model/eval/matrix filenames
    if after_model:
        file_name = os.path.basename(ref_value)
        pattern = rf"{ref_name}_(.+?)_output"
        match = re.search(pattern, file_name)

        if not match:
            raise ValueError(
                f"Could not extract model name from filename '{file_name}' "
                f"using pattern '{pattern}'"
            )

        m_name = match.group(1)
        ref_name = f"{original_ref_name}_{m_name}_output_path"

        if original_ref_name in {"eval", "matrix"} and target_name == "model_output":
            target_name = f"prediction_{m_name}_output_path"
        elif original_ref_name == "matrix" and target_name == "eval_output":
            target_name = f"eval_{m_name}_output_path"

    # Handle vetoFree / vetoTagged prefixes
    if "veto_" in ref_value and (ref_name not in METADATA.columns):
        ref_name = f"veto_{ref_name}"
        if target_name not in METADATA.columns:
            target_name = f"veto_{target_name}"

    if ref_name not in METADATA.columns:
        raise ValueError(
            f"Column '{ref_name}' not found in METADATA.\n"
            f"Available columns: {METADATA.columns.tolist()}"
        )

    if target_name not in METADATA.columns:
        raise ValueError(
            f"Column '{target_name}' not found in METADATA.\n"
            f"Available columns: {METADATA.columns.tolist()}"
        )

    ref_value = Path(ref_value).name
    rows = METADATA.loc[METADATA[ref_name] == ref_value]

    if len(rows) == 0:
        raise ValueError(
            f"No row found where {ref_name} == {ref_value}\n"
            f"Examples: {METADATA[ref_name].head(10).tolist()}"
        )
    elif len(rows) > 1:
        raise ValueError(
            f"Multiple rows found where {ref_name} == {ref_value}\n"
            f"Matching rows:\n{rows[[ref_name, target_name]]}"
        )

    row = rows.iloc[0]
    value = str(row[target_name])

    if target_name in ["digi_path", "geo_path", "raw_path", "data_type", "subfolder","partition","n_event", "split"]:
        return value

    return f"{row['output_base_path'].rstrip('/')}/{value.lstrip('/')}"


rule process_nueAnalysis:
    input:
        #expand("/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/{metadata_csv}", metadata_csv=metadata_csv_list),
        digi=lambda wildcards: get_metadata_value(ref_name = "nueAnalysisFilter_path", ref_value = wildcards.nueAnalysisFilter_path, target_name = "digi_path",  after_model = False),
        geo=lambda wildcards: get_metadata_value(ref_name = "nueAnalysisFilter_path", ref_value = wildcards.nueAnalysisFilter_path, target_name = "geo_path",  after_model = False),
        
    params:
        data_type=lambda wildcards: get_metadata_value("nueAnalysisFilter_path", wildcards.nueAnalysisFilter_path, "data_type"),
        pipeline = f"$SNDSW_ROOT/analysis/analyses/snd_analysis_2024_0mu/pipelines/nueFilterMoriondOrder_withEventLevelOutput.h"
    output:
        nueAnalysisFilter_path="{nueAnalysisFilter_path}"
    threads:1
    resources:
        runtime=2*60*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        # Avoid conda/python conflicts
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u  # Avoid unbound variable errors

        echo "Source SNDSW environment script"
        source {env_script_sndsw_nue}

        export EOSSHIP=root://eosuser.cern.ch/

        tmp_dir=$(mktemp -d)
        filename=$(basename "{output.nueAnalysisFilter_path}")
        tmp_output="${{tmp_dir}}/${{filename}}"

        echo "Running sndEventFilter"
        sndEventFilter \
            --input {input.digi} \
            --geofile {input.geo} \
            --output "${{tmp_output}}" \
            --pipeline  {params.pipeline}


        echo "Transferring output to EOS"
        xrdcp -f "${{tmp_output}}" "{output.nueAnalysisFilter_path}" || {{ echo "xrdcp failed"; exit 1; }}
        echo "Cleaning up"
        rm -rf "${{tmp_dir}}"
        """

rule process_features:
    input:
        #expand("/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/{metadata_csv}", metadata_csv=metadata_csv_list),
        digi=lambda wildcards: get_metadata_value(ref_name = "feature_path", ref_value = wildcards.feature_path, target_name = "digi_path",  after_model = False),
        geo=lambda wildcards: get_metadata_value(ref_name = "feature_path", ref_value = wildcards.feature_path, target_name = "geo_path",  after_model = False),
        script = f"{PERSONAL_WORK_SPACE}/convertData/digi_2_features.py",
        nueAnalysisFilter_path=lambda wildcards: get_metadata_value(ref_name = "feature_path", ref_value = wildcards.feature_path, target_name = "nueAnalysisFilter_path",  after_model = False),
    params:
        data_type=lambda wildcards: get_metadata_value(ref_name = "feature_path", ref_value = wildcards.feature_path, target_name = "data_type",  after_model = False),
    output:
        feature_path="{feature_path}"
    threads:1
    resources:
        runtime=60*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        # Avoid sndsw and conda python conflict 
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        
        # Avoid unbound variable error (only occurs when using Snakemake)
        set +u 
        
        echo "Source SNDSW environment script"
        source {env_script_sndsw_nue}

        export EOSSHIP=root://eosuser.cern.ch/

        # Create a unique temporary directory
        tmp_dir=$(mktemp -d)
        filename=$(basename "{output.feature_path}")
        tmp_output="${{tmp_dir}}/${{filename}}"

        export PYTHONPATH="$SNDSW_ROOT:$PYTHONPATH"
        echo "Running feature generation script"
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            {input.script} \
            -p {input.nueAnalysisFilter_path}\
            -d {input.digi} \
            -g {input.geo} \
            -o "${{tmp_output}}" \
            -t {params.data_type}

        echo "Copying result to final location"
        xrdcp -f "${{tmp_output}}" "{output.feature_path}" || {{ echo "xrdcp failed"; exit 1; }}

        echo "Cleaning up"
        rm -rf "${{tmp_dir}}"
        """


rule process_hits:
    input:
        #expand("/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/{metadata_csv}", metadata_csv=metadata_csv_list),
        digi=lambda wildcards: get_metadata_value(ref_name = "npz_hit_path", ref_value = wildcards.npz_hit_path, target_name = "digi_path",  after_model = False),
        geo=lambda wildcards: get_metadata_value(ref_name = "npz_hit_path", ref_value = wildcards.npz_hit_path, target_name = "geo_path",  after_model = False),
        script = f"{PERSONAL_WORK_SPACE}/convertData/digi_2_hits3D.py",
        nueAnalysisFilter_path=lambda wildcards: get_metadata_value(ref_name = "npz_hit_path", ref_value = wildcards.npz_hit_path, target_name = "nueAnalysisFilter_path",  after_model = False),
    params:
        data_type=lambda wildcards: get_metadata_value(ref_name = "npz_hit_path", ref_value = wildcards.npz_hit_path, target_name = "data_type",  after_model = False),
        split = lambda wildcards: get_metadata_value(ref_name = "npz_hit_path", ref_value = wildcards.npz_hit_path, target_name = "split",  after_model = False),
    output:
        npz_hit_path="{npz_hit_path}"
    threads:1
    resources:
        runtime=60*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        # Avoid sndsw and conda python conflict 
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        
        # Avoid unbound variable error (only occurs when using Snakemake)
        set +u 
        
        echo "Source SNDSW environment script"
        source {env_script_sndsw_nue}

        export EOSSHIP=root://eosuser.cern.ch/

        # Create a unique temporary directory
        tmp_dir=$(mktemp -d)
        filename=$(basename "{output.npz_hit_path}")
        tmp_output="${{tmp_dir}}/${{filename}}"

        export PYTHONPATH="$SNDSW_ROOT:$PYTHONPATH"
        echo "Running hit generation script"
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            {input.script} \
            -p {input.nueAnalysisFilter_path}\
            -d {input.digi} \
            -g {input.geo} \
            -o "${{tmp_output}}" \
            -t {params.data_type} \
            --dataset-split {params.split}

        echo "Copying result to final location"
        xrdcp -f "${{tmp_output}}" "{output.npz_hit_path}" || {{ echo "xrdcp failed"; exit 1; }}

        echo "Cleaning up"
        rm -rf "${{tmp_dir}}"
        """

rule process_muonDIS_features:
    input:
        #expand("/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/{metadata_csv}", metadata_csv=metadata_csv_list),
        digi=lambda wildcards: get_metadata_value(ref_name = "muonDISFeature_path", ref_value = wildcards.muonDISFeature_path, target_name = "digi_path",  after_model = False),
        geo=lambda wildcards: get_metadata_value(ref_name = "muonDISFeature_path", ref_value = wildcards.muonDISFeature_path, target_name = "geo_path",  after_model = False),
        script = f"{PERSONAL_WORK_SPACE}/convertData/digi_2_muonDIS.py",
    params:
        data_type=lambda wildcards: get_metadata_value("muonDISFeature_path", wildcards.muonDISFeature_path, "data_type"),
    output:
        muonDISFeature_path="{muonDISFeature_path}"
    threads:1
    resources:
        runtime=2*60*60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        # Avoid conda/python conflicts
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u  # Avoid unbound variable errors

        echo "Source SNDSW environment script"
        source {env_script_sndsw_nue}

        export EOSSHIP=root://eosuser.cern.ch/

        tmp_dir=$(mktemp -d)
        filename=$(basename "{output.muonDISFeature_path}")
        tmp_output="${{tmp_dir}}/${{filename}}"

        export PYTHONPATH="$SNDSW_ROOT:$PYTHONPATH"
        echo "Running process_muonDIS_features"
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            {input.script} \
            -d {input.digi} \
            -g {input.geo} \
            -o "${{tmp_output}}" \
            -t {params.data_type}


        echo "Transferring output to EOS"
        xrdcp -f "${{tmp_output}}" "{output.muonDISFeature_path}" || {{ echo "xrdcp failed"; exit 1; }}
        echo "Cleaning up"
        rm -rf "${{tmp_dir}}"
        """
