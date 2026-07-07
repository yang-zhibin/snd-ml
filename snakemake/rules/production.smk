
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
        metadata_files = [
            os.path.join(metadata_rootpath, file_name)
            for file_name in (
                raw_metadata_csv_list
                + subset_metadata_csv_list
                + eventbuilder_metadata_csv_list
            )
        ]
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


def collect_eventbuilder_targets(target_name):
    metadata_rootpath = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
    metadata_files = [
        os.path.join(metadata_rootpath, file_name)
        for file_name in eventbuilder_metadata_csv_list
        if os.path.isfile(os.path.join(metadata_rootpath, file_name))
    ]
    if not metadata_files:
        return []

    df = pd.concat((pd.read_csv(file) for file in metadata_files), ignore_index=True)
    if target_name not in df.columns:
        raise ValueError(
            f"Column '{target_name}' not found in EventBuilder metadata.\n"
            f"Available columns: {df.columns.tolist()}"
        )
    if "output_base_path" not in df.columns:
        raise ValueError("EventBuilder metadata is missing output_base_path")

    invalid_base = df[
        ~df["output_base_path"].astype(str).str.startswith(("/", "root://"))
    ]
    if len(invalid_base) > 0:
        examples = invalid_base[["subfolder", "partition", "output_base_path"]].head(10).to_dict("records")
        raise ValueError(
            "EventBuilder metadata has invalid output_base_path values. "
            "Regenerate it with metadata/make_eventbuilder_metadata.py. "
            f"Examples: {examples}"
        )

    valid = df["output_base_path"].notna() & df[target_name].notna()
    paths = (
        df.loc[valid, "output_base_path"].astype(str).str.rstrip("/") + "/" +
        df.loc[valid, target_name].astype(str).str.lstrip("/")
    ).tolist()
    return sorted(set(path for path in paths if str(path).strip().lower() != "nan"))


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

    if target_name in [
        "digi_path",
        "geo_path",
        "raw_path",
        "data_type",
        "subfolder",
        "partition",
        "n_event",
        "split",
        "production_profile",
    ]:
        return value

    return f"{row['output_base_path'].rstrip('/')}/{value.lstrip('/')}"

def get_production_profile(ref_name, ref_value):
    if METADATA is None or "production_profile" not in METADATA.columns:
        data_type = get_metadata_value(ref_name, ref_value, "data_type", after_model=False)
        return "real_data_default" if data_type == "real_data" else "mc_default"

    return get_metadata_value(ref_name, ref_value, "production_profile", after_model=False)

def drop_real_has_veto_has_us_flag(ref_name, ref_value):
    data_type = get_metadata_value(ref_name, ref_value, "data_type", after_model=False)
    production_profile = get_production_profile(ref_name, ref_value)
    if data_type == "real_data" and production_profile != "real_data_2024_skim_run":
        return "--drop-real-has-veto-has-us"
    return ""

NUE_ANALYSIS_PIPELINE = "$SNDSW_ROOT/analysis/analyses/snd_analysis_2024_0mu/pipelines/nueFilterMoriondOrder_withEventLevelOutput.h"
MUONDIS_NUE_ANALYSIS_PIPELINE = "$SNDSW_ROOT/analysis/analyses/snd_analysis_2024_0mu/pipelines/muonDISFilterMoriondOrder_withEventLevelOutput.h"
NUE_ANALYSIS_PIPELINES_BY_DATA_TYPE = {
    "MC_muonDIS": MUONDIS_NUE_ANALYSIS_PIPELINE,
}

def get_nue_analysis_pipeline(wildcards):
    data_type = get_metadata_value(
        "nueAnalysisFilter_path",
        wildcards.nueAnalysisFilter_path,
        "data_type",
    )
    return NUE_ANALYSIS_PIPELINES_BY_DATA_TYPE.get(data_type, NUE_ANALYSIS_PIPELINE)


rule process_newRaw:
    input:
        raw=lambda wildcards: get_metadata_value(
            ref_name="newRaw_path",
            ref_value=wildcards.newRaw_path,
            target_name="raw_path",
            after_model=False,
        ),
        geo=lambda wildcards: get_metadata_value(
            ref_name="newRaw_path",
            ref_value=wildcards.newRaw_path,
            target_name="geo_path",
            after_model=False,
        ),
    output:
        newRaw_path="{newRaw_path}"
    threads: 1
    resources:
        runtime=15 * 60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u

        echo "Source SNDSW EventBuilder environment script"
        source {env_script_sndsw_eventbuilder}
        export EOSSHIP=root://eosuser.cern.ch/

        tmp_dir=$(mktemp -d)
        trap 'status=$?; echo "Cleaning up"; rm -rf "${{tmp_dir}}"; exit $status' EXIT

        filename=$(basename "{output.newRaw_path}")
        tmp_output="${{tmp_dir}}/${{filename}}"

        echo "Processing EventBuilder newRaw file"
        python "$SNDSW_ROOT/shipLHC/run_MCEventBuilder.py" \
            -f {input.raw} \
            -g {input.geo} \
            -o "${{tmp_output}}"

        echo "Transferring newRaw output to EOS"
        xrdcp -f "${{tmp_output}}" "{output.newRaw_path}" || {{ echo "xrdcp failed"; exit 1; }}
        """


rule process_newDigi:
    input:
        newRaw=lambda wildcards: get_metadata_value(
            ref_name="newDigi_path",
            ref_value=wildcards.newDigi_path,
            target_name="newRaw_path",
            after_model=False,
        ),
        geo=lambda wildcards: get_metadata_value(
            ref_name="newDigi_path",
            ref_value=wildcards.newDigi_path,
            target_name="geo_path",
            after_model=False,
        ),
    output:
        newDigi_path="{newDigi_path}"
    threads: 1
    resources:
        runtime=15 * 60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u

        echo "Source SNDSW EventBuilder environment script"
        source {env_script_sndsw_eventbuilder}
        export EOSSHIP=root://eosuser.cern.ch/

        tmp_dir=$(mktemp -d)
        trap 'status=$?; echo "Cleaning up"; rm -rf "${{tmp_dir}}"; exit $status' EXIT
        cd "${{tmp_dir}}"

        filename=$(basename "{input.newRaw}")
        echo "Fetching newRaw input: {input.newRaw}"
        xrdcp -f "{input.newRaw}" "${{filename}}" || {{ echo "xrdcp failed"; exit 1; }}

        expected_output="${{filename/.root/_dig.root}}"
        echo "Processing EventBuilder newDigi file"
        set +e
        python "$SNDSW_ROOT/shipLHC/run_digiSND.py" \
            -f "${{filename}}" \
            -g {input.geo}
        rc=$?
        set -e
        echo "[run_digiSND.py] exit code: ${{rc}}"
        if [[ "${{rc}}" -ne 0 && "${{rc}}" -ne 143 ]]; then
            echo "run_digiSND.py failed"
            exit "${{rc}}"
        fi

        if [[ -f "${{expected_output}}" ]]; then
            produced="${{expected_output}}"
        else
            produced=$(ls -1 *_dig.root 2>/dev/null | head -n1 || true)
            [[ -n "${{produced}}" ]] || {{ echo "ERROR: No *_dig.root produced"; ls -al; exit 1; }}
            echo "Detected output: ${{produced}}"
        fi

        echo "Transferring newDigi output to EOS"
        xrdcp -f "${{produced}}" "{output.newDigi_path}" || {{ echo "xrdcp failed"; exit 1; }}
        """


rule process_nueAnalysis:
    input:
        #expand("/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/{metadata_csv}", metadata_csv=metadata_csv_list),
        digi=lambda wildcards: get_metadata_value(ref_name = "nueAnalysisFilter_path", ref_value = wildcards.nueAnalysisFilter_path, target_name = "digi_path",  after_model = False),
        geo=lambda wildcards: get_metadata_value(ref_name = "nueAnalysisFilter_path", ref_value = wildcards.nueAnalysisFilter_path, target_name = "geo_path",  after_model = False),
        
    params:
        data_type=lambda wildcards: get_metadata_value("nueAnalysisFilter_path", wildcards.nueAnalysisFilter_path, "data_type"),
        pipeline=get_nue_analysis_pipeline
    output:
        nueAnalysisFilter_path="{nueAnalysisFilter_path}"
    threads:1
    resources:
        runtime=20*60,
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
        echo "Using sndEventFilter pipeline: {params.pipeline}"
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
        drop_real_has_veto_has_us_flag=lambda wildcards: drop_real_has_veto_has_us_flag(
            "feature_path",
            wildcards.feature_path,
        ),
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
            -t {params.data_type} \
            {params.drop_real_has_veto_has_us_flag}

        echo "Copying result to final location"
        xrdcp -f "${{tmp_output}}" "{output.feature_path}" || {{ echo "xrdcp failed"; exit 1; }}

        echo "Cleaning up"
        rm -rf "${{tmp_dir}}"
        """


rule process_hit3d:
    input:
        #expand("/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/{metadata_csv}", metadata_csv=metadata_csv_list),
        digi=lambda wildcards: get_metadata_value(ref_name = "hit3d_path", ref_value = wildcards.hit3d_path, target_name = "digi_path",  after_model = False),
        geo=lambda wildcards: get_metadata_value(ref_name = "hit3d_path", ref_value = wildcards.hit3d_path, target_name = "geo_path",  after_model = False),
        script = f"{PERSONAL_WORK_SPACE}/convertData/digi_2_hits3D.py",
        nueAnalysisFilter_path=lambda wildcards: get_metadata_value(ref_name = "hit3d_path", ref_value = wildcards.hit3d_path, target_name = "nueAnalysisFilter_path",  after_model = False),
    params:
        data_type=lambda wildcards: get_metadata_value(ref_name = "hit3d_path", ref_value = wildcards.hit3d_path, target_name = "data_type",  after_model = False),
        drop_real_has_veto_has_us_flag=lambda wildcards: drop_real_has_veto_has_us_flag(
            "hit3d_path",
            wildcards.hit3d_path,
        ),
    output:
        hit3d_path="{hit3d_path}"
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
        filename=$(basename "{output.hit3d_path}")
        tmp_output="${{tmp_dir}}/${{filename}}"

        export PYTHONPATH="$SNDSW_ROOT:$PYTHONPATH"
        echo "Running hit3D ROOT generation script"
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            {input.script} \
            -p {input.nueAnalysisFilter_path}\
            -d {input.digi} \
            -g {input.geo} \
            -o "${{tmp_output}}" \
            -t {params.data_type} \
            {params.drop_real_has_veto_has_us_flag}

        echo "Copying result to final location"
        xrdcp -f "${{tmp_output}}" "{output.hit3d_path}" || {{ echo "xrdcp failed"; exit 1; }}

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

rule process_muonDIS_digi:
    input:
        #expand("/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/{metadata_csv}", metadata_csv=metadata_csv_list),
        digi=lambda wildcards: get_metadata_value(ref_name = "muonDISDigi_path", ref_value = wildcards.muonDISDigi_path, target_name = "digi_path",  after_model = False),
        script = f"{PERSONAL_WORK_SPACE}/convertData/filter_muonDIS.py",
    output:
        muonDISDigi_path="{muonDISDigi_path}"
    threads:1
    resources:
        runtime=1*60*60,
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
        filename=$(basename "{output.muonDISDigi_path}")
        tmp_output="${{tmp_dir}}/${{filename}}"

        export PYTHONPATH="$SNDSW_ROOT:$PYTHONPATH"
        echo "Running process_muonDIS_digi"
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            {input.script} \
            -d {input.digi} \
            -o "${{tmp_output}}" \


        echo "Transferring output to EOS"
        xrdcp -f "${{tmp_output}}" "{output.muonDISDigi_path}" || {{ echo "xrdcp failed"; exit 1; }}
        echo "Cleaning up"
        rm -rf "${{tmp_dir}}"
        """
