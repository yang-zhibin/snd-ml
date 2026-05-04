
rule combine_train_dataset:
    input: 
        script = f"{PERSONAL_WORK_SPACE}/train_3d/prepare_train_dataset.py"
    params:
        training_data_dir = f"{EOS_Work_SPACE}/training_data/",
        split = train_split,
        metadata = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/{train_split}_{split_dataset}.csv"
    output:
        nueAnalysis_out_path=f"{EOS_Work_SPACE}/training_data/{{train_split}}_train.npz"
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
        echo "running combine train dataset"

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
            -m {params.metadata} \
            -o "${{tmp_dir}}" \
            -s {params.split}


        # Copy results to EOS
        mkdir -p {params.training_data_dir}
        xrdcp -rf "${{tmp_dir}}"/* "{params.training_data_dir}/" || {{ echo "xrdcp failed"; exit 1; }}

        # Cleanup
        rm -rf "${{tmp_dir}}"
        """



rule train_3d:
    input:
        script = f'{PERSONAL_WORK_SPACE}/train_3d/train.py',
    params:
        train_data_dir = f"{EOS_Work_SPACE}/training_data/",
        split_version = train_split,
        model_version = train_model,
    threads:2
    resources:
        runtime=23*60*60,
        mem_mb=8000,
        disk_mb=8000,
        nvidia_gpu=1
    shell:
        r"""
        echo "training 3d hits model"
        #avoid unbound variable error (only occur when using snakemake)
        set +u 
        source /afs/cern.ch/work/z/zhibin/snd-ml/snakemake/envs/env_snd_pytorch.sh
        export EOSSHIP=root://eosuser.cern.ch/

        #tmp_dir=$(mktemp -d)
        export CUDA_VISIBLE_DEVICES=0
        #nvidia-smi

        
        python {input.script} \
        --train-data-dir {params.train_data_dir} \
        --split-version {params.split_version}\
        --model-version {params.model_version}

        echo "Cleaning up"
        rm -rf "${{tmp_dir}}"
        """