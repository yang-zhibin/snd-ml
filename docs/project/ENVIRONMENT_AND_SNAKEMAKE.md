# Environment and Snakemake execution guide

Status: authoritative

Use this guide before running Python, ROOT, Snakemake, tests, or HTCondor jobs.
The repository has several runtime stacks with different purposes. An activated
environment belongs only to the current shell process; a new Codex command,
Snakemake rule, HTCondor job, terminal, or SSH login starts without it.

## Select the environment from the workflow

Inspect the relevant Snakemake rule and its wrapper before executing it. Use
the environment already selected there rather than guessing from the script
name or from an earlier command.

| Workload | Existing entry point | Expected role |
| --- | --- | --- |
| General analysis, metadata, uproot, and many ROOT tasks | `snakemake/envs/env_lcg.sh` | LCG 107 Python and analysis packages |
| SND detector software and experiment classes | The exact `env_sndsw_*.sh` referenced by the rule | SNDSW release required by that workflow |
| Legacy local ML training | `snakemake/envs/env_snd_pytorch.sh` | Conda `snd-pytorch` stack |
| Validated metadata command in a container or locally | `snakemake/run_metadata_python.sh` | Container Python when its imports pass; otherwise LCG |
| Current production ML training | The versioned container or LCG CUDA overlay named by its workflow documentation | Pinned PyTorch, PyG, and MinkowskiEngine stack |

The SNDSW scripts select different releases. They are not interchangeable:
use the exact one referenced by the rule. The `snd-pytorch` environment has ML
packages but must not be assumed to have PyROOT. An SNDSW environment has ROOT
and detector software but must not be assumed to have PyTorch. For a task that
needs both, use a documented container or overlay that has passed both import
and operator tests, or split the task into separate processes and exchange a
file between them.

## Activate and execute in one shell

Activation must occur in the same shell block as the Python or ROOT command.
Many CVMFS, alienv, and Conda setup scripts read variables that may be unset,
so disable Bash nounset during activation. Restore strict handling afterward.

For a direct LCG command from the repository root:

```bash
bash -lc '
  set +u
  source snakemake/envs/env_lcg.sh
  set -euo pipefail
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  hostname
  pwd
  command -v python3
  python3 --version
  python3 -c "import ROOT, numpy; print(ROOT.__version__, numpy.__version__)"
  python3 path/to/script.py
'
```

Change the activation script and import probe to match the relevant rule. Do
not source one environment in an earlier command and run Python in a later
command. Do not stack LCG, SNDSW, and Conda environments in the same shell to
make imports appear; library and Python ABI conflicts can then fail later and
produce misleading results.

## Snakemake shell blocks

Every rule must set up its own runtime. A safe pattern is:

```python
shell:
    r"""
    set +u
    source {env_script_lcg}
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1

    echo "host: $(hostname)" >&2
    echo "workdir: $(pwd)" >&2
    command -v python3 >&2
    python3 --version >&2
    python3 -c 'import awkward, numpy, pandas, uproot'

    python3 {input.script} --output {output}
    """
```

Use the imports actually required by the command. Keep the probe small and
fail before expensive input processing or batch submission. Prefer an existing
repository wrapper, such as `snakemake/run_metadata_python.sh`, when it already
owns environment selection and validation.

Do not rely on exports from the shell that launched Snakemake. Local rule
processes and HTCondor workers need explicit setup. For long work, run the
named Snakemake target and profile used by the workflow instead of invoking a
worker script under a guessed environment.

## Diagnose an environment failure

An import or shared-library failure is evidence about the selected runtime,
not immediately a code defect. Check it in this order:

1. Locate the exact rule, wrapper, environment script, container, or overlay
   intended for the command.
2. Reproduce the rule's activation and import in one Bash process. Run it from
   the workflow's expected working directory.
3. Record `hostname`, `pwd`, `command -v python3`, `python3 --version`, and the
   relevant module versions and paths.
4. Set `PYTHONNOUSERSITE=1` so packages in `~/.local` cannot mask the selected
   stack.
5. Compare local and worker logs. A passing login-node probe does not prove
   that an HTCondor job activated the same runtime.
6. If dependencies belong to incompatible stacks, split the operation into
   stages. Do not install packages or modify shared activation scripts without
   explicit user authorization.

For pytest in older shared environments, plugin discovery can itself cause a
version conflict. After confirming that this is the failure, retry the focused
test with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; this does not supply missing test
dependencies or validate optional backends.

## Tests and production jobs

Choose focused tests whose dependencies exist in the selected runtime. A host
that can import ROOT may still be unable to run PyTorch model tests, and a
PyTorch environment may still be unable to read through PyROOT. Tests that
need both stacks belong in the validated container or overlay, or must be
divided by process boundary.

Before submitting a long job, perform the same import probe through the actual
Snakemake rule or job wrapper. Record the target and command, Git commit and
dirty status, environment or image version, scheduler job ID, log paths,
expected outputs, and status command in the active task file.
