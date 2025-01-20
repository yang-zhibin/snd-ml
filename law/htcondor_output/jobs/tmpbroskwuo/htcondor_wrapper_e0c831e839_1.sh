#!/usr/bin/env bash

# Wrapper script that is to be configured as htcondor's main executable file

htcondor_wrapper() {
    # helper to select the correct python executable
    _law_python() {
        command -v python &> /dev/null && python "$@" || python3 "$@"
    }

    #
    # detect variables
    #

    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_file_base="$( basename "${this_file}" )"

    # get the job number
    export LAW_HTCONDOR_JOB_NUMBER="${LAW_HTCONDOR_JOB_PROCESS}"
    if [ -z "${LAW_HTCONDOR_JOB_NUMBER}" ]; then
        >&2 echo "could not determine htcondor job number"
        return "1"
    fi
    # htcondor process numbers start at 0, law job numbers at 1, so increment
    ((LAW_HTCONDOR_JOB_NUMBER++))
    echo "running ${this_file_base} for job number ${LAW_HTCONDOR_JOB_NUMBER}"


    #
    # job argument definitons, depending on LAW_HTCONDOR_JOB_NUMBER
    #

    # definition
    local htcondor_job_arguments_map
    declare -A htcondor_job_arguments_map
    htcondor_job_arguments_map=(
        ['1']="analysis.tasks Digi2Hits LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tbWF4LXJ1bnRpbWU9MDE6MDA6MDA= MA== 1 no LQ=="
        ['2']="analysis.tasks Digi2Hits LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tbWF4LXJ1bnRpbWU9MDE6MDA6MDA= MQ== 1 no LQ=="
        ['3']="analysis.tasks Digi2Hits LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tbWF4LXJ1bnRpbWU9MDE6MDA6MDA= Mg== 1 no LQ=="
        ['4']="analysis.tasks Digi2Hits LS1sb2ctZmlsZT1OT19TVFIgLS1jbGVhci1sb2dzPUZhbHNlIC0tbWF4LXJ1bnRpbWU9MDE6MDA6MDA= Mw== 1 no LQ=="
    )

    # pick
    local htcondor_job_arguments="${htcondor_job_arguments_map[${LAW_HTCONDOR_JOB_NUMBER}]}"
    if [ -z "${htcondor_job_arguments}" ]; then
        >&2 echo "empty htcondor job arguments for LAW_HTCONDOR_JOB_NUMBER ${LAW_HTCONDOR_JOB_NUMBER}"
        return "3"
    fi


    #
    # variable rendering
    #

    # check variables
    local render_variables="eyJqb2JfZmlsZSI6ICJsYXdfam9iXzFkOGNjNWMxZjMuc2giLCAiZXhlY3V0YWJsZV9maWxlIjogImh0Y29uZG9yX3dyYXBwZXJfZTBjODMxZTgzOV8xLnNoIiwgImJvb3RzdHJhcF9maWxlIjogImJvb3RzdHJhcF9hNDE5ODgxZTNmLnNoIiwgImlucHV0X2ZpbGVzIjogImxhd19qb2JfMWQ4Y2M1YzFmMy5zaCBodGNvbmRvcl93cmFwcGVyX2UwYzgzMWU4MzlfMS5zaCBib290c3RyYXBfYTQxOTg4MWUzZi5zaCIsICJpbnB1dF9maWxlc19yZW5kZXIiOiAibGF3X2pvYl8xZDhjYzVjMWYzLnNoIGJvb3RzdHJhcF9hNDE5ODgxZTNmLnNoIiwgImxvZ19maWxlIjogInN0ZGFsbCQobGF3X2pvYl9wb3N0Zml4KS50eHQiLCAiaHRjb25kb3Jfam9iX2FyZ3VtZW50c19tYXAiOiAiWycxJ109XCJhbmFseXNpcy50YXNrcyBEaWdpMkhpdHMgTFMxc2IyY3RabWxzWlQxT1QxOVRWRklnTFMxamJHVmhjaTFzYjJkelBVWmhiSE5sSUMwdGJXRjRMWEoxYm5ScGJXVTlNREU2TURBNk1EQT0gTUE9PSAxIG5vIExRPT1cIlxuICAgICAgICBbJzInXT1cImFuYWx5c2lzLnRhc2tzIERpZ2kySGl0cyBMUzFzYjJjdFptbHNaVDFPVDE5VFZGSWdMUzFqYkdWaGNpMXNiMmR6UFVaaGJITmxJQzB0YldGNExYSjFiblJwYldVOU1ERTZNREE2TURBPSBNUT09IDEgbm8gTFE9PVwiXG4gICAgICAgIFsnMyddPVwiYW5hbHlzaXMudGFza3MgRGlnaTJIaXRzIExTMXNiMmN0Wm1sc1pUMU9UMTlUVkZJZ0xTMWpiR1ZoY2kxc2IyZHpQVVpoYkhObElDMHRiV0Y0TFhKMWJuUnBiV1U5TURFNk1EQTZNREE9IE1nPT0gMSBubyBMUT09XCJcbiAgICAgICAgWyc0J109XCJhbmFseXNpcy50YXNrcyBEaWdpMkhpdHMgTFMxc2IyY3RabWxzWlQxT1QxOVRWRklnTFMxamJHVmhjaTFzYjJkelBVWmhiSE5sSUMwdGJXRjRMWEoxYm5ScGJXVTlNREU2TURBNk1EQT0gTXc9PSAxIG5vIExRPT1cIiJ9"
    if [ -z "${render_variables}" ]; then
        >&2 echo "empty render variables"
        return "4"
    fi

    # decode
    render_variables="$( echo "${render_variables}" | base64 --decode )"

    # check files to render
    local input_files_render=( law_job_1d8cc5c1f3.sh bootstrap_a419881e3f.sh )
    if [ "${#input_files_render[@]}" == "0" ]; then
        >&2 echo "received empty input files for rendering for LAW_HTCONDOR_JOB_NUMBER ${LAW_HTCONDOR_JOB_NUMBER}"
        return "5"
    fi

    # render files
    local input_file_render
    for input_file_render in ${input_files_render[@]}; do
        # skip if the file refers to _this_ one
        local input_file_render_base="$( basename "${input_file_render}" )"
        [ "${input_file_render_base}" = "${this_file_base}" ] && continue
        # render
        echo "render ${input_file_render}"
        cat > _render.py << EOT
import re
repl = ${render_variables}
repl['input_files_render'] = ''
repl['file_postfix'] = '${file_postfix}' or repl.get('file_postfix', '')
repl['log_file'] = ''
content = open('${input_file_render}', 'r').read()
content = re.sub(r'\{\{(\w+)\}\}', lambda m: repl.get(m.group(1), ''), content)
open('${input_file_render_base}', 'w').write(content)
EOT
        _law_python _render.py
        local render_ret="$?"
        rm -f _render.py
        # handle rendering errors
        if [ "${render_ret}" != "0" ]; then
            >&2 echo "input file rendering failed with code ${render_ret}"
            return "6"
        fi
    done


    #
    # run the actual job file
    #

    # check the job file
    local job_file="law_job_1d8cc5c1f3.sh"
    if [ ! -f "${job_file}" ]; then
        >&2 echo "job file '${job_file}' does not exist"
        return "7"
    fi

    # helper to print a banner
    banner() {
        local msg="$1"

        echo
        echo "================================================================================"
        echo "=== ${msg}"
        echo "================================================================================"
        echo
    }

    # debugging: print its contents
    # echo "=== content of job file '${job_file}'"
    # echo
    # cat "${job_file}"
    # echo
    # echo "=== end of job file content"

    # run it
    banner "Start of law job"

    local job_ret
    bash "${job_file}" ${htcondor_job_arguments}
    job_ret="$?"

    banner "End of law job"

    return "${job_ret}"
}

action() {
    # arguments: file_postfix, log_file
    local file_postfix="$1"
    local log_file="$2"

    # create log directory
    if [ ! -z "${log_file}" ]; then
        local log_dir="$( dirname "${log_file}" )"
        [ ! -d "${log_dir}" ] && mkdir -p "${log_dir}"
    fi

    # run the wrapper function
    if [ -z "${log_file}" ]; then
        htcondor_wrapper "$@"
    elif command -v tee &> /dev/null; then
        set -o pipefail
        echo "---" >> "${log_file}"
        htcondor_wrapper "$@" 2>&1 | tee -a "${log_file}"
    else
        echo "---" >> "${log_file}"
        htcondor_wrapper "$@" &>> "${log_file}"
    fi
}

action "$@"
