#!/usr/bin/python3

import sys
import htcondor
from os import makedirs
from os.path import join
from uuid import uuid4
from datetime import datetime
import os

from snakemake.utils import read_job_properties


jobscript = sys.argv[1]
job_properties = read_job_properties(jobscript)
resources = job_properties.get("resources", {})
#print(job_properties)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

UUID = uuid4()  # random UUID
log_dir = "/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/htcondor_log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
jobDir = f"{log_dir}/{job_properties['jobid']}_{UUID}"
makedirs(jobDir, exist_ok=True)

sub = htcondor.Submit(
    {
        "executable": "/bin/bash",
        "arguments": jobscript,
        "max_retries": "0",
        "log": join(jobDir, "condor.log"), #join(jobDir, "condor.log"), "/tmp/zhibin/condor.log"
        "output": join(jobDir, "condor.out"),
        "error": join(jobDir, "condor.err"),
        "should_transfer_files": "NO",
        "getenv": "True",
        "request_cpus": str(max(1, int(job_properties.get("threads", 1)))),
        "+MaxRuntime": str(resources["runtime"]),
        # Add your custom HTCondor settings
        "+AccountingGroup": '"group_u_SNDLHC.users"',
    }
)

requirements = ['(TARGET.OpSysAndVer =?= "AlmaLinux9")']
container_image = str(resources.get("container_image", "")).strip()
if container_image:
    if not os.path.isabs(container_image):
        raise ValueError("container_image must be an absolute shared-filesystem path")
    # Use HTCondor's native container universe.  The legacy SingularityImage
    # ClassAd is only acted on when the execute point enables a corresponding
    # site policy; on CERN workers it can otherwise be preserved in the job ad
    # while the payload still runs on the host.
    sub["universe"] = "container"
    sub["container_image"] = container_image
    # Resolve /bin/bash from inside the image.  Otherwise HTCondor transfers
    # the submit host's executable into the container sandbox, where it can be
    # ABI-incompatible with the image (for example, Python 3.9 vs. 3.10).
    sub["transfer_executable"] = "False"
    # Images configured here live on a shared filesystem.  In particular, the
    # training SIF is several GB, so copying it into every job sandbox would be
    # both unnecessary and larger than the workers' requested scratch space.
    sub["transfer_container"] = "False"
    # Keep credentials and other runtime state from getenv, but do not let the
    # submit host's Conda PATH select its Python inside the container.
    container_path = (
        "/usr/local/nvidia/bin:"
        "/usr/local/cuda/bin:"
        "/usr/local/sbin:"
        "/usr/local/bin:"
        "/usr/sbin:"
        "/usr/bin:"
        "/sbin:"
        "/bin"
    )
    sub["environment"] = f'"PATH={container_path} PYTHONNOUSERSITE=1"'
    requirements.append("(TARGET.HasContainer =?= true)")

# Add GPU request if specified in job properties
if int(resources.get("nvidia_gpu", 0)) != 0:
    sub["request_GPUs"] = str(resources["nvidia_gpu"])
    minimum_capability = resources.get("minimum_gpu_capability")
    if minimum_capability is not None:
        requirements.append(
            f"(TARGET.GPUs_Capability >= {float(minimum_capability)})"
        )
    gpu_device_name = str(resources.get("gpu_device_name", "")).strip()
    if gpu_device_name:
        # Pin timing jobs to one accelerator model so their rates are directly
        # comparable. ClassAd string literals use C-style escaping.
        escaped_name = gpu_device_name.replace("\\", "\\\\").replace('"', '\\"')
        requirements.append(f'(TARGET.GPUs_DeviceName =?= "{escaped_name}")')

sub["requirements"] = " && ".join(requirements)


request_memory = resources.get("mem_mb", None)
if request_memory is not None:
    sub["request_memory"] = f"{int(request_memory)}MB"

request_disk = resources.get("disk_mb", None)
if request_disk is not None:
    sub["request_disk"] = f"{int(request_disk)}MB"

# The current htcondor2 wheel on lxplus has regressions in both Kerberos
# credential production and empty-queue submission.  The co-installed stable
# binding uses the same schedd protocol and supports CERN's credential flow.
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)
sub["MY.SendCredential"] = "True"

schedd = htcondor.Schedd()
submit_result = schedd.submit(sub)
clusterID = submit_result.cluster()

# print jobid for use in Snakemake
print("{}_{}_{}".format(job_properties["jobid"], UUID, clusterID))
