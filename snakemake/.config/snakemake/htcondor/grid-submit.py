#!/usr/bin/env python3

import sys
import htcondor2 as htcondor
from os import makedirs
from os.path import join
from uuid import uuid4
from datetime import datetime
import os

from snakemake.utils import read_job_properties


jobscript = sys.argv[1]
job_properties = read_job_properties(jobscript)
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
        "output":  "condor.out", # join(jobDir, "condor.out"), "condor.out", 
        "error": "condor.err", #join(jobDir, "condor.err"), "condor.err", 
        "should_transfer_files": "NO",
        "getenv": "True",
        "request_cpus": "1",#        "request_cpus": str(job_properties["threads"]),
        "+MaxRuntime": job_properties["resources"]["runtime"],
        # Add your custom HTCondor settings
        "+AccountingGroup": '"group_u_SNDLHC.users"',
        "requirements": '(TARGET.OpSysAndVer =?= "AlmaLinux9")',
    }
)

# Add GPU request if specified in job properties
if "nvidia_gpu" in job_properties["resources"] and job_properties["resources"]["nvidia_gpu"] != 0:
    sub["request_GPUs"] = str(job_properties["resources"]["nvidia_gpu"])


# request_memory = job_properties["resources"].get("mem_mb", None)
# if request_memory is not None:
#     sub["request_memory"] = str(request_memory)

# request_disk = job_properties["resources"].get("disk_mb", None)
# if request_disk is not None:
#     sub["request_disk"] = str(request_disk)

# Add kerberos credentials
# c.f. https://batchdocs.web.cern.ch/local/pythonapi.html
col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)
sub["MY.SendCredential"] = "True"

schedd = htcondor.Schedd()
clusterID = schedd.submit(sub)

# print jobid for use in Snakemake
print("{}_{}_{}".format(job_properties["jobid"], UUID, clusterID))
