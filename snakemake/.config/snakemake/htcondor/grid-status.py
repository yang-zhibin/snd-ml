#!/usr/bin/python3

import sys
import htcondor2 as htcondor
from htcondor2 import JobEventType
import os
from os.path import join

def print_and_exit(s):
    print(s)
    exit()


jobID, UUID, clusterID = sys.argv[1].split("_")

log_dir = "/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/htcondor_log"
jobDir = f"{log_dir}/{jobID}_{UUID}"
jobLog = join(jobDir, "condor.log")

failed_states = [
    JobEventType.JOB_HELD,
    JobEventType.JOB_ABORTED,
    JobEventType.EXECUTABLE_ERROR,
]

try:
    jel = htcondor.JobEventLog(join(jobLog))
    for event in jel.events(stop_after=5):
        if event.type in failed_states:
            print_and_exit("failed")
        if event.type is JobEventType.JOB_TERMINATED:
            if event["ReturnValue"] == 0:
                print_and_exit("success")
            print_and_exit("failed")
except (OSError, htcondor.HTCondorException):
    # A newly submitted job can be visible to the schedd before its event log
    # is initialized on the shared filesystem. This is a transient polling
    # state, so let Snakemake retry instead of aborting the whole workflow.
    print_and_exit("running")

print_and_exit("running")
