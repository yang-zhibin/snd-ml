import os
import math

import luigi
import law

law.contrib.load("htcondor")


class Task(law.Task):

    def local_target(self, path):
        # Directly create a LocalFileTarget using the provided full path
        return law.LocalFileTarget(path)


class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
    """
    Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
    to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
    Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
    the CERN HTCondor environment. In most cases, like in this example, only a minimal amount of
    configuration is required.
    """

    max_runtime = law.DurationParameter(
        default=1.0,
        unit="h",
        significant=False,
        description="maximum runtime; default unit is hours; default: 1",
    )
    poll_interval = law.DurationParameter(
        default=0.5,
        unit="m",
        significant=False,
        description="time between status polls; default unit is minutes; default: 0.5",
    )
    transfer_logs = luigi.BoolParameter(
        default=True,
        significant=False,
        description="transfer job logs to the output directory; default: True",
    )

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        htcondor_path = os.getenv("HTCONDOR_OUTPUT") 
        return law.LocalDirectoryTarget(htcondor_path)

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # configure it to be shared across jobs and rendered as part of the job itself
        bootstrap_file = law.util.rel_path(__file__, "bootstrap.sh")
        return law.JobInputFile(bootstrap_file, share=True, render_job=True)

    def htcondor_job_config(self, config, job_num, branches):
        # render_variables are rendered into all files sent with a job
        #config.render_variables["analysis_path"] = os.getenv("ANALYSIS_PATH")

        # configure to run in a "el7" container
        # https://batchdocs.web.cern.ch/local/submit.html#os-selection-via-containers
        config.custom_content.append(("MY.WantOS", "el9"))

        # maximum runtime
        config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))

        # copy the entire environment
        config.custom_content.append(("getenv", "true"))

        # the CERN htcondor setup requires a "log" config, but we can safely set it to /dev/null
        # if you are interested in the logs of the batch system itself, set a meaningful value here
        htcondor_path = os.getenv("HTCONDOR_OUTPUT") 
        config.custom_content.append(("log", htcondor_path))

        return config