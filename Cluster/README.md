# Cluster
This folder contains job scripts and associated files for running this project on a cluster


## Cluster Setup

### Clone the repo
First, cd into the projects/ directory (under your supervisors directory if required)

Then run:
```bash
git clone https://github.com/lerdman183/Curiosity-by-Design.git
cd projects/def-SUPERVISORNAME/Curiosity-by-Design
```

### Complete the setup
After which, follow the commented pre-run steps found at the top of each slurm file to complete setup before submitting a job.


## Folder paths 
**Paths defined in the folders will not work for other users**
You must replace my username (lerdman), and my supervisor's (def-lutellie), with names that correspond to your account if you wish to run these scripts. I replaced these names with general tokens (%u) where possible, but other places require the names to be replaced.

File paths in the Python scripts should work for a general user, but the job scripts do not have this kind of support


## Submitting Job Scripts
Refer to the README in the top level directory for directions on submitting the slurm scripts in the correct order.

**NOTE:** The steps commented out at the top of the slurm scripts must be run before the scripts are submitted to a scheduler. The mkdir, venv, and pip install steps only need to be run once to set up the project on the cluster. The other steps MUST be run each time before submitting a script to ensure the job runs correctly.