#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 24:00:00
#SBATCH --job-name=Cluster_CKW_Data
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2

# If needed, set the Singularity environment variables here,
# or just leave them commented to be set externally:
# export SINGULARITYENV_PERSONAL_ACCESS_TOKEN="<Github OAuth token>"
# export SINGULARITYENV_RUNNER_NAME="<runner name>"
# export SINGULARITYENV_RUNNER_WORKDIR="/tmp/actions-runner-repo"
# export SINGULARITYENV_GITHUB_ORG="<org or username>"
# export SINGULARITYENV_GITHUB_REPO="<repo name>"

# Absolute path to your project on the host
PROJECT_DIR=/home2/lukas.breiter/poetry_python/CICMC-L4-2

# Where to mount it inside the container
WORKDIR=/workspace

singularity exec \
     --bind ${PROJECT_DIR}:${WORKDIR} \
     python_poetry.sif \
     bash -lc "\
       cd ${WORKDIR} && \
       poetry install --no-interaction && \
       poetry run python run_full_pipeline.py \
     "