#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=cryosage-web
#SBATCH --export=ALL
set -euo pipefail
cd "${CRYOAGENT_REPO_ROOT:?Start the service using cryosage-web with a deployment config}"
export PYTHONPATH="$PWD:$PWD/cryosparc_agent${PYTHONPATH:+:$PYTHONPATH}"
export CRYOAGENT_WEB_GPU_ROUTING=1
export CRYOAGENT_WEB_DYNAMIC_GPUS="${CRYOAGENT_WEB_DYNAMIC_GPUS:-1}"
export CRYOAGENT_WEB_MAX_GPUS="${CRYOAGENT_WEB_MAX_GPUS:-8}"
export CRYOAGENT_WEB_SUPERVISED=1
export CRYOAGENT_UNATTENDED=1
exec "${CRYOAGENT_WEB_PYTHON:-python3}" -u -m webapp.supervisor "$@"
