#!/bin/bash
#SBATCH -J ag-scaling-manifests
#SBATCH -A r00284
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /N/scratch/mjpathar/AudioGroove/logs/scaling_manifests_%j.out
#SBATCH -e /N/scratch/mjpathar/AudioGroove/logs/scaling_manifests_%j.err

set -euo pipefail

module purge
module load python/gpu/3.11.5
source /N/scratch/mjpathar/AudioGroove/venv/bin/activate
cd /N/u/mjpathar/BigRed200/AudioGroove

python -u -m src.data_prep.build_scaling_manifests \
  --source-root /N/scratch/mjpathar/AudioGroove/data/clean_midi \
  --pilot-root /N/u/mjpathar/BigRed200/AudioGroove/data/audit/lmdclean_pilot_250 \
  --output-root /N/scratch/mjpathar/AudioGroove/audits/lmdclean_scaling_v1
