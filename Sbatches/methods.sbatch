#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=50iterXGRF # Name of the job 
#SBATCH --export=ALL              # Export all environment variables
#SBATCH --output=rfxg.log     # Log-file
#SBATCH --cpus-per-task=5         # More CPUs for parallelism
#SBATCH --mem-per-cpu=24G         # Increased memory for larger datasets
#SBATCH --gres=gpu:1              # Keep 1 GPU (or 2 if available)
#SBATCH --time=200:00:00           # Reduced time with optimizations

# Activate the Anaconda environment in which to execute the Jupyter instance.
eval "$(conda shell.bash hook)"
conda activate myenv 


cd /home/sbamboneyeho/thesis 
python train.py