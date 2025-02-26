#!/bin/bash
#SBATCH --partition=gpu-single       # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=16:00:00              # Max runtime
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=4gb                    # Memory
#SBATCH --job-name=mmd_job    # Job name
#SBATCH --output=slurm/mmd_run_%j.out    # Standard output file (%j will be replaced with the job ID)
#SBATCH --error=slurm/mmd_run_%j.err     # Standard error file

# Load the correct CUDA module (12.1)
module load devel/cuda/12.1

# Activate the Python virtual environment
source /home/tu/tu_tu/tu_wwdth01/DPN-SA-MMD_extension/venv/bin/activate

# Move to the repository root directory
cd /home/tu/tu_tu/tu_wwdth01/DPN-SA-MMD_extension/

# Run the main script
python mmd_main.py

