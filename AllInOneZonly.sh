#!/bin/bash
#PBS -P v46
#PBS -q normal
#PBS -l ncpus=4
#PBS -l mem=64GB
#PBS -l walltime=16:00:00
#PBS -l storage=scratch/v46+gdata/v46+gdata/xp65+gdata/rq0
#PBS -l wd
#PBS -j oe
#PBS -m abe
#PBS -M skyler.graap@student.unimelb.edu.au
#PBS -N AllInOne_Test

set -e

# Load python module
# module load python3/3.11.0

module use /g/data/xp65/public/modules
module load conda/analysis3-26.10

# Point to your conda environment's python
# PY=/opt/conda/analysis3-26.10/bin/python3

# Run the script
python3 /home/563/sg3241/Notebooks/PhD/AllInOneZonly.py