#!/bin/bash
#PBS -P v46
#PBS -q normal
#PBS -l ncpus=4
#PBS -l mem=64GB
#PBS -l walltime=04:00:00
#PBS -l storage=scratch/v46+gdata/v46+gdata/xp65+gdata/rq0
#PBS -l wd
#PBS -j oe
#PBS -m abe
#PBS -M skyler.graap@student.unimelb.edu.au

set -e

module use /g/data/xp65/public/modules
module load conda/analysis3-26.10

# RADAR_ID, YEAR, MONTH, DAY, START, END are passed in via qsub -v
python3 -u /home/563/sg3241/Notebooks/PhD/AllInOneZonly.py \
    --radar_id $RADAR_ID \
    --year     $YEAR     \
    --month    $MONTH    \
    --day      $DAY      \
    --start    $START    \
    --end      $END
