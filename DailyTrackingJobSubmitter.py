# TITLE
# DailyTrackingJobSubmitter.py
# TITLE

# This python file creates and submits one job to run all radar QC and gridding and tracking for individual time steps/ time deltas between the dates you set

import subprocess
from datetime import date, timedelta

# --- CHOOSE YOUR DATE RANGE ---
LoopStartDate = date(2024,  1,  1)
LoopEndDate   = date(2024,  1, 31)

# --- CHOOSE YOUR OTHER SETTINGS ---
RadarID  = '19'
StartTime = '0000'
EndTime   = '2355'

# --- LOOP OVER EVERY DAY ---
CurrentDate = LoopStartDate
while CurrentDate <= LoopEndDate:

    year  = CurrentDate.year
    month = CurrentDate.month
    day   = CurrentDate.day

    # Build a unique job name for each day
    job_name = f"AIO_{RadarID}_{year}{str(month).zfill(2)}{str(day).zfill(2)}"

    # Build the qsub command, overriding the job name and passing variables
    qsub_command = f"""qsub \
        -N {job_name} \
        -v RADAR_ID={RadarID},YEAR={year},MONTH={month},DAY={day},START={StartTime},END={EndTime} \
        /home/563/sg3241/Notebooks/PhD/AllInOneZonly.sh"""

    print(f"Submitting job for {CurrentDate}: {job_name}")
    result = subprocess.run(qsub_command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  Submitted: {result.stdout.strip()}")
    else:
        print(f"  ERROR: {result.stderr.strip()}")

    CurrentDate += timedelta(days=1)

print("All jobs submitted.")
