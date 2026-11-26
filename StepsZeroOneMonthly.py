# TITLE
# StepsZeroOneMonthly.py
# TITLE

# this python file takes BOM-provided gridded radar data 
# and performs STEPS ZERO and ONE, creating files for just cell identification and overlap (with midnight bridges)
# for full months of user choice (not including buffer days in adjacent months)

# SPECIAL METHOD TO IMPORT CUSTOM FUNCTIONS AND ELEVATION FROM LOCAL DIRECTORY
import sys
sys.path.append('/home/563/sg3241/Notebooks/PhD/CustomFunctions')
sys.path.append('/home/563/sg3241/Notebooks')
sys.path.insert(0, '/scratch/v46/sg3241/tmp/PyFLEXTRKR/')  # folder *containing* pyflextrkr/
from CustomFunctions1 import *

# IMPORT PYFLEXTRKR
import logging
logging.basicConfig(level=logging.INFO)
from pyflextrkr.idcells_reflectivity import idcells_reflectivity
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import pandas as pd

import pyproj # for projecting radar xyz coordinates to lat/lon

from pathlib import Path      # used to play with pathnames to save
import os                     # used for retrieving file names
import glob
import shutil # I think this is to rename the output file from PyFLEXTRKR
import gc # something to prevent memory leaks
import warnings

import time

import argparse # for grabbing variables values from a shell script

# GRAB THESE VARIABLES FROM A SHELL SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument('--radar_id', type=str)
parser.add_argument('--year',     type=int)
parser.add_argument('--month',    type=int)

args = parser.parse_args()

RadarIDno     = args.radar_id
# CHOOSE THE DATE, ALL IN UTC
RadarYear     = args.year
RadarMonth    = args.month
# CHOOSE THE MINUTES YOU WANT TO LOOP OVER (5-MIN PERIODS) (INCLUSIVE OF START AND END TIMES)
# Convert HHMM string back to HH:MM format
LoopStartTime = '00:00'
LoopEndTime   = '23:59'

# find out how many days are in the user's choice of month
OrdinaryMonthLengths = [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
LeapMonthLengths     = [ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

if (RadarYear%4 == 0):
    MonthLength = LeapMonthLengths
else:
    MonthLength = OrdinaryMonthLengths

MonthLength = MonthLengths[RadarMonth - 1]

# can only be set to 5 as far as I know, since I never made another version
TrackingVersionOption = 5
# but hypothetically can be used to set different configurations of the tracking algorithm

# USER CHOICE FOLLOW-ON SECTION

# RADAR CHOICE FOLLOW-ON
if (RadarIDno == '22'):
    RadarSiteName = 'Mackay'
elif (RadarIDno == '106'):
    RadarSiteName = 'Townsville'
elif (RadarIDno == '66'):
    RadarSiteName = 'Mt Staplyton (Brisbane)'
elif (RadarIDno == '50'):
    RadarSiteName = 'Marburg'
elif (RadarIDno == '19'):
    RadarSiteName = 'Cairns'
elif (RadarIDno == '8'):
    RadarSiteName = 'Gympie'
elif (RadarIDno == '24'):
    RadarSiteName = 'Bowen'
elif (RadarIDno == '23'):
    RadarSiteName = 'Gladstone'
elif (RadarIDno == '74'):
    RadarSiteName = 'Greenvale'
elif (RadarIDno == '98'):
    RadarSiteName = 'Taroom'
elif (RadarIDno == '108'):
    RadarSiteName = 'Towoomba'
elif (RadarIDno == '78'):
    RadarSiteName = 'Weipa'
elif (RadarIDno == '72'):
    RadarSiteName = 'Emerald'
else:
    RadarSiteName = 'Site ' + RadarIDno

# TOWNSVILLE LON NEEDS TO BE SHIFTED
# TOWNSVILLE LON NEEDS TO BE SHIFTED
# TOWNSVILLE LON NEEDS TO BE SHIFTED
# TOWNSVILLE LON NEEDS TO BE SHIFTED
# TOWNSVILLE LON NEEDS TO BE SHIFTED
# Longitude shift correction for known coordinate errors
if RadarSiteName == 'Townsville':
    LonShift = 146.5505 - (-19.4195)
else:
    LonShift = 0.0


# TRACKING
# STEP ZERO
# IDENTIFY OBJECTS

for RadarDay in range(0,MonthLength+1):
    
    # date choice follow-on
    # add leading zeros for strings
    YYYY = str(RadarYear).zfill(4)
    MM = str(RadarMonth).zfill(2)
    DD = str(RadarDay).zfill(2)
    # write out the data in one string with and without dashes
    RadarFileDate  = YYYY + MM + DD
    RadarFileDatePrint = YYYY + '-' + MM + '-' + DD
    
    # when reading in pre-gridded files, they are stored with all 288 (or fewer times) a day in one .nc file
    # so read in one file for the day before looping over each time
    RadarGridFolder = '/g/data/rq0/level_1b/v2026/grid/' + RadarIDno + '/' + YYYY  + '/'
    RadarGridFile   =  RadarIDno + '_' + RadarFileDate + '_grid.nc'
    
    RadarGridPath = RadarGridFolder + RadarGridFile
    
    # --- Check file exists without loading it into memory ---
    if not os.path.exists(RadarGridPath):
        print(f'File missing for {RadarFileTimePrint}, skipping: {RadarGridPath}')
    
    
    # Load the daily file once before the loop
    RadarGridXR = xr.open_dataset(RadarGridPath)
    
    # CALCULATE LAT/LON COORDINATES FROM RADAR COORDINATES
    # Pull radar site coordinates from file attributes
    radar_lat = RadarGridXR.attrs['origin_latitude']
    radar_lon = RadarGridXR.attrs['origin_longitude']
    # Set up azimuthal equidistant projection centred on the radar
    proj = pyproj.Proj(proj='aeqd', lat_0=radar_lat, lon_0=radar_lon, datum='WGS84', units='m')
    # Compute 2D lon/lat grids from the 1D x/y coordinates
    xx, yy = np.meshgrid(RadarGridXR['x'].values, RadarGridXR['y'].values)
    lon2d, lat2d = proj(xx, yy, inverse=True)
    # Add lat/lon and radar position as coordinates to the dataset
    RadarGridXR = RadarGridXR.assign_coords(
        lat            = xr.DataArray(lat2d, dims=['y', 'x']),
        lon            = xr.DataArray(lon2d, dims=['y', 'x']),
        radar_latitude = radar_lat,
        radar_longitude= radar_lon)
    
    
    # Build period bounds as numpy datetime64 for comparison
    time_start = np.datetime64(f'{YYYY}-{MM}-{DD}T{LoopStartTime}:00')
    time_end   = np.datetime64(f'{YYYY}-{MM}-{DD}T{LoopEndTime}:59')
    
    # Filter the time coordinate to the desired window
    all_times     = RadarGridXR['time'].values
    selected_times = all_times[(all_times >= time_start) & (all_times <= time_end)]
    
    print(f"Total times in file:    {len(all_times)}")
    print(f"Times within period:    {len(selected_times)}")
    
    
    for t in selected_times:
    
        # Build human-readable string for printing
        t_pd               = pd.Timestamp(t)
        RadarFileTime      = t_pd.strftime('%H%M%S')
        RadarFileTimePrint = t_pd.strftime('%H:%M:%S')
        scheduled_date     = t_pd.strftime('%Y%m%d')
    
        print(f'Working on {RadarFileTimePrint}')
    
        # --- Output path ---
        output_path = (
            f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V{TrackingVersionOption}/'
            f'QCgrid/Radar{RadarIDno}/{YYYY}/{YYYY}{MM}{DD}/SingleTimes/'
        )
        os.makedirs(output_path, exist_ok=True)
    
        # --- Skip if output already exists ---
        scheduled_outfile = output_path + f'cellidfile_{scheduled_date}_{RadarFileTime}.nc'
        if os.path.exists(scheduled_outfile):
            print(f'Already exists, skipping')
            continue
    
        # --- Slice to this single timestep and write to temp file ---
        RadarGridXR_slice = RadarGridXR.sel(time=t).expand_dims('time')
        tmp_path = f'/scratch/v46/sg3241/tmp/tmp_slice_{RadarFileTime}.nc'
        RadarGridXR_slice.to_netcdf(tmp_path)
    
        # --- Config ---
        pyflextrkr_config = {
            'input_format'          : 'netcdf',
            'reflectivity_varname'  : 'corrected_reflectivity',
            'time_dimname'          : 'time',
            'z_dimname'             : 'z',
            'y_dimname'             : 'y',
            'x_dimname'             : 'x',
            'x_coordname'           : 'x',
            'y_coordname'           : 'y',
            'z_coordname'           : 'z',
            'lon_coordname'         : 'lon',
            'lat_coordname'         : 'lat',
            'radar_lon_varname'     : 'radar_longitude',
            'radar_lat_varname'     : 'radar_latitude',
            'dx'                    : 1000.0,
            'dy'                    : 1000.0,
            'is_3d'                 : True,
            'z_coord_type'          : 'height',
            'sfc_dz_min'            : 500.0,
            'sfc_dz_max'            : 3000.0,
            'echotop_gap'           : 2,
            'default_sfc_height'    : 0.0,
            'radar_sensitivity'     : 0.0,
            'absConvThres'          : 30.0,
            'minZdiff'              : 10.0,
            'truncZconvThres'       : 42.0,
            'mindBZuse'             : 10.0,
            'dBZforMaxConvRadius'   : 45.0,
            'conv_rad_increment'    : 0.2,
            'conv_rad_start'        : 1.0,
            'bkg_refl_increment'    : 5.0,
            'maxConvRadius'         : 1.0,
            'radii_expand'          : [1, 2, 3, 4, 5],
            'weakEchoThres'         : 15.0,
            'bkgrndRadius'          : 11.0,
            'min_corearea'          : 4.0,
            'min_cellarea'          : 4.0,
            'tracking_outpath'      : output_path,
            'cloudid_filebase'      : 'cellidfile_tmp_',
            'fillval'               : -9999,
            'return_diag'           : False,
            'convolve_method'       : 'fft',
            'dilate_method'         : 'orig',
            'expand_method'         : 'orig',
            'echotop_method'        : 'orig',
            'remove_smallcores'     : True,
            'remove_smallcells'     : False,
        }
    
        # --- Run identification ---
        try:
            outfile = idcells_reflectivity(tmp_path, pyflextrkr_config)
            shutil.move(outfile, scheduled_outfile)
            print(f'Output written to: {scheduled_outfile}')
        except Exception as e:
            print(f'FAILED for {RadarFileTimePrint}: {e}')
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            gc.collect()
    
    RadarGridXR.close()






# TRACKING
# STEP ONE
for RadarDay in range(0,MonthLength-27):
    
    # date choice follow-on
    # add leading zeros for strings
    YYYY = str(RadarYear).zfill(4)
    MM = str(RadarMonth).zfill(2)
    DD = str(RadarDay).zfill(2)
    # write out the data in one string with and without dashes
    RadarFileDate  = YYYY + MM + DD
    RadarFileDatePrint = YYYY + '-' + MM + '-' + DD

    cellid_path   = f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V5/QCgrid/Radar{RadarIDno}/{YYYY}/{RadarFileDate}/SingleTimes/'
    tracking_path = f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V5/QCgrid/Radar{RadarIDno}/{YYYY}/{RadarFileDate}/Tracking/'
    stats_path    = f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V5/QCgrid/Radar{RadarIDno}/{YYYY}/{RadarFileDate}/Stats/'
    
    os.makedirs(tracking_path, exist_ok=True)
    os.makedirs(stats_path,    exist_ok=True)
    
    # --- Get start/end times from available cellid files ---
    cellid_files = sorted(glob.glob(os.path.join(cellid_path, 'cellidfile_*.nc')))
    print(f"Found {len(cellid_files)} cellid files")
    
    def datetime_from_cellid(filepath):
        basename = os.path.basename(filepath)           # cellidfile_20241205_120000.nc
        parts    = basename.replace('.nc', '').split('_')
        return pd.Timestamp(f"{parts[1]} {parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:6]}")
    
    t_start = datetime_from_cellid(cellid_files[0])
    t_end   = datetime_from_cellid(cellid_files[-1])
    
    # Convert to epoch seconds (required by tracksingle_driver)
    epoch           = pd.Timestamp('1970-01-01')
    start_basetime  = (t_start - epoch).total_seconds()
    end_basetime    = (t_end   - epoch).total_seconds()
    
    startdate_str   = t_start.strftime('%Y%m%d.%H%M%S')
    enddate_str     = t_end.strftime('%Y%m%d.%H%M%S')
    
    print(f"Time range: {t_start} to {t_end}")
    print(f"Epoch seconds: {start_basetime} to {end_basetime}")
    
    # --- Full config ---
    config = {
        # --- Feature type ---
        'feature_type'                  : 'radar_cells',
    
        # --- Paths ---
        'tracking_outpath'              : cellid_path,    # where cellid files live
        'stats_outpath'                 : stats_path,
    
        # --- File bases ---
        'cloudid_filebase'              : 'cellidfile_',
        'singletrack_filebase'          : 'track_',
        'tracknumbers_filebase'         : 'tracknumbers_',
        'trackstats_filebase'           : 'trackstats_',
        'trackstats_sparse_filebase'    : 'trackstats_sparse_',
        'pixeltracking_filebase'        : 'celltracks_',
    
        # --- Time range ---
        'startdate'                     : startdate_str,
        'enddate'                       : enddate_str,
        'start_basetime'                : start_basetime,
        'end_basetime'                  : end_basetime,
        'time_format'                   : 'yyyymodd.hhmmss',
        'datatimeresolution'            : 5.0 / 60.0,     # hours (5-min data)
    
        # --- Grid parameters ---
        'dx'                            : 1000.0,          # metres
        'dy'                            : 1000.0,          # metres
        'pixel_radius'                  : 1.0,             # km
        'area_method'                   : 'fixed',
    
        # --- Variable names in cellid files ---
        'feature_varname'               : 'feature_number',
        'nfeature_varname'              : 'nfeatures',
        'featuresize_varname'           : 'npix_feature',
        'ref_varname'                   : 'dbz_comp',
    
        # --- Dimension names in cellid files ---
        'x_dimname'                     : 'lon',
        'y_dimname'                     : 'lat',
        'time_dimname'                  : 'time',
    
        # --- Tracking parameters ---
        'timegap'                       : 0.25,            # hours
        'othresh'                       : 0.3,             # 30% overlap
        'nmaxlinks'                     : 10,
        'maxnclouds'                    : 1000,
        'fillval'                       : -9999,
    
        # --- Track duration ---
        'duration_range'             : [4, 288],   # min 2 timesteps, max 288 (full day)
        'duration_range_auto_update' : True,      # automatically adjust based on data
        'duration_range_round_base'  : 1,          # round auto-updated duration to nearest 10
        'remove_shorttracks'         : 1,         # remove tracks shorter than duration_range[0]
    
    
        # --- Output options ---
        'trackstats_dense_netcdf'       : 1,
        'match_pixel_dt_thresh'         : 60.0,            # seconds
    
        # --- Dimension names in output files ---
        'tracks_dimname'                : 'tracks',
        'times_dimname'                 : 'times',
    
        # --- Parallelisation (0=serial, 1=local dask cluster) ---
        'run_parallel'                  : 0,
    
        # --- No terrain file ---
        'terrain_file'                  : None,
        'rangemask_varname'             : None,
    }

    # TRACKING
    # STEP ONE
    
    # ── Step 1: Pairwise linking ───────────────────────────────────────────────────
    print("\n--- Step 1: tracksingle (pairwise linking) ---")
    tracksingle_driver(config)
    print("tracksingle complete.")


# TRACKING
# STEP ONE
# MIDNIGHT BRIDGE
# NOT PARALLELISED
for RadarDay in range(0,MonthLength+1):
    
    # date choice follow-on
    # add leading zeros for strings
    YYYY = str(RadarYear).zfill(4)
    MM = str(RadarMonth).zfill(2)
    DD = str(RadarDay).zfill(2)
    # write out the data in one string with and without dashes
    RadarFileDate  = YYYY + MM + DD
    RadarFileDatePrint = YYYY + '-' + MM + '-' + DD

# --- HELPER: get the cellid files for a given date ---
def get_cellid_files_for_date(radar_id, date_obj):
    path = (f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V{TrackingVersionOption}/'
        f'QCgrid/Radar{radar_id}/{YYYY}/{YYYY}{MM}{DD}/SingleTimes/')
    # Exclude symlinks so previous bridge symlinks do not pollute the file list
    files = sorted([
        f for f in glob.glob(os.path.join(path, 'cellidfile_*.nc'))
        if not os.path.islink(f)
    ])
    return files

# --- HELPER: extract timestamp from cellid filename ---
def datetime_from_cellid(filepath):
    basename = os.path.basename(filepath)           # cellidfile_20240101_120000.nc
    parts    = basename.replace('.nc', '').split('_')
    return pd.Timestamp(f"{parts[1]} {parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:6]}")

# --- LOOP OVER CONSECUTIVE DAY PAIRS ---
CurrentDate = BridgeStartDate
while CurrentDate < BridgeEndDate:

    NextDate = CurrentDate + timedelta(days=1)

    print(f"\n--- Bridge: {CurrentDate} --> {NextDate} ---")

    # Get files for each day
    files_today    = get_cellid_files_for_date(RadarIDno, CurrentDate)
    files_tomorrow = get_cellid_files_for_date(RadarIDno, NextDate)

    if not files_today:
        print(f"  No cellid files found for {CurrentDate}, skipping.")
        CurrentDate += timedelta(days=1)
        continue

    if not files_tomorrow:
        print(f"  No cellid files found for {NextDate}, skipping.")
        CurrentDate += timedelta(days=1)
        continue

    # Grab last scan of today and first scan of tomorrow
    last_file_today    = files_today[-1]
    first_file_tomorrow = files_tomorrow[0]

    t_last  = datetime_from_cellid(last_file_today)
    t_first = datetime_from_cellid(first_file_tomorrow)

    print(f"  Last scan today:      {t_last}  ({os.path.basename(last_file_today)})")
    print(f"  First scan tomorrow:  {t_first} ({os.path.basename(first_file_tomorrow)})")

    # Check the gap is not too large (should be ~5 min, warn if more)
    gap_minutes = (t_first - t_last).total_seconds() / 60
    print(f"  Gap between scans: {gap_minutes:.1f} minutes")
    if gap_minutes > 15:
        print(f"  WARNING: gap is larger than 15 minutes, bridge may not be meaningful.")

    # --- Build a minimal config for just these two files ---
    epoch          = pd.Timestamp('1970-01-01')
    start_basetime = (t_last  - epoch).total_seconds()
    end_basetime   = (t_first - epoch).total_seconds()
    startdate_str  = t_last.strftime('%Y%m%d.%H%M%S')
    enddate_str    = t_first.strftime('%Y%m%d.%H%M%S')

    # Bridge tracking output goes into the EARLIER day's tracking folder
    YYYY = str(CurrentDate.year).zfill(4)
    MM   = str(CurrentDate.month).zfill(2)
    DD   = str(CurrentDate.day).zfill(2)
    RadarFileDate = YYYY + MM + DD

    # The cellid files span two different folders, so we need a combined input path
    # tracksingle_driver searches for files by glob in tracking_outpath,
    # so we temporarily symlink or copy the next day's first file into today's folder
    bridge_file_link = os.path.join(
        os.path.dirname(last_file_today),
        os.path.basename(first_file_tomorrow)
    )

    # Create symlink if it does not already exist
    if not os.path.exists(bridge_file_link):
        os.symlink(first_file_tomorrow, bridge_file_link)
        print(f"  Symlinked: {bridge_file_link}")
    else:
        print(f"  Symlink already exists: {bridge_file_link}")

    bridge_tracking_path = (
        f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V{TrackingVersionOption}/'
        f'QCgrid/Radar{RadarIDno}/{YYYY}/{RadarFileDate}/Tracking/Bridge/'
    )
    os.makedirs(bridge_tracking_path, exist_ok=True)

    # --- Update existing config for this bridge ---
    config['startdate']         = startdate_str
    config['enddate']           = enddate_str
    config['start_basetime']    = start_basetime
    config['end_basetime']      = end_basetime
    config['tracking_outpath'] = os.path.dirname(last_file_today) + '/'
    config['stats_outpath']     = bridge_tracking_path
    config['duration_range']    = [1, 2]
    config['remove_shorttracks']= 0

    # ASSIGN THE MIDNIGHT BRIDGE TRACKING FILES TO THE FORMER DAY
    # THEN MOVE IT TO THE LATTER DAY ONCE IT IS CREATED
    # Snapshot existing track files BEFORE running
    existing_tracks = set(glob.glob(os.path.join(config['tracking_outpath'], 'track_*.nc')))
    
    # --- Run just Step 1 for this bridge ---
    try:
        print(f"  Running tracksingle for bridge...")
        tracksingle_driver(config)
    
        # Find only NEW track files created during this run
        all_tracks_after = set(glob.glob(os.path.join(config['tracking_outpath'], 'track_*.nc')))
        new_tracks = all_tracks_after - existing_tracks
    
        print(f"  New track files written: {new_tracks}")
    
        for f in new_tracks:
            dest = os.path.join(os.path.dirname(first_file_tomorrow), os.path.basename(f))
            shutil.move(f, dest)
            print(f"  Moved track file to: {dest}")
    
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
    finally:
        # Remove the symlink after use so it does not pollute today's folder
        if os.path.islink(bridge_file_link):
            os.unlink(bridge_file_link)
            print(f"  Cleaned up symlink: {bridge_file_link}")

    CurrentDate += timedelta(days=1)


print("\nAll bridges processed.")