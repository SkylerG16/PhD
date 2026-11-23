# TITLE
# AllInOneForGrids.py
# TITLE

# this python file takes BOM-provided gridded radar data 
# and performs all of the steps of tracking precipitation objects

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
parser.add_argument('--day',      type=int)
parser.add_argument('--start', type=str)
parser.add_argument('--end',   type=str)

args = parser.parse_args()

RadarIDno     = args.radar_id
# CHOOSE THE DATE, ALL IN UTC
RadarYear     = args.year
RadarMonth    = args.month
RadarDay      = args.day
# CHOOSE THE MINUTES YOU WANT TO LOOP OVER (5-MIN PERIODS) (INCLUSIVE OF START AND END TIMES)
# Convert HHMM string back to HH:MM format
LoopStartTime = args.start[:2] + ':' + args.start[2:]  # '0000' -> '00:00'
LoopEndTime   = args.end[:2]   + ':' + args.end[2:]    # '2355' -> '23:55'


# TRACKING
# STEP ZERO
# IDENTIFY OBJECTS

logging.basicConfig(level=logging.INFO)

# date choice follow-on
# add leading zeros for strings
YYYY = str(RadarYear).zfill(4)
MM = str(RadarMonth).zfill(2)
DD = str(RadarDay).zfill(2)
# write out the data in one string with and without dashes
RadarFileDate  = YYYY + MM + DD
RadarFileDatePrint = YYYY + '-' + MM + '-' + DD

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
# STEPS One to Three
# required info and configuration

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
# STEPS ONE TWO THREE EXECUTION

# ── Step 1: Pairwise linking ───────────────────────────────────────────────────
print("\n--- Step 1: tracksingle (pairwise linking) ---")
tracksingle_driver(config)
print("tracksingle complete.")

# # ── Step 2: Assemble pairs into full tracks ────────────────────────────────────
# print("\n--- Step 2: gettracknumbers (assemble full tracks) ---")
# gettracknumbers(config)
# print("gettracknumbers complete.")

# # ── Step 3: Track statistics ───────────────────────────────────────────────────
# print("\n--- Step 3: trackstats (compute statistics) ---")
# trackstats_outfile = trackstats_driver(config)
# print(f"trackstats complete.")
# print(f"Output: {trackstats_outfile}")