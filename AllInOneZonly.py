# TITLE
# AllInOneZonly.py
# TITLE

# This python file is a conglomeration of all of the jupyter notebooks I have created which quality control the original radar data, grid the ppi data, and run PyFLEXTRKR on them

# SPECIAL METHOD TO IMPORT CUSTOM FUNCTIONS AND ELEVATION FROM LOCAL DIRECTORY
import sys
sys.path.append('/home/563/sg3241/Notebooks/PhD/CustomFunctions')
sys.path.append('/home/563/sg3241/Notebooks')
sys.path.insert(0, '/scratch/v46/sg3241/tmp/PyFLEXTRKR/')  # folder *containing* pyflextrkr/
from CustomFunctions1 import *
# AND IMPORT LEROI GRIDDING
from leroi.leroi import *
# IMPORT PYFLEXTRKR
import logging
logging.basicConfig(level=logging.INFO)
from pyflextrkr.idcells_reflectivity import idcells_reflectivity
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver

# # OPENING IMPORTS
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr        # used for reading in Net CDF files
import pandas as pd

import zipfile as zp       # used for unzipping ppi files
from pathlib import Path   # used to play with pathnames to save 
from datetime import datetime # used to manipulate time :)

import wradlib as wr          # used for having fun with radar data

from PIL import Image         # used for creating gif loops
import os                     # used for retrieving file names

import h5py                   # used for reading .h5 files (Radar Level 1 data)
import h5netcdf               # used for converting .h5 files to NetCDF

# TAKEN FROM "Part4IntroductionToGridding"
import cartopy.crs as ccrs
import pyart
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import glob
import time

import shutil # I think this is to rename the output file from PyFLEXTRKR
import gc # something to prevent memory leaks
import warnings

# FOR MAP THINGS
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

# for adding lat/lon gridlines on plots
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER 

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

GlobalRadarIDno     = args.radar_id
GlobalRadarYear     = args.year
GlobalRadarMonth    = args.month
GlobalRadarDay      = args.day
# Convert HHMM string back to HH:MM format
GlobalLoopStartTime = args.start[:2] + ':' + args.start[2:]  # '0000' -> '00:00'
GlobalLoopEndTime   = args.end[:2]   + ':' + args.end[2:]    # '2355' -> '23:55'






# GRIDDING STEP ONE
# UNZIPPING


# CHOOSE YOUR RADAR

# Radar Number Catalogue:
# Down The Coast YES Dual-Pol: 22 is Mackay,    106 is Townsville, 66 is Mt Staplyton (Brisbane) , 50 is Marburg (near Bris)
# Down The Coast not Dual-Pol: 19 is Cairns,    8 is Gympie
# Down The Coast NO DOPPLER:   24 is Bowen,     23 is Gladstone
# 41 is Willis Island
#Inland
# Down inland YES Dual-Pol:    74 is Greenvale, 98 is Taroom (PROBLEM UNZIPPING),    108 is Towoomba
# Down inland not Dual-Pol:    78 is Weipa,     72 is Emerald
# RadarIDno = '22'  
GridOrPPI = 'ppi'

# the day in consideration (YYYYMMDD) and time (hhmmss)
# ALL IN UTC !!!
RadarYear  = GlobalRadarYear 
RadarMonth = GlobalRadarMonth
RadarDay   = GlobalRadarDay

for RadarIDnos  in [GlobalRadarIDno]: #22, 106, 66, 50, 19, 8, 24, 23, 74, 98, 108, 78, 72
    RadarIDno = str(RadarIDnos)
    for RadarDay in range(RadarDay,RadarDay+1):
        
        YYYY = str(RadarYear).zfill(4)
        MM = str(RadarMonth).zfill(2)
        DD = str(RadarDay).zfill(2)
        
        RadarFileDate  = YYYY + MM + DD
        RadarFileDatePrint = RadarFileDate[0:4] + '-' + RadarFileDate[4:6] + '-' + RadarFileDate[6:8] # add a string of format YYYY-MM-DD for printing
        print('Working on ' + RadarFileDatePrint)
        
        # THIS BLOCK TAKES A ZIPPED FOLDER FROM /G/DATA/ AND UNZIPS IT TO A FOLDER "zippedRadarFiles" IN SCRATCH
        
        # PATH NAMES
        # ZippedFolder = '/g/data/rq0/level_1b/22/ppi/2024/'
        ZippedFolder = '/g/data/rq0/level_1b/v2026/' + GridOrPPI + '/' + RadarIDno + '/' + str(RadarYear) + '/'
        ZippedFile   =  RadarIDno + '_' + RadarFileDate + '_' + GridOrPPI + '.zip'

        # UPDATED ON 2026-09-09W since the new 2026 version was released

        
        # place where the zipped file lives
        ZippedPath = ZippedFolder + ZippedFile
        # place to extract the files to
        ExtractToDirectory = Path('/scratch/v46/sg3241/tmp/UnzippedRadarFiles/QC0/Radar' + RadarIDno + '/' + YYYY + '/' + RadarIDno + '_' + RadarFileDate + '_' + GridOrPPI + '/')

        # Create the extraction folder (only if you haven't already)
        if not Path(ExtractToDirectory).exists():
            ExtractToDirectory.mkdir(parents=True, exist_ok=True)
        
            # actually do the unzipping
            with zp.ZipFile(ZippedPath, 'r') as ZipReference:
                ZipReference.extractall(ExtractToDirectory)
        
            print('Done')
        else:
            print('Unzipped Folder Already Exists')




# it seems this block might have a reduntant part where it calculates the variances again instead of loading in the existing files
# GRIDDING STEP THREE
# DATA MASKING (QUALITY CONTROL)


# Original Version without Dask

# SHUT UP THE RUN TIME WARNINGS
import warnings # !!!!!
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0', category=RuntimeWarning)


# set your Quality Control Options:
# 0 for no quality control
QualityControlOption = 2

# CHOOSE THE RADAR
RadarIDno = GlobalRadarIDno  # 22 is Mackay, 106 is Townsville, 66 is Mt Staplyton (Brisbane) , 50 is Marabong (near Brisbane)

# CHOOSE THE DATE
# the day in consideration (YYYYMMDD) and time (hhmmss)       ALL IN UTC !!!
RadarYear  = GlobalRadarYear
RadarMonth = GlobalRadarMonth
# RadarDay   = 14 # looped down below

# CHOOSE THE MINUTES YOU WANT TO LOOP OVER (5-MIN PERIODS) (INCLUSIVE OF START AND END TIMES)
LoopStartTime = GlobalLoopStartTime
LoopEndTime   = GlobalLoopEndTime

if (QualityControlOption == 0):
    VarianceGridSize = '3x3'
    MinZ             = 'none' # [dBZ]                           # write 'none' [string] for no limit
    MinRhoHV         = 'none' # [0 to 1 correlation]            # write 'none' [string] for no limit
    
    MaxZvariance     = 'none' # [dBZ^2]                         # write 'none' [string] for no limit
    MaxVvariance     = 'none' # [m^2/s^2]                       # write 'none' [string] for no limit
    MaxZDRvariance   = 'none' # [dB^2]                          # write 'none' [string] for no limit
    MaxRhoHVvariance = 'none' # [0 to 1 correlation squared]    # write 'none' [string] for no limit

    # but maybe you only want to consider those cells with at least a few neighbours!
    MinZcount        =  3     # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinVcount        = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinZDRcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinRhoHVcount    = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinKDPcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    
elif (QualityControlOption == 1):
    VarianceGridSize = '3x3'
    MinZ             =  5     # [dBZ]                           # write 'none' [string] for no limit
    MinRhoHV         =  0.88  # [0 to 1 correlation]            # write 'none' [string] for no limit
    
    MaxZvariance     =  15    # [dBZ^2]                         # write 'none' [string] for no limit
    MaxVvariance     = 'none' # [m^2/s^2]                       # write 'none' [string] for no limit
    MaxZDRvariance   = 'none' # [dB^2]                          # write 'none' [string] for no limit
    MaxRhoHVvariance = 'none' # [0 to 1 correlation squared]    # write 'none' [string] for no limit

    # but maybe you only want to consider those cells with at least a few neighbours!
    MinZcount        =  3     # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinVcount        = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinZDRcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinRhoHVcount    = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinKDPcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    
elif (QualityControlOption == 2):
    VarianceGridSize = '3x3'
    MinZ             =  0     # [dBZ]                           # write 'none' [string] for no limit
    MinRhoHV         =  'none' #0.88  # [0 to 1 correlation]            # write 'none' [string] for no limit
    
    MaxZvariance     =  15    # [dBZ^2]                         # write 'none' [string] for no limit
    MaxVvariance     = 'none' # [m^2/s^2]                       # write 'none' [string] for no limit
    MaxZDRvariance   = 'none' # [dB^2]                          # write 'none' [string] for no limit
    MaxRhoHVvariance = 'none' # [0 to 1 correlation squared]    # write 'none' [string] for no limit

    # but maybe you only want to consider those cells with at least a few neighbours!
    MinZcount        =  2     # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinVcount        = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinZDRcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinRhoHVcount    = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    MinKDPcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
# elif (QualityControlOption == 3):
#     VarianceGridSize = '3x3'
#     MinZ             =  0     # [dBZ]                           # write 'none' [string] for no limit
#     MinRhoHV         =  0.88  # [0 to 1 correlation]            # write 'none' [string] for no limit
    
#     MaxZvariance     =  15    # [dBZ^2]                         # write 'none' [string] for no limit
#     MaxVvariance     = 'none' # [m^2/s^2]                       # write 'none' [string] for no limit
#     MaxZDRvariance   = 'none' # [dB^2]                          # write 'none' [string] for no limit
#     MaxRhoHVvariance = 'none' # [0 to 1 correlation squared]    # write 'none' [string] for no limit

#     # but maybe you only want to consider those cells with at least a few neighbours!
#     MinZcount        =  2     # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
#     MinVcount        = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
#     MinZDRcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
#     MinRhoHVcount    = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
#     MinKDPcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
# elif (QualityControlOption == 4):
    # VarianceGridSize = '3x3'
    # MinZ             =  0     # [dBZ]                           # write 'none' [string] for no limit
    # MinRhoHV         =  0.88  # [0 to 1 correlation]            # write 'none' [string] for no limit
    
    # MaxZvariance     =  15    # [dBZ^2]                         # write 'none' [string] for no limit
    # MaxVvariance     = 'none' # [m^2/s^2]                       # write 'none' [string] for no limit
    # MaxZDRvariance   = 'none' # [dB^2]                          # write 'none' [string] for no limit
    # MaxRhoHVvariance = 'none' # [0 to 1 correlation squared]    # write 'none' [string] for no limit

    # # but maybe you only want to consider those cells with at least a few neighbours!
    # MinZcount        =  2     # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    # MinVcount        = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    # MinZDRcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    # MinRhoHVcount    = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
    # MinKDPcount      = 'none' # [0 to 9 [int] for 3x3 grids, 0 to 25 for 5x5 grids]   # write 'none' [string] for no limit
else:
    print('Please choose QualityControlOption = 0 or 1 or 2') # or 3 or 4


for RadarDay in range(GlobalRadarDay,GlobalRadarDay+1):

    # date choice follow-on
    # add leading zeros for strings
    YYYY = str(RadarYear).zfill(4)
    MM = str(RadarMonth).zfill(2)
    DD = str(RadarDay).zfill(2)
    # write out the data in one string with and without dashes
    RadarFileDate  = YYYY + MM + DD
    RadarFileDatePrint = YYYY + '-' + MM + '-' + DD
    
    
    
    # LOOP OVER EVERY 5 MIN PERIOD IN THE DAY
    # Parse start and end times
    StartHour, StartMin = int(LoopStartTime.split(':')[0]), int(LoopStartTime.split(':')[1])
    EndHour, EndMin     =   int(LoopEndTime.split(':')[0]),   int(LoopEndTime.split(':')[1])
    
    # Convert to total minutes for easy comparison
    StartMinOfDay = StartHour * 60 + StartMin
    EndMinOfDay = EndHour * 60 + EndMin
    # LOOP OVER EVERY 5 MIN PERIOD IN THE DAY
    # Parse start and end times
    StartHour, StartMin = int(LoopStartTime.split(':')[0]), int(LoopStartTime.split(':')[1])
    EndHour, EndMin     =   int(LoopEndTime.split(':')[0]),   int(LoopEndTime.split(':')[1])
    
    # Convert to total minutes for easy comparison
    StartMinOfDay = StartHour * 60 + StartMin
    EndMinOfDay = EndHour * 60 + EndMin
    
    for MinOfDay in range(StartMinOfDay, EndMinOfDay + 1, 5):
        # find the old indicies with which this code was written (0-23 for hours, 0-11 for 5-minute periods within hours)
        houri = MinOfDay // 60
        mini  = MinOfDay % 60
            
        RadarFileTime = str(houri).zfill(2) + str(mini).zfill(2) + '00' # write out the time in 6 digits (like 012040 for 01:20:40 AM)
        # add a string of format hh:mm:ss for printing
        RadarFileTimePrint = str(RadarFileTime)[0:2] + ':' + str(RadarFileTime)[2:4] + ':' + str(RadarFileTime)[4:6] 
        
        print('working on ' + RadarFileDatePrint + ' ' + RadarFileTimePrint)
        
        NetCDFstorageFolder = '/scratch/v46/sg3241/tmp/UnzippedRadarFiles/QC0/Radar' + RadarIDno + '/' + YYYY + '/' + RadarIDno + '_' + RadarFileDate + '_ppi' + '/'
        NetCDFstorageFile = RadarIDno + '_' + RadarFileDate + '_' + RadarFileTime + '_ppi.nc'
        NetCDFstoragePath = NetCDFstorageFolder + NetCDFstorageFile
        
        # try to load in the netcdf file and if it doesn't work, just keep going
        try:
            RadarXR = xr.open_dataset(NetCDFstoragePath, decode_timedelta = False) # add the decode_timedelta to shut up a warning
        except FileNotFoundError:
            print(f'File missing for {RadarFileTimePrint}, skipping: {NetCDFstoragePath}')
            continue
    
    
        if (VarianceGridSize == '3x3'):
            if (MaxZvariance != 'none'):
                Zvar, Zcount = GridStatsFast(RadarXR, 'corrected_reflectivity', 3, fill_value=-32.0)                                   # add variances and counts for Z
            if (MaxVvariance != 'none'):
                Vvar, Vcount = GridStatsFast(RadarXR, 'corrected_velocity',     3, fill_value=[-300, 0.009155552842798897])            # add variances and counts for velocity 
            if (MaxRhoHVvariance != 'none'):
                RhoHVvar, RhoHVcount = GridStatsFast(RadarXR, 'corrected_cross_correlation_ratio', 3, fill_value=[0.0])                # add variances and counts for RhoHV # very specific near 0 value keeps appearing, likely not valid data                                                     
            if (MaxZDRvariance != 'none'):
                ZDRvar, ZDRcount = GridStatsFast(RadarXR, 'corrected_differential_reflectivity', 3, fill_value=[-15.0, 10.00015259254738]) # add variances and counts for ZDR # very specific near +10 value keeps appearing, likely not valid data
        elif (VarianceGridSize == '5x5'):
            if (MaxZvariance != 'none'):
                Zvar, Zcount = GridStatsFast(RadarXR, 'corrected_reflectivity', 5, fill_value=-32.0)                                   # add variances and counts for Z
            if (MaxVvariance != 'none'):
                Vvar, Vcount = GridStatsFast(RadarXR, 'corrected_velocity',     5, fill_value=[-300, 0.009155552842798897])            # add variances and counts for velocity 
            if (MaxRhoHVvariance != 'none'):
                RhoHVvar, RhoHVcount = GridStatsFast(RadarXR, 'corrected_cross_correlation_ratio', 5, fill_value=[0.0])                # add variances and counts for RhoHV # very specific near 0 value keeps appearing, likely not valid data                                                     
            if (MaxZDRvariance != 'none'):
                ZDRvar, ZDRcount = GridStatsFast(RadarXR, 'corrected_differential_reflectivity', 5, fill_value=[-15.0, 10.00015259254738]) # add variances and counts for ZDR # very specific near +10 value keeps appearing, likely not valid data
        else:
            print("VarianceGridSize must be exactly '3x3' or '5x5'")
    
    
        # MASK CREATION
        # --- Initialise mask as all True (keep all points) ---
        # Use a known (time, range) variable to get the right shape/coords
        ReferenceVar = RadarXR['corrected_reflectivity']
        QCmask = xr.ones_like(ReferenceVar, dtype=bool)  # True = keep
        
        # --- Simple threshold conditions ---
        if MinZ != 'none':
            QCmask = QCmask & (RadarXR['corrected_reflectivity'] >= MinZ)
        if MinRhoHV != 'none':
            QCmask = QCmask & (RadarXR['corrected_cross_correlation_ratio'] >= MinRhoHV)
        
        # --- Variance conditions ---
        if MaxZvariance != 'none':
            QCmask = QCmask & (Zvar <= MaxZvariance)
        if MaxVvariance != 'none':
            QCmask = QCmask & (Vvar <= MaxVvariance)
        if MaxZDRvariance != 'none':
            QCmask = QCmask & (ZDRvar <= MaxZDRvariance)
        if MaxRhoHVvariance != 'none':
            QCmask = QCmask & (RhoHVvar <= MaxRhoHVvariance)
        
        # --- Count conditions (minimum valid neighbours) ---
        if MinZcount != 'none':
            QCmask = QCmask & (Zcount >= MinZcount)
        if MinVcount != 'none':
            QCmask = QCmask & (Vcount >= MinVcount)
        if MinZDRcount != 'none':
            QCmask = QCmask & (ZDRcount >= MinZDRcount)
        if MinRhoHVcount != 'none':
            QCmask = QCmask & (RhoHVcount >= MinRhoHVcount)
        if MinKDPcount != 'none':
            QCmask = QCmask & (KDPcount >= MinKDPcount)
    
        
        # APPPLY THE GRAND MASK
        # Build a new dataset, masking only variables with exactly (time, range) dims
        MaskedVars = {}
        for VarName, Da in RadarXR.data_vars.items():
            if Da.dims == ('time', 'range'):
                # Where mask is False, replace with NaN
                MaskedVars[VarName] = Da.where(QCmask)
            else:
                # Keep non-(time, range) variables completely untouched
                MaskedVars[VarName] = Da
    
        RadarXR_QC = RadarXR.assign(MaskedVars)
    
        SaveFolder = '/scratch/v46/sg3241/tmp/UnzippedRadarFiles/QC' + str(QualityControlOption) + '/Radar' + RadarIDno + '/' + YYYY + '/' + RadarIDno + '_' + RadarFileDate + '_ppi' + '/'
        SaveFile = RadarIDno + '_' + RadarFileDate + '_' + RadarFileTime + '_ppi_QC' + str(QualityControlOption) + '.nc'
        SavePath = SaveFolder + SaveFile
    
        # CHAD EDITTED SAVING SECTION WHICH DELETES AND RE-WRITES SAVED FILES
        # Ensure directory exists with proper permissions
        if not Path(SaveFolder).exists():
            Path(SaveFolder).mkdir(parents=True, exist_ok=True)
        
        # Remove existing file to avoid permission conflicts
        if Path(SavePath).exists():
            Path(SavePath).unlink()
        
        # Write with explicit mode
        try:
            RadarXR_QC.to_netcdf(
                SavePath, 
                mode='w',
                encoding={var: {'zlib': True, 'complevel': 4} for var in RadarXR.data_vars}
            )
            print(f"Successfully saved: {SavePath}")
        except PermissionError as e:
            print(f"Permission denied writing to {SavePath}")
            print(f"Error: {e}")





# GRIDDING STEP FOUR
# ACTUAL GRIDDING

# THIS WHOLE BLOCK CREATES AND STORES NET CDF FILES FOR GRIDDED RADAR DATA

# which quality controlled data are you gridding?
QualityControlOption = 2

# the reference number for the radar location
RadarIDno = GlobalRadarIDno  # 22 is Mackay, 106 is Townsville, 66 is Mt Staplyton (Brisbane) , 50 is Marabong (near Bris)

# the day in consideration (YYYYMMDD) and time (hhmmss)       ALL IN UTC !!!
RadarYear  = GlobalRadarYear
RadarMonth = GlobalRadarMonth
RadarDay   = GlobalRadarDay

LoopStartTime = GlobalLoopStartTime
LoopEndTime   = GlobalLoopEndTime

# add leading zeros for strings
YYYY = str(RadarYear).zfill(4)
MM = str(RadarMonth).zfill(2)
DD = str(RadarDay).zfill(2)

RadarFileDate  = YYYY + MM + DD 

# LOOP OVER EVERY 5 MIN PERIOD IN THE DAY
# Parse start and end times
StartHour, StartMin = int(LoopStartTime.split(':')[0]), int(LoopStartTime.split(':')[1])
EndHour, EndMin     =   int(LoopEndTime.split(':')[0]),   int(LoopEndTime.split(':')[1])

# Convert to total minutes for easy comparison
StartMinOfDay = StartHour * 60 + StartMin
EndMinOfDay = EndHour * 60 + EndMin

for MinOfDay in range(StartMinOfDay, EndMinOfDay + 1, 5):
    # find the old indicies with which this code was written (0-23 for hours, 0-11 for 5-minute periods within hours)
    houri = MinOfDay // 60
    mini  = MinOfDay % 60
    
    RadarFileTime = str(houri).zfill(2) + str(mini).zfill(2) + '00' # write out the time in 6 digits (like 012040 for 01:20:40 AM)
    RadarFileTimePrint = str(RadarFileTime)[0:2] + ':' + str(RadarFileTime)[2:4] + ':' + str(RadarFileTime)[4:6] # add a string of format hh:mm:ss for printing
    print('working on ' + RadarFileTimePrint)
    

    # READ IN THE DATA TO A PYART FILE
    
    # the path of where the UNZIPPED radar data now live
    RadarFolder = '/scratch/v46/sg3241/tmp/UnzippedRadarFiles/QC' + str(QualityControlOption) + '/Radar' + RadarIDno + '/' + YYYY + '/' + RadarIDno + '_' + RadarFileDate + '_ppi' + '/'   
    RadarFile   = RadarIDno + '_' + RadarFileDate + '_' + RadarFileTime + '_ppi_QC' + str(QualityControlOption) + '.nc'
    
    # open the radar data
    try:
        radar = pyart.io.read(RadarFolder + RadarFile, delay_field_loading = True)
    except FileNotFoundError:
        print(f'File missing for {RadarFileTimePrint}, skipping: {RadarFolder + RadarFile}')
        continue
 
    # radar = pyart.io.read(RadarFolder + RadarFile, delay_field_loading=False)
    
    # # RETRIEVE SOME FIELDS IN THE RADAR DATA
    # preferred_fields = ['corrected_reflectivity', 'corrected_velocity', 'corrected_differential_reflectivity',\
    #                     'corrected_cross_correlation_ratio', 'corrected_differential_phase', 'corrected_specific_differential_phase']
    
    # field = next((name for name in preferred_fields if name in radar.fields), next(iter(radar.fields)))
    # fields = [field]
    # field

    # Get ALL available fields from the radar
    fields = list(radar.fields.keys())
    print(f"Gridding fields: {fields}")
    
    #DEFINING THE CARTESIAN GRID SHAPE
    grid_shape = (41, 301, 301)
    grid_limits = ((0.0, 20000.0), (-150000.0, 150000.0), (-150000.0, 150000.0))
    coords = [
        np.linspace(lower, upper, n)
        for (lower, upper), n in zip(grid_limits, grid_shape) ]
    
    # SETS THE WEIGHTING AND CREATES A DICTIONARY OF FIELDS OR SOMETHING
    # I DON'T REALLY KNOW
    # ??????
    grid_fields = leroi_interp(
        radar,
        coords,
        field_names=fields,
        weight_type="Barnes",
        Rc=None,
        k=100,
        verbose=True )
    
    grid = build_pyart_grid(radar, grid_fields, grid_shape, grid_limits)
    # grid
    
    xgrid = grid.to_xarray() # change this loaded grid into an xarray data frame so I know how to work with it

    # save the converted gridded data as a re-usable NetCDF file
    NetCDFsaveFolder = '/scratch/v46/sg3241/tmp/CustomRadarGrids/QC' + str(QualityControlOption) + '/Radar' + RadarIDno + '/' + YYYY + MM + DD + '/'
    NetCDFsaveFile   =  RadarIDno + '_' + RadarFileDate + '_' + RadarFileTime + '_QC' + str(QualityControlOption) + '.nc'
    
    NetCDFsavePath = NetCDFsaveFolder + NetCDFsaveFile
    
    if not Path(NetCDFsaveFolder).exists():
        print('Creating Folder: ' + NetCDFsaveFolder)
        Path(NetCDFsaveFolder).mkdir(parents=True, exist_ok=True)
    
    xgrid.to_netcdf(NetCDFsavePath)




# FEATURE TRACKING STEP ZERO
# IDENTIFY OBJECTS IN EACH TIME STEP

# original non-parallelised version

# can only be set to 5 as far as I know, since I never made another version
TrackingVersionOption = 5
# but hypothetically can be used to set different configurations of the tracking algorithm

# RETRIEVE FEATURES THAT EXIST AT EACH TIME STEP

logging.basicConfig(level=logging.INFO)

# CHOOSE YOUR RADAR

# Radar Number Catalogue:
# Down The Coast YES Dual-Pol: 22 is Mackay,    106 is Townsville, 66 is Mt Staplyton (Brisbane) , 50 is Marburg (near Bris)
# Down The Coast not Dual-Pol: 19 is Cairns,    8 is Gympie
# Down The Coast NO DOPPLER:   24 is Bowen,     23 is Gladstone (SPECIAL ELEVATION ANGLES)
# 0.8, 1.6, 2.4
# 3.6, 5.6
# 8.0, 11.5
# 16.0, 22.0, 32.0
#Inland
# Down inland YES Dual-Pol:    74 is Greenvale, 98 is Taroom,    108 is Towoomba
# Down inland not Dual-Pol:    78 is Weipa,     72 is Emerald
RadarIDno = GlobalRadarIDno

QualityControlOption = 2 # [0 (none) or 1,2, o 3 (int)] # which quality control do you want to grid from?

# CHOOSE THE DATE
# the day in consideration (YYYYMMDD) and time (hhmmss)       ALL IN UTC !!!
RadarYear  = GlobalRadarYear
RadarMonth = GlobalRadarMonth
RadarDay   = GlobalRadarDay
# CHOOSE THE MINUTES YOU WANT TO LOOP OVER (5-MIN PERIODS) (INCLUSIVE OF START AND END TIMES)
LoopStartTime = GlobalLoopStartTime
LoopEndTime   = GlobalLoopEndTime

# date choice follow-on
# add leading zeros for strings
YYYY = str(RadarYear).zfill(4)
MM = str(RadarMonth).zfill(2)
DD = str(RadarDay).zfill(2)
# write out the data in one string with and without dashes
RadarFileDate  = YYYY + MM + DD
RadarFileDatePrint = YYYY + '-' + MM + '-' + DD

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

# LOOP OVER EVERY 5 MIN PERIOD IN THE DAY
# Parse start and end times
StartHour, StartMin = int(LoopStartTime.split(':')[0]), int(LoopStartTime.split(':')[1])
EndHour, EndMin     =   int(LoopEndTime.split(':')[0]),   int(LoopEndTime.split(':')[1])

# Convert to total minutes for easy comparison
StartMinOfDay = StartHour * 60 + StartMin
EndMinOfDay = EndHour * 60 + EndMin

for MinOfDay in range(StartMinOfDay, EndMinOfDay + 1, 5):

    houri = MinOfDay // 60
    mini  = MinOfDay % 60

    RadarFileTime      = str(houri).zfill(2) + str(mini).zfill(2) + '00'
    RadarFileTimePrint = RadarFileTime[0:2] + ':' + RadarFileTime[2:4] + ':' + RadarFileTime[4:6]

    print(f'Working on {RadarFileTimePrint}')

    # RadarGridPath = (
    #     '/scratch/v46/sg3241/tmp/CompressedRadarGrids/'
    #     + RadarIDno + '/' + YYYY + MM + DD + '/'
    #     + RadarIDno + '_' + RadarFileDate + '_' + RadarFileTime + '.nc'
    # )

    RadarGridFolder = '/scratch/v46/sg3241/tmp/CustomRadarGrids/QC' + str(QualityControlOption) + '/Radar' + RadarIDno + '/' + YYYY + MM + DD + '/'
    RadarGridFile   =  RadarIDno + '_' + RadarFileDate + '_' + RadarFileTime + '_QC' + str(QualityControlOption) + '.nc'
    
    RadarGridPath = RadarGridFolder + RadarGridFile

    # --- Check file exists without loading it into memory ---
    if not os.path.exists(RadarGridPath):
        print(f'File missing for {RadarFileTimePrint}, skipping: {RadarGridPath}')
        continue

    # --- Extract scheduled date/time from filename ---
    input_basename = os.path.basename(RadarGridPath)
    parts          = input_basename.replace('.nc', '').split('_')
    scheduled_date = parts[1]
    scheduled_time = parts[2]

    # --- Output path ---
    output_path  = (
            f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V{TrackingVersionOption}/'
            f'QC{QualityControlOption}/Radar{RadarIDno}/{YYYY}/{YYYY}{MM}{DD}/SingleTimes/'
        )
    os.makedirs(output_path, exist_ok=True)

    # --- Skip if output already exists ---
    scheduled_outfile = output_path + f'cellidfile_{scheduled_date}_{scheduled_time}.nc'
    if os.path.exists(scheduled_outfile):
        print(f'Already exists, skipping')
        continue

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
        outfile = idcells_reflectivity(RadarGridPath, pyflextrkr_config)
        shutil.move(outfile, scheduled_outfile)
        print(f'Output written to: {scheduled_outfile}')
    except Exception as e:
        print(f'FAILED for {RadarFileTimePrint}: {e}')

    # --- Explicitly release memory after every timestep ---
    finally:
        gc.collect()





# FEATURE TRACKING STEP ONE
# IDENTIFY OBJECT CONTINUITY BETWEEN FRAMES

RadarIDno      = GlobalRadarIDno
YYYY = str(GlobalRadarYear).zfill(4)
MM = str(GlobalRadarMonth).zfill(2)
DD = str(GlobalRadarDay).zfill(2)

PeriodStart           = YYYY + '-' + MM + '-' + DD + 'UTC' + GlobalLoopStartTime + ':00' 
PeriodEnd             = YYYY + '-' + MM + '-' + DD + 'UTC' + GlobalLoopEndTime   + ':59' 
RadarIDno             = GlobalRadarIDno
QualityControlOption  = 2
TrackingVersionNumber = 5

# FUNNELING IN THE FEATURES FROM INDIVIDUAL TIME STEPS TO GET STATS
sys.path.insert(0, '/scratch/v46/sg3241/tmp/PyFLEXTRKR/')
logging.basicConfig(level=logging.WARNING)


# FIND START AND END TIMES OF PERIODS
def parse_period_timestamp(ts_str):
    return pd.Timestamp(ts_str.replace('UTC', ' '), tz=None)

t_start = parse_period_timestamp(PeriodStart)
t_end   = parse_period_timestamp(PeriodEnd)

if t_end <= t_start:
    raise ValueError(
        f'PeriodEnd ({PeriodEnd}) must be after PeriodStart ({PeriodStart}).'
    )

start_label = t_start.strftime('%Y%m%d')
end_label   = t_end.strftime('%Y%m%d')

# --- Paths ---

cellid_path   = f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V{TrackingVersionOption}/QC{QualityControlOption}/Radar{RadarIDno}/{YYYY}/{YYYY}{MM}{DD}/SingleTimes/'
tracking_path = f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V{TrackingVersionOption}/QC{QualityControlOption}/Radar{RadarIDno}/{YYYY}/{YYYY}{MM}{DD}/tracks/'
stats_path    = f'/scratch/v46/sg3241/tmp/PyFLEXTRKRnetCDFs/V{TrackingVersionOption}/QC{QualityControlOption}/Radar{RadarIDno}/Period_{start_label}_{end_label}/Stats/'

os.makedirs(tracking_path, exist_ok=True)
os.makedirs(stats_path,    exist_ok=True)

# Parse period bounds
t_period_start = parse_period_timestamp(PeriodStart)
t_period_end   = parse_period_timestamp(PeriodEnd)

# Filter cellid files to those within the period
def datetime_from_cellid(filepath):
    basename = os.path.basename(filepath)
    parts    = basename.replace('.nc', '').split('_')
    return pd.Timestamp(f"{parts[1]} {parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:6]}")

all_cellid_files = sorted(glob.glob(os.path.join(cellid_path, 'cellidfile_*.nc')))
print(f"Found {len(all_cellid_files)} total cellid files")

cellid_files = [
    f for f in all_cellid_files
    if t_period_start <= datetime_from_cellid(f) <= t_period_end
]
print(f"Found {len(cellid_files)} cellid files within period")

if len(cellid_files) == 0:
    raise FileNotFoundError(
        f"No cellid files found between {PeriodStart} and {PeriodEnd} in {cellid_path}"
    )

# Now derive t_start and t_end from the filtered list
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

# ── Step 1: Pairwise linking ───────────────────────────────────────────────────
print("\n--- Step 1: tracksingle (pairwise linking) ---")
tracksingle_driver(config)
print("tracksingle complete.")

# SKIP THESE STEPS IF YOU ARE SUBMITTING ONE JOB FOR EACH DAY SINCE THEY WILL ONLY TRACK OBJECTS FOR 24-HOURS BEFORE ASSUMING THEY GO OUT OF EXISTENCE EVERY 24 HOURS

# # FEATURE TRACKING STEP TWO
# # ASSEMBLE FULL TRACKS

# # ── Step 2: Assemble pairs into full tracks ────────────────────────────────────
# print("\n--- Step 2: gettracknumbers (assemble full tracks) ---")
# gettracknumbers(config)
# print("gettracknumbers complete.")


# # FEATURE TRACKING STEP THREE
# # COMPUTE TRACK STATISTICS

# # ── Step 3: Track statistics ───────────────────────────────────────────────────
# print("\n--- Step 3: trackstats (compute statistics) ---")
# trackstats_outfile = trackstats_driver(config)
# print(f"trackstats complete.")
# print(f"Output: {trackstats_outfile}")