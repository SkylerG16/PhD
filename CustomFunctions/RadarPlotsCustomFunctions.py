
from scipy.ndimage import map_coordinates
import pandas as pd
from metpy.plots import SkewT
from metpy.units import units
from scipy.ndimage import generic_filter

# OPENING IMPORTS
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import xarray as xr

from pathlib import Path      # used to play with pathnames to save

from PIL import Image         # used for creating gif loops
import os                     # used for retrieving file names

# mapping things
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

import matplotlib.colors as mcolors
import matplotlib.cm as cm

# for adding lat/lon gridlines on plots
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

# for adding a colourful topo base map to the CAPI plots
from custom_elevation import fetch_srtm, fetch_gebco_local
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize

# for projecting radar coordinates to lat and lon
from pyproj import Geod

# RADAR REFLECTIVITY COLOUR MAP
def make_ChadMapZ():
    
    vmin, vmax = -30.0, 100.0
    span = vmax - vmin

    def pos(v):
        return (v - vmin) / span

    stops = [
        # ---- -30 to -20: purple, darker at -30 (far from zero) ----
        (pos(-30), "#3D1A66"),  # dark purple
        (pos(-20), "#B89FE6"),  # light purple

        # ---- SHARP JUMP at -20: beige/tan, darker at -20 ----
        (pos(-20), "#7A6B3D"),  # dark beige
        (pos(-10), "#E8DCA8"),  # light beige

        # ---- SHARP JUMP at -10: pale blue-gray, darker at -10 ----
        (pos(-10), "#8AADC4"),  # darker blue-gray
        (pos(0),   "#E8F4FB"),  # pale icy blue

        # ---- SHARP JUMP at 0: light blue (not cyan), darker toward 10 ----
        (pos(0),   "#C7DCFF"),  # light blue
        (pos(10),  "#3B73DB"),  # medium blue

        # ---- SHARP JUMP at 10: dark blues, darker toward 20 ----
        (pos(10),  "#3355DD"),
        (pos(20),  "#0A1166"),

        # ---- SHARP JUMP at 20: greens, darker toward 30 ----
        (pos(20),  "#7CFF4D"),
        (pos(30),  "#0F4D0F"),

        # ---- SHARP JUMP at 30: yellow, darker toward 40 ----
        (pos(30),  "#FFFF99"),
        (pos(40),  "#CCAA00"),

        # ---- SHARP JUMP at 40: orange, darker toward 50 ----
        (pos(40),  "#FFC04D"),
        (pos(50),  "#CC6600"),

        # ---- SHARP JUMP at 50: red, darker toward 60 ----
        (pos(50),  "#FF6644"),
        (pos(60),  "#880000"),

        # ---- SHARP JUMP at 60: pink/magenta, darker toward 70 ----
        (pos(60),  "#FFAAE0"),
        (pos(70),  "#99006B"),

        # ---- SHARP JUMP at 70: violet/purple, darker toward 80 ----
        (pos(70),  "#C966FF"),
        (pos(80),  "#3D0F70"),

        # ---- SHARP JUMP at 80: cyan, darker toward 90 ----
        (pos(80),  "#B3FFFF"),
        (pos(90),  "#1A7A8C"),

        # ---- SHARP JUMP at 90: salmon/maroon, darker toward 100 ----
        (pos(90),  "#FFAA88"),
        (pos(100), "#5C1A0A"),
    ]

    return LinearSegmentedColormap.from_list("ChadMapZ", stops, N=1024)


# RETRIEVE THE NAME OF THE RADAR SITE FROM THE ID NUMBER
def GrabRadarInfo(radar_id: int) -> str:
    """
    Returns the radar site name for a given radar ID number.

    Args:
        radar_id: The radar ID number as an integer.

    Returns:
        The radar site name, or 'Site <id>' if not found.
    """
    radar_sites = {
        22:  'Mackay',
        106: 'Townsville',
        66:  'Mt Staplyton (Brisbane)',
        50:  'Marburg',
        19:  'Cairns',
        8:   'Gympie',
        24:  'Bowen',
        23:  'Gladstone',
        74:  'Greenvale',
        98:  'Taroom',
        108: 'Toowoomba',
        78:  'Weipa',
        72:  'Emerald',
        41:  'Willis Island',
    }

    radar_site_name = radar_sites.get(radar_id, f'Site {radar_id}')

    # TOWNSVILLE LONGITUDE NEEDS TO BE SHIFTED
    # Longitude shift correction for known coordinate errors
    if radar_site_name == 'Townsville':
        lon_shift = 146.5505 - (-19.4195)
    else:
        lon_shift = 0.0

    return radar_site_name, lon_shift



# RETRIEVE ALL OF THE VARIABLES THAT DEPEND ON THE VARIABLE CHOSEN TO PLOT
def GrabVarInfo(plot_variable: str, value_or_texture_or_count: str, variance_grid_size: str) -> dict:
    """
    Returns a dictionary of plotting information for a given radar variable.

    Args:
        plot_variable:             The base variable name (e.g. 'Z', 'V', 'ZDR', 'RhoHV', 'PhiDP', 'KDP').
        value_or_texture_or_count: Whether to plot the value, texture (variance), or count ('Value', 'Texture', 'Count').
        variance_grid_size:        The grid size string for variance/count lookups (e.g. '3x3', '5x5').

    Returns:
        A dictionary containing all plotting variables for the requested variable.

    Raises:
        ValueError: If the variable combination is not recognised.
    """

    # Build the internal lookup key
    if value_or_texture_or_count == 'Value':
        var_key = plot_variable
    elif value_or_texture_or_count == 'Texture':
        var_key = f'{plot_variable}_variance_{variance_grid_size}grid'
    elif value_or_texture_or_count == 'Count':
        var_key = f'{plot_variable}_count_{variance_grid_size}grid'
    else:
        raise ValueError(
            f"Input ValueOrTextureOrCount '{value_or_texture_or_count}' not recognised.\n"
            f"Please choose from: 'Value', 'Texture', 'Count'."
        )

    # -------------------------------------------------------------------------
    # VARIABLE CATALOGUE
    # -------------------------------------------------------------------------
    var_catalogue = {

        # --- BASE VALUES -----------------------------------------------------

        'Z': {
            'VarName':           'Reflectivity',
            'VarNameLong':       'corrected_reflectivity',
            'VarMinVal':         -30,       # [dBZ]
            'VarMaxVal':          65,       # [dBZ]
            'VarUnit':           'dBZ',
            'VarFillValue':      -32.0,     # [dBZ]
            'VarColourBar':      make_ChadMapZ(),
            'VarColourBar_min':  -30.0,
            'VarColourBar_max':  100.0,
            'VarColourBar_norm': Normalize(vmin=-30.0, vmax=100.0),
            'VarTickSpacing':    10.0,
        },
        'V': {
            'VarName':           'Velocity',
            'VarNameLong':       'corrected_velocity',
            'VarMinVal':         -30,       # [m/s]
            'VarMaxVal':          30,       # [m/s]
            'VarUnit':           'm/s',
            'VarFillValue':      -300.0,    # [m/s]
            'VarColourBar':      'RdBu_r',
            'VarColourBar_min':  -30.0,
            'VarColourBar_max':   30.0,
            'VarColourBar_norm': Normalize(vmin=-30.0, vmax=30.0),
            'VarTickSpacing':    5.0,
        },
        'ZDR': {
            'VarName':           'Differential Reflectivity',
            'VarNameLong':       'corrected_differential_reflectivity',
            'VarMinVal':         -5,        # [dB]
            'VarMaxVal':          5,        # [dB]
            'VarUnit':           'dB',
            'VarFillValue':      -15.0,     # [dB]
            'VarColourBar':      'RdBu',
            'VarColourBar_min':  -5.0,
            'VarColourBar_max':   5.0,
            'VarColourBar_norm': Normalize(vmin=-5.0, vmax=5.0),
            'VarTickSpacing':    1.0,
        },
        'RhoHV': {
            'VarName':           'Correlation Coefficient',
            'VarNameLong':       'corrected_cross_correlation_ratio',
            'VarMinVal':          0.8,      # [0 to 1]
            'VarMaxVal':          1.0,      # [0 to 1]
            'VarUnit':           '0 to 1',
            'VarFillValue':       0.0,      # [0 to 1]
            'VarColourBar':      'nipy_spectral',
            'VarColourBar_min':   0.8,
            'VarColourBar_max':   1.0,
            'VarColourBar_norm': Normalize(vmin=0.8, vmax=1.0),
            'VarTickSpacing':    0.05,
        },
        'PhiDP': {
            'VarName':           'Differential Phase',
            'VarNameLong':       'corrected_differential_phase',
            'VarMinVal':          0,        # [deg]
            'VarMaxVal':         30,        # [deg]
            'VarUnit':           'deg',
            'VarFillValue':      -999.0,    # [deg]
            'VarColourBar':      'nipy_spectral',
            'VarColourBar_min':   0.0,
            'VarColourBar_max':  30.0,
            'VarColourBar_norm': Normalize(vmin=0.0, vmax=30.0),
            'VarTickSpacing':    5.0,
        },
        'KDP': {
            'VarName':           'Specific Differential Phase',
            'VarNameLong':       'corrected_specific_differential_phase',
            'VarMinVal':          0,        # [deg/km]
            'VarMaxVal':         10,        # [deg/km]
            'VarUnit':           'deg / km',
            'VarFillValue':      -5.0,      # [deg/km]
            'VarColourBar':      'nipy_spectral',
            'VarColourBar_min':   0.0,
            'VarColourBar_max':  10.0,
            'VarColourBar_norm': Normalize(vmin=0.0, vmax=10.0),
            'VarTickSpacing':    1.0,
        },

        # --- VARIANCE (TEXTURE) — 3x3 ----------------------------------------

        'Z_variance_3x3grid': {
            'VarName':           'Reflectivity Variance',
            'VarNameLong':       f'corrected_reflectivity_{variance_grid_size}grid_variance',
            'VarMinVal':          0,        # [dBZ²]
            'VarMaxVal':         65,        # [dBZ²]
            'VarUnit':           'dBZ²',
            'VarFillValue':      -99,
            'VarColourBar':      make_ChadMapZ(),
            'VarColourBar_min':  -30.0,
            'VarColourBar_max':  100.0,
            'VarColourBar_norm': Normalize(vmin=-30.0, vmax=100.0),
            'VarTickSpacing':    10.0,
        },
        'V_variance_3x3grid': {
            'VarName':           'Velocity Variance',
            'VarNameLong':       f'corrected_velocity_{variance_grid_size}grid_variance',
            'VarMinVal':          0,        # [m²/s²]
            'VarMaxVal':          1,        # [m²/s²]
            'VarUnit':           'm²/s²',
            'VarFillValue':      -99,
            'VarColourBar':      'RdPu_r',
            'VarColourBar_min':   0.0,
            'VarColourBar_max':   1.0,
            'VarColourBar_norm': Normalize(vmin=0.0, vmax=1.0),
            'VarTickSpacing':    0.1,
        },
        'ZDR_variance_3x3grid': {
            'VarName':           'Differential Reflectivity Variance',
            'VarNameLong':       f'corrected_differential_reflectivity_{variance_grid_size}grid_variance',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -99,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'RhoHV_variance_3x3grid': {
            'VarName':           'Correlation Coefficient Variance',
            'VarNameLong':       f'corrected_cross_correlation_ratio_{variance_grid_size}grid_variance',
            'VarMinVal':          0,
            'VarMaxVal':          0.01,
            'VarUnit':           'variance',
            'VarFillValue':      -99,
            'VarColourBar':      'viridis',
            'VarColourBar_min':   0.0,
            'VarColourBar_max':   0.01,
            'VarColourBar_norm': Normalize(vmin=0.0, vmax=0.01),
            'VarTickSpacing':    0.001,
        },
        'PhiDP_variance_3x3grid': {
            'VarName':           'Differential Phase Variance',
            'VarNameLong':       f'corrected_differential_phase_{variance_grid_size}grid_variance',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -99,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'KDP_variance_3x3grid': {
            'VarName':           'Specific Differential Phase Variance',
            'VarNameLong':       f'corrected_specific_differential_phase_{variance_grid_size}grid_variance',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -99,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },

        # --- VARIANCE (TEXTURE) — 5x5 ----------------------------------------

        'Z_variance_5x5grid': {
            'VarName':           'Reflectivity Variance',
            'VarNameLong':       f'corrected_reflectivity_{variance_grid_size}grid_variance',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -99,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'V_variance_5x5grid': {
            'VarName':           'Velocity Variance',
            'VarNameLong':       f'corrected_velocity_{variance_grid_size}grid_variance',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -99,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'ZDR_variance_5x5grid': {
            'VarName':           'Differential Reflectivity Variance',
            'VarNameLong':       f'corrected_differential_reflectivity_{variance_grid_size}grid_variance',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -99,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'RhoHV_variance_5x5grid': {
            'VarName':           'Correlation Coefficient Variance',
            'VarNameLong':       f'corrected_cross_correlation_ratio_{variance_grid_size}grid_variance',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -99,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'PhiDP_variance_5x5grid': {
            'VarName':           'Differential Phase Variance',
            'VarNameLong':       f'corrected_differential_phase_{variance_grid_size}grid_variance',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -99,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'KDP_variance_5x5grid': {
            'VarName':           'Specific Differential Phase Variance',
            'VarNameLong':       f'corrected_specific_differential_phase_{variance_grid_size}grid_variance',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -99,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },

        # --- COUNT — 3x3 -----------------------------------------------------

        'Z_count_3x3grid': {
            'VarName':           'Reflectivity Counts',
            'VarNameLong':       f'corrected_reflectivity_{variance_grid_size}grid_count',
            'VarMinVal':          0,        # [cells]
            'VarMaxVal':          9,        # [cells]
            'VarUnit':           'cells',
            'VarFillValue':      -9,
            'VarColourBar':      make_ChadMapZ(),
            'VarColourBar_min':  -3.0,
            'VarColourBar_max':  10.0,
            'VarColourBar_norm': Normalize(vmin=-3.0, vmax=10.0),
            'VarTickSpacing':    1.0,
        },
        'V_count_3x3grid': {
            'VarName':           'Velocity Counts',
            'VarNameLong':       f'corrected_velocity_{variance_grid_size}grid_count',
            'VarMinVal':          0,        # [cells]
            'VarMaxVal':          9,        # [cells]
            'VarUnit':           'cells',
            'VarFillValue':      -9,
            'VarColourBar':      'RdPu_r',
            'VarColourBar_min':   0.0,
            'VarColourBar_max':   9.0,
            'VarColourBar_norm': Normalize(vmin=0.0, vmax=9.0),
            'VarTickSpacing':    1.0,
        },
        'ZDR_count_3x3grid': {
            'VarName':           'Differential Reflectivity Counts',
            'VarNameLong':       f'corrected_differential_reflectivity_{variance_grid_size}grid_count',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -9,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'RhoHV_count_3x3grid': {
            'VarName':           'Correlation Coefficient Counts',
            'VarNameLong':       f'corrected_cross_correlation_ratio_{variance_grid_size}grid_count',
            'VarMinVal':          0,        # [cells]
            'VarMaxVal':          9,        # [cells]
            'VarUnit':           'cells',
            'VarFillValue':      -9,
            'VarColourBar':      'RdPu_r',
            'VarColourBar_min':   0.0,
            'VarColourBar_max':   9.0,
            'VarColourBar_norm': Normalize(vmin=0.0, vmax=9.0),
            'VarTickSpacing':    1.0,
        },
        'PhiDP_count_3x3grid': {
            'VarName':           'Differential Phase Counts',
            'VarNameLong':       f'corrected_differential_phase_{variance_grid_size}grid_count',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -9,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'KDP_count_3x3grid': {
            'VarName':           'Specific Differential Phase Counts',
            'VarNameLong':       f'corrected_specific_differential_phase_{variance_grid_size}grid_count',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -9,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },

        # --- COUNT — 5x5 -----------------------------------------------------

        'Z_count_5x5grid': {
            'VarName':           'Reflectivity Counts',
            'VarNameLong':       f'corrected_reflectivity_{variance_grid_size}grid_count',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -9,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'V_count_5x5grid': {
            'VarName':           'Velocity Counts',
            'VarNameLong':       f'corrected_velocity_{variance_grid_size}grid_count',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -9,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'ZDR_count_5x5grid': {
            'VarName':           'Differential Reflectivity Counts',
            'VarNameLong':       f'corrected_differential_reflectivity_{variance_grid_size}grid_count',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -9,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'RhoHV_count_5x5grid': {
            'VarName':           'Correlation Coefficient Counts',
            'VarNameLong':       f'corrected_cross_correlation_ratio_{variance_grid_size}grid_count',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -9,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'PhiDP_count_5x5grid': {
            'VarName':           'Differential Phase Counts',
            'VarNameLong':       f'corrected_differential_phase_{variance_grid_size}grid_count',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -9,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
        'KDP_count_5x5grid': {
            'VarName':           'Specific Differential Phase Counts',
            'VarNameLong':       f'corrected_specific_differential_phase_{variance_grid_size}grid_count',
            'VarMinVal':         'MISSING VALUE',
            'VarMaxVal':         'MISSING VALUE',
            'VarUnit':           'MISSING VALUE',
            'VarFillValue':      -9,
            'VarColourBar':      'MISSING VALUE',
            'VarColourBar_min':  'MISSING VALUE',
            'VarColourBar_max':  'MISSING VALUE',
            'VarColourBar_norm': 'MISSING VALUE',
            'VarTickSpacing':    'MISSING VALUE',
        },
    }

    # -------------------------------------------------------------------------
    # LOOKUP & ERROR HANDLING
    # -------------------------------------------------------------------------
    if var_key not in var_catalogue:
        raise ValueError(
            f"Input variable combination '{var_key}' is not recognised.\n"
            f"Please choose a base variable from:\n"
            f"  Z, V, ZDR, RhoHV, PhiDP, KDP\n"
            f"With acceptable extensions:\n"
            f"  (none)                  — for Value\n"
            f"  _variance_3x3grid       — for Texture with 3x3 grid\n"
            f"  _variance_5x5grid       — for Texture with 5x5 grid\n"
            f"  _count_3x3grid          — for Count with 3x3 grid\n"
            f"  _count_5x5grid          — for Count with 5x5 grid\n"
        )

    return var_catalogue[var_key]



# TOPOGRAPHY BASEMAP FUNCTION FOR PLOTTING FIGURES WITH HORIZONTAL RADAR DATA
def AddTopoShading(ax, lon_min, lon_max, lat_min, lat_max, ElevationModelPath, ColourStyle):
    """
    Adds terrain shading to an existing cartopy axes object using a local GEBCO DEM.
    Falls back to simple land/ocean shading if the DEM cannot be loaded.

    Args:
        ax:                 The existing cartopy axes object to plot onto.
        lon_min:            Minimum longitude of the plot domain.
        lon_max:            Maximum longitude of the plot domain.
        lat_min:            Minimum latitude of the plot domain.
        lat_max:            Maximum latitude of the plot domain.
        ElevationModelPath: Full path to the local GEBCO .nc elevation file.
        ColourStyle:        Either 'Light' or 'Dark'. Controls the colour scheme of the
                            terrain shading, contour lines, and fallback features.
                            Defaults to 'Light'.
    """

    # -------------------------------------------------------------------------
    # Colour scheme definitions
    # -------------------------------------------------------------------------
    if ColourStyle == 'Dark':
        colours = [
            '#1a3a5c',  # 0: ocean (< 0 m)      — medium-dark navy blue
            '#1a4a1a',  # 1: 0–200 m            — medium-dark green
            '#2e4a10',  # 2: 200–400 m          — dark olive green
            '#545040',  # 3: 400–600 m          — muted olive
            '#7a5c3a',  # 4: 600–800 m          — muted brown
            '#a07848',  # 5: 800–1000 m         — muted warm brown
            '#bc9858',  # 6: 1000–1200 m        — muted tan
            '#d4b870',  # 7: > 1200 m           — muted golden
        ]
        contour_colour         = 'white'
        contour_alpha_0m       = 0.5
        contour_alpha_400m     = 0.2
        contour_linewidth_0m   = 0.25
        contour_linewidth_400m = 0.25
        fallback_ocean_colour  = '#0d1f3c'   # dark navy
        fallback_land_colour   = '#2b1d0e'   # dark brown
        border_colour          = 'white'

    else:  # Default to 'Light'
        colours = [
            '#dde4e8',  # 0: pale blue-grey (ocean, < 0 m)
            '#c4dec2',  # 1: 0–200 m,    pale green
            '#e4edc9',  # 2: 200–400 m,  greenish-yellow
            '#f3f0cf',  # 3: 400–600 m,  pale yellow-beige
            '#e9d7bd',  # 4: 600–800 m,  light tan
            '#ddc4aa',  # 5: 800–1000 m, tan
            '#cfb194',  # 6: 1000–1200 m, light brown
            '#b58f6e',  # 7: > 1200 m,   darker brown
        ]
        contour_colour         = 'black'
        contour_alpha_0m       = 1.0
        contour_alpha_400m     = 1.0
        contour_linewidth_0m   = 0.5
        contour_linewidth_400m = 0.3
        fallback_ocean_colour  = 'lightblue'
        fallback_land_colour   = '#E8E8E8'
        border_colour          = 'black'

    # -------------------------------------------------------------------------
    # Shared elevation bounds
    # -------------------------------------------------------------------------
    bounds = [
        -1000.0,  # ocean below 0
            0.0,  # 0–200
          200.0,  # 200–400
          400.0,  # 400–600
          600.0,  # 600–800
          800.0,  # 800–1000
         1000.0,  # 1000–1200
         1200.0,  # > 1200
         5000.0,
    ]

    try:
        dem_da = fetch_gebco_local(ElevationModelPath, lon_min, lon_max, lat_min, lat_max)

        if dem_da is None:
            raise ValueError('GEBCO DEM returned None')

        dem_lon  = dem_da.lon.values
        dem_lat  = dem_da.lat.values
        dem_data = dem_da.values

        # Make 2D lon/lat grids if necessary
        if dem_lon.ndim == 1 and dem_lat.ndim == 1:
            dem_lon_2d, dem_lat_2d = np.meshgrid(dem_lon, dem_lat)
        else:
            dem_lon_2d, dem_lat_2d = dem_lon, dem_lat

        # Ensure we have some valid data
        valid = np.isfinite(dem_data)
        if not np.any(valid):
            raise ValueError('DEM has no finite values in this domain')

        cmap_elev = ListedColormap(colours)
        norm      = BoundaryNorm(bounds, len(colours), clip=True)

        # Plot terrain as background
        elev_plot = ax.pcolormesh(
            dem_lon_2d,
            dem_lat_2d,
            dem_data,
            cmap=cmap_elev,
            norm=norm,
            alpha=1.0,
            transform=ccrs.PlateCarree(),
        )

        # Draw 0 m contour as an accurate coastline
        contour_0m = ax.contour(
            dem_lon_2d,
            dem_lat_2d,
            dem_data,
            levels=[0.0],
            colors=contour_colour,
            alpha=contour_alpha_0m,
            linewidths=contour_linewidth_0m,
            transform=ccrs.PlateCarree(),
            zorder=15,
        )

        # Draw 400 m contour
        contour_400m = ax.contour(
            dem_lon_2d,
            dem_lat_2d,
            dem_data,
            levels=[400.0],
            colors=contour_colour,
            alpha=contour_alpha_400m,
            linewidths=contour_linewidth_400m,
            transform=ccrs.PlateCarree(),
            zorder=15,
        )

    except Exception as e:
        print(f'Terrain shading failed: {e}')
        ax.add_feature(cfeature.OCEAN, facecolor=fallback_ocean_colour, alpha=0.3, zorder=1)
        ax.add_feature(cfeature.LAND,  facecolor=fallback_land_colour,  alpha=0.3, zorder=2)

    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5,
                   edgecolor=border_colour, zorder=3)





# ADD GRID LINES TO TOP-DOWN (HORIZONTAL) PLOTS
def AddGridlines(ax, ThinLineThickness, MediumLineThickness, ThickLineThickness,
                     ThinLineFrequency, MediumLineFrequency, ThickLineFrequency, StandOutColour):
    
    """
    Adds three levels of lat/lon gridlines to an existing cartopy axes object.

    Args:
        ax:                    The existing cartopy axes object to plot onto.
        ThinLineThickness:     Line width for the minor (thin) gridlines.
        MediumLineThickness:   Line width for the mid-level gridlines.
        ThickLineThickness:    Line width for the major (thick) gridlines.
        ThinLineFrequency:     Spacing in degrees for the minor gridlines (e.g. 0.1).
        MediumLineFrequency:   Spacing in degrees for the mid-level gridlines (e.g. 0.5).
        ThickLineFrequency:    Spacing in degrees for the major gridlines (e.g. 1.0).
    """

    # Add MINOR gridlines — thin, no labels
    gl_minor          = ax.gridlines(draw_labels=False, alpha=0.8, zorder=11, linewidth=ThinLineThickness)
    gl_minor.xlocator = mticker.MultipleLocator(ThinLineFrequency)
    gl_minor.ylocator = mticker.MultipleLocator(ThinLineFrequency)

    # Add MID LEVEL gridlines — medium width, no labels
    gl_mid = ax.gridlines(draw_labels=False, alpha=0.8, zorder=12, linewidth=MediumLineThickness)

    gl_mid.xlocator = mticker.MultipleLocator(MediumLineFrequency)
    gl_mid.ylocator = mticker.MultipleLocator(MediumLineFrequency)

    # Add MAJOR gridlines — thick, with labels
    gl_major          = ax.gridlines(draw_labels=True, alpha=0.8, zorder=13, linewidth=ThickLineThickness)
    gl_major.xlocator = mticker.MultipleLocator(ThickLineFrequency)
    gl_major.ylocator = mticker.MultipleLocator(ThickLineFrequency)

    # Format labels
    gl_major.xformatter = LONGITUDE_FORMATTER
    gl_major.yformatter = LATITUDE_FORMATTER

    # Remove labels from top and right
    gl_major.top_labels    = False
    gl_major.right_labels  = False
    gl_major.bottom_labels = True
    gl_major.left_labels   = True

    # Axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # ------------------------------------------------------------------
    # Gridline label colouring — must be applied at draw time because
    # cartopy only creates the label text artists during rendering
    # ------------------------------------------------------------------
    def _colour_gridline_labels(event):
        for gl in [gl_mid, gl_major]:
            for artist in gl.label_artists:
                artist.set_color(StandOutColour)

    ax.figure.canvas.mpl_connect('draw_event', _colour_gridline_labels)



# ADD GATE COORDINATES (LATS AND LONS) TO PPI DATA FOR PLOTTING
def AddGateCoords(RadarXR, LonShift):
    """
    Calculates the latitude, longitude, and altitude of every radar gate
    in a PPI scan using the 4/3 effective Earth radius model, and returns
    the input xarray Dataset with these added as new variables.

    Args:
        RadarXR:  The input xarray Dataset containing radar PPI data.
        LonShift: Longitude correction offset for known coordinate errors.

    Returns:
        RadarXR:  The input xarray Dataset with three new variables added:
                  'gate_latitude', 'gate_longitude', 'gate_altitude'.
    """

    # Calculate the effective radius of the Earth given standard refraction
    RadiusEarth    = 6371000  # [m]
    RadiusEarthEff = RadiusEarth * (4/3)

    # Load in a geodesic calculator
    geod = Geod(ellps='WGS84')

    # Read in the PPI coordinates from the xarray dataset
    Ranges  = RadarXR['range'].values
    AziDegs = RadarXR['azimuth'].values
    EleDegs = RadarXR['elevation'].values

    # Read in the radar site location from the xarray dataset
    SiteLat = float(RadarXR.latitude)
    SiteLon = float(RadarXR.longitude)
    SiteAlt = float(RadarXR.altitude)

    # Convert angles to radians
    AziRads = np.deg2rad(AziDegs)[:, None]
    EleRads = np.deg2rad(EleDegs)[:, None]

    # Distance from centre of the Earth using 4/3 Earth-radius model
    DistsFromCOE = np.sqrt(
        (Ranges**2) + (RadiusEarthEff**2) + (2.0 * Ranges * RadiusEarthEff * np.sin(EleRads))
    )

    # Convert to altitude by subtracting out the Earth radius and adding site elevation
    Alts = (DistsFromCOE - RadiusEarthEff) + SiteAlt

    # Calculate ground range (distance along the surface of the Earth)
    GroundRanges = Ranges * np.cos(EleRads)
    GroundRanges = np.squeeze(GroundRanges)

    # Retrieve number of rays and gates
    NumRays, NumGates = GroundRanges.shape

    # Broadcast radar site lon/lat and azimuth to match GroundRanges shape
    SiteLonRep = np.full_like(GroundRanges, SiteLon, dtype=float)
    SiteLatRep = np.full_like(GroundRanges, SiteLat, dtype=float)
    AziDegsRep = np.repeat(AziDegs[:, None], NumGates, axis=1)

    # Calculate gate lon/lat using geodesic forward projection
    Lons, Lats, _ = geod.fwd(SiteLonRep, SiteLatRep, AziDegsRep, GroundRanges)

    # Package results as xarray DataArrays
    LatsData = xr.DataArray(
        Lats,
        dims=('time', 'range'),
        name='gate_latitude',
        attrs={
            'long_name': 'latitude of radar gates',
            'units':     'degrees_north',
        }
    )

    LonsData = xr.DataArray(
        Lons + LonShift,
        dims=('time', 'range'),
        name='gate_longitude',
        attrs={
            'long_name': 'longitude of radar gates',
            'units':     'degrees_east',
        }
    )

    AltsData = xr.DataArray(
        Alts,
        dims=('time', 'range'),
        name='gate_altitude',
        attrs={
            'long_name':     'altitude of radar gates',
            'units':         'm',
            'standard_name': 'altitude',
        }
    )

    # Add the new variables to the dataset and return
    RadarXR['gate_latitude']  = LatsData
    RadarXR['gate_longitude'] = LonsData
    RadarXR['gate_altitude']  = AltsData

    return RadarXR



def beam_height_curved(s_km, elev_deg, ke=4/3, re_km=6371.0):
    """
    Calculate radar beam centre height above Earth's surface accounting
    for Earth curvature and standard atmospheric refraction.
    Uses the effective Earth radius (4/3 Earth radius) model.

    Parameters
    ----------
    s_km     : float or array  — horizontal ground distance from radar [km]
    elev_deg : float           — radar elevation angle [degrees]
    ke       : float           — effective Earth radius factor (default 4/3)
    re_km    : float           — Earth radius [km] (default 6371)

    Returns
    -------
    height_km : float or array — beam centre height above surface [km]
    """
    theta = np.radians(elev_deg)
    ker   = ke * re_km

    pretend_height_km = s_km * np.tan(theta) # estimated height above radar without earth curve

    slant_km = np.sqrt(s_km**2 + pretend_height_km **2) # estimated slant distance from radar
    
    height_km = np.sqrt(slant_km**2 + ker**2 + 2 * slant_km * ker * np.sin(theta)) - ker # new estimated height with curvature
    return height_km

# LETS PRETEND FOR NOW THAT THE CURVATURE OF THE EARTH HAS NO EFFECT ON THE HORIZONTAL DISTANCE
# def slant_range_curved(s_km, elev_deg, ke=4/3, re_km=6371.0):
#     """
#     Calculate the true slant range from the radar to a point at horizontal
#     ground distance s_km, accounting for Earth curvature.
#     Used for minimum detectable reflectivity calculations.

#     Parameters
#     ----------
#     s_km     : float or array  — horizontal ground distance from radar [km]
#     elev_deg : float           — radar elevation angle [degrees] (not used in
#                                  range-only MDR but kept for completeness)
#     ke       : float           — effective Earth radius factor (default 4/3)
#     re_km    : float           — Earth radius [km] (default 6371)

#     Returns
#     -------
#     slant_km : float or array  — slant range from radar [km]
#     """
#     theta = np.radians(elev_deg)
#     ker   = ke * re_km
#     # slant range from geometry
#     slant_km = ker * np.arcsin(s_km * np.cos(theta) / 
#                                np.sqrt(s_km**2 + ker**2 + 2 * s_km * ker * np.sin(theta)))
#     return slant_km


def calculate_mdr(distance_km):
    """
    Minimum detectable reflectivity as a function of slant range.
    5 dBZ at 80 km, +6 dBZ per doubling of distance.

    Parameters
    ----------
    distance_km : float or array — slant range from radar [km]

    Returns
    -------
    mdr : float or array — minimum detectable reflectivity [dBZ]
    """
    return 5.0 + 6.0 * np.log2(distance_km / 80.0)




# THIS FUNCTION add variables to the xarray data frame
# that rearrange time-based variables to correspond to time of day, not time in feature's life
def AddFrameTimeVars(FeatureXR):
    
    # Derive date from dataset attributes
    RadarFileDatePD = pd.Timestamp(str(FeatureXR.attrs['startdate'])[:8]).date()
    FrameTimes = pd.date_range(start=pd.Timestamp(RadarFileDatePD), periods=288, freq='5min')
    FrameTimesIndices = pd.DatetimeIndex(FrameTimes)
    StartTimeFloors = pd.DatetimeIndex(FeatureXR['start_basetime'].values).floor('5min')

    # Map each track's floored start time to its FrameTimes index
    StartFrameIndices = np.array([FrameTimesIndices.get_loc(t) for t in StartTimeFloors])
    
    # Collect all variables that have 'times' as a dimension
    VarsWithTime= [var for var in FeatureXR.data_vars if 'times' in FeatureXR[var].dims]
    
    NumFrames = len(FrameTimes)  # 288
    NewVars = {}
    
    for var in VarsWithTime:
        OldVar = FeatureXR[var]
        OldValues =  OldVar.values
        OldDims = list(OldVar.dims)
    
        # Determine fill value based on dtype
        if np.issubdtype(OldValues.dtype, np.floating):
            FillValue = np.nan
        else:
            FillValue = -9999
    
        # Replace 'times' with 'FrameTimes' in the dimension list
        NewDims = [d if d != 'times' else 'FrameTimes' for d in OldDims]
    
        # Build the shape of the new array, replacing times axis size with n_frames
        TimeAxis = OldDims.index('times')
        NewShape = list(OldValues.shape)
        NewShape[TimeAxis] = NumFrames
    
        # Initialise new array with fill value
        NewValues = np.full(NewShape, FillValue, dtype=OldValues.dtype)
    
        # --- Fill in data track by track ---
        # Get the index of 'times' axis; handle both (times,) and (tracks, times)
        if 'tracks' in OldDims:
            NumTracks = OldValues.shape[OldDims.index('tracks')]
            for i in range(NumTracks):
                StartIndex = StartFrameIndices[i]
    
                # Slice along tracks axis
                TrackData = np.take(OldValues, i, axis=OldDims.index('tracks'))
    
                # Find valid (non-fill) entries
                if np.issubdtype(OldValues.dtype, np.floating):
                    ValidMask = ~np.isnan(TrackData)
                else:
                    ValidMask = TrackData != -9999
    
                ValidData = TrackData[ValidMask]
                NumValidFrames = len(ValidData)
    
                # How many fit from start_idx to end of day
                EndIndex = min(StartIndex + NumValidFrames, NumFrames)
                TrackFrameLength  = EndIndex - StartIndex
    
                # Place into new array along tracks axis
                if OldDims.index('tracks') == 0:
                    NewValues[i, StartIndex:EndIndex] = ValidData[:TrackFrameLength]
                else:
                    NewValues[StartIndex:EndIndex, i] = ValidData[:TrackFrameLength]
        else:
            # Variable has only (times,) dimension — no track offset to apply
            # Just copy directly; no per-track start index available
            NewValues[:] = OldValues
    
        NewVars[f'{var}_frametimes'] = xr.Variable(
            dims=NewDims,
            data=NewValues,
            attrs=OldVar.attrs
        )
    
    # --- Assign all new variables ---
    FeatureXR = FeatureXR.assign(NewVars)

    return FeatureXR




# THIS FUNCTION add feature propogation speed and direction variables to the xarray data frame from Feature Tracking
def AddPropagationVars(FeatureXR):
    """
    Calculates feature propagation speeds and direction from mean lat/lon
    and base_time, and adds them as new variables to the input xarray Dataset.

    Vectorised using pyproj — no Python loops.

    Parameters
    ----------
    FeatureXR : xr.Dataset
        Feature tracking dataset with dimensions (tracks, times) and variables:
            - base_time : datetime64 timestamps for each feature at each time step
            - meanlat   : mean latitude  of each feature at each time step
            - meanlon   : mean longitude of each feature at each time step

    Returns
    -------
    FeatureXR : xr.Dataset
        Input dataset with four new variables added:
            - u_prop_speed     : Eastward  propagation speed component [m s-1]
            - v_prop_speed     : Northward propagation speed component [m s-1]
            - total_prop_speed : Total propagation speed magnitude     [m s-1]
            - prop_direction   : Propagation direction, clockwise from North [degrees]
    """

    from pyproj import Geod

    geod = Geod(ellps='WGS84')

    n_tracks = len(FeatureXR.coords['tracks'])
    n_times  = len(FeatureXR.coords['times'])

    # --- Load full arrays into memory once ---
    base_time = FeatureXR['base_time'].values   # (tracks, times)
    meanlat   = FeatureXR['meanlat'].values     # (tracks, times)
    meanlon   = FeatureXR['meanlon'].values     # (tracks, times)

    # --- Time differences in seconds along time axis ---
    Tdiff = (base_time[:, 1:] - base_time[:, :-1]) / np.timedelta64(1, 's')
    # Shape: (tracks, times-1)

    # --- Start/end lat/lon slices ---
    StartLat = meanlat[:, :-1]
    StartLon = meanlon[:, :-1]
    EndLat   = meanlat[:, 1:]
    EndLon   = meanlon[:, 1:]

    # --- Flatten to 1D for pyproj ---
    flat_lon1 = StartLon.ravel()
    flat_lat1 = StartLat.ravel()
    flat_lon2 = EndLon.ravel()
    flat_lat2 = EndLat.ravel()
    flat_tdiff = Tdiff.ravel()

    # --- Valid point mask ---
    valid_mask = (
        ~np.isnan(flat_lat1) &
        ~np.isnan(flat_lon1) &
        ~np.isnan(flat_lat2) &
        ~np.isnan(flat_lon2) &
        (flat_tdiff != 0)
    )

    # --- Pre-allocate flat displacement arrays ---
    n_flat       = flat_lon1.shape[0]
    Xtravel_flat = np.full(n_flat, np.nan)
    Ytravel_flat = np.full(n_flat, np.nan)

    # --- Run pyproj only on valid points ---
    az_fwd, _, dist_m = geod.inv(
        flat_lon1[valid_mask],
        flat_lat1[valid_mask],
        flat_lon2[valid_mask],
        flat_lat2[valid_mask]
    )

    az_rad = np.deg2rad(az_fwd)
    Xtravel_flat[valid_mask] = dist_m * np.sin(az_rad)
    Ytravel_flat[valid_mask] = dist_m * np.cos(az_rad)

    # --- Reshape back to (tracks, times-1) ---
    Xtravel = Xtravel_flat.reshape(n_tracks, n_times - 1)
    Ytravel = Ytravel_flat.reshape(n_tracks, n_times - 1)

    # --- Compute speeds [m/s] and direction ---
    Uspeed        = Xtravel / Tdiff
    Vspeed        = Ytravel / Tdiff
    PropSpeed     = (Uspeed**2 + Vspeed**2) ** 0.5
    PropDirection = 180 + np.rad2deg(np.arctan2(Uspeed, Vspeed))

    # --- Pad NaN column at front (no previous step at time_i=0) ---
    nan_col = np.full((n_tracks, 1), np.nan)

    Uspeed_all    = np.hstack([nan_col, Uspeed])
    Vspeed_all    = np.hstack([nan_col, Vspeed])
    PropSpeed_all = np.hstack([nan_col, PropSpeed])
    PropDir_all   = np.hstack([nan_col, PropDirection])

    # --- Assign new variables to dataset ---
    FeatureXR = FeatureXR.assign(

        u_prop_speed = xr.DataArray(
            Uspeed_all,
            dims  = ['tracks', 'times'],
            attrs = {
                'long_name' : 'Eastward Propagation Speed',
                'units'     : 'm s-1',
                'comments'  : 'East-West component of feature propagation speed. '
                              'Positive values indicate eastward motion.',
            }
        ),

        v_prop_speed = xr.DataArray(
            Vspeed_all,
            dims  = ['tracks', 'times'],
            attrs = {
                'long_name' : 'Northward Propagation Speed',
                'units'     : 'm s-1',
                'comments'  : 'North-South component of feature propagation speed. '
                              'Positive values indicate northward motion.',
            }
        ),

        total_prop_speed = xr.DataArray(
            PropSpeed_all,
            dims  = ['tracks', 'times'],
            attrs = {
                'long_name' : 'Total Propagation Speed',
                'units'     : 'm s-1',
                'comments'  : 'Magnitude of the feature propagation speed vector. '
                              'Computed as sqrt(u_prop_speed^2 + v_prop_speed^2).',
            }
        ),

        prop_direction = xr.DataArray(
            PropDir_all,
            dims  = ['tracks', 'times'],
            attrs = {
                'long_name' : 'Propagation Direction',
                'units'     : 'degrees',
                'comments'  : 'Direction of feature propagation as an azimuthal bearing. '
                              'Measured clockwise from North (0-360°). '
                              '0° = North, 90° = East, 180° = South, 270° = West.',
            }
        ),

    )

    return FeatureXR
