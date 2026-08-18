
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
def AddTopoShading(ax, xgrid, LonShift, ElevationModelPath):
    """
    Adds terrain shading to an existing cartopy axes object using a local GEBCO DEM.
    Falls back to simple land/ocean shading if the DEM cannot be loaded.

    Args:
        ax:                 The existing cartopy axes object to plot onto.
        xgrid:              The radar grid xarray object, used to extract lon/lat bounds.
        LonShift:           Longitude correction offset for known coordinate errors.
        ElevationModelPath: Full path to the local GEBCO .nc elevation file.
    """

    # Extract plot bounds from radar grid
    lon_min, lon_max = float(xgrid.lon.min()) + LonShift, float(xgrid.lon.max()) + LonShift
    lat_min, lat_max = float(xgrid.lat.min()), float(xgrid.lat.max())

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

        cmap_elev = ListedColormap(colours)
        norm      = BoundaryNorm(bounds, len(colours), clip=True)

        # Plot as semi-transparent background
        elev_plot = ax.pcolormesh(
            dem_lon_2d,
            dem_lat_2d,
            dem_data,
            cmap=cmap_elev,
            norm=norm,
            alpha=1.0,
            transform=ccrs.PlateCarree(),
            # zorder=0,
        )

        # Draw 0 m contour as an accurate coastline
        contour_0m = ax.contour(
            dem_lon_2d,
            dem_lat_2d,
            dem_data,
            levels=[0.0],
            colors='black',
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=15,
        )

        # Draw 400 m contour
        contour_400m = ax.contour(
            dem_lon_2d,
            dem_lat_2d,
            dem_data,
            levels=[400.0],
            colors='black',
            linewidths=0.3,
            transform=ccrs.PlateCarree(),
            zorder=15,
        )

    except Exception as e:
        print(f'Terrain shading failed: {e}')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3, zorder=1)
        ax.add_feature(cfeature.LAND,  facecolor='#E8E8E8',   alpha=0.3, zorder=2)

    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5, zorder=3)



# ADD GRID LINES TO TOP-DOWN (HORIZONTAL) PLOTS
def AddGridlines(ax, ThinLineThickness, MediumLineThickness, ThickLineThickness,
                     ThinLineFrequency, MediumLineFrequency, ThickLineFrequency):
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

    # Add MID LEVEL gridlines — medium width, with labels
    gl_mid          = ax.gridlines(draw_labels=True, alpha=0.8, zorder=12, linewidth=MediumLineThickness)
    gl_mid.xlocator = mticker.MultipleLocator(MediumLineFrequency)
    gl_mid.ylocator = mticker.MultipleLocator(MediumLineFrequency)

    # Add MAJOR gridlines — thick, with labels
    gl_major          = ax.gridlines(draw_labels=True, alpha=0.8, zorder=13, linewidth=ThickLineThickness)
    gl_major.xlocator = mticker.MultipleLocator(ThickLineFrequency)
    gl_major.ylocator = mticker.MultipleLocator(ThickLineFrequency)

    # Format labels
    gl_mid.xformatter   = LONGITUDE_FORMATTER
    gl_mid.yformatter   = LATITUDE_FORMATTER
    gl_major.xformatter = LONGITUDE_FORMATTER
    gl_major.yformatter = LATITUDE_FORMATTER

    # Remove labels from top and right
    gl_mid.top_labels    = False
    gl_mid.right_labels  = False
    gl_mid.bottom_labels = True
    gl_mid.left_labels   = True

    gl_major.top_labels    = False
    gl_major.right_labels  = False
    gl_major.bottom_labels = True
    gl_major.left_labels   = True

    # Axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
