
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

from scipy.ndimage import map_coordinates

import pandas as pd
from metpy.plots import SkewT
from metpy.units import units

from matplotlib.colors import LinearSegmentedColormap

from scipy.ndimage import generic_filter



def pcolormeshC(x_centers, y_centers, z, ax=None,
                            shading='auto', **pcolor_kwargs):
    """
    # Like pcolormesh, but takes 1D X and Y coordinates for centres of the pixels.
    
    Create a pcolormesh from a 2D array and 1D coordinate-center arrays.

    Parameters
    ----------
    x_centers : 1D array
        X coordinates of cell centers (length = number of columns in z)
    y_centers : 1D array
        Y coordinates of cell centers (length = number of rows in z)
    z : 2D array
        Data array with shape (len(y_centers), len(x_centers))
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on
    shading : str
        Passed to pcolormesh (default: 'auto')
    **pcolor_kwargs
        Extra kwargs passed to pcolormesh

    Returns
    -------
    pcm : QuadMesh
        The pcolormesh object
    """

    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)
    z = np.asarray(z)

    if z.shape != (len(y_centers), len(x_centers)):
        raise ValueError(
            f"z shape {z.shape} does not match "
            f"(len(y_centers), len(x_centers)) = "
            f"({len(y_centers)}, {len(x_centers)})"
        )

    # Convert centers -> edges
    def centers_to_edges(c):
        dc = np.diff(c)

        edges = np.empty(len(c) + 1)

        # Interior edges
        edges[1:-1] = c[:-1] + dc / 2

        # Extrapolate outer edges
        edges[0] = c[0] - dc[0] / 2
        edges[-1] = c[-1] + dc[-1] / 2

        return edges

    x_edges = centers_to_edges(x_centers)
    y_edges = centers_to_edges(y_centers)

    if ax is None:
        fig, ax = plt.subplots()

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        shading=shading,
        **pcolor_kwargs
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return pcm





def temp_crossing_altitude(sounding_df, crossing_temp):
    """
    Find the altitude where the temperature profile crosses a given temperature
    for the first time (from above, i.e. warm to cold).

    Parameters
    ----------
    sounding_df : pd.DataFrame
        Sounding data with columns: 'geopotential height_m', 'temperature_C'
    crossing_temp : float or int
        The temperature (°C) at which to find the crossing altitude.

    Returns
    -------
    crossing_alt_km : float
        Altitude of the temperature crossing in km.
    """

    # Find the first index where temperature drops below the crossing temperature
    first_index = (sounding_df['temperature_C'] < crossing_temp).idxmax()

    # Retrieve altitudes and temperatures on either side of the crossing
    top_alt = sounding_df['geopotential height_m'][first_index]
    bot_alt = sounding_df['geopotential height_m'][first_index - 1]

    top_temp = sounding_df['temperature_C'][first_index]
    bot_temp = sounding_df['temperature_C'][first_index - 1]

    # Linear interpolation to find exact crossing altitude
    crossing_alt = bot_alt + (
        (top_alt - bot_alt) * ((bot_temp - crossing_temp) / (bot_temp - top_temp))
    )

    crossing_alt_km = crossing_alt * 0.001  # convert to km

    return crossing_alt_km





def find_temperature_inversion(sounding_df, depth_m=300):
    """
    Find the base of the first temperature inversion in a sounding profile —
    defined as the first level where temperature increases with height,
    sustained over a given vertical depth.

    Parameters
    ----------
    sounding_df : pd.DataFrame
        Sounding data with columns: 'geopotential height_m', 'temperature_C'
        Assumed to be sorted by altitude ascending.
    depth_m : float, optional
        Minimum vertical depth (metres) over which temperature must continuously
        increase with height to confirm an inversion. Default is 500 m.

    Returns
    -------
    inversion_alt_km : float or None
        Altitude (km) of the first level that is warmer than the level below it,
        at the base of the first sustained inversion. Returns None if no
        inversion is found.
    """

    temps = sounding_df['temperature_C'].values
    alts  = sounding_df['geopotential height_m'].values

    # Boolean array: True where temperature increases compared to level below
    warming_with_height = temps[1:] > temps[:-1]  # shape (n-1,)

    for i in range(len(warming_with_height)):
        if warming_with_height[i]:
            # Potential inversion base found — now check it sustains for depth_m
            base_alt = alts[i + 1]  # first level warmer than the one below

            # Walk upward from here until we either break the inversion or
            # exceed the required depth
            sustained = True
            for j in range(i + 1, len(warming_with_height)):
                if not warming_with_height[j]:
                    # Inversion broke before reaching required depth
                    sustained = False
                    break
                if alts[j + 1] - base_alt >= depth_m:
                    # Inversion has been sustained over the required depth
                    break

            if sustained and (alts[j + 1] - base_alt >= depth_m):
                return base_alt * 0.001

    return None  # No sustained inversion found

    



def plot_skewt(data, title='Skew-T Diagram'):
    """
    Plot a Skew-T diagram from sounding data.
    
    Parameters
    ----------
    data : str or pd.DataFrame
        Path to CSV file OR a pandas DataFrame with columns:
        pressure_hPa, temperature_C, dewpoint_C
    title : str
        Plot title
    """
    
    # Read CSV if string, otherwise assume it's a DataFrame
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data
    
    # Extract data
    pressure = df['pressure_hPa'].values * units.hPa
    temperature = df['temperature_C'].values * units.degC
    dewpoint = df['dew point temperature_C'].values * units.degC
    
    # Create Skew-T plot
    fig = plt.figure(figsize=(8, 10))
    skew = SkewT(fig, rotation=45)
    
    # Plot profiles
    skew.plot(pressure, temperature, 'r-', linewidth=2, label='Temperature')
    skew.plot(pressure, dewpoint, 'g-', linewidth=2, label='Dewpoint')
    
    # Add reference lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    
    # Labels
    skew.ax.set_xlabel('Temperature (°C)')
    skew.ax.set_ylabel('Pressure (hPa)')
    skew.ax.set_title(title)
    skew.ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()





def find_tropopause(sounding_df, depth_km=0.1, max_lapse_rate=2.0):
    """
    Find the tropopause altitude, defined as the lowest level where the
    temperature lapse rate drops below 2°C/km (or inverts), sustained over
    1 km of depth.

    The tropopause is triggered by the first level where temperature increases
    with height OR decreases by less than 2°C/km. Returns the first altitude
    within that layer where temperature strictly increases with height.

    Parameters
    ----------
    sounding_df : pd.DataFrame
        Sounding data with columns: 'geopotential height_m', 'temperature_C'
        Assumed to be sorted by altitude ascending.
    depth_km : float, optional
        Vertical depth (km) over which the lapse rate criterion must be
        sustained. Default is 1.0 km.
    max_lapse_rate : float, optional
        Maximum lapse rate (°C/km) allowed within the tropopause layer.
        Default is 2.0 °C/km.

    Returns
    -------
    tropopause_alt_km : float or None
        Altitude (km) of the first level within the sustained layer where
        temperature increases with height. Returns None if not found.
    """
    # Drop any levels with duplicate altitudes to avoid division by zero
    sounding_df = sounding_df.drop_duplicates(subset='geopotential height_m').copy()

    temps = sounding_df['temperature_C'].values
    alts  = sounding_df['geopotential height_m'].values * 0.001  # work in km

    # Lapse rate between each pair of levels (°C/km), positive = cooling with height
    dz        = alts[1:]  - alts[:-1]
    dT        = temps[1:] - temps[:-1]
    lapse     = -dT / dz  # positive lapse rate = temperature decreasing with height

    # Criterion: lapse rate less than 2°C/km (includes inversions)
    criterion = lapse < max_lapse_rate  # shape (n-1,)

    for i in range(len(criterion)):

        # Must be triggered by a level meeting the criterion
        if not criterion[i]:
            continue

        # Walk upward — criterion must hold for depth_km
        base_alt  = alts[i]
        sustained = False

        for j in range(i, len(criterion)):
            if not criterion[j]:
                break
            if alts[j + 1] - base_alt >= depth_km:
                sustained = True
                break

        if sustained:
            # Return the first level within this layer where temp increases with height
            for k in range(i, j + 1):
                if dT[k] > 0:
                    return alts[k]

    return None





def extract_line_slice(data_2d, x_coords, y_coords, angle_deg, y_intercept, num_points=500):
    """
    Extract an arbitrary line slice from a 2D radar field.

    Parameters
    ----------
    data_2d : array-like, shape (ny, nx)
        2D radar data at a single z-level and time.
        Can be an xarray DataArray — will be converted automatically.
    x_coords : array-like, shape (nx,)
        1D array of x-axis coordinate values.
    y_coords : array-like, shape (ny,)
        1D array of y-axis coordinate values.
    angle_deg : float
        Angle of the slice in degrees, measured clockwise from north (0–180).
    y_intercept : float
        The y-value where the line crosses x=0, in data coordinates.
    num_points : int
        Number of sample points along the slice.

    Returns
    -------
    along_track : np.ndarray
        Along-track distance values in the same units as your grid.
    slice_values : np.ndarray
        Interpolated radar values along the line.
    """

    # --- Force everything to plain NumPy to avoid xarray scalar issues ---
    if hasattr(data_2d, 'values'):
        data_2d = data_2d.values
    if hasattr(x_coords, 'values'):
        x_coords = x_coords.values
    if hasattr(y_coords, 'values'):
        y_coords = y_coords.values

    # Squeeze out any size-1 leading dimensions (time, z, nradar etc.)
    data_2d = np.squeeze(data_2d)

    if data_2d.ndim != 2:
        raise ValueError(
            f"data_2d must be 2D after squeezing, but got shape {data_2d.shape}. "
            "Please select a single time and z-level before passing in."
        )

    x_coords = np.squeeze(x_coords).astype(float)
    y_coords = np.squeeze(y_coords).astype(float)

    angle_rad = np.deg2rad(float(angle_deg))
    y_intercept = float(y_intercept)

    # Grid spacing (assumes regular grid)
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]

    x_min, x_max = float(x_coords[0]),  float(x_coords[-1])
    y_min, y_max = float(y_coords[0]),  float(y_coords[-1])

    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)

    # Find t bounds from grid extents
    if abs(sin_a) > 1e-9:
        tx0 = (x_min) / sin_a
        tx1 = (x_max) / sin_a
    else:
        tx0, tx1 = -1e12, 1e12

    if abs(cos_a) > 1e-9:
        ty0 = (y_min - y_intercept) / cos_a
        ty1 = (y_max - y_intercept) / cos_a
    else:
        ty0, ty1 = -1e12, 1e12

    t_min = max(min(tx0, tx1), min(ty0, ty1))
    t_max = min(max(tx0, tx1), max(ty0, ty1))

    if t_min >= t_max:
        raise ValueError(
            f"Line does not intersect the grid. "
            f"Check your y_intercept ({y_intercept}) is within "
            f"y bounds [{y_min}, {y_max}]."
        )

    t_values = np.linspace(t_min, t_max, num_points)

    x_sample = t_values * sin_a
    y_sample = t_values * cos_a + y_intercept

    # Convert to fractional array indices
    col_indices = (x_sample - x_coords[0]) / dx
    row_indices = (y_sample - y_coords[0]) / dy

    slice_values = map_coordinates(
        data_2d,
        [row_indices, col_indices],
        order=1,
        mode='constant',
        cval=np.nan
    )

    return t_values, slice_values





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





def _get_elevation_groups(RadarXR: xr.Dataset) -> dict:
    """
    Groups time indices by rounded elevation angle (to nearest 0.1 degree).
    Returns a dict mapping rounded elevation -> array of time indices.
    """
    elevations = RadarXR.coords['elevation'].values
    rounded = np.round(elevations, 1)
    unique_elevs = np.unique(rounded)

    groups = {}
    for elev in unique_elevs:
        groups[elev] = np.where(rounded == elev)[0]

    return groups


def _variance_filter(values: np.ndarray, min_valid: int = 1) -> float:
    """
    Population variance of valid (non-NaN) values in a window.
    Returns 0.0 if only one valid value exists, NaN if none exist.
    """
    valid = values[~np.isnan(values)]
    n = len(valid)
    if n == 0:
        return np.nan
    if n == 1:
        return 0.0
    return np.var(valid)  # population variance (ddof=0)


def _count_filter(values: np.ndarray) -> float:
    """
    Count of valid (non-NaN) values in a window.
    """
    return np.sum(~np.isnan(values)).astype(float)


def _apply_grid_function(
    RadarXR: xr.Dataset,
    variable: str,
    grid_size: int,
    fill_value,
    filter_func,
    result_name: str
):
    """
    Core engine. Applies a given filter function over a rolling azimuth-wrapped,
    range-edge-truncated grid for each elevation group separately.

    Parameters
    ----------
    RadarXR    : xr.Dataset — the radar dataset (mutated in place)
    variable   : str        — variable name to operate on
    grid_size  : int        — odd integer, size of the NxN grid
    fill_value : scalar     — fill value to treat as invalid (in addition to NaN)
    filter_func: callable   — function applied to each flattened window
    result_name: str        — name of the output variable added to RadarXR
    """
    data = RadarXR[variable].values.copy()  # shape: (time, range)

    # Replace fill values with NaN (unless fill_value is already NaN,
    # in which case NaNs are already handled)
    if fill_value is not None and not (isinstance(fill_value, float) and np.isnan(fill_value)):
        data = np.where(data == fill_value, np.nan, data)

    n_time, n_range = data.shape
    half = grid_size // 2

    # Output array, initialised to NaN
    result = np.full((n_time, n_range), np.nan, dtype=np.float32)

    elev_groups = _get_elevation_groups(RadarXR)

    for elev, time_indices in elev_groups.items():
        # Extract the 2D slice for this elevation: shape (n_az, n_range)
        # where n_az should be 360 (one full sweep)
        sweep = data[time_indices, :]   # (n_az, n_range)
        n_az = sweep.shape[0]

        # --- Azimuth wrapping ---
        # Pad azimuth dimension circularly by `half` on each side
        padded = np.concatenate(
            [sweep[-half:, :], sweep, sweep[:half, :]],
            axis=0
        )  # shape: (n_az + 2*half, n_range)

        # --- Range: no wrapping, use 'reflect' equivalent via generic_filter
        # We handle range edges by letting generic_filter use only existing cells
        # (we pad with NaN so edge windows naturally shrink)
        range_pad = np.full((padded.shape[0], half), np.nan)
        padded = np.concatenate([range_pad, padded, range_pad], axis=1)
        # shape: (n_az + 2*half, n_range + 2*half)

        sweep_result = np.full((n_az, n_range), np.nan, dtype=np.float32)

        for i in range(n_az):
            for j in range(n_range):
                # Window in padded array:
                # azimuth: i to i + grid_size  (half already added by circular pad)
                # range:   j to j + grid_size  (half already added by NaN pad)
                window = padded[i:i + grid_size, j:j + grid_size].ravel()
                sweep_result[i, j] = filter_func(window)

        result[time_indices, :] = sweep_result

    # Build a DataArray matching the original dimensions
    result_da = xr.DataArray(
        result,
        dims=RadarXR[variable].dims,
        coords=RadarXR[variable].coords,
        attrs={'long_name': result_name, 'grid_size': grid_size}
    )

    RadarXR[result_name] = result_da


def GridVariance(
    RadarXR: xr.Dataset,
    variable: str,
    grid_size: int,
    fill_value
):
    """
    Computes the population variance of `variable` within a grid_size x grid_size
    neighbourhood for every cell, grouped by elevation angle.

    Azimuth dimension is treated as circular (wraps around).
    Range dimension is truncated at edges (no wrapping).
    NaN and fill_value cells are excluded from variance calculation.
    Windows with a single valid cell return variance = 0.0.
    Windows with no valid cells return NaN.

    The result is added to RadarXR as:
        {variable}_{grid_size}x{grid_size}grid_variance

    Parameters
    ----------
    RadarXR    : xr.Dataset — radar dataset, mutated in place
    variable   : str        — name of the variable to process
    grid_size  : int        — must be an odd integer (e.g. 3, 5, 7)
    fill_value : scalar     — value to treat as invalid (use np.nan if applicable)
    """
    if grid_size % 2 == 0:
        raise ValueError(f"grid_size must be an odd integer, got {grid_size}.")

    result_name = f"{variable}_{grid_size}x{grid_size}grid_variance"
    _apply_grid_function(RadarXR, variable, grid_size, fill_value, _variance_filter, result_name)
    print(f"Added '{result_name}' to RadarXR.")


def GridCount(
    RadarXR: xr.Dataset,
    variable: str,
    grid_size: int,
    fill_value
):
    """
    Counts the number of valid cells within a grid_size x grid_size neighbourhood
    for every cell, grouped by elevation angle.

    Valid cells are those that are not NaN and not equal to fill_value.
    The result is added to RadarXR as:
        {variable}_{grid_size}x{grid_size}grid_count

    Parameters
    ----------
    RadarXR    : xr.Dataset — radar dataset, mutated in place
    variable   : str        — name of the variable to process
    grid_size  : int        — must be an odd integer (e.g. 3, 5, 7)
    fill_value : scalar     — value to treat as invalid (use np.nan if applicable)
    """
    if grid_size % 2 == 0:
        raise ValueError(f"grid_size must be an odd integer, got {grid_size}.")

    result_name = f"{variable}_{grid_size}x{grid_size}grid_count"
    _apply_grid_function(RadarXR, variable, grid_size, fill_value, _count_filter, result_name)
    print(f"Added '{result_name}' to RadarXR.")