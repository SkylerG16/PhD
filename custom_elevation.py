"""
Custom elevation data fetcher using SRTM via rasterio
Lightweight alternative to the elevation package
"""

import numpy as np
import urllib.request
import os
from pathlib import Path
import zipfile
import struct

def fetch_srtm(lon_min, lon_max, lat_min, lat_max, cache_dir='./srtm_cache'):
    """
    Fetch SRTM elevation data for a given bounding box.
    Uses USGS SRTM 1 Arc-Second Global data.
    
    Parameters
    ----------
    lon_min, lon_max : float
        Longitude bounds
    lat_min, lat_max : float
        Latitude bounds
    cache_dir : str
        Directory to cache downloaded files
    
    Returns
    -------
    dem_lon, dem_lat, dem_data : arrays
        Longitude, latitude, and elevation data
    """
    
    Path(cache_dir).mkdir(exist_ok=True)
    
    # Calculate tile indices
    lon_tiles = range(int(np.floor(lon_min)), int(np.ceil(lon_max)) + 1)
    lat_tiles = range(int(np.floor(lat_min)), int(np.ceil(lat_max)) + 1)
    
    dem_arrays = []
    
    for lat_tile in lat_tiles:
        for lon_tile in lon_tiles:
            # SRTM filename format: N##W###.hgt or S##E###.hgt
            lat_str = f'N{lat_tile:02d}' if lat_tile >= 0 else f'S{abs(lat_tile):02d}'
            lon_str = f'E{lon_tile:03d}' if lon_tile >= 0 else f'W{abs(lon_tile):03d}'
            
            tile_name = f'{lat_str}{lon_str}'
            cache_file = os.path.join(cache_dir, f'{tile_name}.hgt')
            zip_file = os.path.join(cache_dir, f'{tile_name}.zip')
            
            # Download if not cached
            if not os.path.exists(cache_file):
                # Try multiple SRTM sources
                urls = [
                    f'https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/SRTM_GL1_srtm/{tile_name}.zip',
                    f'https://e4ftl01.cr.usgs.gov/MODV6_Dal/SRTM/SRTMGL1.003/2000.02.11/{tile_name}.SRTMGL1.hgt.zip',
                ]
                
                downloaded = False
                for url in urls:
                    try:
                        print(f'Downloading {tile_name} from {url.split("/")[2]}...')
                        urllib.request.urlretrieve(url, zip_file)
                        
                        # Extract
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            zip_ref.extractall(cache_dir)
                        
                        os.remove(zip_file)
                        print(f'{tile_name} downloaded and extracted')
                        downloaded = True
                        break
                        
                    except Exception as e:
                        print(f'  Source failed: {str(e)[:50]}')
                        if os.path.exists(zip_file):
                            os.remove(zip_file)
                        continue
                
                if not downloaded:
                    print(f'Could not download {tile_name} from any source')
            
            # Load HGT file if it exists
            if os.path.exists(cache_file):
                try:
                    dem_data = np.fromfile(cache_file, np.int16)
                    dem_data = dem_data.byteswap().newbyteorder()  # Convert from big-endian
                    dem_data = dem_data.reshape((3601, 3601))
                    dem_arrays.append((lon_tile, lat_tile, dem_data))
                    print(f'Loaded {tile_name}')
                except Exception as e:
                    print(f'Error reading {tile_name}: {e}')
    
    if not dem_arrays:
        print('No SRTM data downloaded successfully')
        return None, None, None
    
    # Combine tiles into single array
    if len(dem_arrays) == 1:
        lon_tile, lat_tile, dem_data = dem_arrays[0]
    else:
        # Stack multiple tiles
        dem_data = np.vstack([arr[2] for arr in sorted(dem_arrays, key=lambda x: -x[1])])
        lon_tile = min(arr[0] for arr in dem_arrays)
        lat_tile = max(arr[1] for arr in dem_arrays)
    
    # Create coordinate arrays
    dem_lon = np.linspace(lon_tile, lon_tile + len(dem_arrays), dem_data.shape[1])
    dem_lat = np.linspace(lat_tile + 1, lat_tile, dem_data.shape[0])
    
    # Replace invalid values (-32768) with NaN
    dem_data = dem_data.astype(float)
    dem_data[dem_data == -32768] = np.nan
    
    return dem_lon, dem_lat, dem_data


def fetch_gebco_local(gebco_path, lon_min, lon_max, lat_min, lat_max):
    """
    Load GEBCO data from a local file.
    
    Parameters
    ----------
    gebco_path : str
        Path to local GEBCO NetCDF file
    lon_min, lon_max : float
        Longitude bounds
    lat_min, lat_max : float
        Latitude bounds
    
    Returns
    -------
    dem_data : xarray.DataArray
        Elevation data with lon/lat coordinates
    """
    
    try:
        import xarray as xr
        gebco = xr.open_dataset(gebco_path)
        
        buffer = 0.2
        dem_data = gebco['elevation'].sel(
            lon=slice(lon_min - buffer, lon_max + buffer),
            lat=slice(lat_min - buffer, lat_max + buffer)
        )
        
        return dem_data
        
    except Exception as e:
        print(f'Error loading local GEBCO file: {e}')
        return None
