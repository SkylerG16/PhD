
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label

def track_reflectivity_features_to_xr(
    xgrid,
    threshold_dbz     = 20.0,
    overlap_threshold = 0.20,
):
    """
    Track radar reflectivity features across all time steps in xgrid and
    return results as a compressed xarray Dataset (FeaturesXR).

    Parameters
    ----------
    xgrid : xarray.Dataset
        Expected dimensions: (time, y, x).
        Must contain 'origin_latitude' and 'origin_longitude'.
    threshold_dbz : float
    overlap_threshold : float

    Returns
    -------
    FeaturesXR : xarray.Dataset
        Dimensions: (time, feature_id)
        All variables are float32, NaN where feature does not exist.
    """

    # ── coordinate setup ──────────────────────────────────────────────────────
    x_coords      = xgrid['x'].values / 1000.0
    y_coords      = xgrid['y'].values / 1000.0
    cell_area_km2 = (x_coords[1] - x_coords[0]) * (y_coords[1] - y_coords[0])
    timestamps    = xgrid['time'].values
    n_times       = len(timestamps)

    radar_lat = float(xgrid['origin_latitude'].values)
    radar_lon = float(xgrid['origin_longitude'].values)

    print(f"Radar origin: {radar_lat:.4f}°N, {radar_lon:.4f}°E")

    # ── global state ──────────────────────────────────────────────────────────
    next_global_id       = 1
    prev_global_features = {}
    prev_local_to_global = {}
    prev_features_raw    = {}
    prev_timestamp       = None

    # master_history keyed by global_id — pruned when features die
    # Each entry: list of {'t', 'event', ...} dicts
    master_history = {}

    # ── per-timestep raw records ───────────────────────────────────────────────
    # raw_records[t_idx] = list of dicts, one per living feature at that timestep
    # We collect these first, then build the xarray at the end once we know
    # the maximum feature ID
    raw_records = [[] for _ in range(n_times)]

    # ── iterate over timesteps ────────────────────────────────────────────────
    for t_idx in range(n_times):
        timestamp = timestamps[t_idx]
        refl      = xgrid['corrected_reflectivity'][t_idx].values

        new_features_raw, labelled_grid = _detect_features(
            refl, x_coords, y_coords, threshold_dbz, cell_area_km2,
            radar_lat, radar_lon
        )

        new_local_to_global = {}

        motion_info = {
            nid: {
                'u'            : np.nan,
                'v'            : np.nan,
                'prev_mass'    : np.nan,
                'motion_clean' : np.nan,
            }
            for nid in new_features_raw
        }

        # Per-timestep event flags keyed by new local ID
        # These are derived fresh each timestep — no history column needed
        event_flags = {
            nid: {
                'born'         : np.nan,
                'died'         : np.nan,
                'split_from'   : np.nan,
                'merges_with'  : np.nan,   # filled in second pass
                'split_out'    : [],        # list of IDs, up to 5
                'absorbed'     : [],        # list of IDs, up to 5
            }
            for nid in new_features_raw
        }

        # ── first frame ───────────────────────────────────────────────────────
        if t_idx == 0:
            for nid in new_features_raw:
                gid                      = next_global_id
                next_global_id          += 1
                new_local_to_global[nid] = gid
                master_history[gid]      = [{'t': timestamp, 'event': 'born'}]
                event_flags[nid]['born'] = 1.0
                event_flags[nid]['died'] = 0.0

        # ── subsequent frames ─────────────────────────────────────────────────
        else:
            dt_seconds = float(
                (pd.Timestamp(timestamp) - pd.Timestamp(prev_timestamp))
                .total_seconds()
            )

            old_to_new, new_to_old = _compute_overlap_links(
                prev_features_raw, new_features_raw, overlap_threshold
            )

            assigned_new        = {}
            split_or_merge_nids = set()
            merge_loser_gids    = set()

            # ── STEP 2: resolve splits ────────────────────────────────────────
            for oid, linked_nids in old_to_new.items():
                if len(linked_nids) <= 1:
                    continue

                old_gid = prev_local_to_global[oid]

                linked_nids_sorted = sorted(
                    linked_nids,
                    key=lambda nid: new_features_raw[nid]['refl_mass_dBZkm2'],
                    reverse=True,
                )

                for nid in linked_nids_sorted:
                    split_or_merge_nids.add(nid)

                winner_nid = linked_nids_sorted[0]
                loser_nids = linked_nids_sorted[1:]

                loser_gids = []
                for loser_nid in loser_nids:
                    if loser_nid not in assigned_new:
                        new_gid                 = next_global_id
                        next_global_id         += 1
                        assigned_new[loser_nid] = new_gid
                        master_history[new_gid] = [{'t': timestamp, 'event': 'split_from', 'id': old_gid}]
                        event_flags[loser_nid]['born']       = 1.0
                        event_flags[loser_nid]['died']       = 0.0
                        event_flags[loser_nid]['split_from'] = float(old_gid)
                    loser_gids.append(assigned_new[loser_nid])

                if winner_nid not in assigned_new:
                    assigned_new[winner_nid] = old_gid
                    master_history[old_gid].append({
                        't': timestamp, 'event': 'split_winner', 'ids': loser_gids
                    })
                    event_flags[winner_nid]['born']      = 0.0
                    event_flags[winner_nid]['died']      = 0.0
                    event_flags[winner_nid]['split_out'] = loser_gids[:5]

            # ── STEP 3: simple continuation candidates ────────────────────────
            continuation_candidates = {}
            for oid, linked_nids in old_to_new.items():
                if len(linked_nids) != 1:
                    continue
                nid     = linked_nids[0]
                old_gid = prev_local_to_global[oid]
                if nid not in assigned_new:
                    continuation_candidates[nid] = old_gid

            # ── STEP 4: resolve merges and continuations ──────────────────────
            for nid in new_features_raw:
                linked_oids = new_to_old.get(nid, [])

                candidates = {}
                for oid in linked_oids:
                    old_gid             = prev_local_to_global[oid]
                    candidates[old_gid] = prev_global_features[old_gid]['refl_mass_dBZkm2']

                if nid in assigned_new:
                    existing_gid = assigned_new[nid]
                    if existing_gid not in candidates and existing_gid in prev_global_features:
                        candidates[existing_gid] = prev_global_features[existing_gid]['refl_mass_dBZkm2']

                if nid in continuation_candidates:
                    cont_gid = continuation_candidates[nid]
                    if cont_gid not in candidates:
                        candidates[cont_gid] = prev_global_features[cont_gid]['refl_mass_dBZkm2']

                if len(candidates) == 0:
                    continue

                elif len(candidates) == 1:
                    winning_gid = list(candidates.keys())[0]
                    if nid not in assigned_new:
                        assigned_new[nid] = winning_gid
                        event_flags[nid]['born'] = 0.0
                        event_flags[nid]['died'] = 0.0

                else:
                    # Genuine merge
                    split_or_merge_nids.add(nid)
                    winning_gid = max(candidates, key=candidates.get)

                    if nid not in assigned_new or assigned_new[nid] in prev_global_features:
                        assigned_new[nid] = winning_gid

                    loser_gids = [gid for gid in candidates if gid != winning_gid]

                    for gid in loser_gids:
                        merge_loser_gids.add(gid)
                        if gid in master_history:
                            master_history[gid].append({
                                't': timestamp, 'event': 'merged_into', 'id': winning_gid
                            })
                            master_history[gid].append({
                                't': timestamp, 'event': 'died'
                            })

                    master_history[winning_gid].append({
                        't': timestamp, 'event': 'absorbed', 'ids': loser_gids
                    })

                    event_flags[nid]['born']     = 0.0
                    event_flags[nid]['died']     = 0.0
                    event_flags[nid]['absorbed'] = loser_gids[:5]

            # ── STEP 5: births and deaths ─────────────────────────────────────
            for oid, linked_nids in old_to_new.items():
                if len(linked_nids) == 0:
                    old_gid = prev_local_to_global[oid]
                    if old_gid not in merge_loser_gids:
                        master_history[old_gid].append({'t': timestamp, 'event': 'died'})

            for nid in new_features_raw:
                if nid not in assigned_new:
                    new_gid             = next_global_id
                    next_global_id     += 1
                    assigned_new[nid]   = new_gid
                    master_history[new_gid] = [{'t': timestamp, 'event': 'born'}]
                    event_flags[nid]['born'] = 1.0
                    event_flags[nid]['died'] = 0.0

            new_local_to_global = assigned_new

            # ── STEP 6: motion ────────────────────────────────────────────────
            for nid, gid in new_local_to_global.items():
                prev_oid = None
                for oid, old_gid in prev_local_to_global.items():
                    if old_gid == gid:
                        prev_oid = oid
                        break

                if prev_oid is None:
                    continue

                prev_f = prev_features_raw[prev_oid]
                curr_f = new_features_raw[nid]

                dx_m = (curr_f['centre_x_km'] - prev_f['centre_x_km']) * 1000.0
                dy_m = (curr_f['centre_y_km'] - prev_f['centre_y_km']) * 1000.0

                motion_info[nid]['u']            = dx_m / dt_seconds
                motion_info[nid]['v']            = dy_m / dt_seconds
                motion_info[nid]['prev_mass']    = prev_f['refl_mass_dBZkm2']
                motion_info[nid]['motion_clean'] = (
                    0.0 if nid in split_or_merge_nids else 1.0
                )

        # ── store raw records for this timestep ───────────────────────────────
        for nid, gid in new_local_to_global.items():
            f  = new_features_raw[nid]
            mi = motion_info[nid]
            ef = event_flags[nid]

            raw_records[t_idx].append({
                'feature_id'       : gid,
                'n_cells'          : float(f['n_cells']),
                'area_km2'         : f['area_km2'],
                'mean_dbz'         : f['mean_dbz'],
                'centre_x_km'      : f['centre_x_km'],
                'centre_y_km'      : f['centre_y_km'],
                'centre_lat'       : f['centre_lat'],
                'centre_lon'       : f['centre_lon'],
                'refl_mass_dBZkm2' : f['refl_mass_dBZkm2'],
                'u_ms'             : mi['u'],
                'v_ms'             : mi['v'],
                'prev_mass_dBZkm2' : mi['prev_mass'],
                'motion_clean'     : mi['motion_clean'],
                'born'             : ef['born'],
                'died'             : ef['died'],
                'split_from'       : ef['split_from'],
                'merges_with'      : np.nan,   # filled in second pass
                'split_out_1'      : float(ef['split_out'][0]) if len(ef['split_out']) > 0 else np.nan,
                'split_out_2'      : float(ef['split_out'][1]) if len(ef['split_out']) > 1 else np.nan,
                'split_out_3'      : float(ef['split_out'][2]) if len(ef['split_out']) > 2 else np.nan,
                'split_out_4'      : float(ef['split_out'][3]) if len(ef['split_out']) > 3 else np.nan,
                'split_out_5'      : float(ef['split_out'][4]) if len(ef['split_out']) > 4 else np.nan,
                'absorbed_1'       : float(ef['absorbed'][0]) if len(ef['absorbed']) > 0 else np.nan,
                'absorbed_2'       : float(ef['absorbed'][1]) if len(ef['absorbed']) > 1 else np.nan,
                'absorbed_3'       : float(ef['absorbed'][2]) if len(ef['absorbed']) > 2 else np.nan,
                'absorbed_4'       : float(ef['absorbed'][3]) if len(ef['absorbed']) > 3 else np.nan,
                'absorbed_5'       : float(ef['absorbed'][4]) if len(ef['absorbed']) > 4 else np.nan,
            })

        print(f"t={timestamp} — {len(new_local_to_global)} feature(s)")

        # ── update previous-frame state ───────────────────────────────────────
        prev_global_features = {}
        for nid, gid in new_local_to_global.items():
            f = {k: v for k, v in new_features_raw[nid].items() if k != 'mask'}
            prev_global_features[gid] = f

        prev_local_to_global = new_local_to_global
        prev_features_raw    = new_features_raw
        prev_timestamp       = timestamp

        # Prune master_history for dead features
        live_gids = set(new_local_to_global.values())
        for gid in [g for g in master_history if g not in live_gids]:
            del master_history[gid]

    # ── SECOND PASS: fill in merges_with at T-1 ───────────────────────────────
    # For each feature that ends in a merge, find the timestep before it dies
    # and write the winning feature's ID into merges_with
    print("Running second pass to fill merges_with...")

    # Build a lookup: feature_id → list of (t_idx, record) tuples
    feature_record_lookup = {}   # { gid : [(t_idx, record), ...] }
    for t_idx, records in enumerate(raw_records):
        for rec in records:
            gid = rec['feature_id']
            if gid not in feature_record_lookup:
                feature_record_lookup[gid] = []
            feature_record_lookup[gid].append((t_idx, rec))

    # For each feature, check if its last event was a merge
    # If so, write the merge target ID into the second-to-last timestep
    for gid, t_records in feature_record_lookup.items():
        if len(t_records) < 1:
            continue

        # Sort by t_idx
        t_records_sorted = sorted(t_records, key=lambda x: x[0])
        last_t_idx, last_rec = t_records_sorted[-1]

        # Check master_history for this feature — but it's been pruned
        # Instead check the absorbed columns of other features at last_t_idx+1
        # to see if this gid appears there
        if last_t_idx + 1 < n_times:
            next_records = raw_records[last_t_idx + 1]
            for next_rec in next_records:
                absorbed_ids = [
                    next_rec.get(f'absorbed_{i}') for i in range(1, 6)
                    if not np.isnan(next_rec.get(f'absorbed_{i}', np.nan))
                ]
                if gid in absorbed_ids:
                    # This feature was absorbed by next_rec's feature
                    # Write the winning ID into the last record of this feature
                    last_rec['merges_with'] = float(next_rec['feature_id'])
                    # Also set died=1 on the last record
                    last_rec['died'] = 1.0
                    break

        # Also set died=1 for features that simply vanish (no merge)
        if last_rec['died'] == 0.0:
            last_rec['died'] = 1.0

    # ── BUILD XARRAY DATASET ──────────────────────────────────────────────────
    print("Building xarray Dataset...")

    # Maximum feature ID seen — defines the feature_id dimension size
    max_fid = next_global_id - 1
    # Feature IDs are 1-based so dimension runs 1..max_fid
    feature_ids = np.arange(1, max_fid + 1)

    # Variable names to include
    var_names = [
        'n_cells', 'area_km2', 'mean_dbz',
        'centre_x_km', 'centre_y_km', 'centre_lat', 'centre_lon',
        'refl_mass_dBZkm2', 'u_ms', 'v_ms', 'prev_mass_dBZkm2',
        'motion_clean', 'born', 'died',
        'split_from', 'merges_with',
        'split_out_1', 'split_out_2', 'split_out_3', 'split_out_4', 'split_out_5',
        'absorbed_1',  'absorbed_2',  'absorbed_3',  'absorbed_4',  'absorbed_5',
    ]

    # Initialise all arrays as NaN — shape (n_times, max_fid)
    arrays = {
        vname: np.full((n_times, max_fid), np.nan, dtype=np.float32)
        for vname in var_names
    }

    # Fill arrays from raw_records
    for t_idx, records in enumerate(raw_records):
        for rec in records:
            gid    = rec['feature_id']
            fid_i  = gid - 1   # 0-based index into feature_id dimension
            for vname in var_names:
                val = rec.get(vname, np.nan)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    arrays[vname][t_idx, fid_i] = np.float32(val)

    # ── compression encoding ──────────────────────────────────────────────────
    encoding = {
        vname: {
            'zlib'       : True,
            'complevel'  : 3,
            'dtype'      : 'float32',
            '_FillValue' : np.float32(np.nan),
        }
        for vname in var_names
    }

    # ── assemble xarray Dataset ───────────────────────────────────────────────
    data_vars = {}
    for vname in var_names:
        data_vars[vname] = xr.Variable(
            dims    = ('time', 'feature_id'),
            data    = arrays[vname],
            attrs   = {'_FillValue': np.float32(np.nan)},
        )

    FeaturesXR = xr.Dataset(
        data_vars = data_vars,
        coords    = {
            'time'       : timestamps,
            'feature_id' : feature_ids,
        },
        attrs = {
            'description'       : 'Tracked radar reflectivity features',
            'threshold_dbz'     : threshold_dbz,
            'overlap_threshold' : overlap_threshold,
            'radar_lat'         : radar_lat,
            'radar_lon'         : radar_lon,
        }
    )

    print(f"FeaturesXR built: {n_times} timesteps x {max_fid} features.")
    print(FeaturesXR)

    return FeaturesXR




def save_features_xr(FeaturesXR, save_path):
    """
    Save FeaturesXR to a compressed NetCDF file.

    Parameters
    ----------
    FeaturesXR : xarray.Dataset
    save_path : str
    """
    var_names = list(FeaturesXR.data_vars)

    encoding = {
        vname: {
            'zlib'      : True,
            'complevel' : 3,
            'dtype'     : 'float32',
        }
        for vname in var_names
    }

    # Strip _FillValue from attrs on all variables
    for vname in var_names:
        FeaturesXR[vname].attrs.pop('_FillValue', None)

    FeaturesXR.to_netcdf(save_path, encoding=encoding)
    print(f"Saved FeaturesXR to {save_path}")




def save_features_xr_sterile(FeaturesXR, save_path):
    """
    Save FeaturesXR to a compressed NetCDF file.

    Parameters
    ----------
    FeaturesXR : xarray.Dataset
    save_path : str
    """
    # Create a clean copy with no attrs on any variable to avoid
    # xarray's CFMaskCoder _FillValue conflict
    clean_vars = {}
    for vname in FeaturesXR.data_vars:
        clean_vars[vname] = xr.Variable(
            dims  = FeaturesXR[vname].dims,
            data  = FeaturesXR[vname].values,
            attrs = {},   # completely empty attrs
        )

    FeaturesClean = xr.Dataset(
        data_vars = clean_vars,
        coords    = FeaturesXR.coords,
        attrs     = FeaturesXR.attrs,
    )

    encoding = {
        vname: {
            'zlib'      : True,
            'complevel' : 3,
            'dtype'     : 'float32',
        }
        for vname in FeaturesClean.data_vars
    }

    FeaturesClean.to_netcdf(save_path, encoding=encoding)
    print(f"Saved FeaturesXR to {save_path}")




def load_features_xr(load_path):
    """
    Load a previously saved FeaturesXR NetCDF file.

    Parameters
    ----------
    load_path : str

    Returns
    -------
    FeaturesXR : xarray.Dataset
    """
    FeaturesXR = xr.open_dataset(load_path)
    print(f"Loaded FeaturesXR from {load_path}")
    print(FeaturesXR)
    return FeaturesXR


# ── helpers ───────────────────────────────────────────────────────────────────

def _km_offset_to_latlon(centre_x_km, centre_y_km, radar_lat, radar_lon):
    lat = radar_lat + (centre_y_km / 111.32)
    lon = radar_lon + (centre_x_km / (111.32 * np.cos(np.radians(radar_lat))))
    return lat, lon


def _detect_features(refl, x_coords, y_coords, threshold_dbz, cell_area_km2,
                     radar_lat, radar_lon):
    refl = np.squeeze(refl)

    if refl.ndim != 2:
        raise ValueError(
            f"Expected a 2D reflectivity slice after squeezing, "
            f"but got shape {refl.shape}. Check your time indexing."
        )

    mask            = refl > threshold_dbz
    structure_8conn = np.ones((3, 3), dtype=int)
    labelled_grid, n_features = label(mask, structure=structure_8conn)

    x_grid_2d, y_grid_2d = np.meshgrid(x_coords, y_coords)

    features = {}
    for fid in range(1, n_features + 1):
        fmask     = labelled_grid == fid
        refl_vals = refl[fmask]
        linear_z  = 10.0 ** (refl_vals / 10.0)
        total_w   = linear_z.sum()
        n_cells   = int(fmask.sum())

        centre_x = float((x_grid_2d[fmask] * linear_z).sum() / total_w)
        centre_y = float((y_grid_2d[fmask] * linear_z).sum() / total_w)

        centre_lat, centre_lon = _km_offset_to_latlon(
            centre_x, centre_y, radar_lat, radar_lon
        )

        features[fid] = {
            'local_id'         : fid,
            'n_cells'          : n_cells,
            'area_km2'         : float(n_cells * cell_area_km2),
            'mean_dbz'         : float(refl_vals.mean()),
            'centre_x_km'      : centre_x,
            'centre_y_km'      : centre_y,
            'centre_lat'       : round(centre_lat, 5),
            'centre_lon'       : round(centre_lon, 5),
            'refl_mass_dBZkm2' : float(10.0 * np.log10((linear_z * cell_area_km2).sum())),
            'mask'             : fmask,
        }

    return features, labelled_grid


def _compute_overlap_links(old_features, new_features, overlap_threshold=0.20):
    old_to_new = {oid: [] for oid in old_features}
    new_to_old = {nid: [] for nid in new_features}

    for oid, of in old_features.items():
        for nid, nf in new_features.items():
            intersection = int((of['mask'] & nf['mask']).sum())
            if intersection == 0:
                continue
            larger_size = max(of['n_cells'], nf['n_cells'])
            fraction    = intersection / larger_size
            if fraction > overlap_threshold:
                old_to_new[oid].append(nid)
                new_to_old[nid].append(oid)

    return old_to_new, new_to_old


# ── main tracker ──────────────────────────────────────────────────────────────

def track_reflectivity_features(
    xgrid,
    threshold_dbz     = 20.0,
    overlap_threshold = 0.20,
):
    """
    Track radar reflectivity features across all time steps in xgrid.

    Parameters
    ----------
    xgrid : xarray.Dataset
        Expected dimensions: (time, y, x).
        Must contain 'origin_latitude' and 'origin_longitude'.
    threshold_dbz : float
    overlap_threshold : float

    Returns
    -------
    results : dict  { timestamp : pd.DataFrame }
    all_labels : dict { timestamp : np.ndarray }
    """

    # ── coordinate setup ──────────────────────────────────────────────────────
    x_coords      = xgrid['x'].values / 1000.0
    y_coords      = xgrid['y'].values / 1000.0
    cell_area_km2 = (x_coords[1] - x_coords[0]) * (y_coords[1] - y_coords[0])
    timestamps    = xgrid['time'].values
    n_times       = len(timestamps)

    radar_lat = float(xgrid['origin_latitude'].values)
    radar_lon = float(xgrid['origin_longitude'].values)

    print(f"Radar origin: {radar_lat:.4f}°N, {radar_lon:.4f}°E")

    # ── global state ──────────────────────────────────────────────────────────
    next_global_id       = 1
    results              = {}
    all_labels           = {}
    prev_global_features = {}
    prev_local_to_global = {}
    prev_features_raw    = {}
    prev_timestamp       = None

    # Master history store — keyed by global feature ID
    # Accumulates ALL events for a feature across its entire lifetime
    # Pruned when a feature dies to free memory
    master_history = {}   # { global_id : [event_dict, ...] }

    # ── iterate over timesteps ────────────────────────────────────────────────
    for t_idx in range(n_times):
        timestamp = timestamps[t_idx]
        refl      = xgrid['corrected_reflectivity'][t_idx].values

        new_features_raw, labelled_grid = _detect_features(
            refl, x_coords, y_coords, threshold_dbz, cell_area_km2,
            radar_lat, radar_lon
        )

        new_local_to_global = {}

        motion_info = {
            nid: {
                'u'            : np.nan,
                'v'            : np.nan,
                'prev_mass'    : np.nan,
                'motion_clean' : np.nan,
            }
            for nid in new_features_raw
        }

        # ── first frame: everything is a birth ────────────────────────────────
        if t_idx == 0:
            for nid in new_features_raw:
                gid                      = next_global_id
                next_global_id          += 1
                new_local_to_global[nid] = gid
                master_history[gid]      = [{'t': timestamp, 'event': 'born'}]

        # ── subsequent frames: full tracking logic ────────────────────────────
        else:
            dt_seconds = float(
                (pd.Timestamp(timestamp) - pd.Timestamp(prev_timestamp))
                .total_seconds()
            )

            old_to_new, new_to_old = _compute_overlap_links(
                prev_features_raw, new_features_raw, overlap_threshold
            )

            assigned_new        = {}   # { new_local_id : global_id }
            split_or_merge_nids = set()

            # Keep track of which old global IDs have been consumed as merge
            # losers so we don't also mark them as dead in Step 5
            merge_loser_gids = set()

            # ── STEP 2: resolve splits ────────────────────────────────────────
            # An old feature that maps to MORE THAN ONE new feature
            for oid, linked_nids in old_to_new.items():
                if len(linked_nids) <= 1:
                    continue

                old_gid = prev_local_to_global[oid]

                linked_nids_sorted = sorted(
                    linked_nids,
                    key=lambda nid: new_features_raw[nid]['refl_mass_dBZkm2'],
                    reverse=True,
                )

                # Flag ALL new features from this split as contaminated
                for nid in linked_nids_sorted:
                    split_or_merge_nids.add(nid)

                winner_nid = linked_nids_sorted[0]
                loser_nids = linked_nids_sorted[1:]

                # Assign loser IDs first so we can reference them in the
                # winner's history entry
                loser_gids = []
                for loser_nid in loser_nids:
                    if loser_nid not in assigned_new:
                        new_gid                 = next_global_id
                        next_global_id         += 1
                        assigned_new[loser_nid] = new_gid
                        master_history[new_gid] = [{
                            't'     : timestamp,
                            'event' : 'split_from',
                            'id'    : old_gid,
                        }]
                    loser_gids.append(assigned_new[loser_nid])

                # Winner inherits old ID — only write history once
                if winner_nid not in assigned_new:
                    assigned_new[winner_nid] = old_gid
                    master_history[old_gid].append({
                        't'     : timestamp,
                        'event' : 'split_winner',
                        'ids'   : loser_gids,
                    })

            # ── STEP 3: simple continuation candidates ────────────────────────
            # Only note the candidate — do NOT assign yet so Step 4 can see
            # all competing old IDs for every new feature before deciding
            continuation_candidates = {}  # { new_local_id : old_global_id }
            for oid, linked_nids in old_to_new.items():
                if len(linked_nids) != 1:
                    continue
                nid     = linked_nids[0]
                old_gid = prev_local_to_global[oid]
                # Only register if not already claimed by a split
                if nid not in assigned_new:
                    continuation_candidates[nid] = old_gid

            # ── STEP 4: resolve merges and continuations together ─────────────
            for nid in new_features_raw:
                linked_oids = new_to_old.get(nid, [])

                # Gather every old global ID that has a valid overlap link
                # to this new feature, using the OLD feature's mass throughout
                candidates = {}  # { old_global_id : old_refl_mass_dBZkm2 }

                for oid in linked_oids:
                    old_gid = prev_local_to_global[oid]
                    # Always use the previous frame's mass for fair comparison
                    candidates[old_gid] = prev_global_features[old_gid]['refl_mass_dBZkm2']

                # Also include any split-winner already assigned to this nid,
                # again using the OLD feature's mass
                if nid in assigned_new:
                    existing_gid = assigned_new[nid]
                    if existing_gid not in candidates and existing_gid in prev_global_features:
                        candidates[existing_gid] = prev_global_features[existing_gid]['refl_mass_dBZkm2']

                # Also include simple continuation candidates
                if nid in continuation_candidates:
                    cont_gid = continuation_candidates[nid]
                    if cont_gid not in candidates:
                        candidates[cont_gid] = prev_global_features[cont_gid]['refl_mass_dBZkm2']

                if len(candidates) == 0:
                    # No old feature links — birth handled in Step 5
                    continue

                elif len(candidates) == 1:
                    winning_gid = list(candidates.keys())[0]

                    # FIX Bug 3: do NOT overwrite a split loser's brand new ID
                    if nid not in assigned_new:
                        assigned_new[nid] = winning_gid
                    elif assigned_new[nid] != winning_gid:
                        # nid was assigned as a split loser — leave it alone
                        pass

                else:
                    # Genuine merge — flag and pick winner by OLD mass
                    split_or_merge_nids.add(nid)
                    winning_gid = max(candidates, key=candidates.get)

                    # FIX Bug 3: only assign if not already a split loser
                    if nid not in assigned_new or assigned_new[nid] in prev_global_features:
                        assigned_new[nid] = winning_gid

                    loser_gids = [gid for gid in candidates if gid != winning_gid]

                    for gid in loser_gids:
                        merge_loser_gids.add(gid)

                        # Append merged_into to loser's master history
                        if gid in master_history:
                            master_history[gid].append({
                                't'     : timestamp,
                                'event' : 'merged_into',
                                'id'    : winning_gid,
                            })

                        # FIX Bug 5: also write died for merge losers
                        master_history[gid].append({
                            't'     : timestamp,
                            'event' : 'died',
                        })

                    # Winner records all absorbed IDs in one event
                    master_history[winning_gid].append({
                        't'     : timestamp,
                        'event' : 'absorbed',
                        'ids'   : loser_gids,
                    })

            # ── STEP 5: births and deaths ─────────────────────────────────────

            # Deaths — old features with no links at all
            # Exclude merge losers as they already got their died event above
            for oid, linked_nids in old_to_new.items():
                if len(linked_nids) == 0:
                    old_gid = prev_local_to_global[oid]
                    if old_gid not in merge_loser_gids:
                        master_history[old_gid].append({
                            't'     : timestamp,
                            'event' : 'died',
                        })

            # Births — new features with no assignment at all
            for nid in new_features_raw:
                if nid not in assigned_new:
                    new_gid             = next_global_id
                    next_global_id     += 1
                    assigned_new[nid]   = new_gid
                    master_history[new_gid] = [{'t': timestamp, 'event': 'born'}]

            new_local_to_global = assigned_new

            # ── STEP 6: compute motion info ───────────────────────────────────
            for nid, gid in new_local_to_global.items():

                # Find which old local ID this global ID came from (if any)
                prev_oid = None
                for oid, old_gid in prev_local_to_global.items():
                    if old_gid == gid:
                        prev_oid = oid
                        break

                if prev_oid is None:
                    # Born or split loser with brand new ID — no previous position
                    continue

                prev_f = prev_features_raw[prev_oid]
                curr_f = new_features_raw[nid]

                dx_m = (curr_f['centre_x_km'] - prev_f['centre_x_km']) * 1000.0
                dy_m = (curr_f['centre_y_km'] - prev_f['centre_y_km']) * 1000.0

                motion_info[nid]['u']            = dx_m / dt_seconds
                motion_info[nid]['v']            = dy_m / dt_seconds
                motion_info[nid]['prev_mass']    = prev_f['refl_mass_dBZkm2']
                motion_info[nid]['motion_clean'] = (
                    False if nid in split_or_merge_nids else True
                )

        # ── build global labelled grid ─────────────────────────────────────────
        global_labelled = np.zeros_like(labelled_grid)
        for nid, gid in new_local_to_global.items():
            global_labelled[new_features_raw[nid]['mask']] = gid
        all_labels[timestamp] = global_labelled

        # ── build output DataFrame ─────────────────────────────────────────────
        rows = []
        for nid, gid in new_local_to_global.items():
            f  = new_features_raw[nid]
            mi = motion_info[nid]
            rows.append({
                'feature_id'       : gid,
                'n_cells'          : f['n_cells'],
                'area_km2'         : round(f['area_km2'],         2),
                'mean_dbz'         : round(f['mean_dbz'],         2),
                'centre_x_km'      : round(f['centre_x_km'],      3),
                'centre_y_km'      : round(f['centre_y_km'],      3),
                'centre_lat'       : f['centre_lat'],
                'centre_lon'       : f['centre_lon'],
                'refl_mass_dBZkm2' : round(f['refl_mass_dBZkm2'], 2),
                'u_ms'             : mi['u'],
                'v_ms'             : mi['v'],
                'prev_mass_dBZkm2' : mi['prev_mass'],
                'motion_clean'     : mi['motion_clean'],
                # Snapshot of full accumulated history at this point in time
                'history'          : list(master_history.get(gid, [])),
            })

        if len(rows) == 0:
            df = pd.DataFrame(columns=[
                'feature_id', 'n_cells', 'area_km2', 'mean_dbz',
                'centre_x_km', 'centre_y_km', 'centre_lat', 'centre_lon',
                'refl_mass_dBZkm2', 'u_ms', 'v_ms', 'prev_mass_dBZkm2',
                'motion_clean', 'history'
            ])
        else:
            df = pd.DataFrame(rows).sort_values('feature_id').reset_index(drop=True)

        results[timestamp] = df

        print(f"t={timestamp} — {len(rows)} feature(s)")
        if len(rows) > 0:
            print(df.drop(columns='history').to_string(index=False))
        print()

        # ── update previous-frame state ───────────────────────────────────────
        prev_global_features = {}
        for nid, gid in new_local_to_global.items():
            f = {k: v for k, v in new_features_raw[nid].items() if k != 'mask'}
            f['history'] = list(master_history.get(gid, []))
            prev_global_features[gid] = f

        prev_local_to_global = new_local_to_global
        prev_features_raw    = new_features_raw
        prev_timestamp       = timestamp

        # FIX Bug 6: prune master_history for features that are now dead
        # Keep a set of all currently live global IDs
        live_gids = set(new_local_to_global.values())
        dead_gids = [gid for gid in master_history if gid not in live_gids]
        for gid in dead_gids:
            del master_history[gid]

    return results, all_labels



def summarise_features(results):
    """
    Summarise the full lifetime of every feature tracked across all timesteps.

    One row per feature containing birth/death info, mean motion, mean size
    and mean reflectivity mass.

    Parameters
    ----------
    results : dict { timestamp : pd.DataFrame }
        Output from track_reflectivity_features().

    Returns
    -------
    summary : pd.DataFrame
        One row per feature.
    """

    timestamps     = list(results.keys())
    first_ts       = timestamps[0]
    last_ts        = timestamps[-1]

    # ── collect per-feature per-timestep records ──────────────────────────────
    # feature_records[gid] = list of dicts, one per timestep the feature exists
    feature_records = {}   # { global_id : [{'t', 'x', 'y', 'lat', 'lon',
                           #                  'area_km2', 'refl_mass', 'history'}, ...] }

    for ts, df in results.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            gid = row['feature_id']
            if gid not in feature_records:
                feature_records[gid] = []
            feature_records[gid].append({
                't'          : ts,
                'x_km'       : row['centre_x_km'],
                'y_km'       : row['centre_y_km'],
                'lat'        : row['centre_lat'],
                'lon'        : row['centre_lon'],
                'area_km2'   : row['area_km2'],
                'refl_mass'  : row['refl_mass_dBZkm2'],
                'history'    : row['history'],
            })

    # ── build one summary row per feature ─────────────────────────────────────
    rows = []

    for gid, records in feature_records.items():
        # Sort by time just in case
        records = sorted(records, key=lambda r: r['t'])

        birth_rec = records[0]
        death_rec = records[-1]

        birth_t   = birth_rec['t']
        death_t   = death_rec['t']

        # Time alive in seconds
        dt_alive  = float(
            (pd.Timestamp(death_t) - pd.Timestamp(birth_t)).total_seconds()
        )

        # ── mean U and V from bulk displacement ───────────────────────────────
        if dt_alive > 0:
            dx_m  = (death_rec['x_km'] - birth_rec['x_km']) * 1000.0
            dy_m  = (death_rec['y_km'] - birth_rec['y_km']) * 1000.0
            mean_u = dx_m / dt_alive
            mean_v = dy_m / dt_alive
        else:
            mean_u = np.nan
            mean_v = np.nan

        # ── mean size and mass over lifetime ──────────────────────────────────
        mean_area      = float(np.mean([r['area_km2']  for r in records]))
        mean_refl_mass = float(np.mean([r['refl_mass'] for r in records]))

        # ── parse history for boolean flags ───────────────────────────────────
        # Use the most complete history — from the last record
        history = death_rec['history']

        event_types = [e['event'] for e in history]

        started_in_split  = 'split_from'   in event_types
        ended_in_merge    = 'merged_into'  in event_types
        alive_at_first    = birth_t == first_ts
        alive_at_last     = death_t == last_ts

        rows.append({
            'feature_id'        : gid,
            'birth_t'           : birth_t,
            'death_t'           : death_t,
            'time_alive_s'      : dt_alive,
            'birth_x_km'        : round(birth_rec['x_km'],  3),
            'birth_y_km'        : round(birth_rec['y_km'],  3),
            'birth_lat'         : birth_rec['lat'],
            'birth_lon'         : birth_rec['lon'],
            'death_x_km'        : round(death_rec['x_km'],  3),
            'death_y_km'        : round(death_rec['y_km'],  3),
            'death_lat'         : death_rec['lat'],
            'death_lon'         : death_rec['lon'],
            'mean_u_ms'         : mean_u,
            'mean_v_ms'         : mean_v,
            'mean_area_km2'     : round(mean_area,      2),
            'mean_refl_mass_dBZkm2' : round(mean_refl_mass, 2),
            'n_timesteps'       : len(records),
            'started_in_split'  : started_in_split,
            'ended_in_merge'    : ended_in_merge,
            'alive_at_first_frame' : alive_at_first,
            'alive_at_last_frame'  : alive_at_last,
        })

    summary = pd.DataFrame(rows).sort_values('feature_id').reset_index(drop=True)

    print(f"Summarised {len(summary)} unique features.")
    print(summary.to_string(index=False))

    return summary
