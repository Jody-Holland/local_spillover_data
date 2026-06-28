#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AGB pipeline using sample proportions and full-grid total area.

Buffer-zone semantics
----------------------
The exterior of each project is sliced into 500 m ANNULAR rings by
`assign_buffer_zone` (via np.searchsorted, side='left'): a pixel at distance d
lands in exactly one ring, `buffer_X`, where (X - 500) < d <= X. Those rings
are disjoint and each is 500 m wide.

This version produces CUMULATIVE buffers by default: `buffer_X` is the total
over all rings with threshold <= X, i.e. all eligible area within X metres of
the boundary. AGB is extensive (additive over disjoint rings), so cumulative
AGB is the running sum of ring AGB; obs/cf are summed independently and diff is
recomputed as obs - cf. Set CUMULATIVE = False to emit the raw annular rings
instead (useful for distance-decay analysis). The interior ('project') zone is
always passed through unchanged; cumulative buffers are exterior-only.

Memory
------
Each task reads a full grid and a pairs file. Two measures keep memory bounded
across a long run:
  * the heavy parquet reads are column-restricted, and each file's pages are
    evicted from the OS page cache with posix_fadvise once read (otherwise
    cumulative reads inflate page cache / SLURM cgroup memory monotonically);
  * the pool uses maxtasksperchild=1, so each worker's heap is reclaimed by the
    OS after every project rather than ratcheting up to a high-water mark.

Other behaviour (proportion estimation from the 1/8 sample, normalisation to
sum to 1 per zone-year, the LUC 1-4 filter, carbon-density weighting) is
unchanged.
"""

import logging
import os
import re
import shutil
import traceback
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
import shapely
import multiprocessing as mp

try:
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except Exception:
    _HAVE_PYARROW = False

from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path('/scratch/jh2589/wdpa_leakage_analysis')
GEOJSON_DIR = Path('/scratch/jh2589/project_meta/wdpa_tropics')
PROJECT_CSV = Path('/scratch/jh2589/chap1_outputs/csv/filtered_projects.csv')
OUTPUT_DIR = BASE_DIR / 'agb_outputs'

PROJECT_ID_COL = 'project'

PIXEL_AREA_HA = 0.09
LUC_CLASSES = [1, 2, 3, 4]
BUFFER_THRESHOLDS = np.arange(500, 10001, 500)

SIMPLIFY_TOL_M = 20.0
N_WORKERS = 64

# True  -> buffer_X = all eligible area within X metres of the boundary (cumulative)
# False -> buffer_X = the (X-500, X] annulus only (raw rings)
CUMULATIVE = True

# Evict each heavy parquet from the page cache after reading it. Keeps page
# cache / cgroup memory flat across the run. Harmless to leave on.
DROP_PAGE_CACHE = True

_MODE = 'cumulative' if CUMULATIVE else 'annular'
OUTPUT_CSV = OUTPUT_DIR / f'all_projects_wide_agb_{_MODE}.csv'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('agb_pipeline')


# ============================================================================
# HELPERS
# ============================================================================
def _drop_cache(path):
    """Advise the kernel to evict a file's pages from the page cache once we
    have finished reading it. Best-effort; harmless if unsupported. The data is
    already copied into pandas/numpy buffers by the time this is called, so the
    (clean) file pages can be dropped without affecting the in-memory frame."""
    if not DROP_PAGE_CACHE:
        return
    fadvise = getattr(os, 'posix_fadvise', None)
    if fadvise is None:
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def parquet_columns_in_order(path):
    if _HAVE_PYARROW:
        return list(pq.read_schema(str(path)).names)
    return list(pd.read_parquet(path).columns)


def read_parquet_columns(path, wanted_prefixes=(), wanted_exact=()):
    available = parquet_columns_in_order(path)
    cols = [c for c in available
            if c in wanted_exact or any(c.startswith(p) for p in wanted_prefixes)]
    missing_exact = [c for c in wanted_exact if c not in available]
    return cols, missing_exact


def union_all_geometry(gdf):
    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.geometry.unary_union


def utm_crs_for(geom):
    centroid = geom.centroid
    zone = int((centroid.x + 180) / 6) + 1
    base = 32600 if centroid.y >= 0 else 32700
    return f"EPSG:{base + zone}"


def split_inside_outside(df, lng_col, lat_col, project_gdf,
                         max_buffer_m=float(BUFFER_THRESHOLDS.max()),
                         simplify_tol_m=SIMPLIFY_TOL_M):
    project_geom_ll = union_all_geometry(project_gdf)
    utm_crs = utm_crs_for(project_geom_ll)
    project_geom_utm = union_all_geometry(project_gdf.to_crs(utm_crs))

    shapely.prepare(project_geom_utm)
    geom_dist = (project_geom_utm if simplify_tol_m <= 0
                 else project_geom_utm.simplify(simplify_tol_m,
                                                preserve_topology=True))

    transformer = pyproj.Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
    x, y = transformer.transform(df[lng_col].to_numpy(), df[lat_col].to_numpy())

    minx, miny, maxx, maxy = project_geom_utm.bounds
    near = ((x >= minx - max_buffer_m) & (x <= maxx + max_buffer_m) &
            (y >= miny - max_buffer_m) & (y <= maxy + max_buffer_m))

    df_near = df.loc[near].copy()
    pts = shapely.points(x[near], y[near])

    inside_mask = shapely.intersects(project_geom_utm, pts)
    inside_df = df_near.loc[inside_mask].copy()
    outside_df = df_near.loc[~inside_mask].copy()

    if len(outside_df):
        outside_df['dist_m'] = shapely.distance(geom_dist, pts[~inside_mask])

    return inside_df, outside_df


def assign_buffer_zone(df, dist_col='dist_m', thresholds=BUFFER_THRESHOLDS):
    """Assign each exterior pixel to its 500 m annular ring.

    searchsorted(side='left') maps distance d to the ring `buffer_X` with
    (X - 500) < d <= X. Pixels beyond the outermost threshold get None and are
    dropped. These rings are disjoint regardless of the CUMULATIVE flag; the
    cumulative roll-up happens later in `accumulate_zones`.
    """
    if df.empty:
        return df
    dist = df[dist_col].to_numpy()
    zone_idx = np.searchsorted(thresholds, dist, side='left')
    df = df.copy()
    df['buffer_zone'] = [f'buffer_{int(thresholds[i])}' if i < len(thresholds) else None
                         for i in zone_idx]
    df = df[df['buffer_zone'].notna()]
    return df


def get_total_area_per_zone(project_id, project_gdf, start_year):
    grid_path = BASE_DIR / str(project_id) / 'k_grids' / 'all_k_grids.parquet'
    if not grid_path.exists():
        log.warning("[%s] missing k_grids", project_id)
        return None

    cols, missing = read_parquet_columns(
        grid_path, wanted_prefixes=('luc_',), wanted_exact=('lng', 'lat'),
    )
    if missing:
        log.warning("[%s] k_grids missing coords", project_id)
        return None
    df = pd.read_parquet(grid_path, columns=cols)
    _drop_cache(grid_path)

    luc_col = f'luc_{start_year}'
    if luc_col not in df.columns:
        log.warning("[%s] no luc_%d in k_grids", project_id, start_year)
        return None

    df = df[df[luc_col].isin(LUC_CLASSES)].copy()

    inside_df, outside_df = split_inside_outside(df, 'lng', 'lat', project_gdf)
    if len(inside_df) == 0:
        log.warning("[%s] no valid pixels inside project polygon", project_id)
        return None

    # Per-ring (annular) areas. These are the additive building blocks; the
    # cumulative roll-up is applied to the AGB totals downstream.
    total_areas = {'project': len(inside_df) * PIXEL_AREA_HA}
    log.info("[%s] full-grid: project pixels = %d", project_id, len(inside_df))

    if len(outside_df):
        outside_df = assign_buffer_zone(outside_df)
        for zone, group in outside_df.groupby('buffer_zone'):
            total_areas[zone] = len(group) * PIXEL_AREA_HA
            log.info("[%s] full-grid: %s ring pixels = %d", project_id, zone, len(group))
    else:
        log.info("[%s] full-grid: no exterior pixels", project_id)

    return total_areas


def compute_proportions_from_pairs(project_id, project_gdf):
    pairs_path = BASE_DIR / str(project_id) / 'all_pairs.parquet'
    if not pairs_path.exists():
        log.warning("[%s] missing all_pairs.parquet", project_id)
        return None, None, None

    # Read only the columns we use (coords + k_luc_YYYY + s_prop_*), not the
    # whole file. This is the dominant per-task allocation, so trimming it
    # lowers the memory plateau each worker reaches.
    cols, missing = read_parquet_columns(
        pairs_path, wanted_prefixes=('k_luc_', 's_prop_'),
        wanted_exact=('k_lng', 'k_lat'),
    )
    if missing:
        log.warning("[%s] all_pairs missing coords %s", project_id, missing)
        return None, None, None
    df = pd.read_parquet(pairs_path, columns=cols)
    _drop_cache(pairs_path)

    # Log column names for debugging
    luc_cols_sample = [c for c in df.columns if 'luc' in c]
    prop_cols_sample = [c for c in df.columns if 's_prop' in c]
    log.info("[%s] Found %d luc columns, %d s_prop columns",
             project_id, len(luc_cols_sample), len(prop_cols_sample))
    if prop_cols_sample:
        log.info("[%s] Sample s_prop columns: %s", project_id, prop_cols_sample[:5])

    k_luc_cols = [c for c in df.columns if re.match(r'k_luc_\d{4}', c)]
    s_prop_cols = [c for c in df.columns if re.match(r's_prop_\d+_\d{4}', c) or re.match(r's_prop_\d{4}', c)]

    if not k_luc_cols:
        log.warning("[%s] no k_luc_YYYY columns", project_id)
        return None, None, None

    years = sorted({int(c.split('_')[-1]) for c in k_luc_cols})
    start_year = years[0]
    log.info("[%s] start_year = %d, years: %s", project_id, start_year, years)

    # Parse s_prop columns: (class, year) -> column name
    s_prop_map = {}
    for col in s_prop_cols:
        parts = col.split('_')
        # s_prop_X_YYYY has 4 parts, s_prop_YYYY has 2 parts
        if len(parts) == 4:  # s_prop_X_YYYY
            cls = int(parts[2])
            yr = int(parts[3])
        elif len(parts) == 2:  # s_prop_YYYY
            cls = 1
            yr = int(parts[1])
        else:
            continue
        s_prop_map[(cls, yr)] = col

    log.info("[%s] Parsed %d s_prop entries: %s",
             project_id, len(s_prop_map), list(s_prop_map.keys())[:10])

    # Split pixels into zones (annular rings)
    inside_df, outside_df = split_inside_outside(df, 'k_lng', 'k_lat', project_gdf)
    if len(inside_df) == 0:
        log.warning("[%s] no pairs pixels inside project polygon", project_id)
        return None, None, None

    inside_df = inside_df.copy()
    inside_df['zone'] = 'project'
    log.info("[%s] pairs: project pixels = %d", project_id, len(inside_df))

    if len(outside_df):
        outside_df = assign_buffer_zone(outside_df)
        outside_df = outside_df.copy()
        outside_df['zone'] = outside_df['buffer_zone']
        for zone, group in outside_df.groupby('zone'):
            log.info("[%s] pairs: %s ring pixels = %d", project_id, zone, len(group))
    else:
        outside_df = pd.DataFrame()
        log.info("[%s] pairs: no exterior pixels", project_id)

    all_pixels = pd.concat([inside_df, outside_df], ignore_index=True)
    all_pixels = all_pixels[all_pixels['zone'].notna()]
    if all_pixels.empty:
        log.warning("[%s] no pixels after zoning", project_id)
        return None, None, None

    zones = all_pixels['zone'].unique().tolist()
    years_all = [y for y in years if y >= start_year]
    log.info("[%s] Zones: %s, years: %s", project_id, zones, years_all)

    # Build full grid of (zone, year, class)
    all_combos = []
    for zone in zones:
        for yr in years_all:
            for cls in LUC_CLASSES:
                all_combos.append({'zone': zone, 'year': yr, 'class': cls})
    full_grid = pd.DataFrame(all_combos)

    # ---- Observed proportions ----
    obs_records = []
    for zone, group in all_pixels.groupby('zone'):
        n_pixels = len(group)
        for col in k_luc_cols:
            yr = int(col.split('_')[-1])
            if yr < start_year:
                continue
            counts = group[col].value_counts()
            for cls in LUC_CLASSES:
                count = counts.get(cls, 0)
                prop = count / n_pixels if n_pixels > 0 else 0.0
                obs_records.append({'zone': zone, 'year': yr, 'class': cls, 'obs_prop': prop})
    obs_df = pd.DataFrame(obs_records)
    obs_df = full_grid.merge(obs_df, on=['zone', 'year', 'class'], how='left')
    obs_df['obs_prop'] = obs_df['obs_prop'].fillna(0.0)

    # ---- Counterfactual proportions ----
    if s_prop_map:
        cf_records = []
        for zone, group in all_pixels.groupby('zone'):
            for (cls, yr), col in s_prop_map.items():
                if yr < start_year:
                    continue
                if col in group.columns:
                    mean_prop = group[col].mean()
                    if pd.isna(mean_prop):
                        mean_prop = 0.0
                else:
                    mean_prop = 0.0
                cf_records.append({'zone': zone, 'year': yr, 'class': cls, 'cf_prop': mean_prop})
        cf_df = pd.DataFrame(cf_records)
        cf_df = full_grid.merge(cf_df, on=['zone', 'year', 'class'], how='left')
        cf_df['cf_prop'] = cf_df['cf_prop'].fillna(0.0)
    else:
        cf_df = full_grid.copy()
        cf_df['cf_prop'] = 0.0

    # Normalise proportions to sum to 1 per zone-year (transform avoids include_groups error)
    # NB: the fallback below fills a zone-year that has NO sampled LUC 1-4 pixels with a
    # flat 0.25 composition. For sparse outer rings under the 1/8 sample this fabricates a
    # composition rather than leaving it empty. Behaviour retained from the prior version;
    # revisit if outer-ring AGB looks implausible. (Under CUMULATIVE such rings are pooled
    # into the running sum, which dilutes but does not remove the effect.)
    # Observed
    obs_sum = obs_df.groupby(['zone', 'year'])['obs_prop'].transform('sum')
    obs_df['obs_prop'] = obs_df['obs_prop'] / obs_sum
    obs_df.loc[obs_df['obs_prop'].isna(), 'obs_prop'] = 1.0 / len(LUC_CLASSES)

    # Counterfactual
    cf_sum = cf_df.groupby(['zone', 'year'])['cf_prop'].transform('sum')
    cf_df['cf_prop'] = cf_df['cf_prop'] / cf_sum
    cf_df.loc[cf_df['cf_prop'].isna(), 'cf_prop'] = 1.0 / len(LUC_CLASSES)

    return start_year, obs_df, cf_df


def compute_agb_for_project(project_id, project_gdf):
    """Per-ring (annular) AGB. Returns long df with one row per (zone, year).

    The cumulative roll-up, if requested, is applied afterwards in
    `accumulate_zones` so that this function stays the testable building block.
    """
    start_year, obs_prop_df, cf_prop_df = compute_proportions_from_pairs(project_id, project_gdf)
    if start_year is None:
        return None

    total_areas = get_total_area_per_zone(project_id, project_gdf, start_year)
    if total_areas is None:
        return None

    merged = obs_prop_df.merge(cf_prop_df, on=['zone', 'year', 'class'], how='outer')
    merged['obs_prop'] = merged['obs_prop'].fillna(0.0)
    merged['cf_prop'] = merged['cf_prop'].fillna(0.0)

    merged['total_area'] = merged['zone'].map(total_areas)
    merged = merged[merged['total_area'].notna()]
    if merged.empty:
        log.warning("[%s] no zones with total area", project_id)
        return None

    merged['obs_area'] = merged['obs_prop'] * merged['total_area']
    merged['cf_area'] = merged['cf_prop'] * merged['total_area']

    obs_wide = merged.pivot_table(index=['zone', 'year'], columns='class',
                                  values='obs_area', aggfunc='first')
    cf_wide = merged.pivot_table(index=['zone', 'year'], columns='class',
                                 values='cf_area', aggfunc='first')

    obs_wide.columns = [f'obs_luc{int(c)}_area' for c in obs_wide.columns]
    cf_wide.columns = [f'cf_luc{int(c)}_area' for c in cf_wide.columns]

    df = obs_wide.join(cf_wide, how='outer').reset_index()

    # Carbon densities
    carbon_path = BASE_DIR / str(project_id) / 'carbon-density.csv'
    if not carbon_path.exists():
        log.warning("[%s] missing carbon-density.csv", project_id)
        return None
    carbon_df = pd.read_csv(carbon_path)
    if not {'luc', 'agb'}.issubset(carbon_df.columns):
        log.warning("[%s] carbon-density.csv missing columns", project_id)
        return None
    carbon_map = carbon_df.set_index('luc')['agb'].to_dict()
    for c in LUC_CLASSES:
        if c not in carbon_map:
            carbon_map[c] = 0.0
    density = np.array([carbon_map[c] for c in LUC_CLASSES])

    obs_area_cols = [f'obs_luc{c}_area' for c in LUC_CLASSES]
    cf_area_cols = [f'cf_luc{c}_area' for c in LUC_CLASSES]

    df['obs_agb'] = df[obs_area_cols].to_numpy() @ density
    df['cf_agb'] = df[cf_area_cols].to_numpy() @ density
    df['diff_agb'] = df['obs_agb'] - df['cf_agb']

    df['project_id'] = project_id
    return df[['project_id', 'zone', 'year', 'obs_agb', 'cf_agb', 'diff_agb']]


# ============================================================================
# CUMULATIVE ROLL-UP
# ============================================================================
def _zone_threshold(zone):
    """Numeric ring threshold for a buffer zone; NaN for the interior 'project'."""
    m = re.fullmatch(r'buffer_(\d+)', str(zone))
    return int(m.group(1)) if m else np.nan


def accumulate_zones(long_df):
    """Convert per-ring (annular) AGB into cumulative-buffer AGB.

    `buffer_X` becomes the total over all rings with threshold <= X, i.e. all
    eligible area within X metres of the boundary. AGB is extensive over the
    disjoint rings, so cumulative AGB is the running sum of ring AGB. obs and cf
    are accumulated independently and diff is recomputed as obs - cf. The
    interior ('project') zone is passed through untouched.
    """
    if long_df is None or long_df.empty:
        return long_df

    df = long_df.copy()
    df['_thr'] = df['zone'].map(_zone_threshold)

    interior = df[df['_thr'].isna()].drop(columns='_thr')
    rings = df[df['_thr'].notna()].copy()
    if rings.empty:
        return interior

    rings = rings.sort_values(['year', '_thr'])
    rings['obs_agb'] = rings.groupby('year')['obs_agb'].cumsum()
    rings['cf_agb'] = rings.groupby('year')['cf_agb'].cumsum()
    rings['diff_agb'] = rings['obs_agb'] - rings['cf_agb']
    rings = rings.drop(columns='_thr')

    return pd.concat([interior, rings], ignore_index=True)


def pivot_to_wide(df_long, project_id):
    if df_long is None or df_long.empty:
        return pd.DataFrame([{'project_id': project_id}])

    long = df_long.melt(
        id_vars=['zone', 'year'],
        value_vars=['obs_agb', 'cf_agb', 'diff_agb'],
        var_name='metric',
        value_name='agb'
    )
    # Map metric names to short kind
    kind_map = {'obs_agb': 'obs', 'cf_agb': 'cf', 'diff_agb': 'diff'}
    long['kind'] = long['metric'].map(kind_map)
    # Construct column name: zone_kind_agb_year (no duplicate "agb")
    long['colname'] = long['zone'] + '_' + long['kind'] + '_agb_' + long['year'].astype(str)
    long['project_id'] = project_id

    wide = long.pivot(index='project_id', columns='colname', values='agb').reset_index()
    wide.columns.name = None
    return wide


def _col_sort_key(col):
    if col == 'project_id':
        return (-1, 0, 0, 0)
    m = re.fullmatch(r'(project|buffer_(\d+))_(obs|cf|diff)_agb_(\d+)', col)
    if not m:
        return (9, 0, 0, 0)
    zone_rank, dist = (0, 0) if m.group(2) is None else (1, int(m.group(2)))
    kind_rank = {'obs': 0, 'cf': 1, 'diff': 2}[m.group(3)]
    return (zone_rank, dist, int(m.group(4)), kind_rank)


def warn_if_non_monotonic(final):
    """Sanity check: under CUMULATIVE, obs_agb must be non-decreasing with
    distance for each project-year (running sum of non-negative ring AGB). A
    violation indicates a regression in the ring or cumsum logic, not a data
    feature. Logs a warning per offending project-year."""
    pat = re.compile(r'buffer_(\d+)_obs_agb_(\d+)')
    by_year = {}
    for c in final.columns:
        m = pat.fullmatch(c)
        if m:
            by_year.setdefault(int(m.group(2)), []).append((int(m.group(1)), c))
    for items in by_year.values():
        items.sort()
    n_bad = 0
    for _, row in final.iterrows():
        pid = row.get('project_id')
        for year, items in by_year.items():
            seq = [row[c] for _, c in items if pd.notna(row[c])]
            if any(b < a - 1e-6 for a, b in zip(seq, seq[1:])):
                log.warning("[%s] non-monotonic cumulative obs_agb in %s", pid, year)
                n_bad += 1
    if n_bad == 0:
        log.info("Monotonicity check passed for all projects.")


def load_project_polygon(project_id):
    geojson_path = GEOJSON_DIR / f"{project_id}.geojson"
    if not geojson_path.exists():
        log.warning("[%s] missing GeoJSON", project_id)
        return None
    return gpd.read_file(geojson_path)


def process_project(project_id, temp_dir):
    temp_path = temp_dir / f"{project_id}.csv"
    if temp_path.exists():
        log.info("Skipping %s (already processed)", project_id)
        return None

    log.info("Processing %s", project_id)
    try:
        project_gdf = load_project_polygon(project_id)
        if project_gdf is None:
            return None

        long_df = compute_agb_for_project(project_id, project_gdf)
        if long_df is None:
            log.warning("[%s] compute_agb_for_project returned None", project_id)
            return None

        if CUMULATIVE:
            long_df = accumulate_zones(long_df)

        wide = pivot_to_wide(long_df, project_id)
        return wide
    except Exception as e:
        log.error("[%s] Failed with error: %s", project_id, str(e))
        log.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Mode-specific temp dir so switching CUMULATIVE never reuses stale files.
    TEMP_DIR = OUTPUT_DIR / f'temp_{_MODE}'
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Buffer mode: %s -> %s", _MODE.upper(), OUTPUT_CSV.name)

    projects = pd.read_csv(PROJECT_CSV)
    if PROJECT_ID_COL not in projects.columns:
        raise KeyError(f"PROJECT_CSV missing '{PROJECT_ID_COL}'")
    project_ids = projects[PROJECT_ID_COL].tolist()
    log.info("Processing %d projects", len(project_ids))

    # maxtasksperchild=1: recycle each worker after every project so the OS
    # reclaims its heap rather than letting RSS ratchet up across the run.
    with mp.Pool(processes=N_WORKERS, maxtasksperchild=1) as pool:
        process_with_temp = partial(process_project, temp_dir=TEMP_DIR)
        for wide in tqdm(pool.imap_unordered(process_with_temp, project_ids),
                         total=len(project_ids), desc="Projects"):
            if wide is not None:
                pid = wide['project_id'].iloc[0]
                temp_path = TEMP_DIR / f"{pid}.csv"
                wide.to_csv(temp_path, index=False)
                log.info("Saved temp file for %s", pid)

    # Concatenate
    temp_files = sorted(TEMP_DIR.glob("*.csv"))
    if temp_files:
        log.info("Concatenating %d temp files...", len(temp_files))
        df_list = [pd.read_csv(f) for f in tqdm(temp_files, desc="Reading")]
        final = pd.concat(df_list, ignore_index=True)
        final = final.drop_duplicates(subset=['project_id'], keep='first')
        final = final[sorted(final.columns, key=_col_sort_key)]
        final.to_csv(OUTPUT_CSV, index=False)
        log.info("Saved %s (%d projects, %d columns)", OUTPUT_CSV, final.shape[0], final.shape[1])

        # Reconcile against the requested project list so dropped projects are
        # visible rather than silently absent.
        processed = set(final['project_id'].astype(str))
        missing = [str(p) for p in project_ids if str(p) not in processed]
        if missing:
            shown = ', '.join(missing[:20]) + (' ...' if len(missing) > 20 else '')
            log.warning("%d/%d projects produced no output: %s",
                        len(missing), len(project_ids), shown)

        if CUMULATIVE:
            warn_if_non_monotonic(final)
    else:
        log.warning("No projects produced output. Check logs for errors.")
        pd.DataFrame(columns=['project_id']).to_csv(OUTPUT_CSV, index=False)