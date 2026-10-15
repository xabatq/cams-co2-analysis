"""
================================================================
CAMS CO2 Flux Analysis — Complete Pipeline (absolute paths)
================================================================
Reproduces all figures and tables in:
  Ghamari et al. (2025), "Global surface CO2 fluxes from CAMS
  (2006-2025): trends, prior-posterior decomposition, and
  comparison with the Global Carbon Budget 2024"

Runtime: ~5 minutes
================================================================
"""
import os
import re
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from scipy.stats import t as tdist, pearsonr, linregress, kendalltau, norm

warnings.filterwarnings("ignore")


# ================================================================
# CONFIG — ABSOLUTE PATHS
# ================================================================
DATA_DIR = r"C:\Users\Think\Desktop\cams\co2"
OUT_DIR  = r"C:\Users\Think\Desktop\cams\co2\analysis_v8\github_outputs"
FIG_DIR  = os.path.join(OUT_DIR, "figures")
TAB_DIR  = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

print("=" * 72)
print("PATHS")
print("=" * 72)
print(f"  DATA_DIR = {DATA_DIR}")
print(f"  OUT_DIR  = {OUT_DIR}")
print(f"  FIG_DIR  = {FIG_DIR}")
print(f"  TAB_DIR  = {TAB_DIR}")
print("=" * 72)

MAIN_VARS  = ['flux_foss', 'flux_apos_bio', 'flux_apos_ocean']
PRIOR_VARS = ['flux_apri_bio', 'flux_apri_ocean']
ALL_VARS   = MAIN_VARS + PRIOR_VARS

LAT_MIN, LAT_MAX = -60, 80
PG_C_PER_PPM = 2.124
RIVERINE_CORRECTION = 0.65


# ================================================================
# STYLE
# ================================================================
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 10,
    'mathtext.fontset' : 'dejavusans',
    'axes.titlesize'   : 11,
    'axes.titleweight' : 'semibold',
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'savefig.dpi'      : 200,
    'savefig.bbox'     : 'tight',
    'savefig.facecolor': 'white',
})

WATER_COLOR = 'white'

CMAP_FOSS_SOFT = LinearSegmentedColormap.from_list(
    'foss_soft',
    ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84',
     '#fc8d59', '#ef6548', '#d94801'], N=256)

CMAP_MEAN_BO = LinearSegmentedColormap.from_list(
    'mean_bo',
    ['#2b8cbe', '#7bccc4', '#b2e2e2', '#e0f3f3',
     '#ffffff', '#fee6ce', '#fdc086', '#fd8d3c', '#e6550d'], N=256)

CMAP_TREND_SOFT = LinearSegmentedColormap.from_list(
    'trend_soft',
    ['#4393c3', '#92c5de', '#d1e5f0',
     '#ffffff', '#fddbc7', '#f4a582', '#d6604d'], N=256)


# ================================================================
# HELPERS
# ================================================================
def safe_savefig(fig, path, dpi=200):
    """Save figure; fall back to timestamped name if file is locked."""
    try:
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  [saved] {os.path.basename(path)}")
    except (OSError, PermissionError) as e:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(path)
        alt = f"{base}_{stamp}{ext}"
        fig.savefig(alt, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  [saved-alt] {os.path.basename(alt)}  ({e})")


def safe_to_csv(df, path, **kw):
    try:
        df.to_csv(path, **kw)
        print(f"  [saved] {os.path.basename(path)}")
    except (OSError, PermissionError) as e:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(path)
        alt = f"{base}_{stamp}{ext}"
        df.to_csv(alt, **kw)
        print(f"  [saved-alt] {os.path.basename(alt)}")


def multipletests(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, n + 1)
    p_adj_sorted = ranked * n / ranks
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]
    p_adj_sorted = np.minimum(p_adj_sorted, 1.0)
    p_adj = np.empty(n)
    p_adj[order] = p_adj_sorted
    return p_adj < alpha, p_adj


def ar1_coef(y):
    y = np.asarray(y, dtype=float)
    y = y - np.nanmean(y)
    ok = ~np.isnan(y)
    y = y[ok]
    if len(y) < 4:
        return 0.0
    num = np.sum(y[1:] * y[:-1])
    den = np.sum(y ** 2)
    return num / den if den > 0 else 0.0


def ols_ar1(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xm = x.mean()
    xc = x - xm
    sxx = (xc ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    s2 = (resid ** 2).sum() / (n - 2)
    rho = ar1_coef(resid)
    vif = (1 + rho) / (1 - rho) if abs(rho) < 0.99 else 1.0
    se_ols = np.sqrt(s2 / sxx)
    se_adj = se_ols * np.sqrt(vif)
    t_stat = slope / se_adj
    p_val = 2 * tdist.sf(abs(t_stat), n - 2)
    return slope, se_adj, p_val, rho


def theilsen_mk(x, y):
    from scipy.stats import theilslopes
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept, lo, hi = theilslopes(y, x, 0.95)
    tau, p_mk = kendalltau(x, y)
    return slope, lo, hi, tau, p_mk


def modified_mk_hamed_rao(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    tau, p_mk = kendalltau(x, y)
    sen = np.median([(y[j] - y[i]) / (x[j] - x[i])
                     for i in range(n) for j in range(i + 1, n)
                     if x[j] != x[i]])
    resid = y - sen * x
    ranks = np.argsort(np.argsort(resid)).astype(float)
    r1 = np.corrcoef(ranks[:-1], ranks[1:])[0, 1]
    if abs(r1) > 0.99:
        r1 = 0.99 * np.sign(r1)
    n_eff = n / (1 + 2 * r1 / n)
    var_corr = n / n_eff
    z_orig = norm.ppf(p_mk / 2)
    z_corr = z_orig / np.sqrt(var_corr)
    p_corr = 2 * norm.sf(abs(z_corr))
    return tau, p_mk, p_corr, r1, n_eff


def block_bootstrap_ci(x, y, block=4, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    slopes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1,
                              size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block)
                              for s in starts])[:n]
        if len(np.unique(x[idx])) > 2:
            slopes.append(np.polyfit(x[idx], y[idx], 1)[0])
    return np.percentile(slopes, [2.5, 97.5])


def trend_row(x, y, label):
    ols_slope, se_adj, p_ols, rho = ols_ar1(x, y)
    ts_slope, ts_lo, ts_hi, tau, p_mk = theilsen_mk(x, y)
    mk_tau, mk_p_orig, mk_p_corr, mk_r1, mk_neff = modified_mk_hamed_rao(x, y)
    bs_lo, bs_hi = block_bootstrap_ci(x, y)
    return {
        'series'        : label,
        'ols_slope_dec' : ols_slope * 10,
        'ols_se_dec'    : se_adj * 10,
        'ols_p'         : p_ols,
        'ar1_rho'       : rho,
        'theilsen_dec'  : ts_slope * 10,
        'theilsen_lo'   : ts_lo * 10,
        'theilsen_hi'   : ts_hi * 10,
        'mk_tau'        : tau,
        'mk_p'          : p_mk,
        'hamedrao_p'    : mk_p_corr,
        'hamedrao_rho'  : mk_r1,
        'hamedrao_neff' : mk_neff,
        'boot_lo'       : bs_lo * 10,
        'boot_hi'       : bs_hi * 10,
    }


def make_basemap(ax):
    m = Basemap(projection='cyl', resolution='c',
                llcrnrlat=LAT_MIN, urcrnrlat=LAT_MAX,
                llcrnrlon=-180, urcrnrlon=180, ax=ax)
    m.drawcoastlines(linewidth=0.3, color='#4d4d4d')
    m.drawcountries(linewidth=0.15, color='#a0a0a0')
    m.drawparallels([-50, -25, 25, 50, 75], labels=[1, 0, 0, 0],
                    fontsize=6.5, linewidth=0.25,
                    color='#999999', alpha=0.4, dashes=[3, 4])
    m.drawparallels([0], labels=[1, 0, 0, 0],
                    fontsize=7, linewidth=0.6,
                    color='#333333', alpha=0.7, dashes=[6, 3])
    return m


def style_colorbar(cbar, label):
    cbar.set_label(label, fontsize=8, fontweight='medium', labelpad=5)
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.5)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 2))
    cbar.formatter = fmt
    cbar.update_ticks()
    cbar.outline.set_linewidth(0.5)


def global_PgC(flux_map, area):
    return np.nansum(flux_map * area) * 12 / 1e12


def global_masked(flux_map, area, mask):
    return np.nansum(np.where(mask, flux_map, 0.0) * area) * 12 / 1e12


# ================================================================
# STEP 1: LOAD CAMS
# ================================================================
print("\n" + "=" * 72)
print("STEP 1: Loading CAMS data")
print("=" * 72)

pat = re.compile(r"cams73_latest_co2_flux_surface_mm_(\d{4})(\d{2})\.nc")
files_info = []
for f in glob.glob(os.path.join(DATA_DIR,
                                "cams73_latest_co2_flux_surface_mm_*.nc")):
    m = pat.search(os.path.basename(f))
    if m:
        files_info.append((int(m.group(1)), int(m.group(2)), f))
files_info.sort()

if not files_info:
    raise FileNotFoundError(f"No CAMS files found in:\n  {DATA_DIR}")

print(f"  Found {len(files_info)} CAMS files")

with Dataset(files_info[0][2], 'r') as nc:
    lats = np.array(nc.variables['latitude'][:],  dtype=float)
    lons = np.array(nc.variables['longitude'][:], dtype=float)
    lsf  = np.array(nc.variables['lsf'][:],       dtype=float)
    area = np.array(nc.variables['area'][:],      dtype=float)

nlat, nlon = len(lats), len(lons)

if lons.max() > 180:
    lons_w = np.where(lons > 180, lons - 360, lons)
    idx_lon = np.argsort(lons_w)
    lons = lons_w[idx_lon]
    lsf  = lsf[:, idx_lon]
    area = area[:, idx_lon]
else:
    idx_lon = None

years      = sorted(set(y for y, m, f in files_info))
years_arr  = np.array(years, dtype=float)
nyears     = len(years)
year_index = {y: i for i, y in enumerate(years)}

ann_sum   = {v: np.zeros((nyears, nlat, nlon)) for v in ALL_VARS}
ann_count = np.zeros(nyears, dtype=int)

for i, (y, m, f) in enumerate(files_info):
    if i % 48 == 0:
        print(f"  [{i+1:3d}/{len(files_info)}] {y}-{m:02d}")
    yi = year_index[y]
    ann_count[yi] += 1
    with Dataset(f, 'r') as nc:
        for v in ALL_VARS:
            arr = nc.variables[v][:]
            if np.ma.is_masked(arr):
                arr = arr.filled(np.nan)
            arr = np.array(arr, dtype=np.float64)
            if idx_lon is not None:
                arr = arr[:, idx_lon]
            ann_sum[v][yi] += arr

annual = {v: ann_sum[v] / ann_count[:, None, None] for v in ALL_VARS}

LAND_MASK  = (lsf >= 0.5)
OCEAN_MASK = (lsf < 0.5)
LON_GRID, LAT_GRID = np.meshgrid(lons, lats)

print(f"  Years: {years[0]}-{years[-1]}  (n = {nyears})")

# ================================================================
# STEP 2: GLOBAL TOTALS
# ================================================================
print("\n" + "=" * 72)
print("STEP 2: Global totals")
print("=" * 72)

global_ts = {v: np.array([global_PgC(annual[v][i], area)
                          for i in range(nyears)])
             for v in ALL_VARS}

uptake = -(global_ts['flux_apos_bio'] +
           global_ts['flux_apos_ocean']) / global_ts['flux_foss']

print(f"  Fossil mean:    {global_ts['flux_foss'].mean():+.4f} PgC/yr")
print(f"  Biosphere mean: {global_ts['flux_apos_bio'].mean():+.4f} PgC/yr")
print(f"  Ocean mean:     {global_ts['flux_apos_ocean'].mean():+.4f} PgC/yr")

# ================================================================
# STEP 3: GRID TRENDS + FDR
# ================================================================
print("\n" + "=" * 72)
print("STEP 3: Grid-cell trends with FDR")
print("=" * 72)


def compute_ols_grid(arr, x):
    n = arr.shape[0]
    xm = x.mean()
    xc = x - xm
    sxx = (xc ** 2).sum()
    y = arr.reshape(n, -1)
    valid = ~np.isnan(y)
    y0 = np.where(valid, y, 0.0)
    nvalid = valid.sum(axis=0).astype(float)
    ymean = y0.sum(axis=0) / np.maximum(nvalid, 1)
    sxy = (xc[:, None] * (y0 - ymean[None, :]) * valid).sum(axis=0)
    slope = sxy / sxx
    intercept = ymean - slope * xm
    resid = (y0 - (slope[None, :] * x[:, None] + intercept[None, :])) * valid
    sse = (resid ** 2).sum(axis=0)
    dof = np.maximum(nvalid - 2, 1)
    se = np.sqrt((sse / dof) / sxx)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = np.where(se > 0, slope / se, 0.0)
    pval = 2 * tdist.sf(np.abs(t_stat), dof)
    return slope.reshape(arr.shape[1:]), pval.reshape(arr.shape[1:])


def masked_annual(v):
    mask = OCEAN_MASK if v == 'flux_apos_ocean' else LAND_MASK
    return np.array([np.where(mask, annual[v][i], np.nan)
                     for i in range(nyears)])


grid_slope, grid_pval, full_mean = {}, {}, {}
for v in MAIN_VARS:
    m_arr = masked_annual(v)
    slope, pval = compute_ols_grid(m_arr, years_arr)
    valid = ~np.isnan(pval)
    rej, p_adj = multipletests(pval[valid].ravel(), alpha=0.05)
    p_adj_full = np.full(pval.shape, np.nan)
    p_adj_full[valid] = p_adj
    grid_slope[v] = slope * 10
    grid_pval[v] = p_adj_full
    full_mean[v] = np.nanmean(m_arr, axis=0)
    print(f"  {v:<18} sig cells (FDR 0.05): {int(rej.sum())}")

# ================================================================
# STEP 4: TABLE 1 — Global trends
# ================================================================
print("\n" + "=" * 72)
print("STEP 4: Table 1 — Global trends")
print("=" * 72)

table1_rows = []
for v in MAIN_VARS:
    table1_rows.append(trend_row(years_arr, global_ts[v], v))
table1_rows.append(trend_row(years_arr, uptake * 100, 'uptake_fraction'))

table1_df = pd.DataFrame(table1_rows)
safe_to_csv(table1_df,
            os.path.join(TAB_DIR, 'Table1_global_trends.csv'),
            index=False, float_format='%.6f')
print(table1_df[['series', 'ols_slope_dec', 'ols_p',
                 'theilsen_dec', 'mk_p', 'boot_lo', 'boot_hi']]
      .to_string(index=False))

# ================================================================
# STEP 5: TABLE S1 — Sensitivity
# ================================================================
print("\n" + "=" * 72)
print("STEP 5: Table S1 — Sensitivity")
print("=" * 72)

sens_rows = []
for v in MAIN_VARS + ['uptake_fraction']:
    ts = (uptake * 100 if v == 'uptake_fraction' else global_ts[v])
    for excl in [[], [2015], [2024, 2025], [2015, 2024, 2025]]:
        if excl:
            mask = ~np.isin(years_arr, excl)
            label = f"{v}_no_{'_'.join(map(str, excl))}"
        else:
            mask = np.ones(len(years_arr), dtype=bool)
            label = f"{v}_full"
        row = trend_row(years_arr[mask], ts[mask], label)
        row['excluded_years'] = ';'.join(map(str, excl)) if excl else 'none'
        sens_rows.append(row)

safe_to_csv(pd.DataFrame(sens_rows),
            os.path.join(TAB_DIR, 'TableS1_sensitivity.csv'),
            index=False, float_format='%.6f')

# ================================================================
# STEP 6: TABLE S2 — Regional decomposition
# ================================================================
print("\n" + "=" * 72)
print("STEP 6: Table S2 — Regional decomposition")
print("=" * 72)

LAND_AREA = np.where(LAND_MASK, area, 0.0)
TOTAL_LAND_AREA = LAND_AREA.sum()

lat_bands = [
    ('NH high (>50N)',      50,  90),
    ('NH mid (20-50N)',     20,  50),
    ('Tropics (20S-20N)',  -20,  20),
    ('SH mid (50-20S)',    -50, -20),
    ('SH high (>50S)',     -90, -50),
]

band_rows = []
for name, lo, hi in lat_bands:
    lat_idx = (lats >= lo) & (lats <= hi)
    weight = np.outer(lat_idx, np.ones(nlon)) * LAND_MASK * area
    w_sum = weight.sum()
    if w_sum == 0:
        continue
    ts_band = np.array([
        np.nansum(annual['flux_apos_bio'][i] * weight) * 12 / 1e12
        for i in range(nyears)
    ])
    slope = np.polyfit(years_arr, ts_band, 1)[0] * 10
    band_rows.append({
        'Band': name,
        'Mean_flux_PgC_yr': float(np.mean(ts_band)),
        'Trend_PgC_yr_per_dec': slope,
        'Area_fraction': float(w_sum / TOTAL_LAND_AREA),
    })

safe_to_csv(pd.DataFrame(band_rows),
            os.path.join(TAB_DIR, 'TableS2_regional_decomposition.csv'),
            index=False, float_format='%.6f')

# ================================================================
# STEP 7: TABLE S3 — NOAA closure stats
# ================================================================
print("\n" + "=" * 72)
print("STEP 7: Table S3 — NOAA closure statistics")
print("=" * 72)

NOAA_CSV = os.path.join(DATA_DIR, 'noaa_growth.csv')
noaa_data_available = os.path.exists(NOAA_CSV)

if noaa_data_available:
    noaa_df = pd.read_csv(NOAA_CSV)
    noaa_dict = dict(zip(noaa_df['year'].astype(int),
                         noaa_df['growth_ppm'].astype(float)))
    noaa_ppm = np.array([noaa_dict.get(int(y), np.nan) for y in years])

    net_surface = (global_ts['flux_foss'] +
                   global_ts['flux_apos_bio'] +
                   global_ts['flux_apos_ocean'])
    cam_ppm = net_surface / PG_C_PER_PPM

    valid = ~np.isnan(noaa_ppm)
    cam_v = cam_ppm[valid]
    noaa_v = noaa_ppm[valid]
    years_v = years_arr[valid]

    r_full, p_full = pearsonr(cam_v, noaa_v)
    pc = np.polyfit(years_v, cam_v, 1)
    pn = np.polyfit(years_v, noaa_v, 1)
    res_c = cam_v - np.polyval(pc, years_v)
    res_n = noaa_v - np.polyval(pn, years_v)
    r_det, p_det = pearsonr(res_c, res_n)
    lr = linregress(cam_v, noaa_v)
    rmse_full = float(np.sqrt(np.mean((noaa_v - cam_v) ** 2)))
    rmse_det = float(np.sqrt(np.mean((res_n - res_c) ** 2)))
    bias = float(np.mean(noaa_v - cam_v))

    rng = np.random.default_rng(42)
    r_boot = []
    for _ in range(5000):
        idx = rng.choice(len(cam_v), size=len(cam_v), replace=True)
        try:
            r_boot.append(pearsonr(res_c[idx], res_n[idx])[0])
        except Exception:
            pass
    ci_det = np.percentile(r_boot, [2.5, 97.5])

    table_s3 = pd.DataFrame([{
        'n_years':        int(valid.sum()),
        'r_full':         r_full,  'p_full':      p_full,
        'r_detrended':    r_det,   'p_detrended': p_det,
        'r_det_ci_low':   ci_det[0],
        'r_det_ci_high':  ci_det[1],
        'rmse_full':      rmse_full,
        'rmse_detrended': rmse_det,
        'bias_ppm':       bias,
        'ols_slope':      lr.slope,
        'ols_intercept':  lr.intercept,
    }])
    safe_to_csv(table_s3,
                os.path.join(TAB_DIR, 'TableS3_noaa_stats.csv'),
                index=False, float_format='%.6f')
else:
    print(f"  [warning] NOAA file not found: {NOAA_CSV}")

# ================================================================
# STEP 8: TABLE 2 — GCB comparison
# ================================================================
print("\n" + "=" * 72)
print("STEP 8: Table 2 — GCB comparison")
print("=" * 72)

GCB_CSV = os.path.join(DATA_DIR, 'Global_Carbon_Budget_2024_v1.0.csv')
gcb_data_available = os.path.exists(GCB_CSV)

if gcb_data_available:
    gcb = pd.read_csv(GCB_CSV, sep=None, engine='python',
                      encoding='utf-8-sig', skipinitialspace=True)
    gcb.columns = [c.replace('\ufeff', '').strip() for c in gcb.columns]
    gcb['Year'] = pd.to_numeric(gcb['Year'], errors='coerce')
    for c in ['fossil emissions excluding carbonation',
              'land-use change emissions', 'ocean sink', 'land sink']:
        gcb[c] = pd.to_numeric(gcb[c], errors='coerce')
    gcb = gcb[(gcb['Year'] >= 2006) & (gcb['Year'] <= 2023)].dropna()

    mask_cmp = (years_arr >= 2006) & (years_arr <= 2023)
    years_cmp = years_arr[mask_cmp]
    idx_cmp = [year_index[int(y)] for y in years_cmp]

    cams_foss_cmp  = global_ts['flux_foss'][idx_cmp]
    cams_ocean_raw = -global_ts['flux_apos_ocean'][idx_cmp]
    cams_ocean_cor = cams_ocean_raw + RIVERINE_CORRECTION
    cams_land_app  = -global_ts['flux_apos_bio'][idx_cmp]

    gcb_efos  = gcb['fossil emissions excluding carbonation'].values
    gcb_eluc  = gcb['land-use change emissions'].values
    gcb_ocean = gcb['ocean sink'].values
    gcb_land  = gcb['land sink'].values

    cams_land_nat = cams_land_app + gcb_eluc

    table2_df = pd.DataFrame([
        {'Component':        'Fossil emissions',
         'CAMS_Scenario_A':  cams_foss_cmp.mean(),
         'CAMS_Scenario_B':  cams_foss_cmp.mean(),
         'GCB':              gcb_efos.mean(),
         'Delta_A_minus_GCB': cams_foss_cmp.mean() - gcb_efos.mean()},
        {'Component':        'Ocean sink',
         'CAMS_Scenario_A':  cams_ocean_raw.mean(),
         'CAMS_Scenario_B':  cams_ocean_cor.mean(),
         'GCB':              gcb_ocean.mean(),
         'Delta_A_minus_GCB': cams_ocean_raw.mean() - gcb_ocean.mean()},
        {'Component':        'Land sink (LUC-corrected)',
         'CAMS_Scenario_A':  cams_land_nat.mean(),
         'CAMS_Scenario_B':  cams_land_nat.mean(),
         'GCB':              gcb_land.mean(),
         'Delta_A_minus_GCB': cams_land_nat.mean() - gcb_land.mean()},
        {'Component':        'Total natural sink',
         'CAMS_Scenario_A':  cams_land_nat.mean() + cams_ocean_raw.mean(),
         'CAMS_Scenario_B':  cams_land_nat.mean() + cams_ocean_cor.mean(),
         'GCB':              gcb_land.mean() + gcb_ocean.mean(),
         'Delta_A_minus_GCB': (cams_land_nat.mean() + cams_ocean_raw.mean())
                              - (gcb_land.mean() + gcb_ocean.mean())},
    ])
    safe_to_csv(table2_df,
                os.path.join(TAB_DIR, 'Table2_GCB_comparison.csv'),
                index=False, float_format='%.4f')
    print(table2_df.to_string(index=False))
else:
    print(f"  [warning] GCB file not found: {GCB_CSV}")

# ================================================================
# STEP 9: FIGURE 1 — Global budget time series
# ================================================================
print("\n" + "=" * 72)
print("STEP 9: Figure 1 — Global budget time series")
print("=" * 72)

fig, axes = plt.subplots(4, 1, figsize=(9.5, 12.5), sharex=True)
fig.subplots_adjust(hspace=0.18)

panels = [
    ('flux_foss',       '(a)', 'Fossil fuel CO$_2$ emissions', '#c0392b'),
    ('flux_apos_bio',   '(b)', 'Posterior biosphere CO$_2$ flux', '#1c5aa6'),
    ('flux_apos_ocean', '(c)', 'Posterior ocean CO$_2$ flux', '#1f77b4'),
]
for i, (v, letter, title, color) in enumerate(panels):
    ax = axes[i]
    ts = global_ts[v]
    ax.plot(years, ts, 'o-', color=color, markersize=4.5, linewidth=1.5,
            markeredgewidth=0.6, markeredgecolor='white',
            label='Annual', zorder=4)
    slope = np.polyfit(years_arr, ts, 1)[0] * 10
    ax.plot(years, np.polyval(np.polyfit(years_arr, ts, 1), years_arr),
            '--', color='#333333', linewidth=1.0, alpha=0.7,
            label=f'OLS: {slope:+.3f} PgC/yr/dec', zorder=3)
    ax.set_ylabel('Flux (PgC yr$^{-1}$)', fontsize=9)
    ax.set_title(f'{letter}  {title}', loc='left',
                 fontsize=10.5, fontweight='semibold', pad=6)
    ax.grid(alpha=0.5, linewidth=0.4)
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax = axes[3]
ts = uptake * 100
ax.plot(years, ts, 'o-', color='#2ca02c', markersize=4.5, linewidth=1.5,
        markeredgewidth=0.6, markeredgecolor='white',
        label='Annual', zorder=4)
slope = np.polyfit(years_arr, ts, 1)[0] * 10
ax.plot(years, np.polyval(np.polyfit(years_arr, ts, 1), years_arr),
        '--', color='#333333', linewidth=1.0, alpha=0.7,
        label=f'OLS: {slope:+.2f} %/dec', zorder=3)
ax.set_ylabel('Net-surface uptake fraction (%)', fontsize=9)
ax.set_xlabel('Year', fontsize=9.5)
ax.set_title('(d)  Net-surface uptake fraction = '
             r'$-(F_{bio}+F_{ocean})/F_{foss}$', loc='left',
             fontsize=10.5, fontweight='semibold', pad=6)
ax.grid(alpha=0.5, linewidth=0.4)
ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

fig.suptitle('Global CO$_2$ budget from CAMS (2006-2025)',
             fontsize=12.5, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
safe_savefig(fig, os.path.join(FIG_DIR, 'Figure1_global_budget.png'))
plt.close(fig)

# ================================================================
# STEP 10: Figure prior vs posterior
# ================================================================
print("\n" + "=" * 72)
print("STEP 10: Figure prior vs posterior")
print("=" * 72)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

pairs = [
    ('flux_apri_bio',   'flux_apos_bio',   'Biosphere', '#1c5aa6'),
    ('flux_apri_ocean', 'flux_apos_ocean', 'Ocean',     '#1f77b4'),
]
for ax, (v_pri, v_pos, name, color) in zip(axes, pairs):
    ax.plot(years, global_ts[v_pri], 'o--', color='#888888',
            lw=1.6, markersize=4.5,
            markeredgecolor='white', markeredgewidth=0.6,
            label=f'Prior {name.lower()}', zorder=3)
    ax.plot(years, global_ts[v_pos], 'o-', color=color,
            lw=1.8, markersize=5.5,
            markeredgecolor='white', markeredgewidth=0.8,
            label=f'Posterior {name.lower()}', zorder=4)
    ax.set_ylabel('Flux (PgC yr$^{-1}$)')
    ax.set_title(f'{name}: prior vs posterior', loc='left', pad=8)
    ax.grid(alpha=0.35, lw=0.6)
    ax.legend(loc='best', framealpha=0.95)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

axes[1].set_xlabel('Year')
plt.tight_layout()
safe_savefig(fig, os.path.join(FIG_DIR, 'Figure_prior_vs_posterior.png'))
plt.close(fig)

# ================================================================
# STEP 11: Figure GCB comparison (3 panels)
# ================================================================
if gcb_data_available:
    print("\n" + "=" * 72)
    print("STEP 11: Figure GCB comparison")
    print("=" * 72)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    fig.subplots_adjust(wspace=0.30)

    panels = [
        (axes[0], cams_foss_cmp, gcb_efos,
         '(a) Fossil emissions', 'E$_{FOS}$ (PgC yr$^{-1}$)',
         '#c0392b'),
        (axes[1], cams_ocean_raw, gcb_ocean,
         '(b) Ocean sink (Scenario A)', 'S$_{OCEAN}$ (PgC yr$^{-1}$)',
         '#1f77b4'),
        (axes[2], cams_land_nat, gcb_land,
         '(c) Land sink (LUC-corrected)', 'S$_{LAND}$ (PgC yr$^{-1}$)',
         '#2ca02c'),
    ]
    for ax, y_c, y_g, title, ylab, color in panels:
        ax.plot(years_cmp, y_c, 'o-', color=color, lw=1.9, markersize=5.5,
                markeredgecolor='white', label='CAMS (this study)')
        ax.plot(years_cmp, y_g, 's--', color='#555555',
                lw=1.6, markersize=4.5,
                markeredgecolor='white', label='GCB 2024')
        ax.set_xlabel('Year')
        ax.set_ylabel(ylab)
        ax.set_title(title, loc='left', pad=8)
        ax.set_xlim(2005.5, 2023.5)
        ax.grid(alpha=0.35)
        ax.legend(loc='best', framealpha=0.95)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, 'Figure_GCB_comparison.png'))
    plt.close(fig)

# ================================================================
# STEP 12: Figure NOAA closure
# ================================================================
if noaa_data_available:
    print("\n" + "=" * 72)
    print("STEP 12: Figure NOAA closure")
    print("=" * 72)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.subplots_adjust(wspace=0.28)

    ax = axes[0]
    ax.scatter(cam_v, noaa_v, s=75, c='#1f77b4',
               edgecolor='white', linewidth=1.0, zorder=3,
               label='Annual values')
    lo = min(cam_v.min(), noaa_v.min()) - 0.5
    hi = max(cam_v.max(), noaa_v.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.55, label='1:1 line')
    xf = np.linspace(lo, hi, 100)
    ax.plot(xf, lr.slope * xf + lr.intercept, '-', color='#d62728',
            lw=1.8, label=f'OLS fit (R² = {r_full**2:.2f})')
    for yi, yv in zip(years, range(len(years))):
        if valid[yv]:
            ax.annotate(str(int(yi)),
                        (cam_ppm[yv], noaa_ppm[yv]),
                        fontsize=6.5, color='#555555',
                        xytext=(4, 3), textcoords='offset points')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('CAMS net surface flux (ppm yr$^{-1}$)')
    ax.set_ylabel('NOAA observed growth (ppm yr$^{-1}$)')
    ax.set_title('(a) Consistency scatter', loc='left', pad=6)
    ax.grid(alpha=0.35)
    ax.legend(loc='upper left', fontsize=8)
    txt = (f"n      = {int(valid.sum())}\n"
           f"r      = {r_full:+.3f}\n"
           f"r_det  = {r_det:+.3f}\n"
           f"RMSE   = {rmse_full:.3f} ppm")
    ax.text(0.98, 0.04, txt, transform=ax.transAxes,
            fontsize=8.5, ha='right', va='bottom', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='#bbbbbb', alpha=0.95))

    ax = axes[1]
    ax.fill_between(years, np.array(noaa_ppm) - rmse_full,
                    np.array(noaa_ppm) + rmse_full,
                    color='#d62728', alpha=0.10, label='NOAA ± RMSE')
    ax.plot(years, noaa_ppm, 'o-', color='#d62728',
            lw=1.8, markersize=6, markeredgecolor='white',
            label='NOAA observed', zorder=3)
    ax.plot(years, cam_ppm, 's--', color='#1f77b4',
            lw=1.7, markersize=5.5, markeredgecolor='white',
            label='CAMS net surface', zorder=2)
    ax.axhline(0, color='#888888', lw=0.6, ls=':')
    ax.set_xlabel('Year')
    ax.set_ylabel('Atmospheric CO$_2$ growth (ppm yr$^{-1}$)')
    ax.set_title('(b) Interannual comparison', loc='left', pad=6)
    ax.grid(alpha=0.35)
    ax.legend(loc='upper right', fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, 'Figure_NOAA_closure.png'))
    plt.close(fig)

# ================================================================
# STEP 13: Three map figures
# ================================================================
print("\n" + "=" * 72)
print("STEP 13: Map figures (fossil, biosphere, ocean)")
print("=" * 72)

FILE_TAG = {'flux_foss': 'fossil',
            'flux_apos_bio': 'biosphere',
            'flux_apos_ocean': 'ocean'}
TITLES = {'flux_foss': 'Fossil fuel CO$_2$ emissions',
          'flux_apos_bio': 'Posterior biosphere CO$_2$ flux',
          'flux_apos_ocean': 'Posterior ocean CO$_2$ flux'}

for v in MAIN_VARS:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 9.5))
    fig.subplots_adjust(hspace=0.28)

    ax = axes[0]
    arr = full_mean[v]
    valid = arr[~np.isnan(arr)]
    if v == 'flux_foss':
        vmax = np.nanpercentile(valid, 98)
        levels = np.linspace(0, vmax, 30)
        cmap = CMAP_FOSS_SOFT.copy()
        extend = 'max'
    else:
        vmin, vmax = np.nanpercentile(valid, [2, 98])
        vabs = max(abs(vmin), abs(vmax)) or 1e-12
        levels = np.linspace(-vabs, vabs, 30)
        cmap = CMAP_MEAN_BO.copy()
        extend = 'both'
    cmap.set_bad(WATER_COLOR)

    m = make_basemap(ax)
    x, y = m(LON_GRID, LAT_GRID)
    im = m.contourf(x, y, arr, levels=levels, cmap=cmap, extend=extend)
    ax.set_title(f'(a) {TITLES[v]} - mean 2006-2025',
                 loc='left', fontsize=10.5, fontweight='semibold', pad=6)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                        fraction=0.045, pad=0.08, shrink=0.72, aspect=38)
    style_colorbar(cbar, '(kgC m$^{-2}$ month$^{-1}$)')

    ax = axes[1]
    slope_dec = grid_slope[v]
    pval_fdr = grid_pval[v]
    valid = slope_dec[~np.isnan(slope_dec)]
    vmin, vmax = np.nanpercentile(valid, [2, 98])
    vabs = max(abs(vmin), abs(vmax)) or 1e-12
    levels = np.linspace(-vabs, vabs, 30)
    cmap = CMAP_TREND_SOFT.copy()
    cmap.set_bad(WATER_COLOR)
    extend = 'both'

    m = make_basemap(ax)
    x, y = m(LON_GRID, LAT_GRID)
    im = m.contourf(x, y, slope_dec, levels=levels, cmap=cmap, extend=extend)

    if v != 'flux_foss':
        sig = (pval_fdr < 0.05) & ~np.isnan(slope_dec)
        step = max(1, nlat // 45)
        lat_ss, lon_ss = lats[::step], lons[::step]
        LON_SS, LAT_SS = np.meshgrid(lon_ss, lat_ss)
        Xs, Ys = m(LON_SS, LAT_SS)
        m.scatter(Xs[sig[::step, ::step]], Ys[sig[::step, ::step]],
                  marker='*', s=3.5, c='#222222', alpha=0.55,
                  edgecolors='none', zorder=5)

    ax.set_title(f'(b) {TITLES[v]} - trend per decade',
                 loc='left', fontsize=10.5, fontweight='semibold', pad=6)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                        fraction=0.045, pad=0.08, shrink=0.72, aspect=38)
    style_colorbar(cbar, 'Trend (kgC m$^{-2}$ month$^{-1}$ / decade)')

    if v != 'flux_foss':
        ax.text(0.985, 0.04, 'FDR-corrected p < 0.05',
                transform=ax.transAxes, fontsize=7.5,
                ha='right', va='bottom', color='#222222',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', edgecolor='#bbbbbb', alpha=0.9))

    plt.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR,
                f'{FILE_TAG[v]}_mean_and_trend.png'))
    plt.close(fig)

# ================================================================
# DONE
# ================================================================
print("\n" + "=" * 72)
print("DONE")
print(f"  Figures: {FIG_DIR}")
print(f"  Tables:  {TAB_DIR}")
print("=" * 72)

# List generated files
print("\nGenerated figures:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
print("\nGenerated tables:")
for f in sorted(os.listdir(TAB_DIR)):
    print(f"  {f}")