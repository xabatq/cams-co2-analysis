# cams-co2-analysis

Reproducibility code for **Ghamari et al. (2025), ACP**.

> **Ghamari, K., Omidirad, R., Seyedzahedi, A., and Darand, M.** (2025).
> *Global surface CO₂ fluxes from CAMS (2006–2025): trends, prior–posterior
> decomposition, and comparison with the Global Carbon Budget 2024.*
> Atmospheric Chemistry and Physics. DOI: [to be assigned]

---

## Overview

This repository contains the complete Python pipeline that reproduces
all figures and tables in the manuscript. The analysis combines:

- **CAMS73** monthly surface CO₂ fluxes (product identifier v25r1)
- **Global Carbon Budget 2024** annual data
- **NOAA/GML** atmospheric CO₂ annual growth

---

## Key results

- Global fossil emission trend: **+0.98 PgC yr⁻¹ decade⁻¹** (p < 10⁻⁶)
- Posterior ocean sink strengthening: **−0.49 PgC yr⁻¹ decade⁻¹**
- Prior-inherited fraction of ocean trend: **≈ 84 %**
- Biosphere posterior trend not statistically significant (p = 0.29)
- CAMS–GCB total natural sink discrepancy: **−0.19 PgC yr⁻¹** (budget-consistent scenario)

---

## Requirements

- Python ≥ 3.8

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### Note on Basemap

`basemap` may require system-level dependencies:

| OS | Command |
|----|---------|
| Windows | `pip install basemap` |
| Linux | `sudo apt install libgeos-dev && pip install basemap` |
| macOS | `brew install geos && pip install basemap` |

---

## Data

Place the following files in `./data/`:

| File | Source |
|------|--------|
| `cams73_latest_co2_flux_surface_mm_YYYYMM.nc` (240 files) | [CAMS ADS portal](https://ads.atmosphere.copernicus.eu) |
| `Global_Carbon_Budget_2024_v1.0.csv` | [GCB 2024](https://doi.org/10.18160/GCP-2024) |
| `noaa_growth.csv` (columns: `year`, `growth_ppm`) | [NOAA GML](https://gml.noaa.gov/ccgg/trends/gr.html) |

The CAMS files and GCB CSV are publicly available. NOAA growth can be
downloaded from the NOAA GML website or reconstructed from the annual
mean CO₂ growth record.

---

## Usage

Edit the paths at the top of `cams_analysis.py`:

```python
DATA_DIR = r"C:\path\to\your\data"
OUT_DIR  = r"C:\path\to\your\outputs"
```

Then run:

```bash
python cams_analysis.py
```

Runtime: ~5 minutes on a standard laptop.

---

## Outputs

### Figures (7 PNG files)

| File | Description |
|------|-------------|
| `Figure1_global_budget.png` | 4-panel global budget time series |
| `Figure_prior_vs_posterior.png` | Prior vs posterior comparison |
| `Figure_GCB_comparison.png` | 3-panel comparison with GCB 2024 |
| `Figure_NOAA_closure.png` | Consistency check vs NOAA |
| `fossil_mean_and_trend.png` | Fossil mean + trend maps |
| `biosphere_mean_and_trend.png` | Biosphere mean + trend maps |
| `ocean_mean_and_trend.png` | Ocean mean + trend maps |

### Tables (5 CSV files)

| File | Description |
|------|-------------|
| `Table1_global_trends.csv` | Global trend statistics |
| `Table2_GCB_comparison.csv` | GCB comparison (Scenario A/B) |
| `TableS1_sensitivity.csv` | Sensitivity to excluded years |
| `TableS2_regional_decomposition.csv` | Latitude-band decomposition |
| `TableS3_noaa_stats.csv` | NOAA closure statistics |

---

## Statistical methods

- **OLS** with AR(1)-adjusted standard errors
- **Theil–Sen** median slope with 95 % confidence intervals
- **Mann–Kendall** and **Hamed–Rao** modified MK tests
- **Moving-block bootstrap** confidence intervals (block = 4 years)
- **Benjamini–Hochberg FDR** for grid-cell trend maps

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Citation

If you use this code, please cite the manuscript and this repository.

```bibtex
@article{Ghamari2025CAMS,
  title   = {Global surface CO₂ fluxes from CAMS (2006–2025): trends,
             prior–posterior decomposition, and comparison with the
             Global Carbon Budget 2024},
  author  = {Ghamari, Khabat and Omidirad, Reyhan and
             Seyedzahedi, Asrin and Darand, Mohammad},
  journal = {Atmospheric Chemistry and Physics},
  year    = {2025},
  doi     = {to be assigned}
}
```
