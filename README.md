# Seasonal and synoptic predictability of daily solar irradiance across Brazil

Code, data and results for:

> Caneppele FL, Tieghi CP, Almeida CFM, Nogueira CEC, Siqueira JAC, Frigo MM, Malagueta DC, Godinho EZ.
> **Maritime exposure and latitude control different components of daily solar irradiance predictability across Brazil.**
> Submitted to *Theoretical and Applied Climatology*.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fcaneppele/s3-solar-predictability-brazil/blob/main/notebooks/S3_pipeline.ipynb)
<!-- After the first release, add the Zenodo DOI badge here -->

## In one paragraph
Daily global horizontal irradiation at eleven Brazilian state capitals (NASA POWER, 2013–2023) is estimated from near-surface reanalysis variables and astronomical forcing with six regression models, evaluated by year-by-year walk-forward validation (test years 2018–2023). The explained variance is decomposed exactly as R² = f + SS (1 − f), where f is the seasonal share captured by climatology and SS the skill on the remaining day-to-day variance. The seasonal share follows maritime exposure and the phase of the cloudiness cycle; the synoptic skill follows latitude.

## Repository structure
```
data/raw/               11 CSV files, NASA POWER daily data (RE community), 2013-01-01 to 2023-12-31
notebooks/S3_pipeline.ipynb   single notebook: runs the 11 cities and regenerates every table and figure
src/s3pipe.py           pipeline functions called by the notebook
vendor/py-earth/        MARS implementation (py-earth 0.1.0, BSD-3), patched for current NumPy/scikit-learn
results/tables/         metrics, decomposition, tests, Table 2, Online Resources (CSV)
results/predictions/    daily observed and estimated values per city and design
results/figures/        Figs. 1–5 (TIFF 600 dpi, EPS, PDF, PNG)
```

## How to run
**Colab:** click the badge and run all cells (≈15–20 min).

**Locally (Python 3.10):**
```bash
pip install -r requirements.txt
cd vendor/py-earth && python setup.py build_ext --inplace --cythonize && cd ../..
export PYTHONPATH=$PWD/vendor/py-earth   # the notebook does this automatically
jupyter notebook notebooks/S3_pipeline.ipynb      # or: python src/s3pipe.py
```

## Data
Columns of `data/raw/*.csv` (daily values, POWER parameter names):

| Column | Description | Unit |
|---|---|---|
| ALLSKY_SFC_SW_DWN | All-sky surface shortwave downward irradiance (target, GHI) | kWh m⁻² day⁻¹ |
| TOA_SW_DWN | Top-of-atmosphere shortwave downward irradiance | kWh m⁻² day⁻¹ |
| CLRSKY_SFC_SW_DWN | Clear-sky surface shortwave downward irradiance | kWh m⁻² day⁻¹ |
| T2M, T2M_MAX, T2M_MIN | 2-m air temperature: mean, maximum, minimum | °C |
| RH2M | 2-m relative humidity | % |
| WS2M | 2-m wind speed | m s⁻¹ |
| WD2M | 2-m wind direction | degrees from north |
| PRECTOTCORR | Bias-corrected precipitation | mm day⁻¹ |

Source: NASA POWER (https://power.larc.nasa.gov), point API, Renewable Energy community, accessed 1 October 2026.
*These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program.*

## Licences
- Code: MIT (see `LICENSE`). Vendored py-earth: BSD-3-Clause (see `vendor/py-earth/LICENSE.txt`).
- Result tables and figures: CC BY 4.0.
- NASA POWER data: public, see the POWER acknowledgement above.

## Citation
See `CITATION.cff` (GitHub shows a "Cite this repository" button). Please cite the article and the Zenodo record.

## Funding
National Council for Scientific and Technological Development (CNPq), Brazil, Grant 401718/2023-9 (CNPq/MCTI Call No. 10/2023, Category A, Emerging Groups).
