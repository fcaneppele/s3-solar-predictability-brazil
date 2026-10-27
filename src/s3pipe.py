"""
s3pipe — pipeline for:
"Maritime exposure and latitude control different components of daily solar
irradiance predictability across Brazil" (Theoretical and Applied Climatology, submitted).

Steps (all paths relative to the repository root):
  run_models()      six regressors + references, random 80/20 (sets A, B) and walk-forward 2018-2023 (set B)
  coast_distance()  great-circle distance to the nearest ocean pixel (GLOBE land-sea mask)
  descriptors()     climatological descriptors of each site (2013-2023)
  decomposition()   seasonal share f, synoptic skill SS, identity R2 = f + SS(1-f), tests, bootstrap
  tables()          Table 2 and Online Resources 1-2
  figures()         Figs. 1-5 (TIFF 600 dpi, EPS, PDF, PNG)
"""
from pathlib import Path
import os, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("S3_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data" / "raw"
TAB = ROOT / "results" / "tables"
PRED = ROOT / "results" / "predictions"
FIG = ROOT / "results" / "figures"
for p in (TAB, PRED, FIG):
    p.mkdir(parents=True, exist_ok=True)
import sys as _sys
_VENDOR = str(ROOT / "vendor" / "py-earth")
if _VENDOR not in _sys.path:
    _sys.path.insert(0, _VENDOR)
os.environ["PYTHONPATH"] = os.pathsep.join([_VENDOR, str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])

CITIES = ["Manaus", "Belem", "Cuiaba", "Brasilia", "Belo_Horizonte", "Fortaleza", "Recife",
          "Rio_de_Janeiro", "Sao_Paulo", "Curitiba", "Porto_Alegre"]
NAME = {"Manaus": "Manaus", "Belem": "Belém", "Cuiaba": "Cuiabá", "Brasilia": "Brasília",
        "Belo_Horizonte": "Belo Horizonte", "Fortaleza": "Fortaleza", "Recife": "Recife",
        "Rio_de_Janeiro": "Rio de Janeiro", "Sao_Paulo": "São Paulo", "Curitiba": "Curitiba",
        "Porto_Alegre": "Porto Alegre"}
CODE = {"Manaus": "MAO", "Belem": "BEL", "Cuiaba": "CGB", "Brasilia": "BSB", "Belo_Horizonte": "BHZ",
        "Fortaleza": "FOR", "Recife": "REC", "Rio_de_Janeiro": "RIO", "Sao_Paulo": "SAO",
        "Curitiba": "CWB", "Porto_Alegre": "POA"}
TARGET = "ALLSKY_SFC_SW_DWN"
FEAT_A = ["T2M", "RH2M", "WD2M", "WS2M"]                       # preliminary-study predictors
FEAT_B = FEAT_A + ["TOA_SW_DWN", "doy_sin", "doy_cos"]         # primary predictor vector
TEST_YEARS = list(range(2018, 2024))
ENSEMBLE = ["RF", "GBT", "SVM", "ANN", "MARS"]
MODELS = ENSEMBLE + ["CART"]
COASTAL_FAO = {"Belem", "Fortaleza", "Recife", "Rio_de_Janeiro", "Porto_Alegre"}  # FAO-56 kRs = 0.19
COAST_KM = 60.0


# ----------------------------------------------------------------- data & models
def load(city):
    d = pd.read_csv(DATA / f"{city}.csv", parse_dates=["date"])
    d["doy"] = d["date"].dt.dayofyear
    d["year"] = d["date"].dt.year
    d["doy_sin"] = np.sin(2 * np.pi * d["doy"] / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d["doy"] / 365.25)
    return d


def make_models():
    if _VENDOR not in _sys.path:
        _sys.path.insert(0, _VENDOR)
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from pyearth import Earth
    return {
        "RF": RandomForestRegressor(random_state=42),
        "GBT": GradientBoostingRegressor(random_state=42),
        "SVM": Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
        "ANN": Pipeline([("scaler", StandardScaler()),
                         ("model", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                                solver="adam", max_iter=800, random_state=42))]),
        "MARS": Earth(),
        "CART": DecisionTreeRegressor(random_state=42),
    }


def _arr(x):
    """Writable float64 copy (pandas >= 3 returns read-only views; py-earth needs writable buffers)."""
    return np.array(x, dtype=float, copy=True)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    e = p - y
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))), MBE=float(np.mean(e)),
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), n=int(len(y)))


def climatology(train):
    """Training-period day-of-year mean, smoothed with a centred 31-day circular moving average."""
    g = train.groupby("doy")[TARGET].mean().reindex(range(1, 367)).interpolate(limit_direction="both").values
    ext = np.concatenate([g[-15:], g, g[:15]])
    return pd.Series(np.convolve(ext, np.ones(31) / 31, "valid"), index=range(1, 367))


def references(city, train, test):
    clim = climatology(train)
    x_tr = np.sqrt(np.clip(train["T2M_MAX"] - train["T2M_MIN"], 0, None)) * train["TOA_SW_DWN"]
    x_te = np.sqrt(np.clip(test["T2M_MAX"] - test["T2M_MIN"], 0, None)) * test["TOA_SW_DWN"]
    k_fao = 0.19 if city in COASTAL_FAO else 0.16
    k_cal = float((x_tr * train[TARGET]).sum() / (x_tr ** 2).sum())
    return {"CLIM": clim.loc[test["doy"]].values, "HS_FAO": (k_fao * x_te).values,
            "HS_CAL": (k_cal * x_te).values}, k_cal


def run_city(city):
    from sklearn.model_selection import train_test_split
    d = load(city)
    rows, krow = [], []
    # random 80/20
    i_tr, i_te = train_test_split(d.index, test_size=0.2, random_state=42)
    tr, te = d.loc[i_tr], d.loc[i_te]
    pr = pd.DataFrame({"date": te["date"].values, "obs": te[TARGET].values})
    ref, k = references(city, tr, te)
    for name, p in ref.items():
        pr[name] = p
        rows.append(dict(city=city, design="random", year="all", features="-", model=name, **metrics(te[TARGET], p)))
    for fs_name, fs in (("A", FEAT_A), ("B", FEAT_B)):
        for name, m in make_models().items():
            m.fit(_arr(tr[fs]), _arr(tr[TARGET]))
            p = m.predict(_arr(te[fs])); pr[f"{name}_{fs_name}"] = p
            rows.append(dict(city=city, design="random", year="all", features=fs_name, model=name, **metrics(te[TARGET], p)))
    pr.to_csv(PRED / f"{city}_random.csv", index=False)
    # walk-forward, expanding window, set B
    allp = []
    for Y in TEST_YEARS:
        tr, te = d[d.year < Y], d[d.year == Y]
        pr = pd.DataFrame({"date": te["date"].values, "year": Y, "obs": te[TARGET].values})
        ref, k = references(city, tr, te); krow.append(dict(city=city, test_year=Y, kRs_cal=k))
        for name, p in ref.items():
            pr[name] = p
        for name, m in make_models().items():
            m.fit(_arr(tr[FEAT_B]), _arr(tr[TARGET]))
            pr[f"{name}_B"] = m.predict(_arr(te[FEAT_B]))
        allp.append(pr)
    allp = pd.concat(allp, ignore_index=True)
    allp.to_csv(PRED / f"{city}_walkforward.csv", index=False)
    for c in [c for c in allp.columns if c not in ("date", "year", "obs")]:
        model, fs = (c, "-") if c in ("CLIM", "HS_FAO", "HS_CAL") else c.rsplit("_", 1)
        rows.append(dict(city=city, design="walkforward", year="pooled", features=fs, model=model,
                         **metrics(allp["obs"], allp[c])))
        for Y, g in allp.groupby("year"):
            rows.append(dict(city=city, design="walkforward", year=str(Y), features=fs, model=model,
                             **metrics(g["obs"], g[c])))
    return pd.DataFrame(rows), pd.DataFrame(krow)


def run_models(n_jobs=2, cities=None):
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_city)(c) for c in (cities or CITIES))
    m = pd.concat([o[0] for o in out], ignore_index=True); k = pd.concat([o[1] for o in out], ignore_index=True)
    m.to_csv(TAB / "metrics_all.csv", index=False); k.to_csv(TAB / "kRs_calibrated_walkforward.csv", index=False)
    return m, k


# ----------------------------------------------------------------- climatology
def coast_distance():
    from global_land_mask import globe
    rows = []
    for c in CITIES:
        d = load(c); la, lo = float(d.lat[0]), float(d.lon[0])
        for span, step in [(3, 0.01), (8, 0.02), (25, 0.05)]:
            LA, LO = np.meshgrid(np.arange(la - span, la + span, step), np.arange(lo - span, lo + span, step), indexing="ij")
            oc = globe.is_ocean(LA, LO)
            if oc.any():
                p1, p2 = np.radians(la), np.radians(LA[oc]); dl = np.radians(LO[oc] - lo)
                dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))
                rows.append(dict(city=c, lat=la, lon=lo, dist_coast_km=round(float(dist.min()), 1))); break
    df = pd.DataFrame(rows); df.to_csv(TAB / "distance_to_coast.csv", index=False); return df


def descriptors():
    dist = pd.read_csv(TAB / "distance_to_coast.csv").set_index("city")
    rows = []
    for c in CITIES:
        d = load(c); d["m"] = d.date.dt.month
        g = d[TARGET]; kt = g / d.TOA_SW_DWN
        clim = climatology(d).values; anom = pd.Series(g.values - clim[d.doy.values - 1])
        mm = d.groupby("m")[[TARGET, "TOA_SW_DWN"]].mean(); ktm = kt.groupby(d.m).mean()
        rows.append(dict(city=c, lat=d.lat[0], lon=d.lon[0], abs_lat=abs(d.lat[0]), dist_coast_km=dist.loc[c, "dist_coast_km"],
                         maritime=bool(dist.loc[c, "dist_coast_km"] <= COAST_KM),
                         GHI_mean=g.mean(), Kt_mean=kt.mean(), Kt_sd=kt.std(), anom_sd=anom.std(),
                         anom_lag1_acf=anom.autocorr(1), rain_day_freq=(d.PRECTOTCORR > 1).mean(),
                         RH_mean=d.RH2M.mean(), DTR_mean=(d.T2M_MAX - d.T2M_MIN).mean(),
                         r_phase_Kt_TOA=np.corrcoef(ktm, mm.TOA_SW_DWN)[0, 1],
                         TOA_monthly_range=np.ptp(mm.TOA_SW_DWN), Kt_monthly_range=np.ptp(ktm),
                         GHI_monthly_range=np.ptp(mm[TARGET])))
    df = pd.DataFrame(rows); df.to_csv(TAB / "climate_descriptors.csv", index=False); return df


def _components(g, col):
    var = ((g.obs - g.obs.mean()) ** 2).mean(); mc = ((g.CLIM - g.obs) ** 2).mean(); mm = ((g[col] - g.obs) ** 2).mean()
    return 1 - mc / var, 1 - mm / mc, 1 - mm / var


def decomposition(n_boot=2000):
    from scipy.stats import spearmanr, mannwhitneyu
    desc = pd.read_csv(TAB / "climate_descriptors.csv").set_index("city").loc[CITIES]
    rows = []
    for c in CITIES:
        for design in ("walkforward", "random"):
            p = pd.read_csv(PRED / f"{c}_{design}.csv")
            groups = list(p.groupby("year")) + [("pooled", p)] if design == "walkforward" else [("all", p)]
            for Y, g in groups:
                for m in MODELS:
                    f, ss, r2 = _components(g, f"{m}_B")
                    rows.append(dict(city=c, design=design, year=str(Y), model=m, f=f, SS=ss, R2=r2,
                                     identity_err=r2 - (f + ss * (1 - f))))
    d = pd.DataFrame(rows); d.to_csv(TAB / "decomposition_all.csv", index=False)
    ens = d[d.model.isin(ENSEMBLE)]
    med = ens.groupby(["city", "design", "year"])[["f", "SS", "R2"]].median().reset_index()
    med.to_csv(TAB / "decomposition_ensemble_median.csv", index=False)
    pool = med[(med.design == "walkforward") & (med.year == "pooled")].set_index("city").loc[CITIES]
    tests = []
    for comp in ("f", "SS", "R2"):
        for x in ("abs_lat", "dist_coast_km", "r_phase_Kt_TOA", "anom_lag1_acf"):
            r, pv = spearmanr(pool[comp], desc[x]); tests.append(dict(design="walkforward", component=comp, descriptor=x, rho=r, p=pv))
        u = mannwhitneyu(pool.loc[desc.maritime, comp], pool.loc[~desc.maritime, comp], alternative="two-sided")
        tests.append(dict(design="walkforward", component=comp, descriptor="maritime_vs_continental_MWU", rho=np.nan, p=u.pvalue))
    r, pv = spearmanr(pool.f, pool.SS); tests.append(dict(design="walkforward", component="f", descriptor="SS", rho=r, p=pv))
    r, pv = spearmanr(desc.dist_coast_km, desc.abs_lat); tests.append(dict(design="-", component="dist_coast_km", descriptor="abs_lat", rho=r, p=pv))
    rnd = med[med.design == "random"].set_index("city").loc[CITIES]
    for comp, x in (("f", "dist_coast_km"), ("SS", "abs_lat"), ("f", "abs_lat"), ("SS", "dist_coast_km")):
        r, pv = spearmanr(rnd[comp], desc[x]); tests.append(dict(design="random", component=comp, descriptor=x, rho=r, p=pv))
    yr = med[(med.design == "walkforward") & (med.year != "pooled")]
    for Y, g in yr.groupby("year"):
        g = g.set_index("city").loc[CITIES]
        for comp, x in (("f", "dist_coast_km"), ("SS", "abs_lat"), ("f", "abs_lat"), ("SS", "dist_coast_km")):
            tests.append(dict(design=f"walkforward_{Y}", component=comp, descriptor=x, rho=spearmanr(g[comp], desc[x])[0], p=np.nan))
    rng = np.random.default_rng(42)
    for comp in ("f", "SS"):
        for x in ("abs_lat", "dist_coast_km"):
            bs = []
            for _ in range(n_boot):
                v = yr.groupby("city")[comp].apply(lambda s: s.values[rng.integers(0, len(s), len(s))].mean()).loc[CITIES]
                bs.append(spearmanr(v, desc[x])[0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            tests.append(dict(design="walkforward_bootstrap", component=comp, descriptor=x, rho=np.nan, p=np.nan, ci_lo=lo, ci_hi=hi))
    t = pd.DataFrame(tests); t.to_csv(TAB / "decomposition_tests.csv", index=False)
    return d, med, t


def tables():
    m = pd.read_csv(TAB / "metrics_all.csv")
    med = pd.read_csv(TAB / "decomposition_ensemble_median.csv", dtype={"year": str})
    desc = pd.read_csv(TAB / "climate_descriptors.csv").set_index("city")
    wf = m[(m.design == "walkforward") & (m.year == "pooled")]
    dec = pd.read_csv(TAB / "decomposition_all.csv", dtype={"year": str})
    dp = dec[(dec.design == "walkforward") & (dec.year == "pooled") & dec.model.isin(ENSEMBLE)]
    rows = []
    for c in CITIES:
        e = wf[(wf.city == c) & (wf.features == "B") & wf.model.isin(ENSEMBLE)]; s = dp[dp.city == c]
        ref = wf[(wf.city == c) & (wf.features == "-")].set_index("model")
        rows.append(dict(city=c, maritime=desc.loc[c, "maritime"], R2_med=e.R2.median(), R2_min=e.R2.min(), R2_max=e.R2.max(),
                         f=s.f.iloc[0], SS_med=s.SS.median(), SS_min=s.SS.min(), SS_max=s.SS.max(),
                         RMSE_med=e.RMSE.median(), MAE_med=e.MAE.median(), MBE_med=e.MBE.median(),
                         R2_HS_FAO=ref.loc["HS_FAO", "R2"], R2_HS_CAL=ref.loc["HS_CAL", "R2"], MBE_HS_FAO=ref.loc["HS_FAO", "MBE"],
                         R2_CART=wf[(wf.city == c) & (wf.features == "B") & (wf.model == "CART")].R2.iloc[0]))
    t2 = pd.DataFrame(rows).set_index("city"); t2.to_csv(TAB / "Table2.csv")
    rnd = m[(m.design == "random") & (m.features == "B") & m.model.isin(ENSEMBLE)].groupby("city").R2.median()
    (rnd - t2.R2_med).rename("R2_random_minus_walkforward").to_csv(TAB / "random_vs_walkforward_gap.csv")
    k = pd.read_csv(TAB / "kRs_calibrated_walkforward.csv").groupby("city").kRs_cal.agg(["mean", "min", "max"])
    k.to_csv(TAB / "kRs_summary.csv")
    # Online Resources
    m[(m.design == "random") & (m.features.isin(["A", "-"]))].to_csv(TAB / "ESM_1_random_split_preliminary_predictors.csv", index=False)
    m[(m.design == "walkforward") | ((m.design == "random") & (m.features == "B"))].merge(
        dec[["city", "design", "year", "model", "f", "SS"]].rename(columns={"year": "year"}),
        on=["city", "design", "year", "model"], how="left").to_csv(TAB / "ESM_2_all_results.csv", index=False)
    return t2


# ----------------------------------------------------------------- figures
def figures():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.stats import spearmanr
    MM = 1 / 25.4
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "ps.fonttype": 42, "pdf.fonttype": 42})
    COL = {True: "#1f5fa8", False: "#c2571a"}; MK = {True: "o", False: "^"}; INK, MUTED = "#222222", "#6b6b6b"
    desc = pd.read_csv(TAB / "climate_descriptors.csv").set_index("city"); mar = desc.maritime.astype(bool)
    t2 = pd.read_csv(TAB / "Table2.csv", index_col=0)
    tests = pd.read_csv(TAB / "decomposition_tests.csv")
    med = pd.read_csv(TAB / "decomposition_ensemble_median.csv", dtype={"year": str})
    ym = med[(med.design == "walkforward") & (med.year != "pooled")]

    def save(fig, name):
        for ext in ("tif", "eps", "pdf", "png"):
            kw = {"pil_kwargs": {"compression": "tiff_lzw"}, "dpi": 600} if ext == "tif" else ({"dpi": 200} if ext == "png" else {})
            fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", **kw)
        plt.close(fig)

    def exp_handles():
        return [Line2D([], [], marker="o", ls="", color=COL[True], ms=5, label="Maritime-exposed"),
                Line2D([], [], marker="^", ls="", color=COL[False], ms=5, label="Continental")]

    # Fig 1
    from global_land_mask import globe
    lon = np.arange(-76, -30, 0.05); lat = np.arange(-36, 7, 0.05); LO, LA = np.meshgrid(lon, lat); land = globe.is_land(LA, LO)
    fig, ax = plt.subplots(figsize=(84 * MM, 84 * MM))
    ax.imshow(np.where(land, 1, 0), extent=[lon[0], lon[-1], lat[0], lat[-1]], origin="lower",
              cmap=matplotlib.colors.ListedColormap(["#ffffff", "#e3e3e0"]), interpolation="nearest")
    ax.contour(LO, LA, land.astype(float), levels=[0.5], colors="#9a9a9a", linewidths=0.3)
    ax.axhline(-23.44, color=MUTED, lw=0.5, ls="--"); ax.text(-75.5, -23.0, "Tropic of Capricorn", fontsize=6, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.5, ls=":"); ax.text(-75.5, 0.4, "Equator", fontsize=6, color=MUTED)
    off = {"Manaus": (0.8, 0.8), "Belem": (0.8, 0.6), "Cuiaba": (-1.0, -1.8), "Brasilia": (0.9, 0.5), "Belo_Horizonte": (0.9, 0.4),
           "Fortaleza": (-1.5, 1.0), "Recife": (-1.2, -1.9), "Rio_de_Janeiro": (0.9, -1.4), "Sao_Paulo": (0.6, 0.9),
           "Curitiba": (-1.0, -0.4), "Porto_Alegre": (-1.0, -0.4)}
    for c in CITIES:
        ax.plot(desc.lon[c], desc.lat[c], MK[mar[c]], color=COL[mar[c]], ms=5.5, mec="white", mew=0.6, zorder=3)
        dx, dy = off[c]; ha = "left" if dx > 0 else ("right" if c in ("Curitiba", "Porto_Alegre") else "center")
        ax.text(desc.lon[c] + dx, desc.lat[c] + dy, NAME[c], fontsize=6.5, color=INK, ha=ha)
    ax.set_xlim(-76, -30); ax.set_ylim(-36, 7); ax.set_aspect("equal"); ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    for s in ("top", "right"): ax.spines[s].set_visible(True)
    ax.legend(handles=exp_handles(), loc="lower right", frameon=False); save(fig, "Fig1")

    # Fig 2
    fig, axes = plt.subplots(3, 4, figsize=(174 * MM, 115 * MM), sharex=True, sharey=True)
    order = sorted(CITIES, key=lambda c: desc.abs_lat[c])
    for ax, c in zip(axes.flat, order):
        x = load(c); x["m"] = x.date.dt.month
        toa = x.groupby("m").TOA_SW_DWN.mean(); kt = (x[TARGET] / x.TOA_SW_DWN).groupby(x.m).mean()
        z = lambda s: (s - s.mean()) / s.std()
        ax.plot(toa.index, z(toa), color=MUTED, ls="--", lw=1.0)
        ax.plot(kt.index, z(kt), color=COL[mar[c]], lw=1.6, marker=MK[mar[c]], ms=3); ax.axhline(0, color="#cccccc", lw=0.4)
        ax.set_title(f"{NAME[c]}  ({desc.abs_lat[c]:.1f}° S)\nr = {desc.r_phase_Kt_TOA[c]:+.2f}", loc="left")
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND"))
    for ax in axes[:, 0]: ax.set_ylabel("Standardized\nmonthly mean")
    lg = axes.flat[-1]; lg.axis("off")
    lg.legend(handles=[Line2D([], [], color=MUTED, ls="--", label="TOA insolation"),
                       Line2D([], [], color=COL[True], marker="o", ms=3, label="Clearness index $K_T$ (maritime)"),
                       Line2D([], [], color=COL[False], marker="^", ms=3, label="Clearness index $K_T$ (continental)")], loc="center", frameon=False)
    fig.tight_layout(h_pad=0.6, w_pad=0.6); save(fig, "Fig2")

    # Fig 3
    cont = sorted([c for c in CITIES if not mar[c]], key=lambda c: t2.R2_med[c]); mari = sorted([c for c in CITIES if mar[c]], key=lambda c: t2.R2_med[c])
    fig, ax = plt.subplots(figsize=(84 * MM, 92 * MM)); y = 0; yt, yl = [], []
    for c in cont + [None] + mari:
        if c is None: y += 0.6; continue
        f = t2.f[c]; syn = t2.SS_med[c] * (1 - f)
        ax.barh(y, f, color="#4a4a4a", height=0.7, edgecolor="white", lw=0.8)
        ax.barh(y, syn, left=max(f, 0), color="#bdbdbd", height=0.7, edgecolor="white", lw=0.8, hatch="////")
        ax.errorbar(t2.R2_med[c], y, xerr=[[t2.R2_med[c] - t2.R2_min[c]], [t2.R2_max[c] - t2.R2_med[c]]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
        ax.text(t2.R2_max[c] + 0.015, y, f"{t2.R2_med[c]:.2f}", va="center", fontsize=6.5, color=INK)
        yt.append(y); yl.append(NAME[c]); y += 1
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.text(-0.42, (len(cont) - 1) / 2, "Continental", rotation=90, ha="right", va="center", transform=ax.get_yaxis_transform(), fontsize=7, color=COL[False])
    ax.text(-0.42, len(cont) + 0.6 + (len(mari) - 1) / 2, "Maritime-exposed", rotation=90, ha="right", va="center", transform=ax.get_yaxis_transform(), fontsize=7, color=COL[True])
    ax.set_xlim(0, 0.9); ax.set_xlabel("Explained variance $R^2$ (walk-forward, 2018–2023)")
    ax.legend(handles=[Patch(fc="#4a4a4a", label="Seasonal share $f$"), Patch(fc="#bdbdbd", hatch="////", ec="white", label="Synoptic contribution $SS\\,(1-f)$")],
              loc="lower right", frameon=False, bbox_to_anchor=(1.0, -0.32)); save(fig, "Fig3")

    # Fig 4
    def ci(comp, x):
        r = tests[(tests.design == "walkforward_bootstrap") & (tests.component == comp) & (tests.descriptor == x)]
        return (r.ci_lo.iloc[0], r.ci_hi.iloc[0]) if len(r) else None
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 62 * MM))
    panels = [("f", np.log10(desc.dist_coast_km + 1), "dist_coast_km", "Distance to coast (km, log scale)", "Seasonal share $f$"),
              ("f", desc.r_phase_Kt_TOA, "r_phase_Kt_TOA", "Phase correlation $r$ ($K_T$ vs TOA)", "Seasonal share $f$"),
              ("SS", desc.abs_lat, "abs_lat", "Latitude (° S)", "Synoptic skill $SS$")]
    offs = {("b", "Brasilia"): (4, -6), ("b", "Belo_Horizonte"): (-14, 3), ("c", "Belem"): (-4, 5), ("c", "Manaus"): (4, -6)}
    for ax, (comp, xv, xn, xl, ylab), lab in zip(axes, panels, "abc"):
        yv = t2.f if comp == "f" else t2.SS_med
        for c in CITIES:
            ax.plot(xv[c], yv[c], MK[mar[c]], color=COL[mar[c]], ms=5, mec="white", mew=0.5)
            ax.annotate(CODE[c], (xv[c], yv[c]), xytext=offs.get((lab, c), (3, 2)), textcoords="offset points", fontsize=5.5, color=MUTED)
        rho, p = spearmanr(yv.loc[CITIES], xv.loc[CITIES])
        txt = f"ρ = {rho:+.2f} (p = {p:.3f})" if p >= 0.001 else f"ρ = {rho:+.2f} (p < 0.001)"
        b = ci(comp, xn)
        if b: txt += f"\n95% CI [{b[0]:+.2f}, {b[1]:+.2f}]"
        if lab == "a": ax.text(0.97, 0.97, txt, transform=ax.transAxes, va="top", ha="right", fontsize=6.5)
        else: ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=6.5)
        ax.set_xlabel(xl); ax.set_ylabel(ylab); ax.set_title(f"({lab})", loc="left")
        if xn == "dist_coast_km":
            ax.set_xticks(np.log10(np.array([0, 10, 100, 1000]) + 1)); ax.set_xticklabels(["0", "10", "100", "1000"])
    fig.tight_layout(w_pad=1.2, rect=(0, 0.09, 1, 1)); fig.legend(handles=exp_handles(), loc="lower center", ncol=2, frameon=False); save(fig, "Fig4")

    # Fig 5
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 70 * MM), sharex=True)
    for ax, comp, ylab, lab in zip(axes, ("f", "SS"), ("Seasonal share $f$", "Synoptic skill $SS$"), "ab"):
        for i, c in enumerate(order):
            v = ym[ym.city == c][comp].values
            ax.plot(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, "o", color="#9a9a9a", ms=2.5, mew=0)
            ax.plot(i, np.median(v), MK[mar[c]], color=COL[mar[c]], ms=6, mec="white", mew=0.6, zorder=3)
        ax.axhline(0, color="#cccccc", lw=0.4); ax.set_xticks(range(len(order))); ax.set_xticklabels([CODE[c] for c in order])
        ax.set_xlabel("Site (ordered by latitude, north to south)"); ax.set_ylabel(ylab); ax.set_title(f"({lab})", loc="left")
    h = [Line2D([], [], marker="o", ls="", color="#9a9a9a", ms=3, label="Single test year"),
         Line2D([], [], marker="o", ls="", color=COL[True], ms=5, label="Median, maritime-exposed"),
         Line2D([], [], marker="^", ls="", color=COL[False], ms=5, label="Median, continental")]
    fig.tight_layout(w_pad=1.5, rect=(0, 0.08, 1, 1)); fig.legend(handles=h, loc="lower center", ncol=3, frameon=False); save(fig, "Fig5")


def run_all(n_jobs=2):
    run_models(n_jobs=n_jobs); coast_distance(); descriptors(); decomposition(); t2 = tables(); figures()
    return t2


if __name__ == "__main__":
    print(run_all().round(2).to_string())
