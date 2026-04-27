import geopandas as gpd
import pandas as pd
import numpy as np
from libpysal.weights import Queen
from spreg import OLS, ML_Lag, ML_Error
from esda.moran import Moran
import warnings
warnings.filterwarnings("ignore")

tracts = gpd.read_file("tracts_in_cities.geojson")
tracts = tracts.to_crs(epsg=3857)

results = []

for city_name in sorted(tracts["NAME"].dropna().unique()):
    city = tracts[tracts["NAME"] == city_name].copy().reset_index(drop=True)
    city = city.dropna(subset=["PctWhite", "PctNonWhite"]).reset_index(drop=True)
    if len(city) < 10:
        print(f"  Skipping {city_name} — too few tracts")
        continue
    try:
        # --- Spatial weights ---
        w = Queen.from_dataframe(city, silence_warnings=True)
        w.transform = "r"
        centroid = city.geometry.centroid
        city["north"] = (centroid.y - centroid.y.mean()) / centroid.y.std()
        city["east"]  = (centroid.x - centroid.x.mean()) / centroid.x.std()
        y = city["PctWhite"].values.reshape(-1, 1)
        X = city[["north", "east"]].values  # spreg adds intercept automatically
        ols = OLS(y, X, w=w, name_y="PctWhite", name_x=["north", "east"],
                  name_ds=city_name, spat_diag=True)
        lag = ML_Lag(y, X, w=w, name_y="PctWhite", name_x=["north", "east"],
                     name_ds=city_name)
        err = ML_Error(y, X, w=w, name_y="PctWhite", name_x=["north", "east"],
                       name_ds=city_name)
        b_intercept, b_north, b_east = [b[0] for b in ols.betas]
        p_intercept, p_north, p_east = [t[1] for t in ols.t_stat]
        def sig(p):
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        rho    = lag.rho
        lam    = err.lam
        mi_resid = Moran(ols.u.flatten(), w, permutations=999)
        results.append({
            "City":          city_name,
            "Tracts":        len(city),
            "OLS_R2":        round(ols.r2, 4),
            "b_north":       round(b_north, 4),
            "p_north":       round(p_north, 4),
            "sig_north":     sig(p_north),
            "b_east":        round(b_east, 4),
            "p_east":        round(p_east, 4),
            "sig_east":      sig(p_east),
            "Moran_I_resid": round(mi_resid.I, 4),   # Moran's I on residuals
            "p_Moran_resid": round(mi_resid.p_sim, 4),   # p-value
            "Lag_rho":       round(rho, 4),
            "Lag_AIC":       round(lag.aic, 2),
            "Err_lambda":    round(lam, 4),
            "Err_AIC":       round(err.aic, 2),
            "Best_model":    "Lag" if lag.aic < err.aic else "Error",})
        print(f"  {city_name}: b_north={b_north:.3f} ({sig(p_north)}) | "
              f"rho={rho:.3f} | lambda={lam:.3f} | "
              f"Best={'Lag' if lag.aic < err.aic else 'Error'}")
    except Exception as e:
        print(f"  ERROR — {city_name}: {e}")
        continue

df = pd.DataFrame(results)
df = df.sort_values("b_north", ascending=False)

print("\n=== Spatial Regression Results: N-S Gradient on PctWhite ===")
cols = ["City", "Tracts", "OLS_R2", "b_north", "p_north", "sig_north",
        "b_east", "p_east", "sig_east", "Moran_I_resid", "p_Moran_resid",
        "Lag_rho", "Lag_AIC", "Err_lambda", "Err_AIC", "Best_model"]
print(df[cols].to_string(index=False))

df.to_csv("spatial_regression_results.csv", index=False)
print("\nSaved: spatial_regression_results.csv")