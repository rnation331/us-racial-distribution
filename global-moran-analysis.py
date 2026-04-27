import geopandas as gpd
import pandas as pd
import numpy as np
from libpysal.weights import Queen
from esda.moran import Moran
import warnings

warnings.filterwarnings("ignore")

tracts = gpd.read_file("tracts_in_cities.geojson")
tracts = tracts.to_crs(epsg=3857)

results = []

for city_name in sorted(tracts["NAME"].dropna().unique()):
    city = tracts[tracts["NAME"] == city_name].copy().reset_index(drop=True)
    if len(city) < 10:
        print(f"  Skipping {city_name} — too few tracts ({len(city)})")
        continue
    city = city.dropna(subset=["PctWhite", "PctNonWhite"]).reset_index(drop=True)
    try:
        w = Queen.from_dataframe(city, silence_warnings=True)
        w.transform = "r"
        city["centroid_y"] = city.geometry.centroid.y
        city["centroid_y_std"] = (
                (city["centroid_y"] - city["centroid_y"].mean()) /
                city["centroid_y"].std())
        mi_white = Moran(city["PctWhite"], w, permutations=999)
        mi_nonwhite = Moran(city["PctNonWhite"], w, permutations=999)
        mi_ns = Moran(city["centroid_y_std"], w, permutations=999)

        results.append({
            "City": city_name,
            "Tracts": len(city),
            "Moran_I_White": round(mi_white.I, 4),
            "p_White": round(mi_white.p_sim, 4),
            "sig_White": "***" if mi_white.p_sim < 0.001 else
            "**" if mi_white.p_sim < 0.01 else
            "*" if mi_white.p_sim < 0.05 else "ns",
            "Moran_I_NonWhite": round(mi_nonwhite.I, 4),
            "p_NonWhite": round(mi_nonwhite.p_sim, 4),
            "sig_NonWhite": "***" if mi_nonwhite.p_sim < 0.001 else
            "**" if mi_nonwhite.p_sim < 0.01 else
            "*" if mi_nonwhite.p_sim < 0.05 else "ns",
            "Moran_I_NS": round(mi_ns.I, 4),
            "p_NS": round(mi_ns.p_sim, 4),
            "z_White": round(mi_white.z_norm, 4),
            "z_NonWhite": round(mi_nonwhite.z_norm, 4),})
        print(f"  {city_name}: I(white)={mi_white.I:.3f} p={mi_white.p_sim:.3f} "
              f"| I(nonwhite)={mi_nonwhite.I:.3f} p={mi_nonwhite.p_sim:.3f}")
    except Exception as e:
        print(f"  ERROR — {city_name}: {e}")
        continue

df = pd.DataFrame(results)
df = df.sort_values("Moran_I_White", ascending=False)

print("\n=== Global Moran's I Results ===")
print(df.to_string(index=False))

df.to_csv("morans_i_results.csv", index=False)
print("\nSaved: morans_i_results.csv")