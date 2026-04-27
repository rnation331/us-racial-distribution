import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import re

tracts = gpd.read_file("tracts_in_cities.geojson")

red_cmap = plt.cm.Reds
blue_cmap = plt.cm.Blues

def tract_color(row):
    if row["majority"] == "white":
        intensity = 0.3 + 0.7 * (row["PctWhite"] - 0.5) / 0.5
        return red_cmap(np.clip(intensity, 0, 1))
    else:
        intensity = 0.3 + 0.7 * (row["PctNonWhite"] - 0.5) / 0.5
        return blue_cmap(np.clip(intensity, 0, 1))

os.makedirs("city_maps", exist_ok=True)

cities = tracts["NAME"].dropna().unique()
print(f"Found {len(cities)} cities: {sorted(cities)}")

for city_name in sorted(cities):
    city = tracts[tracts["NAME"] == city_name].copy()

    if city.empty:
        print(f"  Skipping {city_name} — no tracts found")
        continue

    city["majority"] = np.where(city["PctWhite"] >= 0.5, "white", "nonwhite")
    city["color"] = city.apply(tract_color, axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))

    city.plot(
        ax=ax,
        color=city["color"],
        edgecolor="white",
        linewidth=0.4)

    red_patch = mpatches.Patch(color=red_cmap(0.7), label="Majority White")
    blue_patch = mpatches.Patch(color=blue_cmap(0.7), label="Majority Non-White")
    ax.legend(
        handles=[red_patch, blue_patch],
        loc="lower right",
        fontsize=11,
        framealpha=0.9)

    ax.set_title(f"{city_name} — Racial Distribution by Census Tract", fontsize=16, pad=14)
    ax.set_axis_off()

    plt.tight_layout()

    safe_name = re.sub(r'[^\w\s-]', '', city_name)  # strip /, (, ), etc.
    safe_name = safe_name.strip().lower().replace(' ', '_')
    filename = f"city_maps/{safe_name}_race_map.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")

print("Done.")