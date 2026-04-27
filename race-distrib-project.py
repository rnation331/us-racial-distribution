import pandas as pd
import geopandas as gpd

tract_data = pd.read_csv('2023-tract-data.csv')
geo_data = gpd.read_file('2023-tracts-500k.shp')

geoid_col = tract_data["geoid"].astype(str).str.zfill(11)
tract_data = tract_data.drop(columns=["geoid"]).apply(pd.to_numeric, errors='coerce')
tract_data["GEOID"] = geoid_col

geo_data["GEOID"] = geo_data["GEOID"].astype(str)

numeric_cols = [
    "TotalPop","WhitePop","BlackPop","NativePop","AsianPop",
    "PacificPop","OtherPop","MultiPop",
    "PctWhite","PctBlack","PctNative","PctAsian",
    "PctPacific","PctOther","PctMulti"]

tract_data = tract_data[["GEOID"] + numeric_cols]

for col in numeric_cols:
    tract_data[col] = pd.to_numeric(tract_data[col], errors='coerce')

geo_data = geo_data[["GEOID", "geometry"]]
tracts_gdf = geo_data.merge(tract_data, on="GEOID", how="left")

tracts_gdf["White"] = tracts_gdf["WhitePop"]
tracts_gdf["NonWhite"] = (
tracts_gdf["BlackPop"] +
    tracts_gdf["NativePop"] +
    tracts_gdf["AsianPop"] +
    tracts_gdf["PacificPop"] +
    tracts_gdf["OtherPop"] +
    tracts_gdf["MultiPop"])
tracts_gdf["Total"] = tracts_gdf["TotalPop"]
tracts_gdf["PctWhite"] = tracts_gdf["White"] / tracts_gdf["Total"]
tracts_gdf["PctNonWhite"] = tracts_gdf["NonWhite"] / tracts_gdf["Total"]
tracts_gdf["PctWhite"] = tracts_gdf["PctWhite"].fillna(0)
tracts_gdf["PctNonWhite"] = tracts_gdf["PctNonWhite"].fillna(0)
tracts_gdf["check_total"] = tracts_gdf["White"] + tracts_gdf["NonWhite"]

cities_gdf = gpd.read_file('2023-place.shp')

top_city_fips = [
    "3502000", "1304000", "4805000", "2404000", "2507000",
    "3712000", "1714000", "0816000", "3918000", "4819000",
    "0820000", "2622000", "4824000", "4827000", "0627000",
    "4835000", "1836003", "1235000", "2938000", "3240000",
    "0644000", "2148006", "4748000", "0446000", "5553000",
    "4752006", "3651000", "4055000", "4260000", "0455000",
    "4159000", "3755000", "0664000", "4865000", "0666000",
    "0667000", "0668000", "5363000", "0477000", "1150000"]

top_city_names = [
    "Albuquerque", "Atlanta", "Austin", "Baltimore", "Boston",
    "Charlotte", "Chicago", "Colorado Springs", "Columbus", "Dallas",
    "Denver", "Detroit", "El Paso", "Fort Worth", "Fresno",
    "Houston", "Indianapolis", "Jacksonville", "Kansas City", "Las Vegas",
    "Los Angeles", "Louisville", "Memphis", "Mesa", "Milwaukee",
    "Nashville", "New York", "Oklahoma City", "Philadelphia", "Phoenix",
    "Portland", "Raleigh", "Sacramento", "San Antonio", "San Diego",
    "San Francisco", "San Jose", "Seattle", "Tucson", "Washington"]

cities_gdf["GEOID"] = cities_gdf["GEOID"].astype(str).str.zfill(7)

tracts_gdf = tracts_gdf.to_crs(epsg=3857)
cities_gdf = cities_gdf.to_crs(epsg=3857)
cities = cities_gdf[cities_gdf["GEOID"].isin(top_city_fips)]

tracts_gdf["centroid"] = tracts_gdf.geometry.centroid
tracts_centroid = gpd.GeoDataFrame(tracts_gdf, geometry="centroid", crs=3857)

tracts_in_cities = gpd.sjoin(
    tracts_centroid,
    cities[["geometry", "GEOID", "NAME"]],
    predicate="within"
).drop(columns=["centroid"])

tracts_in_cities = tracts_in_cities.set_geometry(
    tracts_gdf.set_index("GEOID").loc[tracts_in_cities["GEOID_left"]]["geometry"].values)

tracts_in_cities = tracts_in_cities.rename(columns={
    "GEOID_left": "GEOID",
    "NAME_right": "NAME"})

tracts_in_cities = tracts_in_cities.drop(columns=["centroid"])

tracts_in_cities.to_file("tracts_in_cities.geojson", driver="GeoJSON")