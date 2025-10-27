#!/usr/bin/env python3
"""
visualize_zoomed.py — zoomed-in county maps around a city (e.g., Boston)

Fixes:
- Uses the *latest available year per MeasureId* automatically (so measures missing
  for a specific year still render if any year exists). Still respects --year when
  available for a given measure.
- Keeps "yearless" measures (NaN Survey_Year) like atlas/food-access.
- Case-insensitive city matching; robust union (union_all if available).
- Tight crops, "No data" gray base, labeled colorbar.

Usage example:
  python scripts/visualize_zoomed.py \
    --csv data/all_datasets_county_level.csv \
    --county_shp data/shapes_2010/tl_2010_us_county10.shp \
    --place_shp  data/shapes_2010/tl_2010_us_place10.shp \
    --city "Boston" --statefp 25 \
    --measures "Food Insecurity Rate" "DIABETES" "OBESITY" "lalowi1share" \
    --year 2019 \
    --buffer_km 30 \
    --outdir outputs/figs/zoom
"""

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---- appearance ----
NO_DATA_FACE = "#eeeeee"
NO_DATA_EDGE = "none"

MEASURE_LABELS = {
    "Food Insecurity Rate": "Food insecurity (%)",
    "Child food insecurity rate": "Child food insecurity (%)",
    "Cost Per Meal": "Cost per meal (USD)",
    "PovertyRate": "Poverty rate (%)",
    "MedianFamilyIncome": "Median family income (USD)",
    "lalowi1share": "Low-income & low-access share (%)",
    "ACCESS2": "Couldn’t see doctor due to cost (%)",
    "OBESITY": "Obesity prevalence (%)",
    "DIABETES": "Diabetes prevalence (%)",
    "CHD": "Coronary heart disease (%)",
}

_HAS_MAPCLASSIFY = False
try:
    import mapclassify  # noqa: F401
    _HAS_MAPCLASSIFY = True
except Exception:
    _HAS_MAPCLASSIFY = False


# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Create zoomed-in *county* maps around a city.")
    ap.add_argument("--csv", required=True, help="Path to long-format data CSV (CountyFIPS, MeasureId, Data_Value, ...)")
    ap.add_argument("--county_shp", required=True, help="Path to 2010 county shapefile (or gpkg)")
    ap.add_argument("--place_shp", required=True, help="Path to 2010 places (city boundary) shapefile (or gpkg)")
    ap.add_argument("--city", required=True, help='Target city name (case-insensitive), e.g., "Boston"')
    ap.add_argument("--statefp", required=True, type=int, help="2-digit FIPS of state (e.g., 25 for MA)")
    ap.add_argument("--measures", nargs="+", required=True, help="MeasureId(s) to plot")
    ap.add_argument("--year", type=int, default=None,
                    help="Preferred Survey_Year; will fall back to latest available per measure if not present")
    ap.add_argument("--outdir", default="outputs/figs/zoom", help="Where to save images")
    ap.add_argument("--buffer_km", type=float, default=30.0, help="Buffer radius (km) around city boundary")
    ap.add_argument("--bins", choices=["quantiles", "none"], default="quantiles", help="Use quantile bins if available")
    ap.add_argument("--k", type=int, default=5, help="Number of quantile bins")
    return ap.parse_args()


# ------------- data helpers -------------

def read_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    keep = ["CountyFIPS", "Survey_Year", "MeasureId", "Data_Value", "Data_Value_Unit"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
    df = df.dropna(subset=["CountyFIPS"])
    df["CountyFIPS"] = df["CountyFIPS"].astype(str).str.zfill(5)
    return df


def agg_to_county_latest(df, measures, prefer_year=None):
    """
    For each MeasureId:
      - If there are rows with Survey_Year == prefer_year -> use those.
      - Else if there are other years -> use the max available year.
      - Else if Survey_Year is entirely NaN -> keep those (static datasets).
    Then aggregate to county via mean and pivot wide.
    """
    frames = []
    for m in measures:
        sub = df[df["MeasureId"] == m].copy()
        if sub.empty:
            continue
        # partition by year presence
        has_year = sub["Survey_Year"].notna().any()
        chosen = None
        if prefer_year is not None and has_year and (sub["Survey_Year"] == prefer_year).any():
            chosen = sub[sub["Survey_Year"] == prefer_year]
        elif has_year:
            # latest available year
            latest = sub["Survey_Year"].dropna().max()
            chosen = sub[sub["Survey_Year"] == latest]
        else:
            # all rows are NaN year -> static dataset
            chosen = sub

        # aggregate to county mean (cities -> county)
        g = (chosen.groupby(["CountyFIPS"])["Data_Value"].mean()
             .rename(m)
             .reset_index())
        frames.append(g)

    if not frames:
        raise ValueError("None of the requested measures exist in the CSV.")

    wide = frames[0]
    for t in frames[1:]:
        wide = wide.merge(t, on="CountyFIPS", how="outer")
    return wide


def load_counties(county_path):
    gdf = gpd.read_file(county_path)
    if "GEOID10" in gdf.columns:
        gdf["CountyFIPS"] = gdf["GEOID10"].astype(str).str.zfill(5)
    elif "GEOID" in gdf.columns:
        gdf["CountyFIPS"] = gdf["GEOID"].astype(str).str[-5:].str.zfill(5)
    elif {"STATEFP10", "COUNTYFP10"}.issubset(gdf.columns):
        gdf["CountyFIPS"] = gdf["STATEFP10"].astype(str).str.zfill(2) + gdf["COUNTYFP10"].astype(str).str.zfill(3)
    elif {"STATEFP", "COUNTYFP"}.issubset(gdf.columns):
        gdf["CountyFIPS"] = gdf["STATEFP"].astype(str).str.zfill(2) + gdf["COUNTYFP"].astype(str).str.zfill(3)
    else:
        raise ValueError("Could not build CountyFIPS from county shapes.")
    return gdf


def load_place(place_path, statefp, name):
    g = gpd.read_file(place_path)
    # find columns
    state_col = "STATEFP10" if "STATEFP10" in g.columns else ("STATEFP" if "STATEFP" in g.columns else None)
    name_col = "NAME10" if "NAME10" in g.columns else ("NAME" if "NAME" in g.columns else None)
    if not state_col or not name_col:
        raise ValueError("Could not find STATEFP/NAME columns in place file.")
    sel = g[(g[state_col].astype(str).str.zfill(2) == f"{statefp:02d}") &
            (g[name_col].str.lower() == name.lower())]
    if sel.empty:
        raise ValueError(f'City "{name}" not found in state FIPS {statefp} in places file.')
    # dissolve in case multiple polygons
    sel = sel.dissolve().to_crs(3857)  # project for buffering
    return sel


def buffer_polygon(poly_gdf, km):
    dist_m = km * 1000.0
    buf = poly_gdf.buffer(dist_m)
    buf = gpd.GeoDataFrame(geometry=buf, crs=poly_gdf.crs).to_crs(4326)
    return buf


def crop_to_window(counties_gdf, window_gdf):
    bbox = window_gdf.total_bounds
    cand = counties_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    # union: prefer union_all if available (new), fallback to unary_union
    if hasattr(window_gdf, "union_all"):
        u = window_gdf.union_all()
    else:
        u = window_gdf.unary_union
    cand = cand[cand.intersects(u)]
    return cand


# ------------- plotting -------------

def _tight(ax, gdf, pad_frac=0.02):
    minx, miny, maxx, maxy = gdf.total_bounds
    px = (maxx - minx) * pad_frac
    py = (maxy - miny) * pad_frac
    ax.set_xlim(minx - px, maxx + px)
    ax.set_ylim(miny - py, maxy + py)
    ax.set_aspect("equal")
    ax.axis("off")


def _maybe_scheme_kwargs(bins, k):
    if bins == "quantiles" and _HAS_MAPCLASSIFY:
        return dict(scheme="Quantiles", k=int(k))
    return {}


def plot_zoom_county(zoom_counties, column, outdir, title=None, label=None, bins="quantiles", k=5, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    # base: gray for no data
    zoom_counties.plot(ax=ax, color=NO_DATA_FACE, edgecolor=NO_DATA_EDGE)
    # overlay where data exists
    with_data = zoom_counties.dropna(subset=[column])
    if len(with_data) > 0:
        kw = dict(column=column, legend=True, ax=ax, edgecolor="none",
                  legend_kwds={"label": label or column, "shrink": 0.7}, cmap=cmap)
        kw.update(_maybe_scheme_kwargs(bins, k))
        with_data.plot(**kw)
    ax.set_title(title or column)
    _tight(ax, zoom_counties)
    ax.legend(handles=[Patch(facecolor=NO_DATA_FACE, edgecolor=NO_DATA_EDGE, label="No data")],
              loc="lower left", frameon=True, fontsize=9)
    out = Path(outdir) / f"zoom_{column.replace(' ', '_')}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------- main -------------

def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Data → county wide, using latest per measure (respects --year when available)
    df = read_data(args.csv)
    wide = agg_to_county_latest(df, args.measures, prefer_year=args.year)

    # Counties & places
    counties = load_counties(args.county_shp).to_crs(4326)
    place = load_place(args.place_shp, statefp=args.statefp, name=args.city)  # 3857
    window = buffer_polygon(place, args.buffer_km)  # 4326

    # Merge data onto counties
    merged = counties.merge(wide, on="CountyFIPS", how="left")

    # Crop to zoom window
    zoom = crop_to_window(merged, window)

    # Report which measures are present after the join
    missing = [m for m in args.measures if m not in zoom.columns]
    if missing:
        print("[warn] Measures not present after merge:", missing)
    else:
        print("[info] All requested measures present.")

    # Plot each requested measure
    for m in args.measures:
        if m in zoom.columns:
            plot_zoom_county(
                zoom, m, args.outdir,
                title=f"{m} — {args.city} area",
                label=MEASURE_LABELS.get(m, m),
                bins=args.bins, k=args.k, cmap="viridis"
            )

    print("[done] Zoomed figures in:", args.outdir)


if __name__ == "__main__":
    main()
