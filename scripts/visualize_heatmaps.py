#!/usr/bin/env python3
"""
visualize_heatmaps.py  —  CS506 Final Project (cleaned & robust)

What this does:
- Aggregates city rows -> county by CountyFIPS (mean)
- Makes county×measure matrix + non-map heatmap
- Draws county choropleths with:
  * tight cropping (no giant margins)
  * explicit colorbar labels
  * a gray "No data" base for counties without values
  * CONUS-only by default (removes AK/HI/PR/territories)
- Also supports: side-by-side, difference, and bivariate maps

Usage (example):
  python scripts/visualize_heatmaps.py \
    --csv data/all_datasets_county_level.csv \
    --shp data/shapes_2010/tl_2010_us_county10.shp \
    --measures "Food Insecurity Rate" "Child food insecurity rate" "Cost Per Meal" \
               "PovertyRate" "MedianFamilyIncome" "lalowi1share" "ACCESS2" \
               "OBESITY" "DIABETES" "CHD" \
    --year 2019 \
    --outdir outputs/figs \
    --bivariate "Food Insecurity Rate" "DIABETES"

If you don’t want quantile bins: add --bins none
If you want to include AK/HI/PR: add --conus_only False
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import geopandas as gpd
except ImportError:
    gpd = None

# Optional quantile binning
_HAS_MAPCLASSIFY = False
try:
    import mapclassify  # noqa: F401
    _HAS_MAPCLASSIFY = True
except Exception:
    _HAS_MAPCLASSIFY = False

# Lower-48 only (CONUS) by default; use FIPS prefix so this works on any 2010 county file
NON_CONUS = {"02", "15", "72", "60", "66", "69", "78"}  # AK, HI, PR, AS, GU, MP, VI

# Labels for colorbars
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

# Styling
NO_DATA_FACE = "#eeeeee"   # neutral gray for counties without data
NO_DATA_EDGE = "none"


# ----------------------------- CLI -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="County-level heatmaps and comparison maps with no-data shading.")
    ap.add_argument("--csv", required=True, help="Path to all_datasets_county_level.csv")
    ap.add_argument("--shp", required=True, help="Path to US counties shapefile (2010)")
    ap.add_argument("--measures", nargs="+", required=True,
                    help="MeasureId values to visualize (exact strings as they appear in the CSV)")
    ap.add_argument("--year", type=int, default=None, help="Optional Survey_Year filter (e.g., 2019)")
    ap.add_argument("--outdir", default="outputs/figs", help="Directory to save figures")
    ap.add_argument("--bivariate", nargs=2, default=None,
                    help="Two measures to make a bivariate choropleth (must also be in --measures)")
    ap.add_argument("--conus_only", type=lambda s: s.lower() != "false", default=True,
                    help="Restrict plots to contiguous U.S. (default True). Pass 'False' to include AK/HI/PR.")
    ap.add_argument("--bins", choices=["quantiles", "none"], default="quantiles",
                    help="Quantile classes (pretty) or continuous. Quantiles require `pip install mapclassify`.")
    ap.add_argument("--k", type=int, default=5, help="Number of bins for quantiles")
    return ap.parse_args()


# ----------------------------- IO -----------------------------

def read_data(csv_path, year=None):
    df = pd.read_csv(csv_path, low_memory=False)
    if year is not None and "Survey_Year" in df.columns:
        df = df[df["Survey_Year"] == year]
    keep = ["CountyFIPS", "Survey_Year", "MeasureId", "Data_Value_Unit", "Data_Value", "StateDesc", "CityName"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
    df = df.dropna(subset=["CountyFIPS"])
    df["CountyFIPS"] = df["CountyFIPS"].astype(str).str.zfill(5)
    return df


def list_available_measures(df):
    return df["MeasureId"].dropna().value_counts().index.tolist()


def aggregate_measures_to_county(df, measures):
    sub = df[df["MeasureId"].isin(measures)].copy()
    if sub.empty:
        raise ValueError("None of the requested measures were found in MeasureId.")
    agg = (sub.groupby(["CountyFIPS", "MeasureId"])["Data_Value"]
              .mean()
              .reset_index())
    wide = agg.pivot(index="CountyFIPS", columns="MeasureId", values="Data_Value").reset_index()
    wide = wide.rename_axis(None, axis=1)
    return wide


def try_load_county_shapes(shp_path):
    if gpd is None:
        raise RuntimeError("geopandas not installed; cannot make choropleths.")
    gdf = gpd.read_file(shp_path)
    # normalize FIPS
    if "GEOID10" in gdf.columns:
        gdf["CountyFIPS"] = gdf["GEOID10"].astype(str).str.zfill(5)
    elif "GEOID" in gdf.columns:
        gdf["CountyFIPS"] = gdf["GEOID"].astype(str).str[-5:].str.zfill(5)
    elif "COUNTYFP10" in gdf.columns and "STATEFP10" in gdf.columns:
        gdf["CountyFIPS"] = gdf["STATEFP10"].astype(str).str.zfill(2) + gdf["COUNTYFP10"].astype(str).str.zfill(3)
    elif "COUNTYFP" in gdf.columns and "STATEFP" in gdf.columns:
        gdf["CountyFIPS"] = gdf["STATEFP"].astype(str).str.zfill(2) + gdf["COUNTYFP"].astype(str).str.zfill(3)
    else:
        raise ValueError("Could not identify a FIPS/GEOID column in the shapefile.")
    return gdf


# ----------------------------- Plot helpers -----------------------------

def _maybe_scheme_kwargs(bins, k):
    if bins == "quantiles" and _HAS_MAPCLASSIFY:
        return dict(scheme="Quantiles", k=int(k))
    return {}  # continuous colorbar if mapclassify is not installed


def _tighten_map(ax, gdf, pad_frac=0.01):
    minx, miny, maxx, maxy = gdf.total_bounds
    px = (maxx - minx) * pad_frac
    py = (maxy - miny) * pad_frac
    ax.set_xlim(minx - px, maxx + px)
    ax.set_ylim(miny - py, maxy + py)
    ax.set_aspect('equal')
    ax.axis('off')


def _legend_no_data(ax, label="No data", loc="lower left"):
    handle = Patch(facecolor=NO_DATA_FACE, edgecolor=NO_DATA_EDGE, label=label)
    leg = ax.legend(handles=[handle], loc=loc, frameon=True, fontsize=9)
    ax.add_artist(leg)


# ----------------------------- Plotters -----------------------------

def choropleth_one(gdf_all, column, outdir, title=None, legend_label=None,
                   bins="quantiles", k=5, cmap="viridis"):
    """Always paint ALL counties gray first, then overlay counties with data."""
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # a) base layer: all counties (already CONUS-filtered in main)
    gdf_all.plot(ax=ax, color=NO_DATA_FACE, edgecolor=NO_DATA_EDGE)

    # b) overlay: only counties that have a value for `column`
    gdf_with = gdf_all.dropna(subset=[column]).copy()
    plot_kwargs = dict(column=column, legend=True, ax=ax, edgecolor='none',
                       legend_kwds={'label': legend_label or column, 'shrink': 0.7},
                       cmap=cmap)
    plot_kwargs.update(_maybe_scheme_kwargs(bins, k))
    if len(gdf_with) > 0:
        gdf_with.plot(**plot_kwargs)

    ax.set_title(title or column)
    _tighten_map(ax, gdf_all)
    _legend_no_data(ax)
    out_path = Path(outdir) / f"choropleth_{column.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] {out_path}")


def choropleth_side_by_side(gdf_all, colA, colB, outdir, title=None, labels=("", ""),
                            bins="quantiles", k=5, cmap="viridis"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    for col, ax, lab in [(colA, axes[0], labels[0]), (colB, axes[1], labels[1])]:
        # base: all counties gray
        gdf_all.plot(ax=ax, color=NO_DATA_FACE, edgecolor=NO_DATA_EDGE)
        # overlay: counties that have this column
        gdf_with = gdf_all.dropna(subset=[col]).copy()
        plot_kwargs = dict(column=col, legend=True, ax=ax, edgecolor='none',
                           legend_kwds={'label': lab or col, 'shrink': 0.7},
                           cmap=cmap)
        plot_kwargs.update(_maybe_scheme_kwargs(bins, k))
        if len(gdf_with) > 0:
            gdf_with.plot(**plot_kwargs)
        ax.set_title(col)
        _tighten_map(ax, gdf_all)
        _legend_no_data(ax, loc="lower left")

    if title:
        fig.suptitle(title)
    out_path = Path(outdir) / f"side_by_side_{colA.replace(' ','_')}_vs_{colB.replace(' ','_')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] {out_path}")


def choropleth_difference(gdf_all, colA, colB, outdir, legend_label=None, cmap="coolwarm"):
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # base: all counties gray
    gdf_all.plot(ax=ax, color=NO_DATA_FACE, edgecolor=NO_DATA_EDGE)

    # overlay: counties with both measures
    gdf = gdf_all.dropna(subset=[colA, colB]).copy()
    diff_col = f"diff_{colA.replace(' ','_')}_minus_{colB.replace(' ','_')}"
    gdf[diff_col] = gdf[colA] - gdf[colB]
    if len(gdf) > 0:
        gdf.plot(column=diff_col, cmap=cmap, legend=True, ax=ax, edgecolor='none',
                 legend_kwds={'label': legend_label or f"{colA} − {colB}", 'shrink': 0.7})

    ax.set_title(f"{colA} − {colB}")
    _tighten_map(ax, gdf_all)
    _legend_no_data(ax)
    out_path = Path(outdir) / f"{diff_col}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] {out_path}")


def bivariate_choropleth(gdf_all, col_x, col_y, outdir):
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # base: all counties gray
    gdf_all.plot(ax=ax, color=NO_DATA_FACE, edgecolor=NO_DATA_EDGE)

    # overlay: both present
    gdf = gdf_all.dropna(subset=[col_x, col_y]).copy()
    if len(gdf) > 0:
        x_bins = pd.qcut(gdf[col_x], 3, labels=[0, 1, 2])
        y_bins = pd.qcut(gdf[col_y], 3, labels=[0, 1, 2])
        gdf["_xbin"] = x_bins.astype(int)
        gdf["_ybin"] = y_bins.astype(int)
        colors = [
            ["#e8e8e8", "#b5c0da", "#6c83b5"],
            ["#b8d6be", "#90b2b3", "#567994"],
            ["#73ae80", "#5a9178", "#2a5a5b"],
        ]
        gdf["_bivar_color"] = [colors[yb][xb] for xb, yb in zip(gdf["_xbin"], gdf["_ybin"])]
        gdf.plot(color=gdf["_bivar_color"], ax=ax, edgecolor='none')

    ax.set_title(f"Bivariate: {col_x} (x) vs {col_y} (y)")
    _tighten_map(ax, gdf_all)
    _legend_no_data(ax)
    out_path = Path(outdir) / f"bivariate_{col_x.replace(' ','_')}_vs_{col_y.replace(' ','_')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] {out_path}")


def non_map_heatmap(wide, measures, outdir):
    M = wide.set_index("CountyFIPS")[measures].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
    im = ax.imshow(M, aspect="auto")
    ax.set_title("County × Measure Heatmap")
    ax.set_xlabel("Measures")
    ax.set_ylabel("Counties")
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels(measures, rotation=45, ha="right")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Value")
    out_path = Path(outdir) / "non_map_heatmap.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ----------------------------- Main -----------------------------

def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # --- data ---
    df = read_data(args.csv, year=args.year)
    available = set(list_available_measures(df))
    measures = [m for m in args.measures if m in available]
    if not measures:
        print("[error] None of the requested measures were found. Example MeasureIds:")
        print(pd.Series(sorted(list(available))).head(30).to_list())
        return

    wide = aggregate_measures_to_county(df, measures)

    # --- shapes + join ---
    gdf_shapes = try_load_county_shapes(args.shp)
    merged = gdf_shapes.merge(wide, on="CountyFIPS", how="left")

    # Hard cut to CONUS unless the user opts out
    if args.conus_only:
        merged["STATE2"] = merged["CountyFIPS"].str[:2]
        merged = merged[~merged["STATE2"].isin(NON_CONUS)].copy()

    # save the wide matrix for reference
    Path(outdir).mkdir(parents=True, exist_ok=True)
    wide_out = Path(outdir) / "county_measure_matrix.csv"
    wide.to_csv(wide_out, index=False)

    # --- figures ---
    non_map_heatmap(wide, measures, outdir)

    # Univariate choropleths
    for col in measures:
        if col in merged.columns:
            choropleth_one(
                merged, col, outdir,
                title=f"{col} by County{f' ({args.year})' if args.year else ''}",
                legend_label=MEASURE_LABELS.get(col, col),
                bins=args.bins, k=args.k, cmap="viridis"
            )

    # Side-by-side + difference for first two measures (if present)
    if len(measures) >= 2 and all(m in merged.columns for m in measures[:2]):
        choropleth_side_by_side(
            merged, measures[0], measures[1], outdir,
            title=f"{measures[0]} vs {measures[1]}",
            labels=(MEASURE_LABELS.get(measures[0], measures[0]),
                    MEASURE_LABELS.get(measures[1], measures[1])),
            bins=args.bins, k=args.k, cmap="viridis"
        )
        choropleth_difference(
            merged, measures[0], measures[1], outdir,
            legend_label=f"{MEASURE_LABELS.get(measures[0], measures[0])} − {MEASURE_LABELS.get(measures[1], measures[1])}",
            cmap="coolwarm"
        )

    # Bivariate (optional)
    if args.bivariate:
        x, y = args.bivariate
        if x in measures and y in measures and x in merged.columns and y in merged.columns:
            bivariate_choropleth(merged, x, y, outdir)
        else:
            print("[warn] Bivariate requested but measures not found in merged data.")

    print("[done] Outputs saved under:", outdir)


if __name__ == "__main__":
    main()
