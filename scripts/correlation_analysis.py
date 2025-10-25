"""
Food Atlas vs Feeding America: Direct Year Correlation Analysis (FIXED)
Compares Food Atlas structural data against Feeding America outcomes for the exact same year.

FIXED: 2019 data now properly filters by Year column since the file contains 2019-2023 data.

Author: Kimberly
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
import os
import pathlib

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
YEAR_PAIRS_TO_ANALYZE = [
    (2010, 2010),  # Atlas 2010 vs. FA 2010
    (2015, 2015),  # Atlas 2015 vs. FA 2015
    (2019, 2019)   # Atlas 2019 vs. FA 2019
]

# Map FA Year to the EXACT file configuration: (File Name, Sheet Name/Index, SkipRows)
FA_FILE_CONFIG = {
    2010: ('MMG2012_2010Data_ToShare.xlsx', 0, 0),
    2015: ('MMG2017_2015Data_ToShare.xlsx', 0, 0),
    2019: ('MMG2025_2019-2023_Data_To_Share.xlsx', 'County', 1) 
}

# Food Atlas metrics
FOOD_ATLAS_METRICS = {
    'PovertyRate': 'Tract poverty rate',
    'LowIncomeTracts': 'Low income tract flag',
    'LILATracts_1And10': 'Low income AND low access (1mi/10mi)',
    'LAPOP1_10': 'Population with low access (1mi/10mi)',
    'TractSNAP': 'Housing units receiving SNAP'
}

results_multi_year = []

# --- PATH SETUP ---
BASE_DIR = pathlib.Path(os.getcwd()) 
DATA_DIR = BASE_DIR / 'data'
if not DATA_DIR.is_dir():
    DATA_DIR = BASE_DIR.parent / 'data' 

def generate_correlation_plots(merged_df, fa_col_main, atlas_year, fa_year):
    """Generates and saves the 4-panel correlation plot for the given year pair."""
    comparisons_plot = [
        ('PovertyRate', 'Poverty Rate vs Food Insecurity'),
        ('LILATracts_1And10', 'Low Income/Access vs Food Insecurity'),
        ('TractSNAP', 'SNAP Recipients vs Food Insecurity'),
        ('LAPOP1_10', 'Low Access Population vs Food Insecurity'),
    ]
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (atlas_col, title) in enumerate(comparisons_plot):
        ax = axes[i]
        
        if atlas_col in merged_df.columns and fa_col_main in merged_df.columns:
            
            x_data = pd.to_numeric(merged_df[atlas_col], errors='coerce').dropna()
            y_data = pd.to_numeric(merged_df[fa_col_main], errors='coerce').dropna()
            
            if len(x_data) > 1 and len(y_data) > 1:
                r_val, p_val = pearsonr(x_data, y_data)

                sns.regplot(
                    x=x_data, 
                    y=y_data, 
                    scatter_kws={'alpha': 0.3, 's': 10}, 
                    line_kws={'color': 'red', 'linestyle': '--', 'label': 'Trend line'},
                    ax=ax
                )
                
                ax.set_title(
                    f"{title}\nPearson r = {r_val:.3f} (p = {p_val:.4f})", 
                    fontsize=14
                )
                ax.set_xlabel(f"Food Atlas: {atlas_col}", fontsize=12)
                ax.set_ylabel(f"Feeding America: {fa_year} Insecurity Rate", fontsize=12)
            else:
                ax.set_title(f"Insufficient Data for {title}", fontsize=14)
        else:
            ax.set_title(f"Data Column Missing for {title}", fontsize=14)

    fig.suptitle(
        f"Atlas {atlas_year} vs. FA {fa_year}: County-Level Correlation Analysis (N={len(merged_df)})", 
        fontsize=16, 
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = BASE_DIR.parent / 'outputs/correlation_plots'
    if not output_path.exists(): 
        os.makedirs(output_path)
    filename = output_path / f'atlas_{atlas_year}_vs_fa_{fa_year}_direct.png'
    plt.savefig(filename)
    plt.close()
    print(f"  Plot saved: {filename.relative_to(BASE_DIR.parent)}")


# --- START SCRIPT EXECUTION ---
print("\n--------------------------------------------------------------------")
print("DIRECT YEAR FOOD DATA CORRELATION ANALYSIS (FIXED FOR 2019)")
print("--------------------------------------------------------------------\n")

# Create the necessary output directories if they don't exist
output_dir = BASE_DIR.parent / 'outputs'
plot_dir = output_dir / 'correlation_plots'
if not output_dir.exists():
    os.makedirs(output_dir)
if not plot_dir.exists():
    os.makedirs(plot_dir)


# ========================================================================
# PRIMARY LOOP: Iterate through the defined (Atlas Year, FA Year) pairs
# ========================================================================

for atlas_year, fa_year in YEAR_PAIRS_TO_ANALYZE:
    print(f"\n[STARTING COMPARISON: Atlas {atlas_year} vs. FA {fa_year}]")

    # --- Determine File Names and Header Skip ---
    atlas_filename = f'FoodAccessResearchAtlasData{atlas_year}.xlsx'
    fa_filename, fa_sheet, skiprows_fa = FA_FILE_CONFIG.get(fa_year)
    
    atlas_path = DATA_DIR / atlas_filename 
    fa_path = DATA_DIR / fa_filename

    # --- STEP 1: LOAD YEAR-SPECIFIC DATA ---
    food_atlas = None
    feeding_america = None
    
    try:
        if not atlas_path.exists():
            raise FileNotFoundError(f"Atlas file '{atlas_filename}' not found.")
        
        # Load Food Atlas file 
        food_atlas = pd.read_excel(
            atlas_path, 
            sheet_name='Food Access Research Atlas'
        )
        print(f"  Loaded Food Atlas {atlas_year} from: {atlas_path.name}")

        if not fa_path.exists():
            raise FileNotFoundError(f"FA file '{fa_filename}' not found.")
        
        # Load FA file using the dynamic sheet name and skip rows
        feeding_america = pd.read_excel(fa_path, sheet_name=fa_sheet, skiprows=skiprows_fa)
        print(f"  Loaded FA data from: {fa_path.name}, sheet: {fa_sheet}")
        
        # CRITICAL FIX: Filter 2019 data by Year column since file contains 2019-2023
        if fa_year == 2019 and 'Year' in feeding_america.columns:
            before_filter = len(feeding_america)
            feeding_america = feeding_america[feeding_america['Year'] == 2019].copy()
            print(f"  Filtered from {before_filter} to {len(feeding_america)} rows for year {fa_year}")
        
        if food_atlas.empty or feeding_america.empty:
            print(f"  ERROR: No data found after loading files. Skipping.")
            continue

    except FileNotFoundError as e:
        print(f"  ERROR: Could not find required data files for this pair. Skipping.")
        print(f"    - Error Detail: {e}")
        continue
    except Exception as e:
        print(f"  ERROR reading file for pair {atlas_year} vs {fa_year}: {e}. Skipping.")
        continue

    # ========================================================================
    # STEP 2: DYNAMICALLY IDENTIFY KEY COLUMNS 
    # ========================================================================

    available_fa_cols = {col: desc for col, desc in FOOD_ATLAS_METRICS.items() if col in food_atlas.columns}

    # Identify FIPS column (in FA data)
    fips_col = next((col for col in feeding_america.columns if 'fips' in str(col).lower() or 'county' in str(col).lower()), None)
    if not fips_col:
        print(f"  ERROR: Cannot find FIPS/County column in Feeding America data for {fa_year}. Skipping.")
        print(f"  DEBUG: FA Columns: {list(feeding_america.columns)}")
        continue

    # Dynamically find the main Food Insecurity Rate column
    # FIXED: 2019 file has 'Overall Food Insecurity Rate' without year in column name
    feeding_col_main = None
    
    if fa_year == 2019:
        # For 2019, look for 'Overall Food Insecurity Rate' specifically
        for col in feeding_america.columns:
            if 'overall food insecurity rate' in str(col).lower():
                feeding_col_main = col
                print(f"  Found 2019 FA column: {feeding_col_main}")
                break
    else:
        # For 2010 and 2015, look for year in column name
        for col in feeding_america.columns:
            col_lower = str(col).lower()
            if str(fa_year) in col_lower and 'food insecurity rate' in col_lower:
                try:
                    if pd.to_numeric(feeding_america[col], errors='coerce').notna().sum() > 100:
                        feeding_col_main = col
                        print(f"  Found {fa_year} FA column: {feeding_col_main}")
                        break
                except:
                    continue 
    
    if not feeding_col_main:
        print(f"  ERROR: Could not find Food Insecurity Rate column for year {fa_year}.")
        print(f"  DEBUG: FA Columns: {list(feeding_america.columns)}")
        continue

    # ========================================================================
    # STEP 3/4: AGGREGATE AND MERGE
    # ========================================================================

    # Identify Tract ID column (in Food Atlas data)
    id_col = 'CensusTract' if 'CensusTract' in food_atlas.columns else ('GEOID' if 'GEOID' in food_atlas.columns else None)
    if not id_col:
        print(f"  ERROR: Cannot find tract identifier column in Food Atlas {atlas_year}. Skipping.")
        print(f"  DEBUG: Atlas Columns: {list(food_atlas.columns)}")
        continue
    
    # Extract county FIPS code (first 5 digits)
    food_atlas['County_FIPS'] = food_atlas[id_col].astype(str).str[:5]

    agg_dict = {
        col: ('mean' if any(keyword in col for keyword in ['Rate', 'Tracts', 'Flag']) else 'sum')
        for col in available_fa_cols.keys()
    }

    food_atlas_county = food_atlas.groupby('County_FIPS').agg(agg_dict).reset_index()

    # Clean FIPS codes
    food_atlas_county['County_FIPS'] = food_atlas_county['County_FIPS'].astype(str).str.zfill(5)
    feeding_america[fips_col] = feeding_america[fips_col].astype(str).str.zfill(5)

    merged = pd.merge(
        food_atlas_county,
        feeding_america,
        left_on='County_FIPS',
        right_on=fips_col,
        how='inner'
    )
    
    print(f"  Merged {len(merged):,} counties for comparison.")
    
    generate_correlation_plots(merged, feeding_col_main, atlas_year, fa_year)

    # ========================================================================
    # STEP 5: CALCULATE CORRELATIONS FOR YEAR PAIR
    # ========================================================================

    comparisons = [
        ('PovertyRate', feeding_col_main, 'Poverty Rate vs Food Insecurity'),
        ('LILATracts_1And10', feeding_col_main, 'Low Income/Access vs Food Insecurity'),
        ('LAPOP1_10', feeding_col_main, 'Low Access Population vs Food Insecurity'),
        ('TractSNAP', feeding_col_main, 'SNAP Recipients vs Food Insecurity'),
    ]

    for fa_col, feeding_col, description in comparisons:
        if fa_col in merged.columns and feeding_col in merged.columns:
            x = pd.to_numeric(merged[fa_col], errors='coerce')
            y = pd.to_numeric(merged[feeding_col], errors='coerce')
            mask = ~(x.isna() | y.isna())
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) >= 10:
                try:
                    pearson_r, pearson_p = pearsonr(x_clean, y_clean)
                    
                    results_multi_year.append({
                        'Atlas_Year': atlas_year,
                        'FA_Year': fa_year,
                        'Comparison': description,
                        'Pearson_r': pearson_r,
                        'Pearson_p': pearson_p,
                        'N': len(x_clean),
                    })
                except Exception as e:
                    print(f"    Error calculating correlation for {description}: {e}")

# ============================================================================
# FINAL STEP: REPORT GENERATION (OUTSIDE THE LOOP)
# ============================================================================
print("\n[FINAL STEP] Generating Multi-Year Report...")

if not results_multi_year:
    print("FATAL: No results generated. Check file paths and column names.")
    exit()

results_df = pd.DataFrame(results_multi_year)
results_df['Significant'] = results_df['Pearson_p'].apply(lambda p: 'Yes' if p < 0.05 else 'No')
results_df = results_df.sort_values(['Atlas_Year', 'Pearson_r'], ascending=[True, False], key=abs)

pivot_df = results_df.pivot_table(
    index='Comparison', 
    columns=['Atlas_Year', 'FA_Year'], 
    values='Pearson_r',
    aggfunc='mean' 
)

report_content = (
    "="*80 + "\n" +
    "DIRECT YEAR CORRELATION SUMMARY (Pearson r)\n" +
    f"Atlas Years: {list(set(results_df['Atlas_Year']))}, FA Years: {list(set(results_df['FA_Year']))}\n" +
    "="*80 + "\n\n" +
    "CORRELATION CONSISTENCY ACROSS TIME (Direct Match):\n" +
    "This table shows how the relationship (Pearson r) holds when comparing\n" +
    "Food Atlas structural data against Feeding America outcomes for the exact same year.\n\n" +
    pivot_df.to_string(float_format="%.3f") + "\n\n" +
    "="*80 + "\n" +
    "Detailed Results:\n" +
    results_df[['Atlas_Year', 'FA_Year', 'Comparison', 'Pearson_r', 'Significant', 'N']].to_string(index=False, float_format="%.3f") +
    "\n\nAnalysis completed by Kimberly\n" +
    "="*80 + "\n"
)

try:
    with open(output_dir / 'direct_year_correlation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Report saved: {output_dir.name}/direct_year_correlation_report.txt")
except Exception as e:
    print(f"FATAL ERROR: Could not write report file: {e}")

print("\n--------------------------------------------------------------------")
print("ANALYSIS COMPLETE!")
print("--------------------------------------------------------------------")
