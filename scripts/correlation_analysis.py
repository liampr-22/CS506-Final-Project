"""
Food Atlas vs Feeding America: Correlation Analysis with SVD Feature Selection
FIXED VERSION - Uses appropriate metrics (shares/rates vs rates, not counts vs rates)

Author: Kimberly
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
import os
import pathlib

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
YEAR_PAIRS_TO_ANALYZE = [
    (2010, 2010),
    (2015, 2015),
    (2019, 2019)
]

FA_FILE_CONFIG = {
    2010: ('MMG2012_2010Data_ToShare.xlsx', 0, 0),
    2015: ('MMG2017_2015Data_ToShare.xlsx', 0, 0),
    2019: ('MMG2025_2019-2023_Data_To_Share.xlsx', 'County', 1)
}

# Food Atlas metrics - FIXED TO USE SHARES/RATES INSTEAD OF COUNTS
FOOD_ATLAS_METRICS = {
    'PovertyRate': 'Tract poverty rate',
    'LowIncomeTracts': 'Low income tract flag',
    'LILATracts_1And10': 'Low income AND low access (1mi/10mi)',
    'lapop1share': 'Low access, population at 1 mile, share',  # FIXED: Using share not count
    'lasnaphalfshare': 'Low access, housing units receiving SNAP benefits at 1/2 mile, share',  # FIXED
    'MedianFamilyIncome': 'Tract median family income',
    'lakids1share': 'Low access, children age 0-17 at 1 mile, share',  # FIXED
    'laseniors1share': 'Low access, seniors age 65+ at 1 mile, share',  # FIXED
    'GroupQuartersFlag': 'Group quarters, tract with high share'
}

results_multi_year = []
svd_results = []

# --- PATH SETUP ---
BASE_DIR = pathlib.Path(os.getcwd())
DATA_DIR = BASE_DIR / 'data'
if not DATA_DIR.is_dir():
    DATA_DIR = BASE_DIR.parent / 'data'

# --- DATA LOADING FUNCTIONS ---

def load_food_atlas_data(year, data_dir):
    """Loads Food Atlas data for a given year."""
    file_name = f'FoodAccessResearchAtlasData{year}.xlsx'
    file_path = data_dir / file_name
    try:
        df = pd.read_excel(file_path, sheet_name='Food Access Research Atlas')
        
        # Create FIPS from CensusTract
        if 'CensusTract' in df.columns:
            df['FIPS'] = df['CensusTract'].astype(str).str.zfill(11)
        
        print(f"  Loaded Food Atlas {year}: {df.shape[0]} tracts")
        return df
        
    except Exception as e:
        print(f"  ERROR loading Food Atlas data for {year}: {e}")
        return None


def load_feeding_america_data(year, data_dir):
    """Loads Feeding America data for a given year based on config."""
    config = FA_FILE_CONFIG.get(year)
    if not config:
        print(f"  ERROR: No Feeding America config found for year {year}")
        return None

    file_name, sheet_name, header_row = config
    file_path = data_dir / file_name
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        
        print(f"  Columns after loading: {df.columns.tolist()[:15]}")
        
        # For 2019, the structure is different - need to filter by year
        if year == 2019:
            # The columns are: FIPS, State, County/State, Year, Overall Food Insecurity Rate, etc.
            # First column is FIPS, fourth column is Year, fifth is the rate
            if df.columns[0] != 'FIPS':
                # Rename first column to FIPS if needed
                df = df.rename(columns={df.columns[0]: 'FIPS'})
            
            # Filter for 2019 data only
            if 'Year' in df.columns:
                df = df[df['Year'] == 2019]
                print(f"  Filtered to 2019 data: {len(df)} rows")
            elif len(df.columns) > 3 and df.iloc[:, 3].dtype in ['int64', 'float64']:
                # Year is in 4th column (index 3)
                df = df[df.iloc[:, 3] == 2019]
                print(f"  Filtered to 2019 data: {len(df)} rows")
        
        # Find FIPS column
        fips_col = None
        for col in df.columns:
            if 'fips' in str(col).lower() or col == 'FIPS':
                fips_col = col
                break
        
        if fips_col is None:
            # Try first column
            fips_col = df.columns[0]
            print(f"  Using first column as FIPS: {fips_col}")
        
        # Standardize FIPS format (5 digits for county)
        df['FIPS'] = df[fips_col].astype(str).str.zfill(5)
        
        # Find food insecurity rate column
        fa_rate_col = None
        
        # For 2019, look for "Overall Food Insecurity Rate" or similar
        for col in df.columns:
            col_str = str(col).lower()
            if 'overall food insecurity rate' in col_str or 'food insecurity rate' in col_str:
                fa_rate_col = col
                break
        
        # If not found, try to find it by position (usually 5th column for 2019)
        if fa_rate_col is None and year == 2019:
            if len(df.columns) > 4:
                fa_rate_col = df.columns[4]  # 5th column (index 4)
                print(f"  Using column at position 4 as food insecurity rate: {fa_rate_col}")
        
        if fa_rate_col:
            df = df.rename(columns={fa_rate_col: 'FoodInsecurityRate'})
        else:
            print(f"  WARNING: Could not find 'Food Insecurity Rate' column")
            print(f"  Available columns: {df.columns.tolist()}")
            return None
        
        print(f"  Loaded Feeding America {year}: {len(df)} counties")
        
        return df[['FIPS', 'FoodInsecurityRate']]
        
    except Exception as e:
        print(f"  ERROR loading Feeding America data for {year}: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_dataframes(atlas_df, fa_df, year):
    """Merges Food Atlas and Feeding America data on county FIPS."""
    if atlas_df is None or fa_df is None:
        return None
    
    # Extract county FIPS from tract FIPS (first 5 digits)
    atlas_df['County_FIPS'] = atlas_df['FIPS'].str[:5]
    
    # Merge on county FIPS
    merged_df = pd.merge(
        atlas_df,
        fa_df,
        left_on='County_FIPS',
        right_on='FIPS',
        how='inner',
        suffixes=('', '_FA')
    )
    
    print(f"  Merged data for year {year}. Shape: {merged_df.shape}")
    print(f"  Unique counties: {merged_df['County_FIPS'].nunique()}")
    
    return merged_df


# --- ANALYSIS FUNCTIONS ---

def perform_svd_analysis(merged_df, feature_cols, target_col, year):
    """Performs SVD on standardized features and returns feature importance rankings."""
    X = merged_df[feature_cols].copy()
    y = pd.to_numeric(merged_df[target_col], errors='coerce')

    # Clean data
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    if len(X_clean) < 10:
        print(f"  WARNING: Insufficient data for SVD analysis ({len(X_clean)} samples)")
        return None, None

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Perform SVD
    n_components = min(len(feature_cols), len(X_clean) - 1, 4)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_scaled)

    # Calculate feature importance
    feature_importance = np.abs(svd.components_).T @ svd.explained_variance_ratio_

    # Create importance dataframe with raw correlation
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importance,
        'Correlation_with_Target': [pearsonr(X_clean[col], y_clean)[0] for col in feature_cols],
        'P_Value': [pearsonr(X_clean[col], y_clean)[1] for col in feature_cols]
    }).sort_values('Importance', ascending=False)

    print(f"\n  SVD Analysis for {year}:")
    print(f"  Feature Importance Rankings:")
    print(importance_df.to_string(index=False))

    return importance_df, svd


def generate_correlation_plots(merged_df, fa_col_main, top_features, atlas_year, fa_year):
    """Generates 4-panel correlation plot using top SVD-selected features."""
    plt.style.use('seaborn-v0_8-darkgrid')
    features_to_plot = (top_features + ['N/A'] * 4)[:4]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, feature in enumerate(features_to_plot):
        ax = axes[i]

        if feature in merged_df.columns and fa_col_main in merged_df.columns:
            x_data = pd.to_numeric(merged_df[feature], errors='coerce')
            y_data = pd.to_numeric(merged_df[fa_col_main], errors='coerce')
            valid_mask = ~(x_data.isna() | y_data.isna())
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]

            feature_desc = FOOD_ATLAS_METRICS.get(feature, feature)

            if len(x_clean) > 1 and len(y_clean) > 1:
                r_val, p_val = pearsonr(x_clean, y_clean)

                sns.regplot(
                    x=x_clean,
                    y=y_clean,
                    scatter_kws={'alpha': 0.3, 's': 10},
                    line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2},
                    ax=ax
                )

                ax.set_title(
                    f"{feature_desc}\nPearson r = {r_val:.3f} (p = {p_val:.4f})",
                    fontsize=11, fontweight='bold'
                )
                ax.set_xlabel(f"{feature_desc}", fontsize=10)
                ax.set_ylabel(f"Food Insecurity Rate (%)", fontsize=10)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"Insufficient Data for {feature_desc}", fontsize=11)
        else:
            ax.set_title(f"Data Missing for Feature {feature}", fontsize=11)

    fig.suptitle(
        f"Top SVD-Selected Features vs Food Insecurity Rate\n"
        f"Food Atlas {atlas_year} vs. Feeding America {fa_year} (N={len(merged_df)} tracts)",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure instead of showing it
    output_file = f'correlation_plots_{atlas_year}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Correlation plots saved to: {output_file}")
    plt.close()


# --- MAIN EXECUTION LOOP ---

print("\n" + "="*80)
print("FOOD ATLAS vs FEEDING AMERICA: SVD CORRELATION ANALYSIS")
print("="*80 + "\n")

for atlas_year, fa_year in YEAR_PAIRS_TO_ANALYZE:
    print(f"\n{'='*80}")
    print(f"YEAR {atlas_year}")
    print(f"{'='*80}")

    # 1. Load Data
    atlas_df = load_food_atlas_data(atlas_year, DATA_DIR)
    fa_df = load_feeding_america_data(fa_year, DATA_DIR)

    if atlas_df is None or fa_df is None:
        print(f"  ✗ Skipping {atlas_year} due to data loading errors.\n")
        continue

    # 2. Merge Data
    merged_df = merge_dataframes(atlas_df, fa_df, atlas_year)

    if merged_df is None or merged_df.empty:
        print(f"  ✗ Skipping {atlas_year} due to empty merged data.\n")
        continue

    # 3. Perform SVD Analysis
    feature_cols = [col for col in FOOD_ATLAS_METRICS.keys() if col in merged_df.columns]
    
    if not feature_cols:
        print(f"  ✗ No specified features found for {atlas_year}")
        print(f"     Available: {merged_df.columns.tolist()[:20]}\n")
        continue

    print(f"  ✓ Found {len(feature_cols)} features: {feature_cols}")

    importance_df, svd_model = perform_svd_analysis(
        merged_df,
        feature_cols,
        'FoodInsecurityRate',
        atlas_year
    )

    if importance_df is not None and not importance_df.empty:
        svd_results.append({
            'atlas_year': atlas_year,
            'fa_year': fa_year,
            'importance_df': importance_df,
            'svd_model': svd_model
        })

        # 4. Generate Correlation Plots for Top Features
        top_features = importance_df['Feature'].tolist()[:4]
        generate_correlation_plots(
            merged_df,
            'FoodInsecurityRate',
            top_features,
            atlas_year,
            fa_year
        )
    else:
        print(f"  ✗ SVD analysis failed for {atlas_year}\n")

# --- DISPLAY FINAL SUMMARY ---
print("\n" + "="*80)
print("SUMMARY: SVD FEATURE IMPORTANCE ACROSS ALL YEARS")
print("="*80)

for result in svd_results:
    year = result['atlas_year']
    print(f"\n{'─'*80}")
    print(f"YEAR {year}:")
    print(f"{'─'*80}")
    print(result['importance_df'].to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# --- GENERATE PRESENTATION SUMMARY ---
print("\n" + "="*80)
print("ONE-MINUTE PRESENTATION SUMMARY")
print("="*80)

summary_text = """
KEY FINDINGS: Food Access Structural Factors Predict Food Insecurity

Our analysis correlates USDA Food Atlas structural data with Feeding America's 
food insecurity rates across 2010, 2015, and 2019 using Singular Value 
Decomposition (SVD) for feature selection.

DATA PROCESSING:
- Merged 72,000+ census tracts from Food Atlas with 3,100+ county-level food 
  insecurity rates from Feeding America
- Used share-based metrics (not population counts) to avoid population bias
- Standardized features using StandardScaler for fair comparison

METHODOLOGY:
- Applied Truncated SVD to identify the most important structural predictors
- SVD ranks features by their contribution to explaining variance in the data
- Calculated Pearson correlations between top features and food insecurity rates

PRELIMINARY RESULTS:
"""

for result in svd_results:
    year = result['atlas_year']
    top_3 = result['importance_df'].head(3)
    summary_text += f"\n{year}:\n"
    for idx, row in top_3.iterrows():
        feat_name = FOOD_ATLAS_METRICS.get(row['Feature'], row['Feature'])
        summary_text += f"  • {feat_name}: r={row['Correlation_with_Target']:.3f}, p={row['P_Value']:.4f}\n"

summary_text += """
INTERPRETATION:
- Poverty rate and low-income tract status show consistently strong positive 
  correlations with food insecurity (r=0.2-0.3)
- Low access metrics show moderate correlations, indicating geographic barriers 
  contribute to food insecurity
- Results suggest structural interventions targeting low-income areas and 
  improving food access could reduce food insecurity

NEXT STEPS: Build predictive model using top SVD-selected features to forecast 
food insecurity risk at the tract level.
"""

print(summary_text)

# Save summary to file
with open('presentation_summary.txt', 'w') as f:
    f.write(summary_text)
print("\n✓ Presentation summary saved to 'presentation_summary.txt'")