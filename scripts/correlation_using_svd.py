"""
Food Atlas vs Feeding America: Correlation Analysis with SVD Feature Selection
Compares Food Atlas structural data against Feeding America outcomes.
Uses SVD to identify the most important features for predicting food insecurity.

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

# Food Atlas metrics for feature selection
FOOD_ATLAS_METRICS = {
    'PovertyRate': 'Tract poverty rate',
    'LowIncomeTracts': 'Low income tract flag',
    'LILATracts_1And10': 'Low income AND low access (1mi/10mi)',
    'LAPOP1_10': 'Population with low access (1mi/10mi)',
    'TractSNAP': 'Housing units receiving SNAP',
    'MedianFamilyIncome': 'Median family income',
    'lakids1': 'Kids with low access at 1 mile',
    'laseniors1': 'Seniors with low access at 1 mile',
    'GroupQuartersFlag': 'Group quarters flag'
}

results_multi_year = []
svd_results = []

# --- PATH SETUP ---
BASE_DIR = pathlib.Path(os.getcwd()) 
DATA_DIR = BASE_DIR / 'data'
if not DATA_DIR.is_dir():
    DATA_DIR = BASE_DIR.parent / 'data' 


def perform_svd_analysis(merged_df, feature_cols, target_col, year):
    """
    Performs SVD on features and returns feature importance rankings.
    """
    # Prepare feature matrix
    X = merged_df[feature_cols].copy()
    y = pd.to_numeric(merged_df[target_col], errors='coerce')
    
    # Remove rows with missing values
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
    
    # Calculate feature importance based on SVD components
    # Weighted by explained variance ratio
    feature_importance = np.abs(svd.components_).T @ svd.explained_variance_ratio_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importance,
        'Correlation_with_Target': [pearsonr(X_clean[col], y_clean)[0] for col in feature_cols]
    }).sort_values('Importance', ascending=False)
    
    # Calculate correlations of SVD components with target
    component_correlations = []
    for i in range(n_components):
        corr, _ = pearsonr(X_svd[:, i], y_clean)
        component_correlations.append(corr)
    
    print(f"\n  SVD Analysis for {year}:")
    print(f"  Explained variance ratio: {svd.explained_variance_ratio_}")
    print(f"  Component correlations with target: {component_correlations}")
    print(f"\n  Feature Importance Rankings:")
    print(importance_df.to_string(index=False))
    
    return importance_df, svd


def plot_svd_feature_importance(importance_df, year):
    """Creates visualization of feature importance from SVD."""
    plt.figure(figsize=(10, 6))
    
    colors = ['green' if corr > 0 else 'red' for corr in importance_df['Correlation_with_Target']]
    
    plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors, alpha=0.7)
    plt.xlabel('SVD-based Feature Importance', fontsize=12)
    plt.ylabel('Food Atlas Features', fontsize=12)
    plt.title(f'Feature Importance for Predicting Food Insecurity ({year})\n'
              f'Green = Positive correlation, Red = Negative correlation', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = BASE_DIR.parent / 'outputs/svd_analysis'
    if not output_path.exists():
        os.makedirs(output_path)
    filename = output_path / f'feature_importance_{year}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Feature importance plot saved: {filename.relative_to(BASE_DIR.parent)}")


def generate_correlation_plots(merged_df, fa_col_main, top_features, atlas_year, fa_year):
    """Generates 4-panel correlation plot using top SVD-selected features."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Use top 4 features from SVD
    features_to_plot = top_features[:4] if len(top_features) >= 4 else top_features
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        
        if feature in merged_df.columns and fa_col_main in merged_df.columns:
            x_data = pd.to_numeric(merged_df[feature], errors='coerce')
            y_data = pd.to_numeric(merged_df[fa_col_main], errors='coerce')
            
            # Remove NaN values
            valid_mask = ~(x_data.isna() | y_data.isna())
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]
            
            if len(x_clean) > 1 and len(y_clean) > 1:
                r_val, p_val = pearsonr(x_clean, y_clean)

                sns.regplot(
                    x=x_clean, 
                    y=y_clean, 
                    scatter_kws={'alpha': 0.3, 's': 10}, 
                    line_kws={'color': 'red', 'linestyle': '--'},
                    ax=ax
                )
                
                ax.set_title(
                    f"{feature}\nPearson r = {r_val:.3f} (p = {p_val:.4f})", 
                    fontsize=12, fontweight='bold'
                )
                ax.set_xlabel(f"Food Atlas: {feature}", fontsize=10)
                ax.set_ylabel(f"Food Insecurity Rate", fontsize=10)
            else:
                ax.set_title(f"Insufficient Data for {feature}", fontsize=12)
        else:
            ax.set_title(f"Data Missing for {feature}", fontsize=12)

    fig.suptitle(
        f"Top SVD-Selected Features vs Food Insecurity\n"
        f"Atlas {atlas_year} vs. FA {fa_year} (N={len(merged_df)})", 
        fontsize=16, 
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = BASE_DIR.parent / 'outputs/correlation_plots'
    if not output_path.exists(): 
        os.makedirs(output_path)
    filename = output_path / f'atlas_{atlas_year}_vs_fa_{fa_year}_svd_selected.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Correlation plot saved: {filename.relative_to(BASE_DIR.parent)}")


# --- START SCRIPT EXECUTION ---
print("\n" + "="*80)
print("FOOD INSECURITY CORRELATION ANALYSIS WITH SVD FEATURE SELECTION")
print("="*80 + "\n")

# Create output directories
output_dir = BASE_DIR.parent / 'outputs'
plot_dir = output_dir / 'correlation_plots'
svd_dir = output_dir / 'svd_analysis'
for dir_path in [output_dir, plot_dir, svd_dir]:
    if not dir_path.exists():
        os.makedirs(dir_path)


# ========================================================================
# PRIMARY LOOP: Iterate through year pairs
# ========================================================================

for atlas_year, fa_year in YEAR_PAIRS_TO_ANALYZE:
    print(f"\n{'='*80}")
    print(f"PROCESSING: Atlas {atlas_year} vs. FA {fa_year}")
    print(f"{'='*80}")

    atlas_filename = f'FoodAccessResearchAtlasData{atlas_year}.xlsx'
    fa_filename, fa_sheet, skiprows_fa = FA_FILE_CONFIG.get(fa_year)
    
    atlas_path = DATA_DIR / atlas_filename 
    fa_path = DATA_DIR / fa_filename

    # --- STEP 1: LOAD DATA ---
    try:
        if not atlas_path.exists():
            raise FileNotFoundError(f"Atlas file '{atlas_filename}' not found.")
        
        food_atlas = pd.read_excel(atlas_path, sheet_name='Food Access Research Atlas')
        print(f"✓ Loaded Food Atlas {atlas_year}: {atlas_path.name}")

        if not fa_path.exists():
            raise FileNotFoundError(f"FA file '{fa_filename}' not found.")
        
        feeding_america = pd.read_excel(fa_path, sheet_name=fa_sheet, skiprows=skiprows_fa)
        print(f"✓ Loaded FA data: {fa_path.name}, sheet: {fa_sheet}")
        
        # Filter 2019 data by Year column
        if fa_year == 2019 and 'Year' in feeding_america.columns:
            before_filter = len(feeding_america)
            feeding_america = feeding_america[feeding_america['Year'] == 2019].copy()
            print(f"✓ Filtered to year {fa_year}: {before_filter} → {len(feeding_america)} rows")
        
        if food_atlas.empty or feeding_america.empty:
            print(f"✗ ERROR: No data after loading. Skipping.")
            continue

    except Exception as e:
        print(f"✗ ERROR loading data: {e}")
        continue

    # --- STEP 2: IDENTIFY COLUMNS ---
    
    # Find available features
    available_features = [col for col in FOOD_ATLAS_METRICS.keys() if col in food_atlas.columns]
    print(f"✓ Found {len(available_features)} features: {available_features}")
    
    # Find FIPS column
    fips_col = next((col for col in feeding_america.columns 
                     if 'fips' in str(col).lower() or 'county' in str(col).lower()), None)
    if not fips_col:
        print(f"✗ ERROR: Cannot find FIPS column. Skipping.")
        continue

    # Find food insecurity column
    feeding_col_main = None
    if fa_year == 2019:
        for col in feeding_america.columns:
            if 'overall food insecurity rate' in str(col).lower():
                feeding_col_main = col
                break
    else:
        for col in feeding_america.columns:
            col_lower = str(col).lower()
            if str(fa_year) in col_lower and 'food insecurity rate' in col_lower:
                try:
                    if pd.to_numeric(feeding_america[col], errors='coerce').notna().sum() > 100:
                        feeding_col_main = col
                        break
                except:
                    continue
    
    if not feeding_col_main:
        print(f"✗ ERROR: Could not find Food Insecurity Rate column.")
        continue
    
    print(f"✓ Target variable: {feeding_col_main}")

    # --- STEP 3: AGGREGATE TO COUNTY LEVEL ---
    
    id_col = 'CensusTract' if 'CensusTract' in food_atlas.columns else 'GEOID'
    food_atlas['County_FIPS'] = food_atlas[id_col].astype(str).str[:5]

    agg_dict = {
        col: ('mean' if any(kw in col for kw in ['Rate', 'Tracts', 'Flag', 'Income']) else 'sum')
        for col in available_features
    }

    food_atlas_county = food_atlas.groupby('County_FIPS').agg(agg_dict).reset_index()
    food_atlas_county['County_FIPS'] = food_atlas_county['County_FIPS'].astype(str).str.zfill(5)
    feeding_america[fips_col] = feeding_america[fips_col].astype(str).str.zfill(5)

    merged = pd.merge(
        food_atlas_county,
        feeding_america,
        left_on='County_FIPS',
        right_on=fips_col,
        how='inner'
    )
    
    print(f"✓ Merged dataset: {len(merged):,} counties")

    # --- STEP 4: SVD FEATURE SELECTION ---
    
    importance_df, svd_model = perform_svd_analysis(
        merged, 
        available_features, 
        feeding_col_main, 
        atlas_year
    )
    
    if importance_df is not None:
        # Store SVD results
        for idx, row in importance_df.iterrows():
            svd_results.append({
                'Year': atlas_year,
                'Feature': row['Feature'],
                'SVD_Importance': row['Importance'],
                'Correlation': row['Correlation_with_Target']
            })
        
        # Plot feature importance
        plot_svd_feature_importance(importance_df, atlas_year)
        
        # Get top features for correlation plots
        top_features = importance_df['Feature'].tolist()
        
        # Generate correlation plots with top features
        generate_correlation_plots(merged, feeding_col_main, top_features, atlas_year, fa_year)
        
        # Calculate correlations for top 4 features
        for feature in top_features[:4]:
            if feature in merged.columns:
                x = pd.to_numeric(merged[feature], errors='coerce')
                y = pd.to_numeric(merged[feeding_col_main], errors='coerce')
                mask = ~(x.isna() | y.isna())
                x_clean = x[mask]
                y_clean = y[mask]
                
                if len(x_clean) >= 10:
                    try:
                        pearson_r, pearson_p = pearsonr(x_clean, y_clean)
                        results_multi_year.append({
                            'Atlas_Year': atlas_year,
                            'FA_Year': fa_year,
                            'Feature': feature,
                            'Pearson_r': pearson_r,
                            'Pearson_p': pearson_p,
                            'N': len(x_clean),
                        })
                    except Exception as e:
                        print(f"    Error calculating correlation: {e}")

# ============================================================================
# FINAL REPORT GENERATION
# ============================================================================
print(f"\n{'='*80}")
print("GENERATING FINAL REPORTS")
print(f"{'='*80}")

if not results_multi_year or not svd_results:
    print("✗ FATAL: No results generated.")
    exit()

# Correlation results
results_df = pd.DataFrame(results_multi_year)
results_df['Significant'] = results_df['Pearson_p'].apply(lambda p: 'Yes' if p < 0.05 else 'No')
results_df = results_df.sort_values(['Atlas_Year', 'Pearson_r'], ascending=[True, False])

# SVD results
svd_df = pd.DataFrame(svd_results)
svd_summary = svd_df.pivot_table(
    index='Feature',
    columns='Year',
    values='SVD_Importance',
    aggfunc='mean'
)

# Generate comprehensive report
report_content = f"""
{'='*80}
FOOD INSECURITY CORRELATION ANALYSIS WITH SVD FEATURE SELECTION
{'='*80}

ANALYSIS OVERVIEW:
- Years analyzed: {sorted(set(results_df['Atlas_Year']))}
- Total counties analyzed: {results_df['N'].max():,}
- Features evaluated: {len(set(svd_df['Feature']))}

{'='*80}
SVD FEATURE IMPORTANCE SUMMARY
{'='*80}

Feature importance across years (higher = more predictive):

{svd_summary.to_string(float_format='%.4f')}

Top 3 Features by Year:
"""

for year in sorted(set(svd_df['Year'])):
    year_data = svd_df[svd_df['Year'] == year].nlargest(3, 'SVD_Importance')
    report_content += f"\n{year}:\n"
    for idx, row in year_data.iterrows():
        report_content += f"  {idx+1}. {row['Feature']}: {row['SVD_Importance']:.4f} (corr: {row['Correlation']:.3f})\n"

report_content += f"""
{'='*80}
CORRELATION RESULTS (TOP SVD-SELECTED FEATURES)
{'='*80}

{results_df[['Atlas_Year', 'FA_Year', 'Feature', 'Pearson_r', 'Significant', 'N']].to_string(index=False, float_format='%.3f')}

KEY FINDINGS:
- Features with highest predictive power identified via SVD
- Correlations shown are statistically significant (p < 0.05) unless marked 'No'
- SVD helps identify multicollinear features and reduce dimensionality

Analysis completed by Kimberly
{'='*80}
"""

# Save report
try:
    with open(output_dir / 'svd_correlation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✓ Report saved: {output_dir.name}/svd_correlation_report.txt")
except Exception as e:
    print(f"✗ ERROR writing report: {e}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}\n")
