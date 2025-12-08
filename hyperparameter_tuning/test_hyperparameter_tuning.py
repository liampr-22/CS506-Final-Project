"""
Test Suite for Hyperparameter Tuning
Tests key functionality without running full hyperparameter tuning
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, silhouette_score
import os

# Test 1: Check if data file exists
def test_data_file_exists():
    """Test that the required data file exists"""
    assert os.path.exists('atlas_cdc_combined_county_level.csv'), \
        "Data file 'atlas_cdc_combined_county_level.csv' not found"

# Test 2: Load and validate data
def test_load_data():
    """Test that data loads correctly and has expected structure"""
    data = pd.read_csv('atlas_cdc_combined_county_level.csv')
    
    # Check data is not empty
    assert len(data) > 0, "Data file is empty"
    
    # Check required columns exist
    required_cols = ['MeasureId', 'Data_Value', 'StateDesc', 'Survey_Year']
    for col in required_cols:
        assert col in data.columns, f"Required column '{col}' not found in data"
    
    # Check DIABETES measure exists
    assert 'DIABETES' in data['MeasureId'].values, \
        "DIABETES measure not found in data"

# Test 3: Data filtering
def test_data_filtering():
    """Test that filtering to DIABETES works correctly"""
    data = pd.read_csv('atlas_cdc_combined_county_level.csv')
    diabetes_data = data[data['MeasureId'] == 'DIABETES'].copy()
    
    # Check filtered data is not empty
    assert len(diabetes_data) > 0, "No DIABETES data found after filtering"
    
    # Check all rows are DIABETES
    assert (diabetes_data['MeasureId'] == 'DIABETES').all(), \
        "Filtered data contains non-DIABETES rows"
    
    # Check Data_Value column has valid data
    assert diabetes_data['Data_Value'].notna().any(), \
        "No valid Data_Value entries found"

# Test 4: Feature creation
def test_feature_creation():
    """Test that features can be created from data"""
    data = pd.read_csv('atlas_cdc_combined_county_level.csv')
    diabetes_data = data[data['MeasureId'] == 'DIABETES'].copy()
    
    # Create features
    year_feature = diabetes_data[['Survey_Year']].copy()
    state_dummies = pd.get_dummies(diabetes_data['StateDesc'], prefix='State')
    
    # Check features were created
    assert year_feature.shape[0] > 0, "Year feature is empty"
    assert state_dummies.shape[1] > 0, "State features not created"
    
    # Check feature dimensions match
    assert year_feature.shape[0] == state_dummies.shape[0], \
        "Feature dimensions don't match"

# Test 5: Train-test split
def test_train_test_split():
    """Test that train-test split works"""
    # Create sample data
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Check split ratios
    assert len(X_train) == 80, "Train set size incorrect"
    assert len(X_test) == 20, "Test set size incorrect"
    assert len(y_train) == 80, "Train target size incorrect"
    assert len(y_test) == 20, "Test target size incorrect"

# Test 6: Standardization
def test_standardization():
    """Test that feature standardization works"""
    # Create sample data
    X = np.random.randn(100, 10) * 10 + 5
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check standardization (mean ≈ 0, std ≈ 1)
    assert np.abs(X_scaled.mean()) < 0.1, "Scaled data mean not near 0"
    assert np.abs(X_scaled.std() - 1.0) < 0.1, "Scaled data std not near 1"

# Test 7: PCA
def test_pca():
    """Test that PCA dimensionality reduction works"""
    # Create sample data
    X = np.random.randn(100, 50)
    
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Check dimensions were reduced
    assert X_pca.shape[1] < X.shape[1], "PCA did not reduce dimensions"
    
    # Check variance explained is close to 95%
    variance_explained = sum(pca.explained_variance_ratio_)
    assert variance_explained >= 0.94, "PCA retained less than 94% variance"

# Test 8: Ridge regression
def test_ridge_regression():
    """Test that Ridge regression can be trained and predicts"""
    # Create sample data
    X_train = np.random.randn(80, 10)
    X_test = np.random.randn(20, 10)
    y_train = X_train[:, 0] * 2 + np.random.randn(80) * 0.1
    y_test = X_test[:, 0] * 2 + np.random.randn(20) * 0.1
    
    # Train Ridge model
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    
    # Make predictions
    y_pred = ridge.predict(X_test)
    
    # Check predictions have correct shape
    assert y_pred.shape == y_test.shape, "Prediction shape mismatch"
    
    # Check R² is reasonable (should be high for this simple data)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.5, f"Ridge R² too low: {r2}"

# Test 9: K-Means clustering
def test_kmeans_clustering():
    """Test that K-Means clustering works"""
    # Create sample data with clear clusters
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(50, 10) + 5,
        np.random.randn(50, 10) - 5
    ])
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Check labels
    assert len(labels) == 100, "Wrong number of cluster labels"
    assert len(set(labels)) == 2, "Wrong number of clusters"
    
    # Check silhouette score is reasonable
    score = silhouette_score(X, labels)
    assert score > 0.3, f"Silhouette score too low: {score}"

# Test 10: Results files can be created
def test_results_files():
    """Test that result DataFrames can be created and saved"""
    # Create sample results
    results_df = pd.DataFrame({
        'Model': ['Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting'],
        'CV Score (R²)': [0.5, 0.48, 0.45, 0.46],
        'Test Score (R²)': [0.52, 0.49, 0.43, 0.44]
    })
    
    clustering_df = pd.DataFrame({
        'Model': ['K-Means', 'DBSCAN', 'Agglomerative'],
        'Best Parameters': ['k=5', "{'eps': 0.5}", 'k=4, ward'],
        'Silhouette Score': [0.45, 0.78, 0.52]
    })
    
    # Save to CSV
    results_df.to_csv('test_regression_results.csv', index=False)
    clustering_df.to_csv('test_clustering_results.csv', index=False)
    
    # Check files exist
    assert os.path.exists('test_regression_results.csv'), \
        "Regression results CSV not created"
    assert os.path.exists('test_clustering_results.csv'), \
        "Clustering results CSV not created"
    
    # Clean up test files
    os.remove('test_regression_results.csv')
    os.remove('test_clustering_results.csv')

# Run all tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])