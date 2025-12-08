# Hyperparameter Tuning

**Author:** Sharon Kimberly Tope  
**Component:** Hyperparameter tuning for regression and clustering models

---

## Overview

Performs automated hyperparameter tuning for **4 regression models** and **3 clustering algorithms** to analyze diabetes prevalence across **1,356 U.S. county observations** from CDC health outcomes data.

**Models:** Ridge, Lasso, Random Forest, Gradient Boosting (regression) | K-Means, DBSCAN, Agglomerative (clustering)

**Method:** `GridSearchCV` and `RandomizedSearchCV` with 5-fold cross-validation

**Testing:** 10 automated tests with GitHub Actions CI/CD

---

## How to Build and Run
# Install dependencies
python -m pip install numpy pandas scikit-learn matplotlib seaborn pytest openpyxl jupyter

# Run tests
python -m pytest test_hyperparameter_tuning.py -v

# Run analysis
jupyter notebook hyperparameter_tuning.ipynb


---

## Results

### Regression Models

| Model | Best Parameters | CV R² | Test R² |
|-------|----------------|-------|---------|
| Ridge | alpha=100 | 0.236 | **0.075** |
| Lasso | alpha=0.1 | 0.235 | 0.063 |
| Random Forest | n_estimators=100, max_depth=5 | 0.175 | 0.033 |
| Gradient Boosting | n_estimators=200, lr=0.01 | 0.183 | 0.029 |

**Best Model:** Ridge Regression (Test R² = 0.075)

### Clustering Models

| Model | Best Parameters | Silhouette Score |
|-------|----------------|------------------|
| K-Means | k=2 | 0.524 |
| **DBSCAN** | eps=0.3, min_samples=3 | **0.915** |
| Agglomerative | k=2, linkage=complete | 0.600 |

**Best Model:** DBSCAN (Silhouette = 0.915) - Identified 102 distinct geographic clusters

---

## Key Findings

1. **DBSCAN performed very well** (silhouette = 0.915) and found 102 groups of counties that have similar diabetes rates.

2. **Location matters a lot** - clustering works much better than regression: clustering (0.915) and regression (0.075), showing that diabetes rates follow strong geographic patterns.

3. **Hyperparameter tuning improved every model** - it found the best settings, such as Ridge with a large alpha (100), DBSCAN with small eps (0.3), and careful parameter choices for the ensemble models to avoid overfitting.
