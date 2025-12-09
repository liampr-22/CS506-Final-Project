# Hyperparameter Tuning  
**Author:** Sharon Kimberly Tope  
**Component:** Hyperparameter tuning for Link 2 (Food Insecurity → Health Outcomes)

---

## Overview

Performs automated **hyperparameter tuning** for:
- **Regression models:** Ridge, Lasso, Random Forest, Gradient Boosting  
- **Clustering models:** K-Means, DBSCAN, Agglomerative  

The objective is to improve the **midterm baseline Random Forest R² = 0.64** and identify the most effective models for predicting chronic disease prevalence from food insecurity and socioeconomic data.

**Data:** Link 2 dataset (Map the Meal Gap 2017 × CDC PLACES 2017)  
**Methods:** `GridSearchCV` and `RandomizedSearchCV` with 5-fold cross-validation  

---

## Results

### Regression Models

| Model | CV R² | Test R² |
|-------|--------|----------|
| **Ridge** | 0.936884 | 0.935506 |
| **Lasso** | 0.936861 | 0.935541 |
| **Random Forest** | 0.910819 | 0.918246 |
| **Gradient Boosting** | **0.949952** | **0.950816** |

**Best Regression Model:** **Gradient Boosting**  
**Improvement Over Midterm Baseline (0.64 → 0.95):** ~48.6% increase

---

### Clustering Models

| Model | Best Parameters | Silhouette Score |
|-------|-----------------|------------------|
| K-Means | `k=2` | 0.256249 |
| DBSCAN | `eps=2.0`, `min_samples=3` | -0.167028 |
| **Agglomerative** | `k=2`, `linkage='complete'` | **0.706458** |

**Best Clustering Model:** **Agglomerative Clustering (Silhouette = 0.706)**  

---

## Key Findings

1. **Gradient Boosting achieved the strongest predictive performance**, reaching an R² of 0.95 and outperforming all other models by a significant margin. This represents a 48.6% improvement over the midterm baseline.

2. **Agglomerative clustering revealed meaningful county groupings**, indicating strong geographic and socioeconomic structure in health outcomes. The silhouette score of 0.706 demonstrates well-separated clusters.

3. **Food insecurity + socioeconomic variables carry strong predictive signal**, validating the Link 2 causal pathway: *Low Access → Food Insecurity → Health Outcomes.* The model performance demonstrates that these factors are highly predictive of chronic disease prevalence.

4. **Hyperparameter tuning provided major gains** - Systematic optimization using GridSearchCV and RandomizedSearchCV explored parameter spaces to identify optimal configurations, boosting R² from 0.64 (midterm baseline) to 0.95.

---

## Summary of Results

Hyperparameter tuning significantly improved the predictive performance of all regression models, with Gradient Boosting emerging as the best performer (Test R² = 0.9508), a 48.6% improvement over the midterm baseline Random Forest model (R² = 0.64). Ridge and Lasso also performed strongly (R² ≈ 0.936), while the tuned Random Forest reached R² ≈ 0.918.

For clustering, Agglomerative Clustering achieved the highest silhouette score (0.706), revealing clear county-level groupings in health outcomes, while K-Means provided weaker separation and DBSCAN failed to identify meaningful density-based clusters.

Overall, the results show that food insecurity and socioeconomic factors strongly predict chronic disease prevalence, and that tuning model hyperparameters yields substantial accuracy gains across the board.
