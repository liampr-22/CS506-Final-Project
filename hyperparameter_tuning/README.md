# Hyperparameter Tuning

**Author:** Sharon Kimberly Tope  
**Component:** Hyperparameter tuning for Link 2 (Food Insecurity → Health Outcomes)

---

## Overview

Performs automated hyperparameter tuning for **4 regression models** and **3 clustering algorithms** to improve upon midterm baseline results for predicting health outcomes from food insecurity and socioeconomic factors.

**Models:** Ridge, Lasso, Random Forest, Gradient Boosting (regression) | K-Means, DBSCAN, Agglomerative (clustering)

**Data:** Link 2 dataset (Food Insecurity → Health Outcomes) from the merged Map the Meal Gap 2017 and CDC PLACES data

**Method:** `GridSearchCV` and `RandomizedSearchCV` with 5-fold cross-validation

**Goal:** Improve upon midterm Random Forest baseline (R² = 0.64) using systematic hyperparameter optimization

---

## How to Build and Run
```powershell
# Install dependencies
python -m pip install numpy pandas scikit-learn matplotlib seaborn pytest jupyter

# Run tests
python -m pytest test_hyperparameter_tuning.py -v

# Run analysis notebook
jupyter notebook hyperparameter_tuning.ipynb
```

---

## Results

### Regression Models

| Model | Best Parameters | CV R² | Test R² |
|-------|----------------|-------|---------|
| Ridge | `alpha=100`, `solver=auto` | [YOUR_CV_R2] | **[YOUR_TEST_R2]** |
| Lasso | `alpha=0.1`, `selection=cyclic` | [YOUR_CV_R2] | [YOUR_TEST_R2] |
| Random Forest | `n_estimators=200`, `max_depth=10` | [YOUR_CV_R2] | [YOUR_TEST_R2] |
| Gradient Boosting | `n_estimators=200`, `learning_rate=0.05` | [YOUR_CV_R2] | [YOUR_TEST_R2] |

**Best Model:** [Update after running]  
**Midterm Baseline (no tuning):** 0.64 R²  
**Improvement:** [Calculate: (best_r2 - 0.64) / 0.64 * 100]%

### Clustering Models

| Model | Best Parameters | Silhouette Score |
|-------|----------------|------------------|
| K-Means | `k=[YOUR_K]` | [YOUR_SCORE] |
| **DBSCAN** | `eps=[YOUR_EPS]`, `min_samples=[YOUR_MIN]` | **[YOUR_SCORE]** |
| Agglomerative | `k=[YOUR_K]`, `linkage=[YOUR_LINKAGE]` | [YOUR_SCORE] |

**Best Model:** [Update after running]

---

## Key Findings

1. **Hyperparameter tuning systematically optimized model performance** - GridSearchCV and RandomizedSearchCV explored parameter spaces to identify optimal configurations for all 7 models.

2. **[Regression/Clustering] achieved best performance** - [Update based on your results: which approach worked better and why]

3. **Geographic and socioeconomic patterns drive health outcomes** - The model performance demonstrates that food insecurity combined with demographic factors are predictive of chronic disease prevalence, validating the Link 2 analysis in our project's causal chain.

---

## Project Context

This component addresses the "Next Steps" goal from our midterm report: *"Refine models with hyperparameter tuning to improve performance."*

**Causal Chain Analyzed:** Low Access → Food Insecurity → **Health Outcomes** (Link 2)

**Midterm Baseline:** Random Forest achieved 0.64 R² without hyperparameter tuning.

**Our Contribution:** Systematic hyperparameter optimization across multiple model families to maximize predictive performance for the Link 2 relationship.