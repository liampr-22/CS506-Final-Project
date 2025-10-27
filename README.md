# CS506 Project — Food Insecurity & Chronic Disease (Midterm Report)

**YouTube Presentation:** link

## 1. Project summary
The goal of this project is to examine how food access relates to food insecurity, and in turn how food insecurity is associated with adverse health outcomes:

> Low Access → Food Insecurity → Adverse Health Outcomes

We analyze county and census tract level data in the US to understand patterns of vulnerability and the potential predictive relationships between these variables.

## 2. Data Overview
We focus on two main "links" in the causal pathway chain, Low Access → Food Insecurity → Adverse Health Outcomes

### **Link 1: Low Access → Food Insecurity**
- Data sources:
  - USDA Food Access Research Atlas (2015)
  - Feeding America Map the Meal Gap (2015)
- Reasoning: Both datasets align in time, allowing a direct assessment of the relationship between poor food access and higher food insecurity.

### **Link 2: Food Insecurity → Health Outcomes**
- Data sources:
  - Feeding America Map the Meal Gap (2017)
  - CDC PLACES 2020 release (uses 2017/2018 BRFSS and 2013–2018 ACS)
  - Reasoning: Although years differ, assuming gradual change over time, these data provide a reasonable approximation to explore associations between food insecurity and health outcomes.

## 3. Data processing
The project started with various years and release for the three aforementioned data sources. A single merge for all three datasets was not possible due to inconsistent years. Thus, two years were selected for the two causal relationships. This resulted in the following data:
- USDA Food Access Research Atlas (2015)
- Feeding America Map the Meal Gap (2015)
- Feeding America Map the Meal Gap (2017)
- CDC PLACES 2020 releaseaw datasets

The FIPS code served as the merge key. For the USDA Atlas, which is at the census tract level, FIPS codes were derived from tract codes. Census tract-level features were aggregated to the county level, either using sums or weighted means, to create comparable county-level features for analysis.

This process allowed us to align datasets for modeling while respecting the limitations of the data. In the end, we processed two clean sets of data,
1. link1.csv (merged MMG and Atlas)
2. link2.csv (merged MMG and PLACES)

## 4. Preliminary visualizations
During EDA and modeling, several informative figures were produced. These can be found in the `/figures` directory. 

One important visualization was heatmaps of food insecurity rates, food deserts, and chronic illness prevalence. These maps consistently show similar areas affected, particularly in the Deep South/Southeast, suggesting overlap between poor food access, higher food insecurity, and adverse health outcomes. 
![Food Desert Population Proportions by County](https://github.com/liampr-22/CS506-Final-Project/blob/main/figures/EDA/county_food_desert.png)
![Food Insecurity Rate by County](https://github.com/liampr-22/CS506-Final-Project/blob/main/figures/EDA/county_food_insecurity_rate.png)
![Average Chronic Disease Prevalence by County](https://github.com/liampr-22/CS506-Final-Project/blob/main/figures/EDA/county_chronic.png)

## 5. Modeling Methodology

### Predictive Modeling
We implemented predictive models for both causal links:
1. Baseline Model: Predicted the mean of the dependent variable as a reference.
2. Linear Regression: Captured linear relationships and provided interpretable coefficients.
3. Random Forest Regressor: Captured nonlinear relationships and handled complex interactions.
4. XGBoost Regressor: Alternative tree-based method for comparison and potential performance improvement.

All models were trained on 80% of the data and evaluated on 20% held-out data. Features were standardized, and principal component analysis was used to reduce dimensionality and capture 95% of variance before modeling.

### Clustering
- Singular Value Decomposition (SVD) was applied to USDA structural data to identify key predictors of food insecurity.
- Census tract-level data were processed using share-based metrics to avoid population bias and standardized for comparability.
- Top features from SVD were correlated with food insecurity rates to understand which structural factors (e.g., low-income tracts, SNAP participation, poverty rate) are most strongly associated with food insecurity.
- Insights from this clustering analysis highlighted that poverty and low-income status consistently predict food insecurity, while low access metrics provide moderate explanatory power. These findings support the importance of targeting low-income areas and improving food access.

## 6. Preliminary results
### Predictive Models
1. Link 1 (Low Access → Food Insecurity):
Random Forest was the strongest model, explaining ~77% of variance. Linear regression performed well (~68% R²) but missed some nonlinear patterns. XGBoost achieved similar performance to Random Forest (~76% R²).

2. Link 2 (Food Insecurity → Health Outcomes):
Random Forest again performed best (~64% R²), capturing nonlinear relationships between food insecurity and chronic illness. Linear regression explained ~55% of variance.

### Clusters
- Structural features identified by SVD (e.g., low-income population, poverty rate, SNAP participation) are consistently correlated with food insecurity.
- These results suggest that interventions addressing low-income areas and improving access may reduce food insecurity.
- Clustering/feature selection complements predictive modeling by identifying the most influential structural predictors.

## 7. Next steps
- Refine models with hyperparameter tuning to improve performance
- Conduct subgroup analyses (e.g., children, seniors, racial groups)
- Address temporal mismatches in datasets for more accurate interpretations
- Explore feature importance for interpretability
