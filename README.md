# CS506-Final-Project
Project Description:
We will analyze county-level patterns of food insecurity and chronic disease data in the US. Since there are many potential confounding factors for these two variables, we will attempt to account for and explore several economic and demographic aspects to determine whether food insecurity is a good predictor of chronic illness.

Goal:
- Using clustering techniques to identify vulnerable communities in America, as well as areas with similar food access and health profiles
- Predictive modeling for short-term future trends on insecurity within major clusters
- Exploring and interpreting any correlations between food insecurity and chronic illness, and testing if the relationship is valid after controlling for covariates

Data Collection:
Collecting data from the sources below, potentially grouping by county (Request needed for Feeding America dataset)
- https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data
- https://www.cdc.gov/places/tools/data-portal.html
- https://www.feedingamerica.org/research/map-the-meal-gap/by-county
- https://data.census.gov/profile?q=United+States&g=010XX00US
- https://wonder.cdc.gov/welcomet.html
- https://www.countyhealthrankings.org/health-data

Data Modeling:
For clustering purposes, we can use the k-means algorithm. For exploring correlations and predictive models, various techniques may be useful, such as linear models, logarithmic models, decision trees, and XGBoost. It is also interesting to see if deep learning improves or marginally improves predictive power.

Data Visualization:
Scatterplots may be useful for bidimensional data between any two variables, whereas heatmaps of the country could be useful for clustering.

Test plan:
For predictive models, we will withhold 20% of the data via a random train/test split. This is to validate model generalization to unseen locations. We may also use stratified sampling so the test set contains representative examples of each county. 
