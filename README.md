## Project Goal

We'd like to approach analysis from the perspective of wanting to ascertain what determines whether or not a wine is good quality, and also what characteristics contribute to a wine being more alcoholic than others, in order to inform wine producers of what the general market prefers. 

## Project Approach

This project uses the [Red Wine Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data) from Kaggle to train and test both multiple linear regression models and logistic regression models.

## Project Summary
After conducting exploratory data analysis on the 1600 wine dataset, this project conducts backward elimination on our features to select an OLS model. After testing OLS validations, addressing outliers, we use weighted least squares regression to refine the model's predictive power. From there we conduct logistic regression to regress quality on other relevant features and predict which wines are more likely to be considered high-quality wines.

## Relevant Files
Please check out the winefunctions.py file to see the helper functions used, and requirements.txt for module/package information. 
