# Enhanced-AQI-Prediction-and-Analysis-Using-Random-Forest-and-XGBoost
This repository provides a comprehensive implementation for analyzing, classifying, and predicting air quality using advanced machine learning techniques. The project focuses on leveraging air quality data to classify AQI (Air Quality Index) into predefined categories and predict AQI values with high accuracy.

Key Features:
Data Preprocessing and Feature Engineering:

Handling missing values in pollutant data by using mean imputation and grouping techniques.
Adding engineered features such as pollutant ratios to capture additional insights.
Filtering data by city for targeted analysis.
Exploratory Data Analysis (EDA):

Visualization of pollutant correlations with AQI using heatmaps.
Identification of key features influencing air quality.
Binary Classification:

Predicts whether AQI is above or below the median, categorized as 'High' or 'Low'.
Implements Random Forest for binary classification.
Provides evaluation metrics including accuracy, classification reports, and confusion matrices.
ROC curve visualization for model performance.
Multi-Class Classification:

Classifies AQI into six categories (e.g., Good, Moderate, Hazardous).
Employs XGBoost with hyperparameter tuning for optimized classification.
Evaluates performance using confusion matrices and classification reports.
Feature Importance Analysis:

Uses Random Forest and XGBoost to identify the most significant features influencing predictions.
Provides visualizations of feature importance rankings.
Model Evaluation:

Detailed comparison of model performances using metrics such as accuracy and feature relevance.
Visualization of confusion matrices for both binary and multi-class classifications.
This repository is a valuable resource for researchers, data scientists, and environmental analysts interested in air quality monitoring and predictive modeling.
