import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv('C:/Users/sneha/OneDrive/Documents/city_day.csv')

# Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
target = 'AQI'

# Handle missing values
df[pollutants] = df[pollutants].apply(pd.to_numeric, errors='coerce')
df[target] = pd.to_numeric(df[target], errors='coerce')

# Filter data for a specific city
selected_city = 'Delhi'
city_data = df[df['City'] == selected_city].copy()

# Add engineered features
city_data['PM2.5/PM10'] = city_data['PM2.5'] / (city_data['PM10'] + 1e-5)
city_data['NO/NO2'] = city_data['NO'] / (city_data['NO2'] + 1e-5)
city_data['SO2/CO'] = city_data['SO2'] / (city_data['CO'] + 1e-5)

# Correlation matrix visualization
corr_matrix = city_data[pollutants + ['PM2.5/PM10', 'NO/NO2', 'SO2/CO', target]].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title(f"Correlation Matrix for {selected_city}")
plt.show()

# Categorize AQI into classes
def categorize_aqi(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Moderate'
    elif aqi <= 150: return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200: return 'Unhealthy'
    elif aqi <= 300: return 'Very Unhealthy'
    else: return 'Hazardous'

city_data['AQI_Category'] = city_data[target].apply(categorize_aqi)

# Encode AQI categories
label_encoder = LabelEncoder()
city_data['AQI_Category_Encoded'] = label_encoder.fit_transform(city_data['AQI_Category'])

# Binary Classification (High/Low AQI)
city_data['High_Low_AQI'] = (city_data[target] > city_data[target].median()).astype(int)

# Binary classification features and target
X_binary = city_data[pollutants + ['PM2.5/PM10', 'NO/NO2', 'SO2/CO']]
y_binary = city_data['High_Low_AQI']

# Multi-class classification features and target
X_multi = X_binary.copy()
y_multi = city_data['AQI_Category_Encoded']

# Train-test split
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.3, stratify=y_multi, random_state=42)

# Feature Importance with Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_bin, y_train_bin)
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X_binary.columns)

# Plot feature importances
plt.figure(figsize=(12, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title("Feature Importance (Random Forest - Binary Classification)")
plt.ylabel("Importance")
plt.show()

# Retain features with importance > 0.02
selected_features = feature_importances[feature_importances > 0.02].index
X_train_bin_selected = X_train_bin[selected_features]
X_test_bin_selected = X_test_bin[selected_features]
X_train_multi_selected = X_train_multi[selected_features]
X_test_multi_selected = X_test_multi[selected_features]

# Model 1: Binary Classification with Random Forest
rf_binary = RandomForestClassifier(random_state=42)
rf_binary.fit(X_train_bin_selected, y_train_bin)
y_pred_bin = rf_binary.predict(X_test_bin_selected)

# Evaluate Binary Model
bin_accuracy = accuracy_score(y_test_bin, y_pred_bin)
print(f"Binary Classification Accuracy: {bin_accuracy:.2f}")
print("\nBinary Classification Report:\n")
print(classification_report(y_test_bin, y_pred_bin))

# ROC Curve for Binary Classification
fpr, tpr, thresholds = roc_curve(y_test_bin, rf_binary.predict_proba(X_test_bin_selected)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.title("ROC Curve (Binary Classification)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Model 2: Multi-Class Classification with XGBoost
xgb_multi = XGBClassifier(
    objective='multi:softmax', 
    num_class=len(label_encoder.classes_), 
    random_state=42
)

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid_search_xgb = GridSearchCV(
    xgb_multi, 
    param_grid_xgb, 
    cv=StratifiedKFold(n_splits=5), 
    verbose=1, 
    n_jobs=-1
)
grid_search_xgb.fit(X_train_multi_selected, y_train_multi)

# Display the best parameters
print("Best Parameters:", grid_search_xgb.best_params_)


# Train the best XGBoost model
best_xgb = grid_search_xgb.best_estimator_
y_pred_multi = best_xgb.predict(X_test_multi_selected)

# Evaluate Multi-Class Model
multi_class_accuracy = accuracy_score(y_test_multi, y_pred_multi)
print(f"Multi-Class Classification Accuracy: {multi_class_accuracy:.2f}")
print("\nMulti-Class Classification Report:\n")
print(classification_report(y_test_multi, y_pred_multi, target_names=label_encoder.classes_))

# Confusion Matrix for Multi-Class Classification
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_multi, annot=True, cmap="Blues", fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (Multi-Class Classification)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance for XGBoost
xgb_importances = pd.Series(best_xgb.feature_importances_, index=selected_features)

plt.figure(figsize=(12, 6))
xgb_importances.sort_values(ascending=False).plot(kind='bar', color='lightcoral')
plt.title("Feature Importance (XGBoost - Multi-Class Classification)")
plt.ylabel("Importance")
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix for Binary Classification
cm_binary = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_binary, annot=True, cmap="Blues", fmt="d", xticklabels=['Low AQI', 'High AQI'], yticklabels=['Low AQI', 'High AQI'])
plt.title("Confusion Matrix (Binary Classification)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix for Multi-Class Classification
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_multi, annot=True, cmap="Blues", fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (Multi-Class Classification)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


