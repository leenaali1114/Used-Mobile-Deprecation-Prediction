import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Create directories if they don't exist
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('static/images'):
    os.makedirs('static/images')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('used_mobile_data_with_deprecation.csv')

# ================ EXPLORATORY DATA ANALYSIS ================
print("\n===== EXPLORATORY DATA ANALYSIS =====")

# Basic information
print("\nDataset shape:", df.shape)
print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Calculate percentage of missing values
missing_percentage = (missing_values / len(df)) * 100
print("\nPercentage of missing values:")
print(missing_percentage[missing_percentage > 0])

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig('static/images/missing_values.png')

# Distribution of target variables
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['normalized_used_price'], kde=True)
plt.title('Distribution of Used Price')

plt.subplot(1, 2, 2)
sns.histplot(df['deprecation (percentage loss in value)'], kde=True)
plt.title('Distribution of Deprecation')

plt.tight_layout()
plt.savefig('static/images/target_distributions.png')

# Correlation analysis
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('static/images/correlation_matrix.png')

# Relationship between key features and target variables
plt.figure(figsize=(16, 12))

# Used price vs key features
plt.subplot(2, 2, 1)
sns.scatterplot(x='release_year', y='normalized_used_price', data=df)
plt.title('Used Price vs Release Year')

plt.subplot(2, 2, 2)
sns.scatterplot(x='days_used', y='normalized_used_price', data=df)
plt.title('Used Price vs Days Used')

plt.subplot(2, 2, 3)
sns.boxplot(x='os', y='normalized_used_price', data=df)
plt.title('Used Price by OS')
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
sns.boxplot(x='5g', y='normalized_used_price', data=df)
plt.title('Used Price by 5G Support')

plt.tight_layout()
plt.savefig('static/images/price_relationships.png')

# Brand analysis
plt.figure(figsize=(16, 8))
brand_price = df.groupby('device_brand')['normalized_used_price'].mean().sort_values(ascending=False)
sns.barplot(x=brand_price.index, y=brand_price.values)
plt.title('Average Used Price by Brand')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('static/images/brand_price.png')

# ================ FEATURE ENGINEERING ================
print("\n===== FEATURE ENGINEERING =====")

# Create a copy of the dataframe for feature engineering
df_processed = df.copy()

# 1. Handle missing values
print("\nHandling missing values...")
# For numeric columns, fill with median
numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if df_processed[col].isnull().sum() > 0:
        median_value = df_processed[col].median()
        df_processed[col].fillna(median_value, inplace=True)
        print(f"Filled {col} missing values with median: {median_value}")

# For categorical columns, fill with mode
categorical_cols = df_processed.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_processed[col].isnull().sum() > 0:
        mode_value = df_processed[col].mode()[0]
        df_processed[col].fillna(mode_value, inplace=True)
        print(f"Filled {col} missing values with mode: {mode_value}")

# 2. Create new features
print("\nCreating new features...")

# Age of device in years
df_processed['device_age_years'] = (df_processed['days_used'] / 365).round(2)
print("Created 'device_age_years' feature")

# Ratio of days used to release year (usage intensity)
current_year = pd.Timestamp.now().year
df_processed['usage_intensity'] = df_processed['days_used'] / (current_year - df_processed['release_year'] + 1)
print("Created 'usage_intensity' feature")

# Memory per RAM ratio
df_processed['memory_ram_ratio'] = df_processed['internal_memory'] / df_processed['ram']
print("Created 'memory_ram_ratio' feature")

# Camera quality score (combining front and rear)
df_processed['camera_score'] = df_processed['rear_camera_mp'] + (0.5 * df_processed['front_camera_mp'])
print("Created 'camera_score' feature")

# High-end flag (based on RAM and memory)
df_processed['is_high_end'] = ((df_processed['ram'] >= 6) & 
                              (df_processed['internal_memory'] >= 128)).astype(int)
print("Created 'is_high_end' feature")

# Battery per screen size ratio (battery efficiency)
df_processed['battery_efficiency'] = df_processed['battery'] / df_processed['screen_size']
print("Created 'battery_efficiency' feature")

# Weight per screen size ratio (portability)
df_processed['portability'] = df_processed['weight'] / df_processed['screen_size']
print("Created 'portability' feature")

# Network technology score
df_processed['network_score'] = 0
df_processed.loc[df_processed['4g'] == 'yes', 'network_score'] += 1
df_processed.loc[df_processed['5g'] == 'yes', 'network_score'] += 2
print("Created 'network_score' feature")

# 3. Log transform skewed numeric features
print("\nTransforming skewed features...")
skewed_features = ['internal_memory', 'ram', 'days_used']
for feature in skewed_features:
    # Add a small constant to handle zeros
    df_processed[f'{feature}_log'] = np.log1p(df_processed[feature])
    print(f"Created log-transformed feature: '{feature}_log'")

# Visualize new features
plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
sns.scatterplot(x='device_age_years', y='normalized_used_price', data=df_processed)
plt.title('Used Price vs Device Age (Years)')

plt.subplot(2, 3, 2)
sns.scatterplot(x='usage_intensity', y='normalized_used_price', data=df_processed)
plt.title('Used Price vs Usage Intensity')

plt.subplot(2, 3, 3)
sns.scatterplot(x='camera_score', y='normalized_used_price', data=df_processed)
plt.title('Used Price vs Camera Score')

plt.subplot(2, 3, 4)
sns.boxplot(x='is_high_end', y='normalized_used_price', data=df_processed)
plt.title('Used Price by High-End Flag')

plt.subplot(2, 3, 5)
sns.scatterplot(x='battery_efficiency', y='normalized_used_price', data=df_processed)
plt.title('Used Price vs Battery Efficiency')

plt.subplot(2, 3, 6)
sns.boxplot(x='network_score', y='normalized_used_price', data=df_processed)
plt.title('Used Price by Network Score')

plt.tight_layout()
plt.savefig('static/images/engineered_features.png')

# ================ MODEL TRAINING ================
print("\n===== MODEL TRAINING =====")

# Define features and target variables
print("\nPreparing features and targets...")

# Features to use (including engineered features)
categorical_features = ['device_brand', 'os', '4g', '5g']
numerical_features = [
    'screen_size', 'rear_camera_mp', 'front_camera_mp', 'internal_memory', 
    'ram', 'battery', 'weight', 'release_year', 'days_used',
    'device_age_years', 'usage_intensity', 'memory_ram_ratio', 'camera_score',
    'is_high_end', 'battery_efficiency', 'portability', 'network_score',
    'internal_memory_log', 'ram_log', 'days_used_log'
]

# Remove rows with NaN in target variables
df_processed = df_processed.dropna(subset=['normalized_used_price', 'normalized_new_price', 'deprecation (percentage loss in value)'])

X = df_processed[categorical_features + numerical_features]
y_used_price = df_processed['normalized_used_price']
y_deprecation = df_processed['deprecation (percentage loss in value)']

# Split the data
X_train, X_test, y_train_price, y_test_price = train_test_split(
    X, y_used_price, test_size=0.2, random_state=42
)
_, _, y_train_dep, y_test_dep = train_test_split(
    X, y_deprecation, test_size=0.2, random_state=42
)

# Create preprocessor with proper handling of missing values
print("\nCreating preprocessing pipeline...")
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ================ HYPERPARAMETER TUNING ================
print("\n===== HYPERPARAMETER TUNING =====")

# 1. Used Price Model Tuning
print("\nTuning Used Price Model...")

# Define the model pipeline
price_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Define the parameter grid - simplified to reduce memory usage
param_grid_price = {
    'regressor__n_estimators': [100],
    'regressor__max_depth': [None, 10],
    'regressor__min_samples_split': [2]
}

# Perform grid search with cross-validation - using n_jobs=1 to avoid parallel processing issues
grid_search_price = GridSearchCV(
    price_pipeline, param_grid_price, cv=3, 
    scoring='neg_mean_squared_error', n_jobs=1, verbose=1
)

try:
    grid_search_price.fit(X_train, y_train_price)
    
    # Get the best parameters and model
    best_params_price = grid_search_price.best_params_
    best_model_price = grid_search_price.best_estimator_
    
    print(f"Best parameters for Used Price model: {best_params_price}")
    
    # Evaluate the model
    y_pred_price = best_model_price.predict(X_test)
    price_mse = mean_squared_error(y_test_price, y_pred_price)
    price_rmse = np.sqrt(price_mse)
    price_r2 = r2_score(y_test_price, y_pred_price)
    
    print(f"Used Price Model - RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")
except Exception as e:
    print(f"Error during price model tuning: {e}")
    print("Falling back to default model...")
    # Create a default model if grid search fails
    best_model_price = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    best_model_price.fit(X_train, y_train_price)
    
    # Evaluate the default model
    y_pred_price = best_model_price.predict(X_test)
    price_mse = mean_squared_error(y_test_price, y_pred_price)
    price_rmse = np.sqrt(price_mse)
    price_r2 = r2_score(y_test_price, y_pred_price)
    
    print(f"Default Used Price Model - RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}")

# 2. Deprecation Model Tuning
print("\nTuning Deprecation Model...")

# Define the model pipeline
dep_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Define the parameter grid - simplified to reduce memory usage
param_grid_dep = {
    'regressor__n_estimators': [100],
    'regressor__learning_rate': [0.1],
    'regressor__max_depth': [3, 5]
}

# Perform grid search with cross-validation - using n_jobs=1 to avoid parallel processing issues
grid_search_dep = GridSearchCV(
    dep_pipeline, param_grid_dep, cv=3, 
    scoring='neg_mean_squared_error', n_jobs=1, verbose=1
)

try:
    grid_search_dep.fit(X_train, y_train_dep)
    
    # Get the best parameters and model
    best_params_dep = grid_search_dep.best_params_
    best_model_dep = grid_search_dep.best_estimator_
    
    print(f"Best parameters for Deprecation model: {best_params_dep}")
    
    # Evaluate the model
    y_pred_dep = best_model_dep.predict(X_test)
    dep_mse = mean_squared_error(y_test_dep, y_pred_dep)
    dep_rmse = np.sqrt(dep_mse)
    dep_r2 = r2_score(y_test_dep, y_pred_dep)
    
    print(f"Deprecation Model - RMSE: {dep_rmse:.4f}, R²: {dep_r2:.4f}")
except Exception as e:
    print(f"Error during deprecation model tuning: {e}")
    print("Falling back to default model...")
    # Create a default model if grid search fails
    best_model_dep = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
    best_model_dep.fit(X_train, y_train_dep)
    
    # Evaluate the default model
    y_pred_dep = best_model_dep.predict(X_test)
    dep_mse = mean_squared_error(y_test_dep, y_pred_dep)
    dep_rmse = np.sqrt(dep_mse)
    dep_r2 = r2_score(y_test_dep, y_pred_dep)
    
    print(f"Default Deprecation Model - RMSE: {dep_rmse:.4f}, R²: {dep_r2:.4f}")

# ================ FEATURE IMPORTANCE ================
print("\n===== FEATURE IMPORTANCE =====")

# Get feature names after preprocessing
feature_names = (
    numerical_features +
    list(best_model_price.named_steps['preprocessor']
         .named_transformers_['cat']
         .named_steps['onehot']
         .get_feature_names_out(categorical_features))
)

# Plot feature importance for Used Price model
plt.figure(figsize=(12, 8))
importances_price = best_model_price.named_steps['regressor'].feature_importances_
indices_price = np.argsort(importances_price)[::-1]

plt.title('Feature Importance for Used Price Prediction')
plt.bar(range(len(importances_price)), 
        importances_price[indices_price],
        align='center')
plt.xticks(range(min(20, len(importances_price))), 
           [feature_names[i] for i in indices_price][:20], 
           rotation=90)
plt.tight_layout()
plt.savefig('static/images/price_feature_importance.png')

# Plot feature importance for Deprecation model
plt.figure(figsize=(12, 8))
importances_dep = best_model_dep.named_steps['regressor'].feature_importances_
indices_dep = np.argsort(importances_dep)[::-1]

plt.title('Feature Importance for Deprecation Prediction')
plt.bar(range(len(importances_dep)), 
        importances_dep[indices_dep],
        align='center')
plt.xticks(range(min(20, len(importances_dep))), 
           [feature_names[i] for i in indices_dep][:20], 
           rotation=90)
plt.tight_layout()
plt.savefig('static/images/deprecation_feature_importance.png')

# ================ SAVE MODELS ================
print("\n===== SAVING MODELS =====")

# Save the best models
pickle.dump(best_model_price, open('models/used_price_model.pkl', 'wb'))
pickle.dump(best_model_dep, open('models/deprecation_model.pkl', 'wb'))

# Save feature lists for the app
model_info = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'feature_ranges': {
        'screen_size': [df['screen_size'].min(), df['screen_size'].max()],
        'rear_camera_mp': [df['rear_camera_mp'].min(), df['rear_camera_mp'].max()],
        'front_camera_mp': [df['front_camera_mp'].min(), df['front_camera_mp'].max()],
        'internal_memory': [df['internal_memory'].min(), df['internal_memory'].max()],
        'ram': [df['ram'].min(), df['ram'].max()],
        'battery': [df['battery'].min(), df['battery'].max()],
        'weight': [df['weight'].min(), df['weight'].max()],
        'release_year': [int(df['release_year'].min()), int(df['release_year'].max())],
        'days_used': [0, 1500]  # Reasonable range for days used
    },
    'unique_values': {
        'device_brand': df['device_brand'].unique().tolist(),
        'os': df['os'].unique().tolist(),
    }
}

pickle.dump(model_info, open('models/model_info.pkl', 'wb'))

print("\nModels and feature information saved successfully!")
print("\n===== TRAINING COMPLETE =====") 