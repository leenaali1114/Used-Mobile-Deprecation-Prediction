# Mobile Phone Price Prediction App

A Flask web application that predicts the used price and depreciation percentage of mobile phones based on their specifications. The application uses machine learning models trained on a dataset of mobile phone prices.

## Features

- Predicts used price of mobile phones
- Predicts depreciation percentage
- Interactive web interface
- Data visualization and exploratory data analysis
- API endpoint for programmatic access
- Responsive design for mobile and desktop

## Implementation Details

### Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Visualization**: matplotlib, seaborn
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **API**: JSON REST API

### Machine Learning Models

The application uses two separate machine learning models:

1. **Used Price Model**: A Random Forest Regressor that predicts the used price of a mobile phone based on its specifications.
2. **Depreciation Model**: A Gradient Boosting Regressor that predicts the depreciation percentage of a mobile phone.

Both models are trained using hyperparameter tuning with GridSearchCV to optimize performance.

### Feature Engineering

The application performs several feature engineering steps:

- **Device Age**: Calculated from days used
- **Usage Intensity**: Ratio of days used to device age
- **Memory-RAM Ratio**: Ratio of internal memory to RAM
- **Camera Score**: Weighted sum of rear and front camera megapixels
- **High-End Flag**: Binary indicator for high-end devices
- **Battery Efficiency**: Ratio of battery capacity to screen size
- **Portability**: Ratio of weight to screen size
- **Network Score**: Numerical score based on network capabilities
- **Log Transformations**: Applied to skewed numerical features

### Data Preprocessing

- Missing values are handled using median imputation for numerical features and mode imputation for categorical features
- Categorical features are one-hot encoded
- Numerical features are standardized

### Exploratory Data Analysis

The application includes an EDA page that displays visualizations of:

- Target variable distributions
- Missing values
- Correlation matrix
- Feature relationships
- Brand price analysis
- Feature importance

## Project Structure 