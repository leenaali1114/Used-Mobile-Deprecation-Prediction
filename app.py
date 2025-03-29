from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the models and model info
@app.before_first_request
def load_models():
    global used_price_model, deprecation_model, model_info
    try:
        used_price_model = pickle.load(open('models/used_price_model.pkl', 'rb'))
        deprecation_model = pickle.load(open('models/deprecation_model.pkl', 'rb'))
        model_info = pickle.load(open('models/model_info.pkl', 'rb'))
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# Get unique values and ranges for form fields
def get_form_data():
    try:
        return {
            'device_brands': sorted(model_info['unique_values']['device_brand']),
            'os_types': sorted(model_info['unique_values']['os']),
            'ranges': {
                'screen_size': [round(model_info['feature_ranges']['screen_size'][0], 1), 
                               round(model_info['feature_ranges']['screen_size'][1], 1)],
                'rear_camera_mp': [round(model_info['feature_ranges']['rear_camera_mp'][0], 1), 
                                  round(model_info['feature_ranges']['rear_camera_mp'][1], 1)],
                'front_camera_mp': [round(model_info['feature_ranges']['front_camera_mp'][0], 1), 
                                   round(model_info['feature_ranges']['front_camera_mp'][1], 1)],
                'internal_memory': [round(model_info['feature_ranges']['internal_memory'][0], 1), 
                                   round(model_info['feature_ranges']['internal_memory'][1], 1)],
                'ram': [round(model_info['feature_ranges']['ram'][0], 1), 
                       round(model_info['feature_ranges']['ram'][1], 1)],
                'battery': [int(model_info['feature_ranges']['battery'][0]), 
                           int(model_info['feature_ranges']['battery'][1])],
                'weight': [round(model_info['feature_ranges']['weight'][0], 1), 
                          round(model_info['feature_ranges']['weight'][1], 1)],
                'release_year': model_info['feature_ranges']['release_year'],
                'days_used': [0, 1500]
            }
        }
    except Exception as e:
        logger.error(f"Error getting form data: {e}")
        return {
            'device_brands': ['Samsung', 'Apple', 'Huawei'],
            'os_types': ['Android', 'iOS'],
            'ranges': {
                'screen_size': [5.0, 17.0],
                'rear_camera_mp': [8.0, 64.0],
                'front_camera_mp': [5.0, 32.0],
                'internal_memory': [16.0, 512.0],
                'ram': [2.0, 12.0],
                'battery': [3000, 5000],
                'weight': [150, 250],
                'release_year': [2018, 2023],
                'days_used': [0, 1500]
            }
        }

@app.route('/')
def home():
    form_data = get_form_data()
    return render_template('index.html', form_data=form_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = {
            'device_brand': request.form.get('device_brand'),
            'os': request.form.get('os'),
            'screen_size': float(request.form.get('screen_size')),
            '4g': request.form.get('4g'),
            '5g': request.form.get('5g'),
            'rear_camera_mp': float(request.form.get('rear_camera_mp')),
            'front_camera_mp': float(request.form.get('front_camera_mp')),
            'internal_memory': float(request.form.get('internal_memory')),
            'ram': float(request.form.get('ram')),
            'battery': float(request.form.get('battery')),
            'weight': float(request.form.get('weight')),
            'release_year': int(request.form.get('release_year')),
            'days_used': int(request.form.get('days_used'))
        }
        
        # Create engineered features
        features['device_age_years'] = round(features['days_used'] / 365, 2)
        features['usage_intensity'] = features['days_used'] / (2023 - features['release_year'] + 1)
        features['memory_ram_ratio'] = features['internal_memory'] / features['ram']
        features['camera_score'] = features['rear_camera_mp'] + (0.5 * features['front_camera_mp'])
        features['is_high_end'] = 1 if (features['ram'] >= 6 and features['internal_memory'] >= 128) else 0
        features['battery_efficiency'] = features['battery'] / features['screen_size']
        features['portability'] = features['weight'] / features['screen_size']
        
        features['network_score'] = 0
        if features['4g'] == 'yes':
            features['network_score'] += 1
        if features['5g'] == 'yes':
            features['network_score'] += 2
            
        # Log transformations
        features['internal_memory_log'] = np.log1p(features['internal_memory'])
        features['ram_log'] = np.log1p(features['ram'])
        features['days_used_log'] = np.log1p(features['days_used'])
        
        # Create a DataFrame with the input features
        input_df = pd.DataFrame([features])
        
        # Make predictions
        used_price = used_price_model.predict(input_df)[0]
        deprecation = deprecation_model.predict(input_df)[0]
        
        # Ensure predictions are within reasonable bounds
        used_price = max(0, used_price)
        deprecation = max(0, min(100, deprecation))
        
        # Get phone image based on brand
        brand = features['device_brand'].lower()
        phone_image = f"https://source.unsplash.com/300x300/?{brand}+smartphone"
        
        return render_template('result.html', 
                              used_price=round(used_price, 2),
                              deprecation=round(deprecation, 2),
                              features=features,
                              phone_image=phone_image)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        
        # Add engineered features
        data['device_age_years'] = round(data['days_used'] / 365, 2)
        data['usage_intensity'] = data['days_used'] / (2023 - data['release_year'] + 1)
        data['memory_ram_ratio'] = data['internal_memory'] / data['ram']
        data['camera_score'] = data['rear_camera_mp'] + (0.5 * data['front_camera_mp'])
        data['is_high_end'] = 1 if (data['ram'] >= 6 and data['internal_memory'] >= 128) else 0
        data['battery_efficiency'] = data['battery'] / data['screen_size']
        data['portability'] = data['weight'] / data['screen_size']
        
        data['network_score'] = 0
        if data['4g'] == 'yes':
            data['network_score'] += 1
        if data['5g'] == 'yes':
            data['network_score'] += 2
            
        # Log transformations
        data['internal_memory_log'] = np.log1p(data['internal_memory'])
        data['ram_log'] = np.log1p(data['ram'])
        data['days_used_log'] = np.log1p(data['days_used'])
        
        input_df = pd.DataFrame([data])
        
        used_price = used_price_model.predict(input_df)[0]
        deprecation = deprecation_model.predict(input_df)[0]
        
        # Ensure predictions are within reasonable bounds
        used_price = max(0, used_price)
        deprecation = max(0, min(100, deprecation))
        
        return jsonify({
            'used_price': round(used_price, 2),
            'deprecation': round(deprecation, 2)
        })
    
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/eda')
def eda():
    """Display EDA visualizations"""
    return render_template('eda.html')

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Check if models exist, if not train them
    if not (os.path.exists('models/used_price_model.pkl') and 
            os.path.exists('models/deprecation_model.pkl') and
            os.path.exists('models/model_info.pkl')):
        logger.info("Models not found. Training new models...")
        import model_training
        
    app.run(debug=True) 