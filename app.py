"""
Flask Web Application for ML Pipeline Deployment
Handles both binary and multiclass classification with a smart prediction endpoint.
"""

from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import json
import joblib
import os
from werkzeug.utils import secure_filename
import traceback
from datetime import datetime
from ml_utils import PreprocessingPipeline, RFTransformer

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models and metadata
binary_model = None
multiclass_model = None
model_metadata = None

def load_models():
    """Load trained models and metadata on application startup"""
    global binary_model, multiclass_model, model_metadata
    
    models_dir = "models"
    
    try:
        # Load models
        binary_model = joblib.load(os.path.join(models_dir, 'binary_model.pkl'))
        multiclass_model = joblib.load(os.path.join(models_dir, 'multiclass_model.pkl'))
        
        # Load metadata
        with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
            model_metadata = json.load(f)
            
        print("✅ Models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return False

def convert_form_data(data):
    """Convert form data strings to appropriate numeric types"""
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, str) and value.strip():
            try:
                # Try to convert to float first
                float_val = float(value)
                # If it's a whole number, convert to int
                if float_val.is_integer():
                    converted_data[key] = int(float_val)
                else:
                    converted_data[key] = float_val
            except (ValueError, TypeError):
                converted_data[key] = value  # Keep as string if conversion fails
        else:
            converted_data[key] = value
    return converted_data

def prepare_input_data(data):
    """Prepare input data for prediction"""
    try:
        # Convert to DataFrame if it's a dictionary
        if isinstance(data, dict):
            # Handle single prediction
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Handle batch predictions
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Ensure all required features are present
        required_features = model_metadata['feature_schema']['feature_names']
        
        # Add missing features with default values
        for feature in required_features:
            if feature not in df.columns:
                # Use mean for numeric, mode for categorical
                if feature in model_metadata['feature_schema']['sample_values']:
                    sample_val = model_metadata['feature_schema']['sample_values'][feature]
                    if isinstance(sample_val, dict) and 'mean' in sample_val:
                        df[feature] = sample_val['mean']
                    elif isinstance(sample_val, list) and len(sample_val) > 0:
                        df[feature] = sample_val[0]
                    else:
                        df[feature] = 0
                else:
                    df[feature] = 0
        
        # Reorder columns to match training data
        df = df[required_features]
        
        # Apply the same preprocessing as during training
        df_processed = df.copy()
        
        # Handle datetime columns
        for col in df_processed.columns:
            if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                df_processed[col] = pd.to_datetime(df_processed[col]).astype(np.int64) // 10**9
            elif col in model_metadata.get('label_encoders', {}):
                # Apply label encoding for categorical columns
                le_classes = model_metadata['label_encoders'][col]
                # Handle unseen categories by using the first class
                df_processed[col] = df_processed[col].astype(str)
                for i, val in enumerate(df_processed[col]):
                    if val in le_classes:
                        df_processed.iloc[i, df_processed.columns.get_loc(col)] = le_classes.index(val)
                    else:
                        df_processed.iloc[i, df_processed.columns.get_loc(col)] = 0  # Default to first class
        
        return df_processed
        
    except Exception as e:
        raise ValueError(f"Error preparing input data: {str(e)}")

@app.route('/')
def home():
    """Main page with model selection and input form"""
    if not binary_model or not multiclass_model:
        return render_template('error.html', 
                             error="Models not loaded. Please ensure model files are available.")
    
    return render_template('index.html', metadata=model_metadata)

@app.route('/predict-binary', methods=['POST'])
def predict_binary():
    """Binary classification endpoint"""
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = convert_form_data(request.form.to_dict())
        
        # Prepare input data
        input_df = prepare_input_data(data)
        
        # Make prediction
        prediction = binary_model.predict(input_df)[0]
        prediction_proba = binary_model.predict_proba(input_df)[0]
        
        # Prepare response
        prediction_int = int(prediction)
        response = {
            'prediction': prediction_int,
            'prediction_label': model_metadata['model_info']['binary_model']['class_labels'][str(prediction_int)],
            'probabilities': {
                'class_0': float(prediction_proba[0]),
                'class_1': float(prediction_proba[1])
            },
            'confidence': float(max(prediction_proba)),
            'model_type': 'binary'
        }
        
        if request.is_json:
            return jsonify(response)
        else:
            return render_template('result.html', result=response, input_data=data)
            
    except Exception as e:
        error_msg = f"Error in binary prediction: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        else:
            flash(error_msg, 'error')
            return redirect(url_for('home'))

@app.route('/predict-multiclass', methods=['POST'])
def predict_multiclass():
    """Multiclass classification endpoint"""
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = convert_form_data(request.form.to_dict())
        
        # Prepare input data
        input_df = prepare_input_data(data)
        
        # Make prediction
        prediction = multiclass_model.predict(input_df)[0]
        prediction_proba = multiclass_model.predict_proba(input_df)[0]
        
        # Get class labels
        classes = multiclass_model.classes_
        
        # Prepare response
        prediction_int = int(prediction)
        response = {
            'prediction': prediction_int,
            'prediction_label': model_metadata['model_info']['multiclass_model']['class_labels'][str(prediction_int)],
            'probabilities': {f'class_{int(cls)}': float(prob) for cls, prob in zip(classes, prediction_proba)},
            'confidence': float(max(prediction_proba)),
            'model_type': 'multiclass'
        }
        
        if request.is_json:
            return jsonify(response)
        else:
            return render_template('result.html', result=response, input_data=data)
            
    except Exception as e:
        error_msg = f"Error in multiclass prediction: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        else:
            flash(error_msg, 'error')
            return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def smart_predict():
    """Smart prediction endpoint that auto-selects model based on task type"""
    try:
        # Get data and task type
        if request.is_json:
            data = request.get_json()
            task_type = data.pop('task_type', 'multiclass')  # Default to multiclass
        else:
            data = request.form.to_dict()
            task_type = data.pop('task_type', 'multiclass')
            data = convert_form_data(data)
        
        # Route to appropriate model
        if task_type == 'binary':
            # Prepare for binary prediction
            input_df = prepare_input_data(data)
            prediction = binary_model.predict(input_df)[0]
            prediction_proba = binary_model.predict_proba(input_df)[0]
            
            prediction_int = int(prediction)
            response = {
                'prediction': prediction_int,
                'prediction_label': model_metadata['model_info']['binary_model']['class_labels'][str(prediction_int)],
                'probabilities': {
                    'class_0': float(prediction_proba[0]),
                    'class_1': float(prediction_proba[1])
                },
                'confidence': float(max(prediction_proba)),
                'model_type': 'binary'
            }
        else:
            # Prepare for multiclass prediction
            input_df = prepare_input_data(data)
            prediction = multiclass_model.predict(input_df)[0]
            prediction_proba = multiclass_model.predict_proba(input_df)[0]
            classes = multiclass_model.classes_
            
            prediction_int = int(prediction)
            response = {
                'prediction': prediction_int,
                'prediction_label': model_metadata['model_info']['multiclass_model']['class_labels'][str(prediction_int)],
                'probabilities': {f'class_{int(cls)}': float(prob) for cls, prob in zip(classes, prediction_proba)},
                'confidence': float(max(prediction_proba)),
                'model_type': 'multiclass'
            }
        
        if request.is_json:
            return jsonify(response)
        else:
            return render_template('result.html', result=response, input_data=data)
            
    except Exception as e:
        error_msg = f"Error in smart prediction: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        else:
            flash(error_msg, 'error')
            return redirect(url_for('home'))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload for batch predictions"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('home'))
        
        file = request.files['file']
        task_type = request.form.get('task_type', 'multiclass')
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('home'))
        
        if file and file.filename.lower().endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(file)
            
            # Limit to first 100 rows for demo
            if len(df) > 100:
                df = df.head(100)
                flash(f'Limited to first 100 rows for demonstration', 'info')
            
            # Make predictions
            if task_type == 'binary':
                predictions = binary_model.predict(df)
                probabilities = binary_model.predict_proba(df)
                model_info = model_metadata['model_info']['binary_model']
            else:
                predictions = multiclass_model.predict(df)
                probabilities = multiclass_model.predict_proba(df)
                model_info = model_metadata['model_info']['multiclass_model']
            
            # Prepare results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                pred_int = int(pred)
                result = {
                    'row': i + 1,
                    'prediction': pred_int,
                    'prediction_label': model_info['class_labels'][str(pred_int)],
                    'confidence': float(max(prob))
                }
                results.append(result)
            
            return render_template('batch_results.html', 
                                 results=results, 
                                 task_type=task_type,
                                 total_rows=len(results))
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(url_for('home'))
            
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(f"❌ {error_msg}")
        flash(error_msg, 'error')
        return redirect(url_for('home'))

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    if not model_metadata:
        return jsonify({'error': 'Models not loaded'}), 500
    
    api_info = {
        'status': 'active',
        'models': {
            'binary': {
                'endpoint': '/predict-binary',
                'method': 'POST',
                'description': 'Binary classification (Status 3 vs Others)'
            },
            'multiclass': {
                'endpoint': '/predict-multiclass', 
                'method': 'POST',
                'description': 'Multiclass classification (Status 1, 2, or 3)'
            },
            'smart': {
                'endpoint': '/predict',
                'method': 'POST',
                'description': 'Auto-select model based on task_type parameter'
            }
        },
        'feature_schema': model_metadata['feature_schema'],
        'sample_request': {
            'task_type': 'multiclass',  # or 'binary'
            **{feature: 'sample_value' for feature in model_metadata['feature_schema']['feature_names'][:5]}
        }
    }
    
    return jsonify(api_info)

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

# Load models on startup (for production deployment)
load_models()

if __name__ == '__main__':
    print("="*60)
    print("ML PIPELINE WEB APPLICATION")
    print("="*60)
    
    # Load models on startup
    if load_models():
        print("🚀 Starting Flask application...")
        print("📊 Both binary and multiclass models are ready!")
        print("🌐 Access the web interface at: http://localhost:5000")
        print("📡 API endpoints available:")
        print("   - POST /predict-binary")
        print("   - POST /predict-multiclass")
        print("   - POST /predict (smart)")
        print("   - GET /api/info")
        print("="*60)
        
        # Use environment port for deployment or default to 5000
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("❌ Failed to load models. Please run model_preparation.py first.")
