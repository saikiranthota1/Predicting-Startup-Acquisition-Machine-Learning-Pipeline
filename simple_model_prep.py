"""
Simple Model Preparation Script
This script creates compatible models using your existing trained pipeline.
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from ml_utils import PreprocessingPipeline, RFTransformer

def create_simple_models():
    """Create simple models that are compatible with the web app"""
    print("="*60)
    print("CREATING SIMPLE MODELS FOR WEB APP")
    print("="*60)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("fe_outcomes2.csv")
    
    # Basic preprocessing (similar to your notebook)
    df_processed = df.copy()
    label_encoders = {}
    
    for col in df_processed.columns:
        if np.issubdtype(df_processed[col].dtype, np.datetime64):  
            df_processed[col] = pd.to_datetime(df_processed[col]).astype(np.int64) // 10**9
        elif df_processed[col].dtype == 'object':  
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Define meaningful business features (exclude identifiers and irrelevant columns)
    meaningful_features = [
        'category_code',           # Business category
        'country_code',           # Geographic location
        'state_code',             # State/province
        'region',                 # Region within state
        'city',                   # City location
        'first_investment_at',    # Investment timing
        'first_funding_at',       # Funding timing
        'first_milestone_at',     # Milestone timing
        'founded_year',           # When company was founded
        'founded_month',          # Founding month
        'founded_age'             # Age of company
    ]
    
    # Filter to only meaningful features that exist in the dataset
    available_features = [col for col in meaningful_features if col in df_processed.columns]
    print(f"Using {len(available_features)} meaningful features: {available_features}")
    
    # Separate features and targets using only meaningful features
    X = df_processed[available_features]
    y_multiclass = df_processed["status"]
    y_binary = (y_multiclass == 3).astype(int)  # Binary: Status 3 vs Others
    
    # Train/test split
    X_train, X_test, y_multi_train, y_multi_test, y_bin_train, y_bin_test = train_test_split(
        X, y_multiclass, y_binary, test_size=0.2, random_state=42
    )
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Create Binary Model
    print("\nTraining Binary Model...")
    binary_model = LogisticRegression(random_state=42, max_iter=1000)
    binary_model.fit(X_train, y_bin_train)
    
    # Evaluate binary model
    y_bin_pred = binary_model.predict(X_test)
    binary_accuracy = accuracy_score(y_bin_test, y_bin_pred)
    print(f"Binary Model Accuracy: {binary_accuracy:.4f}")
    
    # Create Multiclass Model (simpler version)
    print("\nTraining Multiclass Model...")
    multiclass_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    multiclass_model.fit(X_train, y_multi_train)
    
    # Evaluate multiclass model
    y_multi_pred = multiclass_model.predict(X_test)
    multiclass_accuracy = accuracy_score(y_multi_test, y_multi_pred)
    print(f"Multiclass Model Accuracy: {multiclass_accuracy:.4f}")
    
    # Save models
    joblib.dump(binary_model, 'models/binary_model.pkl')
    joblib.dump(multiclass_model, 'models/multiclass_model.pkl')
    
    # Save label encoders and feature info
    model_info = {
        'label_encoders': {k: v.classes_.tolist() for k, v in label_encoders.items()},
        'feature_names': X.columns.tolist(),
        'binary_accuracy': binary_accuracy,
        'multiclass_accuracy': multiclass_accuracy
    }
    
    # Create feature schema
    feature_schema = {
        'feature_names': X.columns.tolist(),
        'feature_types': {col: str(dtype) for col, dtype in X.dtypes.items()},
        'sample_values': {}
    }
    
    # Add sample values for each feature
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            feature_schema['sample_values'][col] = {
                'min': float(X[col].min()),
                'max': float(X[col].max()),
                'mean': float(X[col].mean())
            }
        else:
            feature_schema['sample_values'][col] = X[col].unique()[:5].tolist()
    
    # Save metadata
    metadata = {
        'feature_schema': feature_schema,
        'model_info': {
            'binary_model': {
                'type': 'Binary Classification',
                'classes': [0, 1],
                'class_labels': {'0': 'Not Status 3', '1': 'Status 3'},
                'accuracy': binary_accuracy
            },
            'multiclass_model': {
                'type': 'Multiclass Classification', 
                'classes': [1, 2, 3],
                'class_labels': {'1': 'Status 1', '2': 'Status 2', '3': 'Status 3'},
                'accuracy': multiclass_accuracy
            }
        },
        'preprocessing_info': {
            'datetime_handling': 'Converted to Unix timestamp',
            'categorical_handling': 'Label encoding',
            'scaling': 'None (raw features used)'
        },
        'label_encoders': model_info['label_encoders']
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("MODELS CREATED SUCCESSFULLY!")
    print("="*60)
    print("Files created:")
    print("- models/binary_model.pkl")
    print("- models/multiclass_model.pkl")
    print("- models/model_metadata.json")
    print(f"\nBinary Model Accuracy: {binary_accuracy:.4f}")
    print(f"Multiclass Model Accuracy: {multiclass_accuracy:.4f}")
    print("\nReady to run the web application! 🚀")
    
    return metadata

if __name__ == "__main__":
    create_simple_models()
