"""
Model Preparation Module
This module prepares and exports both binary and multiclass models with proper preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """Custom preprocessing pipeline for consistent data transformation"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        """Fit the preprocessing pipeline"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_processed = X.copy()
        else:
            X_processed = pd.DataFrame(X)
            
        # Handle datetime columns
        for col in X_processed.columns:
            if pd.api.types.is_datetime64_any_dtype(X_processed[col]):
                X_processed[col] = pd.to_datetime(X_processed[col]).astype(np.int64) // 10**9
            elif X_processed[col].dtype == 'object':
                # Apply label encoding to strings
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Fit scaler on processed data
        self.scaler.fit(X_processed)
        return self
    
    def transform(self, X):
        """Transform the data using fitted preprocessing"""
        if isinstance(X, pd.DataFrame):
            X_processed = X.copy()
        else:
            X_processed = pd.DataFrame(X, columns=self.feature_names if self.feature_names else None)
            
        # Apply same transformations as in fit
        for col in X_processed.columns:
            if pd.api.types.is_datetime64_any_dtype(X_processed[col]):
                X_processed[col] = pd.to_datetime(X_processed[col]).astype(np.int64) // 10**9
            elif X_processed[col].dtype == 'object':
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_vals = set(X_processed[col].astype(str))
                    known_vals = set(self.label_encoders[col].classes_)
                    unknown_vals = unique_vals - known_vals
                    
                    if unknown_vals:
                        # Replace unknown values with most frequent known value
                        most_frequent = self.label_encoders[col].classes_[0]
                        X_processed[col] = X_processed[col].astype(str).replace(
                            list(unknown_vals), most_frequent
                        )
                    
                    X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
        
        # Apply scaling
        return self.scaler.transform(X_processed)

class RFTransformer(BaseEstimator, TransformerMixin):
    """Random Forest Transformer for feature extraction"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf = None

    def fit(self, X, y):
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        self.rf.fit(X, y)
        return self

    def transform(self, X):
        return self.rf.predict_proba(X)

class ModelPreparation:
    """Main class for preparing and exporting models"""
    
    def __init__(self, data_path="fe_outcomes2.csv"):
        self.data_path = data_path
        self.df = None
        self.preprocessor = None
        self.binary_pipeline = None
        self.multiclass_pipeline = None
        self.feature_schema = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Revert normalized status back to integer classes 0, 1, 2, 3
        if self.df["status"].max() <= 1.0:
            self.df["status"] = (self.df["status"] * 3).round().astype(int)
            
        # Separate features and target
        X = self.df.drop("status", axis=1)
        y = self.df["status"]
        
        # Create binary target (convert multiclass to binary: 1,2 -> 0, 3 -> 1)
        y_binary = (y == 3).astype(int)
        
        # Store feature schema for API documentation
        self.feature_schema = {
            'feature_names': X.columns.tolist(),
            'feature_types': X.dtypes.to_dict(),
            'sample_values': {}
        }
        
        # Add sample values for each feature
        for col in X.columns:
            if X[col].dtype == 'object':
                self.feature_schema['sample_values'][col] = X[col].dropna().unique()[:5].tolist()
            else:
                self.feature_schema['sample_values'][col] = {
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'mean': float(X[col].mean())
                }
        
        return X, y, y_binary
    
    def create_binary_pipeline(self):
        """Create binary classification pipeline"""
        print("Creating binary classification pipeline...")
        
        # Simple binary classifier with preprocessing
        self.binary_pipeline = Pipeline([
            ('preprocessor', PreprocessingPipeline()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        return self.binary_pipeline
    
    def create_multiclass_pipeline(self):
        """Create multiclass classification pipeline (RF -> ET)"""
        print("Creating multiclass classification pipeline...")
        
        self.multiclass_pipeline = Pipeline([
            ('preprocessor', PreprocessingPipeline()),
            ('rf_transformer', RFTransformer(n_estimators=100, random_state=42)),
            ('classifier', ExtraTreesClassifier(n_estimators=100, random_state=42))
        ])
        
        return self.multiclass_pipeline
    
    def train_and_evaluate_models(self):
        """Train and evaluate both models"""
        X, y_multiclass, y_binary = self.load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_multi_train, y_multi_test, y_bin_train, y_bin_test = train_test_split(
            X, y_multiclass, y_binary, test_size=0.2, random_state=42
        )
        
        # Create and train binary model
        print("\n" + "="*50)
        print("TRAINING BINARY CLASSIFICATION MODEL")
        print("="*50)
        
        binary_pipeline = self.create_binary_pipeline()
        binary_pipeline.fit(X_train, y_bin_train)
        
        # Evaluate binary model
        y_bin_pred = binary_pipeline.predict(X_test)
        binary_accuracy = accuracy_score(y_bin_test, y_bin_pred)
        
        print(f"Binary Model Accuracy: {binary_accuracy:.4f}")
        print("\nBinary Classification Report:")
        print(classification_report(y_bin_test, y_bin_pred))
        
        # Create and train multiclass model
        print("\n" + "="*50)
        print("TRAINING MULTICLASS CLASSIFICATION MODEL")
        print("="*50)
        
        multiclass_pipeline = self.create_multiclass_pipeline()
        multiclass_pipeline.fit(X_train, y_multi_train)
        
        # Evaluate multiclass model
        y_multi_pred = multiclass_pipeline.predict(X_test)
        multiclass_accuracy = accuracy_score(y_multi_test, y_multi_pred)
        
        print(f"Multiclass Model Accuracy: {multiclass_accuracy:.4f}")
        print("\nMulticlass Classification Report:")
        print(classification_report(y_multi_test, y_multi_pred))
        
        # Store models
        self.binary_pipeline = binary_pipeline
        self.multiclass_pipeline = multiclass_pipeline
        
        return {
            'binary_accuracy': binary_accuracy,
            'multiclass_accuracy': multiclass_accuracy,
            'sample_test_data': {
                'X_test_sample': X_test.head(3).to_dict('records'),
                'y_binary_sample': y_bin_test[:3].tolist(),
                'y_multiclass_sample': y_multi_test[:3].tolist()
            }
        }
    
    def save_models_and_metadata(self, models_dir="models"):
        """Save trained models and metadata"""
        os.makedirs(models_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.binary_pipeline, os.path.join(models_dir, 'binary_model.pkl'))
        joblib.dump(self.multiclass_pipeline, os.path.join(models_dir, 'multiclass_model.pkl'))
        
        # Save feature schema and metadata
        metadata = {
            'feature_schema': self.feature_schema,
            'model_info': {
                'binary_model': {
                    'type': 'Binary Classification',
                    'classes': [0, 1],
                    'class_labels': {'0': 'Not Status 3', '1': 'Status 3'}
                },
                'multiclass_model': {
                    'type': 'Multiclass Classification', 
                    'classes': [1, 2, 3],
                    'class_labels': {'1': 'Status 1', '2': 'Status 2', '3': 'Status 3'}
                }
            },
            'preprocessing_info': {
                'datetime_handling': 'Converted to Unix timestamp',
                'categorical_handling': 'Label encoding',
                'scaling': 'StandardScaler applied'
            }
        }
        
        with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\nModels and metadata saved to '{models_dir}' directory:")
        print("- binary_model.pkl")
        print("- multiclass_model.pkl") 
        print("- model_metadata.json")
        
        return metadata

def main():
    """Main function to prepare and export models"""
    print("="*60)
    print("ML PIPELINE MODEL PREPARATION")
    print("="*60)
    
    # Initialize model preparation
    model_prep = ModelPreparation()
    
    # Train and evaluate models
    results = model_prep.train_and_evaluate_models()
    
    # Save models and metadata
    metadata = model_prep.save_models_and_metadata()
    
    print("\n" + "="*60)
    print("MODEL PREPARATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Binary Model Accuracy: {results['binary_accuracy']:.4f}")
    print(f"Multiclass Model Accuracy: {results['multiclass_accuracy']:.4f}")
    print("\nReady for deployment! 🚀")
    
    return model_prep, results, metadata

if __name__ == "__main__":
    main()
