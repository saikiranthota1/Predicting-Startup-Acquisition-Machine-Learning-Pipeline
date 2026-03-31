"""
Shared ML utilities and custom classes for the pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

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
