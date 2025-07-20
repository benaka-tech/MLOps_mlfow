"""
Data preprocessing utilities for the Health Insurance MLOps project.
"""

import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE, TARGET
from utils import hash_data

def create_preprocessing_pipeline():
    """
    Create the preprocessing pipeline with both numeric and categorical transformations.
    """
    numeric_features = ['age', 'bmi']
    categorical_features = ['smoker', 'region', 'gender']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, numeric_features, categorical_features

def preprocess_data(df: pd.DataFrame, track_mlflow=True):
    """
    Preprocess the input data for model training.
    
    Args:
        df (pd.DataFrame): Raw input data
        track_mlflow (bool): Whether to track preprocessing in MLflow
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    if track_mlflow:
        mlflow.log_param("data_version", hash_data(df))
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", len(df.columns) - 1)
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Create and fit preprocessing pipeline
    preprocessor, numeric_features, categorical_features = create_preprocessing_pipeline()
    
    # Split features and target
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Fit and transform the data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names after transformation
    numeric_cols = numeric_features
    categorical_cols = []
    for feature in categorical_features:
        unique_values = X_train[feature].unique()
        categorical_cols.extend([f"{feature}_{val}" for val in unique_values[1:]])
    
    # Convert to DataFrame to maintain column names
    feature_names = numeric_cols + categorical_cols
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    if track_mlflow:
        # Log preprocessing info
        mlflow.log_param("preprocessing_steps", "fillna,standardization,onehot")
        mlflow.log_param("numeric_features", numeric_features)
        mlflow.log_param("categorical_features", categorical_features)
        mlflow.sklearn.log_model(preprocessor, "preprocessor")
    
    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
