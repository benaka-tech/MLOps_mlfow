"""
Model training script for the Health Insurance MLOps project.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

from data_loader import load_data
from preprocessing import preprocess_data
from config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAME, RANDOM_STATE
from utils import setup_logging, log_model_access

logger = setup_logging()

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and calculate performance metrics.
    """
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def train_model(experiment_name=EXPERIMENT_NAME):
    """
    Train the machine learning model and log artifacts with MLflow.
    """
    logger.info("Starting model training process")
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        data = load_data()
        
        with mlflow.start_run() as run:
            # Log model training start
            log_model_access(logger, "training_service", "new_training", "training_start")
            
            # Preprocess data
            X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
                data, track_mlflow=True
            )
            
            # Initialize and train model
            logger.info("Training model")
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE
            )
            
            # Log model parameters
            mlflow.log_params({
                "model_type": "RandomForestRegressor",
                "n_estimators": 100,
                "random_state": RANDOM_STATE
            })
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate model
            logger.info("Evaluating model")
            metrics = evaluate_model(model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log model and preprocessor
            logger.info("Logging model artifacts")
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=MODEL_NAME
            )
            
            # Log training completion
            log_model_access(
                logger,
                "training_service",
                f"run_id_{run.info.run_id}",
                "training_complete"
            )
            
            logger.info(f"Model training completed. Run ID: {run.info.run_id}")
            logger.info(f"Model metrics: {metrics}")
            
            return run.info.run_id, metrics
            
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise
    
if __name__ == "__main__":
    train_model()
