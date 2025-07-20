"""
Prediction module for the Health Insurance MLOps project.
"""

import mlflow
import mlflow.pyfunc
import pandas as pd
import logging
from datetime import datetime

from config import MLFLOW_TRACKING_URI, MODEL_NAME
from utils import (
    setup_logging,
    validate_input_data,
    log_model_access,
    monitor_prediction,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY
)

logger = setup_logging()

def get_latest_model():
    """
    Get the latest production model from MLflow model registry.
    
    Returns:
        object: Latest production model
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
        log_model_access(
            logger,
            "prediction_service",
            "production_model",
            "model_loaded"
        )
        return model
    except Exception as e:
        logger.error(f"Error loading production model: {str(e)}")
        raise

def load_model(run_id: str = None, stage: str = "Production"):
    """
    Load a trained model from MLflow.
    
    Args:
        run_id (str, optional): MLflow run ID for the model to load
        stage (str, optional): Model stage to load (Production, Staging, None)
        
    Returns:
        object: Loaded model
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        if run_id:
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        else:
            model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{stage}")
            
        log_model_access(
            logger,
            "prediction_service",
            run_id or stage,
            "model_loaded"
        )
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@monitor_prediction
def predict(model, data: pd.DataFrame):
    """
    Make predictions using the loaded model.
    
    Args:
        model: Loaded ML model
        data (pd.DataFrame): Input data for predictions
        
    Returns:
        array-like: Model predictions
    """
    try:
        # Validate input data
        if not validate_input_data(data):
            raise ValueError("Invalid input data format")
        
        # Make predictions
        predictions = model.predict(data)
        
        # Log successful prediction
        log_model_access(
            logger,
            "prediction_service",
            "production_model",
            "prediction_success"
        )
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        log_model_access(
            logger,
            "prediction_service",
            "production_model",
            f"prediction_error: {str(e)}"
        )
        raise
    try:
        # Log prediction request
        mlflow.log_metric("prediction_requests", 1)
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        mlflow.log_metric("prediction_errors", 1)
        raise e
