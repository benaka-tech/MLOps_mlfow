"""
Utility functions for the Health Insurance MLOps project.
"""

import logging
import json
from datetime import datetime
from prometheus_client import Counter, Histogram, start_http_server
import hashlib

# Prometheus metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')
DATA_VALIDATION_ERRORS = Counter('data_validation_errors', 'Number of data validation errors')

def setup_logging(log_file='mlops.log'):
    """
    Set up logging configuration for the project.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('health_insurance_mlops')

def setup_monitoring(port=8000):
    """
    Setup Prometheus monitoring server
    """
    start_http_server(port)
    logging.info(f"Prometheus metrics server started on port {port}")

def validate_input_data(data):
    """
    Validate input data format and contents.
    
    Args:
        data: Input data to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    try:
        required_columns = ['age', 'bmi', 'children', 'smoker', 'region', 'gender']
        
        # Check for required columns
        if not all(col in data.columns for col in required_columns):
            DATA_VALIDATION_ERRORS.inc()
            return False
            
        # Validate data types and ranges
        if not (0 <= data['age'].min() <= 100):
            DATA_VALIDATION_ERRORS.inc()
            return False
            
        if not (10 <= data['bmi'].min() <= 50):
            DATA_VALIDATION_ERRORS.inc()
            return False
            
        if not (0 <= data['children'].min() <= 10):
            DATA_VALIDATION_ERRORS.inc()
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Data validation error: {str(e)}")
        DATA_VALIDATION_ERRORS.inc()
        return False

def log_model_access(logger, user_id, model_version, action):
    """
    Log model access for auditing.
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'model_version': model_version,
        'action': action
    }
    logger.info(f"Model Access: {json.dumps(log_entry)}")

def hash_data(data):
    """
    Create a hash of the data for versioning and tracking.
    """
    return hashlib.sha256(data.to_json().encode()).hexdigest()

def monitor_prediction(func):
    """
    Decorator to monitor prediction latency and count.
    """
    def wrapper(*args, **kwargs):
        PREDICTION_COUNTER.inc()
        with PREDICTION_LATENCY.time():
            return func(*args, **kwargs)
    return wrapper
