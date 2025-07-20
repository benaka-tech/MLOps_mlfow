"""
Data loading utilities for the Health Insurance MLOps project.
"""

import pandas as pd
from config import HEALTH_CLAIMS_FILE

def load_data():
    """
    Load the health claims dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    try:
        df = pd.read_csv(HEALTH_CLAIMS_FILE)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {HEALTH_CLAIMS_FILE}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
