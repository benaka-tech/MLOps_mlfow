"""
Configuration settings for the Health Insurance MLOps project.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data files
DATA_PATH = "data/health_claims.csv"
HEALTH_CLAIMS_FILE = os.path.join(BASE_DIR, DATA_PATH)

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "health-claim-prediction"
MODEL_NAME = "health_claim_cost_model"

# Model parameters
TARGET = "claim_amount"
RANDOM_STATE = 42
TEST_SIZE = 0.2
