#!/bin/bash

# Main MLOps orchestration script
set -e  # Exit on error

# Load environment variables
source .env 2>/dev/null || echo "No .env file found"

# Configuration
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://localhost:5000"}
export MODEL_STAGE=${MODEL_STAGE:-"Production"}
export DATA_VERSION=${DATA_VERSION:-"v1"}

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if MLflow server is running
check_mlflow_server() {
    if ! curl -s ${MLFLOW_TRACKING_URI} > /dev/null; then
        error "MLflow server is not running. Please start it first."
        exit 1
    fi
    log "MLflow server is running"
}

# Generate synthetic data
generate_data() {
    log "Generating synthetic data..."
    python health-insurance-mlops/src/generate_data.py
}

# Train model
train_model() {
    log "Training model..."
    python health-insurance-mlops/src/train.py
}

# Start API service
start_api() {
    log "Starting FastAPI service..."
    uvicorn health-insurance-mlops.app.api:app --reload --port 8000
}

# Run tests
run_tests() {
    log "Running tests..."
    python -m pytest health-insurance-mlops/tests/
}

# Main execution
case "$1" in
    "setup")
        log "Setting up environment..."
        pip install -r health-insurance-mlops/requirements.txt
        ;;
    "data")
        generate_data
        ;;
    "train")
        check_mlflow_server
        train_model
        ;;
    "serve")
        check_mlflow_server
        start_api
        ;;
    "test")
        run_tests
        ;;
    "pipeline")
        check_mlflow_server
        generate_data
        train_model
        run_tests
        log "Pipeline completed successfully!"
        ;;
    *)
        echo "Usage: $0 {setup|data|train|serve|test|pipeline}"
        echo "  setup    - Install requirements and setup environment"
        echo "  data     - Generate synthetic data"
        echo "  train    - Train and register model"
        echo "  serve    - Start the prediction API"
        echo "  test     - Run tests"
        echo "  pipeline - Run full pipeline (data generation, training, testing)"
        exit 1
        ;;
esac
