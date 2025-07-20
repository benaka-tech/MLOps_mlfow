#!/bin/bash

# Model deployment script
set -e

# Load environment variables
source .env 2>/dev/null || echo "No .env file found"

# Configuration
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://localhost:5000"}
MODEL_NAME="health_claim_cost_model"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Deploy model to staging
deploy_staging() {
    log "Deploying to staging..."
    
    # Transition model to staging in MLflow
    python -c "
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
latest_version = client.get_latest_versions(name='${MODEL_NAME}', stages=['None'])[0]
client.transition_model_version_stage(
    name='${MODEL_NAME}',
    version=latest_version.version,
    stage='Staging'
)
"
    
    # Kill existing staging process if running
    pkill -f "uvicorn.*:8001" || true
    
    # Deploy API to staging environment
    export MODEL_STAGE=Staging
    nohup uvicorn app.api:app --host 0.0.0.0 --port 8001 > staging_api.log 2>&1 &
    log "Staging API started on port 8001. Check staging_api.log for details."
}

# Promote to production
promote_to_prod() {
    log "Promoting model to production..."
    
    # Transition model to production in MLflow
    python -c "
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
latest_staging = client.get_latest_versions(name='${MODEL_NAME}', stages=['Staging'])[0]
client.transition_model_version_stage(
    name='${MODEL_NAME}',
    version=latest_staging.version,
    stage='Production'
)
"
    
    # Kill existing production process if running
    pkill -f "uvicorn.*:8000" || true
    
    # Deploy API to production environment
    export MODEL_STAGE=Production
    nohup uvicorn app.api:app --host 0.0.0.0 --port 8000 > production_api.log 2>&1 &
    log "Production API started on port 8000. Check production_api.log for details."
}

# Rollback to previous version
rollback() {
    log "Rolling back to previous version..."
    
    # Get previous production version and restore it
    python -c "
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
versions = client.get_latest_versions(name='${MODEL_NAME}')
prod_versions = [v for v in versions if v.current_stage == 'Production']
if len(prod_versions) > 1:
    client.transition_model_version_stage(
        name='${MODEL_NAME}',
        version=prod_versions[1].version,
        stage='Production'
    )
"
    
    # Restart production API
    pkill -f "uvicorn.*:8000" || true
    export MODEL_STAGE=Production
    nohup uvicorn app.api:app --host 0.0.0.0 --port 8000 > production_api.log 2>&1 &
    log "Production API restarted on port 8000. Check production_api.log for details."
}

# Main execution
case "$1" in
    "staging")
        deploy_staging
        ;;
    "promote")
        promote_to_prod
        ;;
    "rollback")
        rollback
        ;;
    *)
        echo "Usage: $0 {build|staging|promote|rollback}"
        echo "  build    - Build Docker image"
        echo "  staging  - Deploy to staging"
        echo "  promote  - Promote staging to production"
        echo "  rollback - Rollback to previous version"
        exit 1
        ;;
esac
