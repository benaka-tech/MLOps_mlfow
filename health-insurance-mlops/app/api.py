"""
FastAPI application for health insurance predictions.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import mlflow
import pandas as pd
import time
from datetime import datetime

from src.predict import get_latest_model, predict
from src.utils import (
    setup_logging,
    validate_input_data,
    setup_monitoring,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY
)

# Setup logging and monitoring
logger = setup_logging()
setup_monitoring(port=8000)

app = FastAPI(title="Health Insurance Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    age: int
    bmi: float
    children: int
    smoker: bool
    region: str
    gender: str
    
    @validator('age')
    def validate_age(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Age must be between 0 and 100')
        return v
    
    @validator('bmi')
    def validate_bmi(cls, v):
        if not 10 <= v <= 50:
            raise ValueError('BMI must be between 10 and 50')
        return v
    
    @validator('children')
    def validate_children(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('Number of children must be between 0 and 10')
        return v
    
    @validator('region')
    def validate_region(cls, v):
        valid_regions = ['northeast', 'northwest', 'southeast', 'southwest']
        if v.lower() not in valid_regions:
            raise ValueError(f'Region must be one of {valid_regions}')
        return v.lower()
    
    @validator('gender')
    def validate_gender(cls, v):
        valid_genders = ['male', 'female']
        if v.lower() not in valid_genders:
            raise ValueError(f'Gender must be one of {valid_genders}')
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "bmi": 25.5,
                "children": 2,
                "smoker": False,
                "region": "southwest",
                "gender": "female"
            }
        }

class PredictionOutput(BaseModel):
    prediction: float
    model_version: str
    prediction_time: str

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.on_event("startup")
async def startup_event():
    """
    Load the model on startup
    """
    global model
    try:
        model = get_latest_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Error loading model on startup: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionOutput)
async def make_prediction(input_data: PredictionInput):
    """
    Make a prediction using the trained model.
    """
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        start_time = time.time()
        prediction = predict(model, data)
        prediction_time = time.time() - start_time
        
        # Log latency
        PREDICTION_LATENCY.observe(prediction_time)
        
        return PredictionOutput(
            prediction=float(prediction[0]),
            model_version=getattr(model, 'version', 'unknown'),
            prediction_time=datetime.now().isoformat()
        )
    
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Validate input data
        if not validate_input_data(data):
            raise HTTPException(status_code=400, detail="Invalid input data format")
        
        # Make prediction
        prediction = predict(model, data)
        
        return PredictionOutput(
            prediction=float(prediction[0]),
            model_version=model.metadata.get("mlflow.modelVersion", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
