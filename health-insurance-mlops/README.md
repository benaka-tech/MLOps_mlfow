# Health Insurance MLOps Project

This project implements an end-to-end MLOps pipeline for health insurance claim predictions using MLflow.

## Project Structure

```
health-insurance-mlops/
├── data/               # Data files
│   └── health_claims.csv
├── mlruns/            # MLflow logging directory
├── models/            # Optional local model cache
├── src/               # Source code
│   ├── config.py      # Configuration settings
│   ├── data_loader.py # Data loading utilities
│   ├── preprocessing.py # Data preprocessing
│   ├── train.py       # Model training
│   ├── predict.py     # Prediction utilities
│   └── utils.py       # Helper functions
├── app/               # FastAPI application
│   └── api.py         # API endpoints
├── requirements.txt   # Project dependencies
├── Dockerfile         # Container definition
└── README.md         # This file
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure MLflow tracking URI in `src/config.py`

## Usage

1. Train the model:
```bash
python src/train.py
```

2. Start the API:
```bash
uvicorn app.api:app --reload
```

## Docker

Build the container:
```bash
docker build -t health-insurance-api .
```

Run the container:
```bash
docker run -p 8000:8000 health-insurance-api
```
