import uvicorn
from fastapi import FastAPI, HTTPException, Header
from app.inference import ICUModel
from app.schema import PredictionRequest, PredictionResponse
import numpy as np
import sys
from pathlib import Path
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT, RISK_SCORE_DIST
import time 
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import os

ROOT_DIR = Path(__file__).resolve().parents[2]  # ICU-Deterioration-MLOps
sys.path.append(str(ROOT_DIR))

METRICS_TOKEN = os.getenv("METRICS_TOKEN")

app = FastAPI(title="ICU Deterioration Prediction API")

#DATA_PATH = Path(os.getenv("DATA_PATH", ROOT_DIR / "data" / "processed" / "set_c_processed.pkl"))
DATA_PATH = "https://dagshub.com/unstableme/ICU-Deterioration-MLOps/raw/main/data/processed/set_c_processed.pkl"
model = ICUModel(data_path=DATA_PATH)

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the ICU Deterioration Prediction API"}

@app.get("/patients")
def list_patients(sample_size:int=6):
    """List 6 random patient IDs from the dataset."""
    all_ids = model.patient_data_dict['record_ids']
    if sample_size <= len(all_ids):
        sampled_ids = all_ids
    else:
        sampled_ids = list(np.random.choice(all_ids, size=sample_size, replace=False))
    return {"patient_ids": sampled_ids}
    

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make prediction for a given patient and end hour."""
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()

    try:
        prediction = model.predict(
            record_id=request.record_id,
            end_hour=request.end_hour
            )
        RISK_SCORE_DIST.observe(prediction["risk_score"])
        return PredictionResponse(**prediction)
    except ValueError as e:
        ERROR_COUNT.labels(endpoint="/predict").inc()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

@app.get("/metrics")
def metrics(authorization:str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=403, detail="Forbidden")
        
    if not authorization.starts_with("Bearer "):
        raise HTTPException(status_code=403, detail="Forbidden")
    
    token = authorization.replace("Bearer ", "")
    if x_metrics_token != METRICS_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    return Response(
        generate_latest(),
        media_type = CONTENT_TYPE_LATEST
    )

    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)