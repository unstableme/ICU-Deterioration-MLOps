import uvicorn
from fastapi import FastAPI, HTTPException
from app.inference import ICUModel
from app.schema import PredictionRequest, PredictionResponse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # ICU-Deterioration-MLOps
sys.path.append(str(ROOT_DIR))

app = FastAPI(title="ICU Deterioration Prediction API")

DATA_PATH = Path(os.getenv("DATA_PATH", ROOT_DIR / "data" / "processed" / "set_c_processed.pkl"))
model = ICUModel(data_path=DATA_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the ICU Deterioration Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = model.predict(record_id=request.record_id, end_hour=request.end_hour)
        return PredictionResponse(**prediction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)