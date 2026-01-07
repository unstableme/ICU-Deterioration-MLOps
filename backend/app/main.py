import uvicorn
from fastapi import FastAPI, HTTPException
from app.inference import ICUModel
from app.schema import PredictionRequest, PredictionResponse
from pathlib import Path
import os

app = FastAPI(title="ICU Deterioration Prediction API")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "artifacts" / "cnn_gru_model.pth"))
DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "data" / "processed" / "set_c_processed.pkl"))

model = ICUModel(model_path=MODEL_PATH, data_path=DATA_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the ICU Deterioration Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = model.predict(patient_id=request.patient_id, end_hour=request.end_hour)
        return PredictionResponse(**prediction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)