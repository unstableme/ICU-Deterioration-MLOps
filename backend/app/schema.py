from pydantic import BaseModel

class PredictionRequest(BaseModel):
    patient_id: str
    end_hour: int

class PredictionResponse(BaseModel):
    risk_score: float
    alert: str
    vitals: list