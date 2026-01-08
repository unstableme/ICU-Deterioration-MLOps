import torch
import pickle
import mlflow.pytorch 
from pathlib import Path
from io import BytesIO

mlflow.set_tracking_uri("https://dagshub.com/unstableme/ICU-Deterioration-MLOps.mlflow")
# PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ICU-Deterioration-MLOps
class ICUModel():
    """ICU Deterioration Prediction Model Inference Class"""
    def __init__(self, data_path, window_size=12, threshold=0.49):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = mlflow.pytorch.load_model(
            model_uri="models:/CNN_GRU_ICU_Deterioration_Model/7",
            map_location=self.device
        )

        # Download patient data if it's a URL: downloading set_c_processed.pkl
        if str(data_path).startswith("http"):
            response = requests.get(data_path)
            response.raise_for_status()
            self.patient_data_dict = pickle.loads(response.content)
        else:
            with open(data_path, "rb") as f:
                self.patient_data_dict = pickle.load(f)

        # Load scaler (from URL or local)
        if scaler_path is None:
            scaler_path = "https://dagshub.com/unstableme/ICU-Deterioration-MLOps/src/main/data/processed/scaler.pkl"

        if str(scaler_path).startswith("http"):
            response = requests.get(scaler_path)
            response.raise_for_status()
            self.scaler = pickle.loads(response.content)
        else:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        self.threshold = threshold
        self.window_size = window_size
        self.model.eval()
    
    def get_patient_data(self, record_id):
        """Retrieve patient time-series data by RecordID."""
        try:
            idx = self.patient_data_dict['record_ids'].index(str(record_id))
            patient_ts_data = self.patient_data_dict['X'][idx]
            return patient_ts_data
        except ValueError:
            raise ValueError(f"RecordID {record_id} not found in the dataset.")
    
    def predict(self, record_id, end_hour):
        """Make prediction for a given record up to the specified end hour."""
        patient_ts_data = self.get_patient_data(record_id)
        
        if end_hour < self.window_size:
            raise ValueError(f"Not enough history to make prediction. Need at least {self.window_size} hours, got {end_hour}.")
        
        if end_hour > len(patient_ts_data):
            raise ValueError(f"end_hour {end_hour} exceeds available data length {len(patient_ts_data)}.")
        
        window_np = patient_ts_data[end_hour - self.window_size:end_hour]
         # Convert numpy array to torch tensor
        x = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            risk_score = torch.sigmoid(logits).item()
        
        # Inverse scale
        window_original = self.scaler.inverse_transform(window_np)

        # Return as list for API
        vitals_list = window_original.tolist() 

        alert = (
            "LOW" if risk_score < 0.4 else
            "MEDIUM" if 0.4 <= risk_score < 0.7 else
            "HIGH" 
        )
        
        return {
            "risk_score": risk_score,
            "alert": alert,
            "vitals": vitals_list
        }