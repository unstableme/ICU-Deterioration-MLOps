import torch
import pickle

class ICUModel():
    """ICU Deterioration Prediction Model Inference Class"""
    def __init__(self, model_path, data_path, window_size=12, threshold=0.49):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        with open(data_path, 'rb') as f:
            self.patient_data_dict = pickle.load(f)

        self.threshold = threshold
        self.window_size = window_size
        self.model.eval()

    def get_patient_data(self, patient_id):
        """Retrieve patient time-series data by patient ID."""

        if patient_id not in self.patient_data_dict:
            raise ValueError("Patient ID not found in the dataset.")
        return self.patient_data_dict[patient_id]

    def predict(self, patient_id, end_hour):
        """Make prediction for a given patient up to the specified end hour."""

        patient_ts_data = self.get_patient_data(patient_id)

        if end_hour < self.window_size:
            raise ValueError("Not Enough history to make prediction.")
        
        window = patient_ts_data[end_hour - self.window_size:end_hour]
        x = window.to(self.device).float().unsqueeze(0)  

        with torch.no_grad():
            logits = self.model(x)
            risk_score = torch.sigmoid(logits).item()

        alert = (
            "LOW" if risk_score < 0.4 else
            "MEDIUM" if 0.4 <= risk_score < 0.7 else
            "HIGH" 
        )

        return {
            "risk_score": risk_score,
            "alert": alert,
            "vitals": window.tolist()
        }

        

    
