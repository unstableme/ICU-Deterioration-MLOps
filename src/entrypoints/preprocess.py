import numpy as np 
import os
import pickle 
from src.data.data_ingestion import ICUDataIngestion
from src.data.data_preprocessing import (
    all_patients_to_wide, get_global_columns, align_the_patients,
    fit_and_scale_patients_data, scale_patients_data_with_existing_scaler,
    scaled_patients_to_padded_array, save_processed_datasets
)   
from src.logger import get_logger
from pathlib import Path

logger = get_logger(__name__, log_file='entrypoints_preprocessing.log')

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ICU-Deterioration-MLOps
save_scaler_here = PROJECT_ROOT / "data" / "processed" / "scaler.pkl"
def main():
    """Main function to orchestrate data preprocessing."""
    
    ingest = ICUDataIngestion() 
    sets = ['a', 'b', 'c']
    all_patients = {}
    global_columns = {}
    aligned_patients = {}
    scaled_patients = {}
    labels = {}
    datasets = {}

    for set_name in sets:
        # Step 1: Convert to wide format
        all_patients[set_name] = all_patients_to_wide(set_name)

        # Step 2: Determine global columns
        global_columns[set_name] = get_global_columns(all_patients[set_name])

        # Step 3: Align all patients to global columns
        aligned_patients[set_name] = align_the_patients(all_patients[set_name], global_columns[set_name])


    scaled_patients['a'], scaler = fit_and_scale_patients_data(aligned_patients['a'])
    
    #save the scaler to later use for two set and also inverse transform for UI
    os.makedirs(save_scaler_here.parent, exist_ok=True)
    scaler_info = {
        "scaler": scaler,
        "columns": global_columns['c'].tolist()
    }


    with open(save_scaler_here, "wb") as f:
        pickle.dump(scaler_info, f)

    scaled_patients['b'] = scale_patients_data_with_existing_scaler(
        aligned_patients['b'], scaler
    )

    scaled_patients['c'] = scale_patients_data_with_existing_scaler(
        aligned_patients['c'], scaler
)


    for set_name in sets:
        
        # Step 4: Convert scaled patients to 3D NumPy array
        X, record_ids = scaled_patients_to_padded_array(scaled_patients[set_name])

        # Step 5: Check for NaNs in the final array (just to be safe)
        if np.isnan(X).any():
            logger.error(f"NaNs found in the final array for set {set_name}")
            raise ValueError(f"NaNs found in the final array for set {set_name}")

        # Step 6: Map labels to record_ids
        outcomes_file = ingest.get_outcomes_file(set_name)
        labels_map = {str(k).strip(): v for k, v in zip(outcomes_file['RecordID'], outcomes_file['In-hospital_death'])}
        y = np.array([labels_map.get(rid, 0) for rid in record_ids])

        # Step 7: Store processed dataset
        datasets[set_name] = {'X': X, 'y': y, 'record_ids': record_ids}

    # Step 8: Save datasets to disk
    save_processed_datasets(datasets)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred in the data preprocessing process.")
        raise
