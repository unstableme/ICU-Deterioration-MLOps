import numpy as np 
from src.data.data_ingestion import ICUDataIngestion
from src.data.data_preprocessing import (
    all_patients_to_wide, get_global_columns, align_the_patients,
    scale_patient_data, scaled_patients_to_padded_array, save_processed_datasets
)   
from src.logger import get_logger

logger = get_logger(__name__, log_file='entrypoints_preprocessing.log')

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

        # Step 4: Scale patient data
        scaled_patients[set_name] = scale_patient_data(aligned_patients[set_name])

        # Step 5: Check NaNs individually per patient (before converting to array)
        # for rid, df in scaled_patients[set_name]:
        #     if df.isna().any().any():
        #         logger.error(f"NaNs found in patient {rid} of set {set_name}")
        #         raise ValueError(f"NaNs found in patient {rid} of set {set_name}")

        # Step 6: Convert scaled patients to 3D NumPy array
        X, record_ids = scaled_patients_to_padded_array(scaled_patients[set_name])

        # Step 7: Check for NaNs in the final array (just to be safe)
        if np.isnan(X).any():
            logger.error(f"NaNs found in the final array for set {set_name}")
            raise ValueError(f"NaNs found in the final array for set {set_name}")

        # Step 8: Map labels to record_ids
        outcomes_file = ingest.get_outcomes_file(set_name)
        labels_map = {str(k).strip(): v for k, v in zip(outcomes_file['RecordID'], outcomes_file['In-hospital_death'])}
        y = np.array([labels_map.get(rid, 0) for rid in record_ids])

        # Step 9: Store processed dataset
        datasets[set_name] = {'X': X, 'y': y}

    # Step 10: Save datasets to disk
    save_processed_datasets(datasets)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred in the data preprocessing process.")
        raise
