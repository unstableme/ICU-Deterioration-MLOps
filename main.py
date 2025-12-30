import numpy as np

from src.data.data_ingestion import ICUDataIngestion
from src.data.data_preprocessing import (
    all_patients_to_wide, get_global_columns, align_the_patients,
    scale_patient_data, scaled_patients_to_padded_array, save_processed_datasets
)   
from src.data.dataloader_for_torch import get_dataloader
from src.model.train_utils import Trainer
from src.config import load_params

from src.logger import get_logger

logger = get_logger(__name__, log_file='data_preprocessing.log')


def main():
    """Main function to orchestrate data preprocessing, model training, and evaluation."""

    ingest = ICUDataIngestion() 
    PARAMS = load_params()
    sets = ['a', 'b', 'c']
    all_patients = {}
    global_columns = {}
    aligned_patients = {}
    scaled_patients = {}
    labels = {}
    datasets = {}


    for set_name in sets:
        all_patients[set_name] = all_patients_to_wide(set_name)
        global_columns[set_name] = get_global_columns(all_patients[set_name])
        aligned_patients[set_name] = align_the_patients(all_patients[set_name], global_columns[set_name])
        scaled_patients[set_name] = scale_patient_data(aligned_patients[set_name])

        X, record_ids = scaled_patients_to_padded_array(scaled_patients[set_name])
        outcomes_file = ingest.get_outcomes_file(set_name)
        labels[set_name] = {str(k).strip(): v for k, v in zip(outcomes_file['RecordID'], outcomes_file['In-hospital_death'])}
        y = np.array([labels[set_name].get(rid, np.nan) for rid in record_ids])

        datasets[set_name] = {
                'X': X,
                'y': y
            }
        
    save_processed_datasets(datasets)

    # Training 

    train_loaders, val_loaders, test_loaders = get_dataloader()

    trainer = Trainer(train_loaders, val_loaders, test_loaders)
    
    trainer.train_val_epochs(epochs=PARAMS['training']['num_epochs'])
    test_metrics = trainer.test_time()
    print(test_metrics)
    trainer.save_model()
    logger.info("Model training and saving completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred in the main execution.")
        raise