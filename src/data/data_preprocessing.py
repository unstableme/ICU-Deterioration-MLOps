import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import torch
from torch.utils.data import IterableDataset
from src.config import load_params

from src.logger import get_logger
from src.data.data_ingestion import ICUDataIngestion

ingest = ICUDataIngestion()
logger = get_logger(__name__, log_file='data_preprocessing.log')
PARAMS = load_params()

def process_patient(file_path):
    """Process a single patient record from a CSV file into a wide-format DataFrame and attach record id."""
    record_id = os.path.basename(file_path).split('.')[0]
    #logger.info(f"Processing patient record: {record_id} from file: {file_path}")
     
    try:
         
        df = pd.read_csv(file_path)

        def time_to_hours(time_str):
            """Convert time string 'HH:MM' to hours as float."""
            h, m = map(int, time_str.split(':'))
            return h + m / 60
        
        try:
            df['Time'] = df['Time'].apply(time_to_hours)      
        except Exception as e:
            logger.error(f"Error converting Time to hours for record {record_id}")
            raise 

        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df.sort_values('Time', inplace=True)
        wide_df = df.pivot_table(
            index='Time',
            columns='Parameter',
            values='Value',
            aggfunc='last'
        )
        wide_df = wide_df.drop(columns=['RecordID'], errors='ignore')
        wide_df.sort_index(inplace=True)
        return record_id, wide_df
    
    except Exception as e:
        logger.exception("Error processing patient file {file_path}")
        raise 


def all_patients_to_wide(set_name):
    """Process all patient files in the specified set into wide-format DataFrames."""
    patients = []
    try:
        for file_path in ingest.get_patient_files(set_name):
            record_id, wide_df = process_patient(file_path)
            patients.append((record_id, wide_df))
        logger.info(f"Converted all patients in set {set_name} to wide format with record_id attached.")
        return patients
    except Exception as e:
        logger.exception(f"Error processing all patients in set {set_name}")
        raise 


def get_global_columns(all_patients):
    """Determine the global set of columns across a patient DataFrames
       and later we will add these cols in sorted order"""
    global_columns = set()
    logger.info("Determining global columns across all patients.")
    for record_id, wide_df in all_patients:
        global_columns.update(wide_df.columns)
    logger.info(f"Total unique columns found: {len(global_columns)}")
    return sorted(global_columns)


def align_the_patients(all_patients, global_columns):
    """ Align each patient's DataFrame to have the same columns as the global set."""
    
    aligned_patients = []
    try:
        
        for record_id, wide_df in all_patients:
            aligned_df = wide_df.reindex(columns=global_columns, fill_value=0)
            aligned_patients.append((record_id, aligned_df))
        return aligned_patients
    except Exception as e:
        logger.exception("Error aligning patient data to global columns.")
        raise


def fit_and_scale_patients_data(aligned_patients):
    """ Scale patient data using StandardScaler fitted on the training data."""
    scaler = StandardScaler()
    try:
        logger.info("Fitting scaler on training data.")

        train_concat = pd.concat([df for _, df in aligned_patients])
        scaler.fit(train_concat)

        scaled_patients = []
        for rid, df in aligned_patients:
            scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
            scaled_patients.append((rid, scaled_df))
        logger.info("Scaled patient data using fitted scaler.")
        return scaled_patients, scaler
    except Exception as e:
        logger.exception("Error scaling patient data.")
        raise

def scale_patients_data_with_existing_scaler(aligned_patients, scaler):
    scaled_patients = []
    for rid, df in aligned_patients:
        scaled_df = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns,
            index=df.index
        )
        scaled_patients.append((rid, scaled_df))

    return scaled_patients



def scaled_patients_to_padded_array(scaled_patients):
    """ Convert scaled patient data into a 3D numpy array with padding."""

    try:

        # scaled_patients is a list of tuples (record_id, dataframe)
        dataframes_only = [df for _, df in scaled_patients]
        max_time_steps = max(df.shape[0] for df in dataframes_only)
        num_features = dataframes_only[0].shape[1]

        #now we first create the full length zero array
        array = np.zeros((len(dataframes_only), max_time_steps, num_features))
        logger.info(f"Creating padded array of shape {array.shape}")
        record_ids = []

        for i, (record_id, df) in enumerate(scaled_patients):
            if df.isna().any().any():
              logger.warning(f"NaNs found in patient {record_id}. Filling with 0.")
            df = df.fillna(0)  # fill missing values
            record_ids.append(record_id)
            time_steps = df.shape[0]
            array[i, :time_steps, :] = df.values
        logger.info("Converted scaled patient data to padded array.")
        return array, record_ids

    except Exception as e:
        logger.exception("Error converting scaled patient data to padded array.")
        raise



class SlidingWindowDataset(IterableDataset):
    """A PyTorch IterableDataset that generates sliding windows from patient time series data.
       Looks past `window_size` time steps to predict the outcome at `horizon` time steps ahead."""
    def __init__(self, X, y, window_size=PARAMS['data']['window_size'], horizon=PARAMS['data']['horizon']):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.horizon = horizon

        logger.info(f"Initialized SlidingWindowDataset with window_size={window_size}, horizon={horizon}")

    def __iter__(self):
        num_patients = self.X.shape[0]
        for i in range(num_patients):
            seq_len = self.X[i].shape[0]
            for t in range(self.window_size, seq_len - self.horizon + 1):
                X_window = self.X[i, t - self.window_size:t, :]
                if np.isnan(X_window).any():
                    logger.warning(f"NaNs in X_window for patient index {i}, time {t}")
                y_window = self.y[i]
                if np.isnan(X_window).any():
                    logger.warning(f"NaNs in X_window for patient index {i}, time {t}")
                
                yield X_window, y_window




def save_processed_datasets(dataset, processed_dir='data/processed'):
    """
    Save each dataset (set a, b, c) to the processed directory using pickle.
    This saved version is actually X and y not the windowed version..because storing that is heavy and 
    now I used class SlidingWindowDataset to generate windows on the fly.
    
    Args:
        datasets (dict): Dictionary containing processed data for each set.
        processed_dir (str): Path to save processed files.
    """
    os.makedirs(processed_dir, exist_ok=True)
    try:

        for set_name, data in dataset.items():
            file_path = os.path.join(processed_dir, f'set_{set_name}_processed.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)   
            logger.info(f"Saved processed dataset for set {set_name} to {file_path}")
    except Exception as e:
        logger.exception("Error saving processed datasets.")
        raise
    