import torch
from torch.utils.data import DataLoader
import pickle
from pathlib import Path

from src.data.data_preprocessing import SlidingWindowDataset
from src.logger import get_logger
from src.config import load_params

logger = get_logger(__name__, log_file='dataloader_for_torch.log')
PARAMS = load_params()

def get_dataloader(window_size=PARAMS['data']['window_size'], horizon=PARAMS['data']['horizon'], batch_size=PARAMS['data']['batch_size']):
    """Create a DataLoader for the sliding window dataset."""
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2] # parent[0] is 'data', parent[1] is 'src', parent[2] is project root
    PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed'

    logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}")

    try:
        #../ moves one directory up and following does two directory up
        with open(PROCESSED_DATA_PATH/'set_a_processed.pkl', 'rb') as f:
            data_a = pickle.load(f) #returns in this format {'X': X_a, 'y': y_a}
        with open(PROCESSED_DATA_PATH/'set_b_processed.pkl', 'rb') as f:
            data_b = pickle.load(f)
        with open(PROCESSED_DATA_PATH/'set_c_processed.pkl', 'rb') as f:
            data_c = pickle.load(f)
        
        logger.info("Loaded processed datasets for sets a, b, and c.")

    except Exception as e:
        logger.exception(f"Error loading processed data: {e}")
        raise

    X_train = SlidingWindowDataset(data_a['X'], data_a['y'], window_size=window_size, horizon=horizon)
    train_loader = DataLoader(X_train, batch_size=batch_size)
    logger.info("Created DataLoader for training set.")

    X_val = SlidingWindowDataset(data_b['X'], data_b['y'], window_size=window_size, horizon=horizon)
    val_loader = DataLoader(X_val, batch_size=batch_size)
    logger.info("Created DataLoader for validation set.")

    X_test = SlidingWindowDataset(data_c['X'], data_c['y'], window_size=window_size, horizon=horizon)
    test_loader = DataLoader(X_test, batch_size=batch_size)
    logger.info("Created DataLoader for test set.")

    return train_loader, val_loader, test_loader
    