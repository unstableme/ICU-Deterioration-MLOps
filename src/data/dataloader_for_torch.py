import torch
from torch.utils.data import DataLoader
import pickle

from src.data.data_preprocessing import SlidingWindowDataset
from src.logger import get_logger

logger = get_logger(__name__, log_file='dataloader_for_torch.log')

def get_dataloader(window_size=12, horizon=6, batch_size=64, shuffle=True):
    """Create a DataLoader for the sliding window dataset."""
    
    try:
    
        #../ moves one directory up and following does two directory up
        with open('../..data/processed/set_a_processed.pkl', 'rb') as f:
            data_a = pickle.load(f) #returns in this format {'X': X_a, 'y': y_a}
        with open('../..data/processed/set_b_processed.pkl', 'rb') as f:
            data_b = pickle.load(f)
        with open('../..data/processed/set_c_processed.pkl', 'rb') as f:
            data_c = pickle.load(f)
        
        logger.info("Loaded processed datasets for sets a, b, and c.")

        X_train = SlidingWindowDataset(data_a['X'], data_a['y'], window_size=window_size, horizon=horizon)
        train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=shuffle)
        logger.info("Created DataLoader for training set.")

        X_val = SlidingWindowDataset(data_b['X'], data_b['y'], window_size=window_size, horizon=horizon)
        val_loader = DataLoader(X_val, batch_size=batch_size, shuffle=shuffle)
        logger.info("Created DataLoader for validation set.")

        X_test = SlidingWindowDataset(data_c['X'], data_c['y'], window_size=window_size, horizon=horizon)
        test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=shuffle)
        logger.info("Created DataLoader for test set.")

        return train_loader, val_loader, test_loader
    
    except Exception as e:
        logger.exception(f"Error creating DataLoader: {e}")
        raise