import os
import pandas as pd
from src.logger import get_logger
from src.config import load_params

logger = get_logger(__name__, log_file='data_ingestion.log')
PARAMS = load_params()
class ICUDataIngestion:
    def __init__(self, raw_data_dir=PARAMS['data']['raw_data_dir']):
        """Initializes the data ingestion class with paths to raw data directories and outcome files."""
        
        self.raw_data_dir = raw_data_dir

        try:
            if not os.path.exists(self.raw_data_dir):
                raise FileNotFoundError(f"Raw data directory '{self.raw_data_dir}' does not exist.")
            logger.info(f"Initialized ICUDataIngestion with raw data directory: {self.raw_data_dir}")

            #dict inside dict to hold folder paths and outcome file paths
            self.sets = {
                'a': {
                    'folder': os.path.join(self.raw_data_dir, 'set-a'),
                    'outcomes_file': os.path.join(self.raw_data_dir, 'outcomes-a.txt')
                },
                'b': {
                    'folder': os.path.join(self.raw_data_dir, 'set-b'),
                    'outcomes_file': os.path.join(self.raw_data_dir, 'outcomes-b.txt')
                },
                'c': {
                    'folder': os.path.join(self.raw_data_dir, 'set-c'),
                    'outcomes_file': os.path.join(self.raw_data_dir, 'outcomes-c.txt')
                }
            }
            logger.info("Set paths for data ingestion initialized successfully.")

        except Exception as e:
            logger.error(f"Error initializing ICUDataIngestion: {e}")
            raise 

    def get_set_folder(self, set_name):
        """Returns the folder path for the specified set."""

        try:

            if set_name in self.sets:
                return self.sets[set_name]['folder']
            else:
                raise ValueError(f"Set name '{set_name}' is not recognized.")

        except Exception as e:
            logger.exception(f"Error getting folder for set '{set_name}': {e}")
            raise 
    
    def get_patient_files(self, set_name):
        """Returns a list of patient file paths for the specified set."""

        try: 

            folder_path = self.get_set_folder(set_name)
            return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        except Exception as e:
            logger.exception(f"Error getting patient files for set '{set_name}': {e}")
            raise


    def get_outcomes_file(self, set_name):
        """Returns the outcomes file path for the specified set."""

        try:
   
            if set_name in self.sets:
                df = pd.read_csv(self.sets[set_name]['outcomes_file'])
                return df
            else:
                raise ValueError(f"Set name '{set_name}' is not recognized.")
        
        except Exception as e:
            logger.exception(f"Error getting outcomes file for set '{set_name}': {e}")
            raise


