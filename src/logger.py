import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name:str, log_file:str='app.log',
               console_level:int=logging.INFO,
               file_level:int=logging.DEBUG,
               max_bytes:int=5*1024*1024,
               backup_count:int=3
               ):
    """Create and return a logger with both console and file handlers.
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        console_level (int): Logging level for console output.
        file_level (int): Logging level for file output.
        max_bytes (int): Maximum size of log file before rotation.
        backup_count (int): Number of backup files to keep. 
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  
    
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(os.path.join(LOG_DIR, log_file), maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger