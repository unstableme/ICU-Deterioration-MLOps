from src.data.dataloader_for_torch import get_dataloader
from src.model.train_utils import Trainer
from src.config import load_params
from src.logger import get_logger

logger = get_logger(__name__, log_file='Entrypoints_train.log')

def main():
    """Main function to orchestrate model training. """

    PARAMS = load_params()
    
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
        logger.exception("An error occurred in the main training process.")
        raise