from src.data.dataloader_for_torch import get_dataloader
from src.model.train_utils import Trainer
from src.config import load_params
from src.logger import get_logger
import mlflow
import dagshub

import dagshub
dagshub.init(repo_owner='unstableme', repo_name='ICU-Deterioration-MLOps', mlflow=True)

logger = get_logger(__name__, log_file='Entrypoints_train.log')
PARAMS = load_params()

def main():
    """Main function to orchestrate model training. """
    # mlflow.set_tracking_uri("https://dagshub.com/unstableme/ICU-Deterioration-MLOps.mlflow")
    # dont need this line because of above dagshub init mlflow true line
    
    mlflow.set_experiment("ICU_Deterioration_Prediction_CNN_GRU")

    train_loaders, val_loaders, test_loaders = get_dataloader()
    trainer = Trainer(train_loaders, val_loaders, test_loaders)

    with mlflow.start_run(run_name="CNN_GRU_Training"):
        logger.info("MLflow run started for CNN_GRU_Training")
    
        mlflow.set_tags({
            "Title": "CNN_GRU Model Training",
            "Author": "Santosh Sapkota",
            "Version": "1.0",
            "Description": "Training a CNN_GRU model for ICU patient deterioration prediction.",
            "model_type": "1D-CNN_GRU",
            "dataset": "ICU Patient Data from Physionet 2012 Challenge",
            "framework": "PyTorch",
            "tracking_uri": "DagsHub"
        })

        mlflow.log_params({
            "learning_rate": PARAMS['training']['learning_rate'],
            "weight_decay": float(PARAMS['training']['weight_decay']),
            "num_epochs": PARAMS['training']['num_epochs'],
            "batch_size": PARAMS['data']['batch_size']
        })
        
        
        for epoch, train_metrics, val_metrics in trainer.train_val_epochs(epochs=PARAMS['training']['num_epochs']):
            mlflow.log_metric("Train_Acc", train_metrics['Train_Acc'], step=epoch)
            mlflow.log_metric("Val_Acc", val_metrics['Val_Acc'], step=epoch)
            mlflow.log_metric("Val_Recall", val_metrics['Recall'], step=epoch)
            mlflow.log_metric("Val_Precision", val_metrics['Precision'], step=epoch)
            mlflow.log_metric("Val_F1", val_metrics['F1'], step=epoch)
            mlflow.log_metric("Best_Threshold", trainer.best_threshold, step=epoch)
            mlflow.log_metric("Val_ROC_AUC", val_metrics['ROC_AUC'], step=epoch)
            mlflow.log_metric("Val_PR_AUC", val_metrics['PR_AUC'], step=epoch)
        

        test_metrics = trainer.test_time()
        mlflow.log_metric("Test_Acc", test_metrics['Test_Acc'])
        mlflow.log_metric("Test_Recall", test_metrics['Recall'])
        mlflow.log_metric("Test_Precision", test_metrics['Precision'])
        mlflow.log_metric("Test_F1", test_metrics['F1'])
        mlflow.log_metric("Test_ROC_AUC", test_metrics['ROC_AUC'])
        mlflow.log_metric("Test_PR_AUC", test_metrics['PR_AUC'])

        trainer.save_model()
        logger.info("Model training and saving completed.")

        mlflow.log_artifact("artifacts/cnn_gru_model.pth", artifact_path="raw_model")
        mlflow.pytorch.log_model(trainer.model, artifact_path="mlflow_organized_model")
        logger.info("Model and artifacts logged to MLflow.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred in the main training process.")
        raise