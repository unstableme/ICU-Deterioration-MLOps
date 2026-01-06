from src.data.dataloader_for_torch import get_dataloader
from src.model.train_utils import Trainer
from src.config import load_params
from src.logger import get_logger
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import os 

#import dagshub
# dagshub.init(repo_owner='unstableme',
#              repo_name='ICU-Deterioration-MLOps',
#              mlflow=True
#             )

dagshub_token = os.environ.get("DAGSHUB_TOKEN")
dagshub_user = os.environ.get("DAGSHUB_USER")

print(f"DAGSHUB_USER inside container: {dagshub_user}")
print(f"DAGSHUB_TOKEN present: {'Yes' if dagshub_token else 'No'}")

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/unstableme/ICU-Deterioration-MLOps.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user or ""
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token or ""

if not dagshub_token or not dagshub_user:
    raise ValueError("DAGSHUB_TOKEN and DAGSHUB_USER environment variables must be set")


logger = get_logger(__name__, log_file='Entrypoints_train.log')
PARAMS = load_params()
client = MlflowClient()

def main():
    """Main function to orchestrate model training. """
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    logger.info(f"MLflow Tracking URI set to: {os.environ['MLFLOW_TRACKING_URI']}") 
    
    mlflow.set_experiment("ICU_Deterioration_Prediction_CNN_GRU")

    train_loaders, val_loaders, test_loaders = get_dataloader()
    trainer = Trainer(train_loaders, val_loaders, test_loaders)
    
    run_name = f"CNN_GRU_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        current_run_id = run.info.run_id
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
            "batch_size": PARAMS['data']['batch_size'],
            "window_size": PARAMS['data']['window_size'],
            "horizon": PARAMS['data']['horizon'],
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
        
        MODEL_NAME = "CNN_GRU_ICU_Deterioration_Model"
        if (test_metrics['PR_AUC'] >= PARAMS['mlflow_model_registration']['mlflow_pr_auc_threshold'] 
            and test_metrics['Recall'] >= PARAMS['mlflow_model_registration']['mlflow_recall_threshold']
            and test_metrics['ROC_AUC'] >= PARAMS['mlflow_model_registration']['mlflow_roc_auc_threshold']
            ):

            logger.info("Model passed the first criteria and now is being checked for second criteria.")

            def get_metrics_from_staged_model(model_name, client):
                """Retrieve metrics from the latest model version in 'Staging' stage, 
                   later to be compared against current model's metrics."""
                try:
                    versions = client.get_latest_versions(model_name, stages=["Staging"])
                    if not versions:
                        logger.info("No versions found in 'Staging' stage.")
                        return None
                    staged_run_id = versions[0].run_id
                    run = mlflow.get_run(staged_run_id)
                    pr_auc = run.data.metrics.get("Test_PR_AUC")
                    roc_auc = run.data.metrics.get("Test_ROC_AUC")
                    return {"PR_AUC": pr_auc, "ROC_AUC": roc_auc}
                
                except Exception as e:
                    logger.exception(f"Error retrieving metrics from staged model: {e}")
                    return None

            staged_model_metrics = get_metrics_from_staged_model(MODEL_NAME, client)
            should_register = False
            if staged_model_metrics is None:
                should_register = True  
                logger.info("No staged model found. Proceeding to register the current model.")
            else:
                staged_model_pr_auc = staged_model_metrics['PR_AUC']
                staged_model_roc_auc = staged_model_metrics['ROC_AUC']

                if (test_metrics['PR_AUC'] < staged_model_pr_auc 
                    or test_metrics['ROC_AUC'] < staged_model_roc_auc):
                    logger.info("Current model did not outperform the staged model. Registration aborted.")
                    logger.info(f"Staged Model PR_AUC: {staged_model_pr_auc}, Current Model PR_AUC: {test_metrics['PR_AUC']}")
                    logger.info(f"Staged Model ROC_AUC: {staged_model_roc_auc}, Current Model ROC_AUC: {test_metrics['ROC_AUC']}")
                    should_register = False
                else:
                    should_register = True
                    logger.info("Current model outperformed the staged model. Proceeding with registration & staging.")

            if should_register:
                model_uri = f"runs:/{current_run_id}/mlflow_organized_model"
                result = mlflow.register_model(model_uri, MODEL_NAME)
                logger.info(f"Model registered under the name: {MODEL_NAME}")

                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=result.version,
                    stage="Staging",
                    archive_existing_versions=True
                )
                logger.info(f"Model version {result.version} transitioned to 'Staging' stage.")
            else:
                logger.info("Model did not meet the criteria for registration.")
        else:
            logger.info("Model did not meet the initial criteria for registration based on test metrics. so not registering.")
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred in the main training process.")
        raise