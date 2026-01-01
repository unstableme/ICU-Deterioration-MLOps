import os
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
from src.model.model_architecture import CNN_GRU_Model
from src.data.dataloader_for_torch import get_dataloader
from src.config import load_params
from src.logger import get_logger

logger = get_logger(__name__, log_file='train_utils.log')
PARAMS = load_params()

def compute_pos_weight(train_loader, device):
    """Returns the weight of class..basically there is less data in 
       positive class and it returns by how much. spoiler: around 14% of data is only positive"""
    
    try:
        all_labels = []

        for _, y in train_loader:
            all_labels.append(y)

        y = torch.cat(all_labels).to(device)
        num_pos = (y==1).sum()
        num_neg = (y==0).sum()

        pos_weight = (num_neg / num_pos).float()
        logger.info(f"Computed positive class weight: {pos_weight.item()}")
        return pos_weight
    
    except Exception:
        logger.exception("Error computing positive class weight.")
        raise

class Trainer:
    """Trainer class to handle training, validation, and testing of the model."""
    def __init__(self, train_loader, val_loader, test_loader, lr=PARAMS['training']['learning_rate']):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN_GRU_Model().to(self.device)
        logger.info(f"Model initialized on device: {self.device}")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
  
        pos_weight = compute_pos_weight(train_loader, self.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device))
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=float(PARAMS['training']['weight_decay'])
        )

        self.best_threshold = 0.41

        self.artifact_dir = PARAMS['artifacts']['model_dir']
        os.makedirs(self.artifact_dir, exist_ok=True)

        logger.info(f"Initialized Trainer with best_threshold={self.best_threshold}")

    
    def train_one_epoch(self):
        """The baseline training function for one epoch."""

        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in self.train_loader:

            try:
                x_batch, y_batch = x_batch.to(self.device).float(), y_batch.to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()

                logits = self.model(x_batch)
                loss = self.loss_fn(logits, y_batch)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * x_batch.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)

            except Exception:
                logger.exception("Error during training step.")
                raise

        train_acc = train_correct / train_total
        train_loss /= train_total

        return {"Train_Acc": train_acc, "Train_Loss":train_loss}


    def val_one_epoch(self):
        """The baseline validation function for one epoch."""

        self.model.eval()
        val_loss = 0.0

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                try:
                    x_batch, y_batch = x_batch.to(self.device).float(), y_batch.to(self.device).float().unsqueeze(1)
                    assert not torch.isnan(x_batch).any()

                    logits = self.model(x_batch)
                    loss = self.loss_fn(logits, y_batch)
                    val_loss += loss.item() * x_batch.size(0)

                    probs = torch.sigmoid(logits)
                    preds = (probs >= self.best_threshold).int()

                    tp += ((preds == 1) & (y_batch == 1)).sum().item()
                    fp += ((preds == 1) & (y_batch == 0)).sum().item()
                    fn += ((preds == 0) & (y_batch == 1)).sum().item()
                    tn += ((preds == 0) & (y_batch == 0)).sum().item()

                    all_probs.append(probs.cpu())
                    all_labels.append(y_batch.cpu())

                except Exception:
                    logger.exception("Error during validation step.")
                    raise

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        thresholds = np.linspace(0.05, 0.95, 91)
        target_recall = 0.75

        best_precision = 0.0
        best_t = self.best_threshold  # fallback

        labels = all_labels.astype(np.int8)

        for t in thresholds:
            preds = (all_probs >= t).astype(np.int8)

            tp_t = np.sum((preds == 1) & (labels == 1))
            fp_t = np.sum((preds == 1) & (labels == 0))
            fn_t = np.sum((preds == 0) & (labels == 1))

            recall_t = tp_t / (tp_t + fn_t + 1e-8)
            if recall_t >= target_recall:
                precision_t = tp_t / (tp_t + fp_t + 1e-8)
                if precision_t > best_precision:
                    best_precision = precision_t
                    best_t = t


        self.best_threshold = best_t

        val_loss /= (tp + tn + fp + fn)

        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        val_acc = (tp + tn) / (tp + tn + fp + fn)

        val_prauc = average_precision_score(all_labels, all_probs)
        val_auc = roc_auc_score(all_labels, all_probs)

        return {
            "Val_Acc": val_acc,
            "ROC_AUC": val_auc,
            "PR_AUC": val_prauc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "Val_Loss": val_loss,
            "best_threshold": self.best_threshold
        }


    def train_val_epochs(self, epochs):
        """Train and validate the model for a given number of epochs."""
        for epoch in range(epochs):
            train_metrics = self.train_one_epoch()
            val_metrics = self.val_one_epoch()

            # Epoch-level logging
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_metrics['Train_Loss']:.4f}, "
                f"Train Acc: {train_metrics['Train_Acc']:.4f} | "
                f"Val Loss: {val_metrics['Val_Loss']:.4f}, "
                f"Val Acc: {val_metrics['Val_Acc']:.4f}, "
                f"Recall: {val_metrics['Recall']:.4f}, "
                f"Precision: {val_metrics['Precision']:.4f}, "
                f"F1: {val_metrics['F1']:.4f}, "
                f"ROC_AUC: {val_metrics['ROC_AUC']:.4f}, "
                f"PR_AUC: {val_metrics['PR_AUC']:.4f}, "
                f"Best Threshold: {val_metrics['best_threshold']:.3f}"
            )

            logger.info(f"Threshold {self.best_threshold:.3f} will be applied from next epoch")

        logger.info(
            f"Validation Complete after {epochs} epochs. Best threshold: {self.best_threshold:.3f}, "
            f"Recall: {val_metrics['Recall']}, Precision: {val_metrics['Precision']}, "
            f"F1: {val_metrics['F1']}, acc: {val_metrics['Val_Acc']},"
            f" PR_AUC: {val_metrics['PR_AUC']}, ROC_AUC: {val_metrics['ROC_AUC']}"
        )
        yield epoch, train_metrics, val_metrics

    
    def test_time(self):
        """The baseline testing function."""
        self.model.eval()
        test_loss = 0.0

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:

                try:
                    x_batch, y_batch = x_batch.to(self.device).float(), y_batch.to(self.device).float().unsqueeze(1)

                    logits = self.model(x_batch)
                    loss = self.loss_fn(logits, y_batch)
                    test_loss += loss.item() * x_batch.size(0)

                    probs = torch.sigmoid(logits)
                    preds = (probs >= self.best_threshold).int()

                    tp += ((preds == 1) & (y_batch == 1)).sum().item()
                    fp += ((preds == 1) & (y_batch == 0)).sum().item()
                    fn += ((preds == 0) & (y_batch == 1)).sum().item()
                    tn += ((preds == 0) & (y_batch == 0)).sum().item()

                    all_probs.append(probs.cpu())
                    all_labels.append(y_batch.cpu())

                except Exception:
                    logger.exception("Error during testing step.")
                    raise

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        test_acc = (tp + tn) / (tp + tn + fp + fn)

        test_prauc = average_precision_score(all_labels, all_probs)
        test_auc = roc_auc_score(all_labels, all_probs)

        logger.info(
            f"Test Complete. Test Acc: {test_acc}, ROC_AUC: {test_auc}, PR_AUC: {test_prauc}, "
            f"Recall: {recall}, Precision: {precision}, F1: {f1}"
        )

        return {
            "Test_Acc": test_acc,
            "ROC_AUC": test_auc,
            "PR_AUC": test_prauc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1
        }
        
    def save_model(self, filename="cnn_gru_model.pth"):
        """Saves the model state dictionary to the specified filename."""
        try:
            filepath = os.path.join(self.artifact_dir, filename)
            torch.save(self.model.state_dict(), filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception:
            logger.exception("Error saving the model.")
            raise
