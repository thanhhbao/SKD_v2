import torch
import torch.nn as nn
import lightning as L
from typing import Iterable
from lightning.pytorch import seed_everything
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class TrainerWrapper:
    """
    A wrapper around PyTorch Lightning's Trainer for consistent interface and setup.
    """
    def __init__(self, seed: int=42, checkpoint: str=None, config=None, **kwargs):
        """
        Initialize the trainer wrapper.
        """
        self.config = config  # Lưu config để sử dụng trong evaluate
        self.trainer = L.Trainer(**kwargs)
        self.checkpoint = checkpoint
        seed_everything(seed, workers=True)

    def fit(self, model: nn.Module, train_dataloader: Iterable, val_dataloader: Iterable):
        """
        Train the model.
        """
        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=self.checkpoint)

    def evaluate_with_thresholds(self, model: nn.Module, val_dataloader: Iterable):
        """
        Evaluate model on validation set with multiple thresholds to find the best one.
        Uses config for thresholds, metric_priority, and average.
        Assumes binary classification; outputs logits[:,1] for positive class.
        Returns best threshold and results dict.
        """
        # Lấy từ config (nếu không có, dùng default)
        metrics_config = self.config.get('metrics', {})
        thresholds = metrics_config.get('thresholds', np.arange(0.1, 1.0, 0.1))
        metric_priority = metrics_config.get('metric_priority', 'f1')
        average = metrics_config.get('average', 'binary')

        model.eval()  # Eval mode
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in val_dataloader:
                x = batch['pixel_values'].to(model.device)
                y = batch['label'].view(-1).long().cpu().numpy()
                outputs = model(x)  # (B, 2)
                probs_pos = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs_pos)
                all_targets.extend(y)

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        print("prob stats:", all_probs.min(), all_probs.max(), all_probs.mean())
        print("target mean:", all_targets.mean())


        best_threshold = None
        best_score = 0
        results = {}

        for th in thresholds:
            preds_binary = (all_probs >= th).astype(int)
            metrics = {
                'accuracy': accuracy_score(all_targets, preds_binary),
                'precision': precision_score(all_targets, preds_binary, average='binary', pos_label=1, zero_division=0),
                'recall': recall_score(all_targets, preds_binary, average='binary', pos_label=1, zero_division=0),
                'f1': f1_score(all_targets, preds_binary, average='binary', pos_label=1, zero_division=0),
                'confusion_matrix': confusion_matrix(all_targets, preds_binary).tolist()
            }
            score = metrics[metric_priority]
            results[th] = metrics
            if score > best_score:
                best_score = score
                best_threshold = th

        print(f"Best threshold based on {metric_priority}: {best_threshold} (score: {best_score:.4f})")

        # Log vào TensorBoard nếu có
        if self.trainer.logger:
            loggers = self.trainer.logger
            if not isinstance(loggers, list):
                loggers = [loggers]
            for logger in loggers:
                if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_scalar"):
                    for th, mets in results.items():
                        logger.experiment.add_scalar(f"threshold_{th}/{metric_priority}", mets[metric_priority])

        return best_threshold, results
