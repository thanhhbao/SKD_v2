import torch
import torch.nn as nn
import lightning as L

from .registry import build_model
from loss.registry import build_loss
from utils.metrics import MetricsLogger


class ModelWrapper(L.LightningModule):
    def __init__(self, name: str = "", num_labels: int = 1000, lr: float = 1e-4, device: str = 'cuda', **kwargs):
        super().__init__()
        self.lr = float(lr)
        self.weight_decay = float(kwargs.pop('weight_decay', 0.05))
        self.num_labels = num_labels

        # Build model
        model_name = name
        self.model = build_model(model_name, self.num_labels, **kwargs)

        # === Reset classifier to correct output dim ===
        try:
            # Works for standard timm models
            in_features = self.model.get_classifier().in_features
            self.model.reset_classifier(self.num_labels)
        except:
            # Handle models with get_classifier() = Identity (e.g., EfficientViT)
            dummy = torch.randn(1, 3, 224, 224).to(device)
            self.model = self.model.to(device)
            with torch.no_grad():
                features = self.model.forward_features(dummy)
            in_features = features.shape[1]
            self.model.head = nn.Linear(in_features, self.num_labels)

        # Build loss
        loss_name = kwargs.pop('loss_name', 'CrossEntropyLoss')
        loss_kwargs = kwargs.pop('loss_kwargs', {})
        self.loss = build_loss(loss_name, **loss_kwargs)

        # Build metrics
        metrics_config = kwargs.get('_metrics_config', {})
        self.metrics_logger = MetricsLogger(metrics_config, device=device)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        x, y = batch['pixel_values'], batch['label'].long()
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.metrics_logger.update('train', torch.softmax(y_pred, dim=1)[:, 1], y.float())  # prob(class=1)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch['pixel_values'], batch['label'].long()
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.metrics_logger.update('val', torch.softmax(y_pred, dim=1)[:, 1], y.float())
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch['pixel_values'], batch['label'].long()
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)
        self.metrics_logger.update('test', torch.softmax(y_pred, dim=1)[:, 1], y.float())
        return {'loss': loss}

    def on_train_epoch_end(self):
        self.metrics_logger.compute_and_log('train', self.log, prefix='epoch_')
        self.metrics_logger.reset('train')

    def on_validation_epoch_end(self):
        self.metrics_logger.compute_and_log('val', self.log, prefix='epoch_')
        self.metrics_logger.reset('val')

    def on_test_epoch_end(self):
        self.metrics_logger.compute_and_log('test', self.log, prefix='epoch_')
        self.metrics_logger.reset('test')
