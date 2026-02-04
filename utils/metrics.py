import torch
import torchmetrics

class ConfusionMatrix:
    def __init__(self, task, num_classes, device):
        self.metric = torchmetrics.ConfusionMatrix(
            task=task,
            num_classes=num_classes
        ).to(device)

    def update(self, preds, targets):
        if preds.ndim == 2 and preds.shape[1] == 2:
            preds = preds.argmax(dim=1)
        return self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        return self.metric.reset()

           

class MetricsLogger:
    def __init__(self, config, device='cuda'):
        self.device = device
        self.metrics = {}
        self.stages = ['train', 'val', 'test']
        self.config = config

        for stage in self.stages:
            self.metrics[stage] = self._create_metrics_for_stage(config)

        self.all_preds = {stage: [] for stage in self.stages}
        self.all_targets = {stage: [] for stage in self.stages}

    def _create_metrics_for_stage(self, config):
        task = config.get('task', 'binary')
        num_classes = config.get('num_classes', 2)
        average = config.get('average', 'macro')

        metrics_dict = {}

        metrics_dict['accuracy'] = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(self.device)

        if task in ['binary', 'multiclass', 'multilabel']:
            metrics_dict['auroc'] = torchmetrics.AUROC(task=task, num_classes=num_classes, average=average).to(self.device)
            metrics_dict['f1'] = torchmetrics.F1Score(task=task, num_classes=num_classes, average=average).to(self.device)

        if config.get('include_precision', False):
            metrics_dict['precision'] = torchmetrics.Precision(task=task, num_classes=num_classes, average=average).to(self.device)

        if config.get('include_recall', False):
            metrics_dict['recall'] = torchmetrics.Recall(task=task, num_classes=num_classes, average=average).to(self.device)

        if config.get('include_confusion_matrix', False):
            metrics_dict['confusion_matrix'] = ConfusionMatrix(task, num_classes, self.device)

        return metrics_dict

    def update(self, stage: str, y_pred: torch.Tensor, y_true: torch.Tensor):
        if stage not in self.metrics:
            raise ValueError(f"Unknown stage: {stage}")

        self.all_preds[stage].append(y_pred.detach().cpu())
        self.all_targets[stage].append(y_true.detach().cpu())

        for _, metric in self.metrics[stage].items():
            metric.update(y_pred, y_true)

    def compute_and_log(self, stage: str, logger_fn, prefix: str = ''):
        if stage not in self.metrics:
            raise ValueError(f"Unknown stage: {stage}")

        computed_scalar_metrics = {}

        for name, metric in self.metrics[stage].items():
            if name == 'confusion_matrix':
                continue

            value = metric.compute()
            metric_name = f"{prefix}{stage}_{name}"
            logger_fn(metric_name, value, prog_bar=True)
            computed_scalar_metrics[metric_name] = value.item()

        if 'confusion_matrix' in self.metrics[stage]:
            all_preds_stage = torch.cat(self.all_preds[stage]).to(self.device)
            all_targets_stage = torch.cat(self.all_targets[stage]).to(self.device)

            cm_metric = self.metrics[stage]['confusion_matrix']
            cm_metric.update(all_preds_stage, all_targets_stage)
            confusion_matrix_tensor = cm_metric.compute()

            cm_array = confusion_matrix_tensor.int().cpu().numpy()

            for i in range(cm_array.shape[0]):
                for j in range(cm_array.shape[1]):
                    logger_fn(f"{prefix}{stage}_confmat_{i}{j}", cm_array[i, j], prog_bar=False)

            print(f"\n{prefix}{stage}_confusion_matrix:\n{cm_array}")
            cm_metric.reset()

        return computed_scalar_metrics

    def reset(self, stage=None):
        if stage is None:
            for s in self.stages:
                self._reset_stage(s)
        elif stage in self.metrics:
            self._reset_stage(stage)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _reset_stage(self, stage):
        for metric in self.metrics[stage].values():
            metric.reset()
        self.all_preds[stage] = []
        self.all_targets[stage] = []
