import torch
import torch.nn as nn
import timm
from typing import Any
from .registry import BACKBONE

class ConvNeXt(nn.Module):
  def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.2):
    super().__init__()
    self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    self.dropout = nn.Dropout(dropout_rate)
    self.classifier = nn.Linear(self.backbone.num_features, num_labels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.backbone(x)
    x = self.dropout(x)
    logits = self.classifier(x)
    return logits

# Subclass registrations
@BACKBONE.register()
class ConvNeXtV2Tiny(ConvNeXt):
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    dropout_rate = kwargs.pop("dropout_rate", 0.2)
    super().__init__('convnextv2_tiny', num_labels, dropout_rate)

@BACKBONE.register()
class ConvNeXtV2Small(ConvNeXt):
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    dropout_rate = kwargs.pop("dropout_rate", 0.2)
    super().__init__('convnextv2_small', num_labels, dropout_rate)

@BACKBONE.register()
class ConvNeXtV2Pico(ConvNeXt):
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    dropout_rate = kwargs.pop("dropout_rate", 0.2)
    super().__init__('convnextv2_pico', num_labels, dropout_rate)

@BACKBONE.register()
class ConvNeXtV2Nano(ConvNeXt):
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    dropout_rate = kwargs.pop("dropout_rate", 0.2)
    super().__init__('convnextv2_nano', num_labels, dropout_rate)

@BACKBONE.register()
class ConvNeXtV2Base(ConvNeXt):
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    dropout_rate = kwargs.pop("dropout_rate", 0.2)
    super().__init__('convnextv2_base', num_labels, dropout_rate)

@BACKBONE.register()
class ConvNeXtV2Huge(ConvNeXt):
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    dropout_rate = kwargs.pop("dropout_rate", 0.2)
    super().__init__('convnextv2_huge', num_labels, dropout_rate)
