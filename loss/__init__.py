from .registry import LOSS, build_loss
from .base_losses import CrossEntropyLoss, FocalLoss  # hoặc các loss bạn cần

__all__ = [
    'LOSS',
    'build_loss',
    'CrossEntropyLoss',
    'FocalLoss',
]
