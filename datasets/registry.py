from typing import Dict, Type, Optional, Any, Callable
import torch.nn as nn
from utils.base_registry import Registry
from preprocessor.registry import build_preprocessor

class DatasetRegistry(Registry):
  """
  Registry for dataset classes.
  """
  def __init__(self):
    super().__init__('DatasetRegistry')

  def get_dataset(self, name: str, *args: Any, **kwargs: Any) -> nn.Module:
    return self.get(name, *args, **kwargs)

  def list_dataset(self) -> list:
    return self.list_all()

# Create global registry instance
DATASET_REGISTRY = DatasetRegistry() 

def build_dataset(name, *args, **kwargs):
  transform = build_preprocessor(**kwargs['preprocessor'])
  kwargs['transform'] = transform
  return DATASET_REGISTRY.get_dataset(name, *args, **kwargs)
