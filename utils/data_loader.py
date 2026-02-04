import os
from typing import Iterable

import torch
from torch.utils.data import DataLoader
import lightning as L
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize
import numpy as np

from datasets.registry import build_dataset

class DataLoaderWrapper(L.LightningDataModule):
  """
  Lightning DataModule that handles dataset loading, processing and creating dataloaders.
  """
  def __init__(self,
              dataset_name: str,
              cache_dir: str='.cache',
              train_batch_size: int=32,
              test_batch_size: int=32,
              val_batch_size: int=32,
              test_size: float=0.2,
              **kwargs):
    """
    Initialize the DataLoader wrapper.
    """
    super().__init__()

    # init props
    self.dataset_name = dataset_name
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.val_batch_size = val_batch_size
    self.cache_dir = cache_dir
    self.test_size = test_size
    self.kwargs = kwargs
  
  def setup(self, stage=None):
    """
    Set up the dataloaders for the specified stage.
    """
    # Check if this is a HuggingFace dataset or custom dataset
    if self.dataset_name.startswith('hf:'):
      self.setup_hf_dataset()
    else:
      self.setup_base_dataset()
  
  def setup_base_dataset(self):
    """
    Set up a custom dataset from our registry.
    """
    # load dataset
    self.dataset = build_dataset(self.dataset_name, **self.kwargs['dataset_kwargs'])

  def transforms(self, examples):
    """
    Apply transforms to HuggingFace dataset examples.
    """
    examples["pixel_values"] = [self._transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

  def train_dataloader(self):
    """
    Get train dataloader without WeightedRandomSampler.
    """
    train_dataset = self.dataset['train']
    print(f'Loading train dataloader: {len(train_dataset)} samples')
    return DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)

  def test_dataloader(self):
    """
    Get test dataloader.
    """
    print(f'Loading test dataloader: {len(self.dataset["test"])} samples')
    return DataLoader(self.dataset['test'], batch_size=self.test_batch_size)

  def val_dataloader(self):
    """
    Get validation dataloader.
    """
    print(f'Loading validation dataloader: {len(self.dataset["test"])} samples')
    return DataLoader(self.dataset['test'], batch_size=self.val_batch_size)
