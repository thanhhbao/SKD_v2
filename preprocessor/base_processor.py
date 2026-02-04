import torch
import torchvision.transforms as T
from typing import Dict, Any, Optional, Tuple, Union, List
from PIL import Image

from .registry import PREPROCESSOR

class BasePreprocessor:
  """
  Base class for all transforms.
  """
  def __init__(self, image_size: int = 224, **kwargs):
    self.image_size = image_size
    self.train_transform = None
    self.val_transform = None
    self.test_transform = None
    self._build_transforms()
  
  def _build_transforms(self):
    """
    Build the transforms for training, validation, and testing.
    Should be implemented by subclasses.
    """
    raise NotImplementedError
  
  def __call__(self, img: Union[Image.Image, torch.Tensor], is_train: bool=False) -> Dict[str, Any]:
    if is_train:
      transform = self.train_transform
    else:  # test, val
      transform = self.test_transform
    return transform(img)

@PREPROCESSOR.register()
class BasicImagePreprocessor(BasePreprocessor):
  """
  Basic image transform with minimal augmentation.
  """
  def _build_transforms(self):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Training transforms with some augmentation
    self.train_transform = T.Compose([
      T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
      T.RandomHorizontalFlip(p=0.5),
      T.RandomVerticalFlip(p=0.2),  # thêm flip dọc để tăng diversity
      T.RandomRotation(degrees=15),  # tăng nhẹ rotation
      T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
      T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # tăng translate & scale
      T.ToTensor(),
      normalize,
])

    
    # Validation transforms (no augmentation)
    self.val_transform = T.Compose([
      T.Resize((self.image_size, self.image_size)),
      T.ToTensor(),
      normalize,
    ])
    
    # Test transforms (same as validation)
    self.test_transform = self.val_transform