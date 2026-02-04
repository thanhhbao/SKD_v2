import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, Any, Callable, Optional

from .registry import DATASET_REGISTRY

class ISIC2016Dataset(Dataset):
  """
  Dataset class for ISIC 2016 skin lesion dataset
  """
  def __init__(self, csv_path: str, 
              img_dir: str, 
              img_path_col: str, 
              target_col: Optional[str] = None,
              transform=None, 
              img_ext: str='jpg',
              img_size: int=224,
              is_train: bool=False,
              **kwargs
              ):
    """
    Initialize ISIC2016 dataset
    
    Args:
      csv_path: Path to CSV file with dataset metadata
      img_dir: Directory containing images
      img_path_col: Column name in CSV containing image path/id
      target_col: Column name in CSV containing target labels
      transform: Optional transform to apply to images
      img_ext: Image file extension
      img_size: Size to resize images to
      is_train: Train mode
      **kwargs: Additional arguments
    """
    self.df = pd.read_csv(csv_path)
    self.img_dir = img_dir
    self.img_path_col = img_path_col
    self.target_col = target_col
    self.img_ext = img_ext
    self.img_size = img_size
    self.is_train = False
    self.transform = self.setup_transform(transform)
  
  def setup_transform(self, transform: Callable=None) -> Callable:
    if transform:
      return transform
    raise Exception(f'Must define transformation for input image.')

  def __len__(self) -> int:
    """
    Get dataset length
    
    Returns:
      Number of samples in dataset
    """
    return len(self.df)
  
  def __getitem__(self, idx: int) -> Dict[str, Any]:
    """
    Get dataset item
    
    Args:
      idx: Item index
        
    Returns:
      Dict containing image and label
    """
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    img_path = os.path.join(self.img_dir, f'{self.df.iloc[idx][self.img_path_col]}.{self.img_ext}')
    image = Image.open(img_path).convert('RGB')

    # transformation
    # if self.transform:
    image = self.transform(image, self.is_train)
    
    label = self.df.iloc[idx][self.target_col]
    label = int(label == 'malignent') if isinstance(label, str) else int(label)
    label = [1, 0] if label == 0 else [0, 1]

    return {'pixel_values': image, 'label': torch.tensor(label, dtype=torch.float)}

@DATASET_REGISTRY.register()
def load_isic2016(*args, **kwargs):
  """
  Load ISIC 2016 dataset
  
  Returns:
      Dict containing train and test datasets
  """
  dataset = {}
  dataset['train'] = ISIC2016Dataset(csv_path=kwargs['train_csv_path'], 
                                  img_dir=kwargs['train_img_dir'], 
                                  img_path_col=kwargs['train_img_col'],
                                  target_col=kwargs['train_target_col'],
                                  is_train=True,
                                  **kwargs)
  dataset['test'] = ISIC2016Dataset(csv_path=kwargs['test_csv_path'], 
                                  img_dir=kwargs['test_img_dir'], 
                                  img_path_col=kwargs['test_img_col'],
                                  target_col=kwargs['test_target_col'],
                                  is_train=False,
                                  **kwargs)
  return dataset 