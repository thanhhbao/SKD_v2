from typing import Dict, Type, Optional, Any
from utils.base_registry import Registry

class PreprocessorRegistry(Registry):
  """
  Registry for preprocessor classes.
  """
  def __init__(self):
    super().__init__('PreprocessorRegistry')

# Create global registry instance
PREPROCESSOR = PreprocessorRegistry() 

def build_preprocessor(name, **kwargs):
  return PREPROCESSOR.get(name, **kwargs)