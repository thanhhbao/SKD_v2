import os
import yaml
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
  """
  Load configuration from a YAML file.
  """
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
  
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  
  return config

def get_default_config_path() -> str:
  """
  Get the path to the default configuration file.
  """
  return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default_config.yaml')

def save_config(config: Dict[str, Any], output_path: str) -> None:
  """
  Save configuration to a YAML file.
  """
  with open(output_path, 'w') as f:
      yaml.dump(config, f, default_flow_style=False)

def merge_configs(base_config: Dict[str, Any], override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  """
  Merge two configurations, with override_config taking precedence.
  """
  if override_config is None:
    return base_config.copy()
  
  result = base_config.copy()
  
  for key, value in override_config.items():
    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
      result[key] = merge_configs(result[key], value)
    else:
      result[key] = value
  
  return result 