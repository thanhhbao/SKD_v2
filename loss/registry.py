from utils.base_registry import Registry

class LossRegistry(Registry):
  """
  Registry for loss classes.
  """
  def __init__(self):
    super().__init__('LossRegistry')

# Create global registry instance
LOSS = LossRegistry() 

def build_loss(loss_name: str, **kwargs):
  return LOSS.get(loss_name, **kwargs)