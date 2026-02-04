from typing import Dict, Type, Optional, Any, Callable

class Registry:
  """
  Registry template.
  """
  def __init__(self, name: str='Module'):
    self.name = name
    self._modules: Dict[str, Type[Any]] = {}

  def register(self, arg1: Optional[Any] = None) -> Callable:
    """
    Register a module class, supporting custom name.
    Usage:
    - @registry.register  # Uses class.__name__
    - @registry.register('CustomName')  # Uses 'CustomName' as key
    
    Args:
      arg1: Either the class to register or a string for custom name.
        
    Returns:
      Decorator function or the registered class.
    """
    def _register(module_cls: Type[Any]) -> Type[Any]:
      if custom_name:
        module_class_name = custom_name
      else:
        module_class_name = module_cls.__name__
      if module_class_name in self._modules:
        raise ValueError(f"{self.name} {module_class_name} is already registered")
      self._modules[module_class_name] = module_cls
      return module_cls

    if callable(arg1) and not isinstance(arg1, str):  # @register class Foo (arg1 is class)
      custom_name = None
      return _register(arg1)
    elif isinstance(arg1, str):  # @register('Name') -> return decorator
      custom_name = arg1
      return _register
    else:  # @register() -> return _register
      custom_name = None
      return _register

  def get(self, name: str, *args, **kwargs) -> Any:
    """
    Get a module instance by name.
    """
    if name not in self._modules:
      raise KeyError(f"{self.name} {name} is not registered. Available modules: {list(self._modules.keys())}")
    
    module_class = self._modules[name]
    print(f'[BUILD] {self.name}: {name}')
    return module_class(*args, **kwargs)

  def list_all(self) -> list:
    """
    List all registered module names.
    """
    return list(self._modules.keys())