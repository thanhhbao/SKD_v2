import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import LOSS

if 'MSE' not in LOSS._modules:
    @LOSS.register('MSE')
    class MSE:
        """
        Mean Squared Error loss.
        """
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return nn.functional.mse_loss(*args, **kwargs)


@LOSS.register('BinaryCrossEntropy')
class BinaryCrossEntropy:
  """
  Binary Cross Entropy loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.binary_cross_entropy(*args, **kwargs)

@LOSS.register('BinaryCrossEntropyWithLogits')
class BinaryCrossEntropyWithLogits:
  """
  Binary Cross Entropy with Logits loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.binary_cross_entropy_with_logits(*args, **kwargs)
  
@LOSS.register('CrossEntropyLoss')
class CrossEntropyLoss:
  """
  Cross Entropy loss.
  """
  def __init__(self, *args, **kwargs):
    pass
  
  def __call__(self, *args, **kwargs):
    return nn.functional.cross_entropy(*args, **kwargs)

@LOSS.register('FocalLoss')
class FocalLoss(nn.Module):
  def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            raise TypeError("alpha must be float or list")
        self.gamma = gamma
        self.reduction = reduction

  def forward(self, inputs, targets):
        if targets.dim() != 1:
            targets = torch.argmax(targets, dim=1)
        targets = targets.long()

        # Ensure alpha on correct device/dtype
        self.alpha = self.alpha.to(inputs.device).type_as(inputs)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Avoid NaN by clamping pt
        pt = pt.clamp(min=1e-6, max=1.0)

        alpha_t = self.alpha[targets]
        loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # Final reduction
        if self.reduction == 'mean':
            return torch.nanmean(loss)
        elif self.reduction == 'sum':
            return torch.nansum(loss)
        return loss


