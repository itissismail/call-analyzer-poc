import torch
print(torch.backends.mps.is_available())      # True if supported
print(torch.backends.mps.is_built())          # True if PyTorch is MPS-enabled